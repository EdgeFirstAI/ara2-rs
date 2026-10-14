use crate::{Abi, ComponentVersion, DvapiVersion, Endpoint, error::Error, status::RawStatus};
use ara2_sys::{araclient, dv_endpoint, dv_product_version, dv_session, dv_status_code_t};
use std::{ffi::c_char, net::Ipv4Addr, sync::Arc};

/// The Ara-2 proxy supports session connections over either a TCP or UNIX
/// named socket.
#[derive(Debug, Clone, Copy)]
pub enum SocketType {
    /// A TCP/IPv4 connection.
    Tcp,
    /// A UNIX domain socket.
    Unix,
}

/// Internal session state holding the FFI handles.
/// This is wrapped in Arc to enable shared ownership across Session, Endpoint,
/// and Model.
pub(crate) struct SessionInner {
    pub(crate) lib: araclient,
    pub(crate) ptr: *mut dv_session,
    pub(crate) socket_type: SocketType,
    /// Probed once when the library is opened. Every struct read through
    /// `lib` is interpreted against this, so it must not be re-derived
    /// per call.
    pub(crate) abi: Abi,
    pub(crate) dvapi_version: Option<DvapiVersion>,
    /// The functions only `abi`'s version has, from the same library.
    pub(crate) vlib: VersionLib,
}

/// The version-specific function table for the loaded library.
pub(crate) enum VersionLib {
    // No call is routed through the 1.1 table yet; it has no detailed-status variants.
    #[allow(dead_code)]
    V1_1(ara2_sys::v1_1::araclient_v1_1),
    V1_3(ara2_sys::v1_3::araclient_v1_3),
    V3_0(ara2_sys::v3_0::araclient_v3_0),
}

impl VersionLib {
    /// Opens the version-specific function table from the same library.
    /// dlopen reference-counts, so this maps no second copy.
    pub(crate) fn open(abi: Abi, name: &str) -> Result<Self, Error> {
        // SAFETY: `name` is the library the common table already loaded.
        unsafe {
            Ok(match abi {
                Abi::V1_1 => VersionLib::V1_1(ara2_sys::v1_1::araclient_v1_1::new(name)?),
                Abi::V1_3 => VersionLib::V1_3(ara2_sys::v1_3::araclient_v1_3::new(name)?),
                Abi::V3_0 => VersionLib::V3_0(ara2_sys::v3_0::araclient_v3_0::new(name)?),
            })
        }
    }
}

impl SessionInner {
    /// A failing status code as an [`Error`], named by the library.
    pub(crate) fn status_error(&self, code: dv_status_code_t) -> Error {
        crate::status::status_error_from(&self.lib, code)
    }

    /// A failing detailed status as an [`Error`], named by the library.
    pub(crate) fn detailed_error(&self, status: RawStatus) -> Error {
        Error::Ara2 {
            code: status.code as i32,
            name: self.status_name(status.code),
            detail: Some(crate::StatusDetail {
                sub_code: status.sub_code,
                component: crate::StatusComponent::from_raw(status.component),
                severity: crate::Severity::from_raw(status.severity),
            }),
        }
    }

    /// The library's own name for `code`.
    pub(crate) fn status_name(&self, code: dv_status_code_t) -> String {
        crate::status::status_name(&self.lib, code)
    }
}

/// `Ok` for a successful detailed status, the named error otherwise.
pub(crate) fn status_result(session: &SessionInner, status: RawStatus) -> Result<(), Error> {
    if status.code == 0 {
        Ok(())
    } else {
        Err(session.detailed_error(status))
    }
}

// Safety: The C library is thread-safe for session operations.
// All FFI calls go through the library handle which is internally synchronized.
unsafe impl Send for SessionInner {}
unsafe impl Sync for SessionInner {}

impl Drop for SessionInner {
    fn drop(&mut self) {
        unsafe {
            self.lib.dv_session_close(self.ptr);
        }
    }
}

/// ARA-2 session - connection to the NPU proxy service.
///
/// Sessions are cheaply cloneable (reference counted). Multiple `Session`
/// handles can exist, all sharing the same underlying connection.
///
/// # Example
/// ```no_run
/// use ara2::Session;
///
/// let session = Session::connect()?;
/// let endpoints = session.list_endpoints()?;
/// # Ok::<(), ara2::Error>(())
/// ```
#[derive(Clone)]
pub struct Session {
    pub(crate) inner: Arc<SessionInner>,
}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session")
            .field("socket_type", &self.inner.socket_type)
            .field("abi", &self.inner.abi)
            .finish()
    }
}

impl Session {
    /// Connect to the ARA-2 proxy, choosing the endpoint by inspection.
    ///
    /// The endpoint is the first of: `ARA2_SOCKET` (a socket path,
    /// `unix:/path` or `tcp:IPV4:PORT`), the first existing entry of
    /// [`crate::SOCKET_PATHS`], or a running proxy found by
    /// [`crate::discover`], preferring UNIX endpoints over TCP. Exactly one
    /// connection attempt is made: a failed attempt is not free, as the
    /// DVAPI 1.1 client keeps state across one and a later create on the
    /// same handle can come back `DV_SESSION_INVALID_HANDLE`.
    ///
    /// This is the recommended way to connect: it lets a deployment pick
    /// the proxy without a recompile. A TCP session cannot pass file
    /// descriptors; see [`Session::supports_fd_passing`].
    pub fn connect() -> Result<Self, Error> {
        let endpoint = crate::proxy_endpoint()?;
        if let crate::ProxyEndpoint::Tcp(ip, port) = &endpoint {
            log::warn!(
                "connecting to the ARA-2 proxy over TCP at {ip}:{port}; shared-memory and \
                 DMA-BUF tensors are unavailable on TCP (set interface_type: SOCKET in the \
                 proxy configuration to enable them)"
            );
        }
        endpoint.connect()
    }

    /// Whether this session can pass file descriptors to the proxy, which
    /// `TensorMemory::Shm` and `TensorMemory::DmaBuf` tensors require. True
    /// for UNIX-socket sessions, false for TCP.
    ///
    /// When false, tensors the client allocates are heap memory with no file
    /// descriptor: `clone_fd()` fails on them and they must be accessed
    /// through `map()`.
    pub fn supports_fd_passing(&self) -> bool {
        matches!(self.inner.socket_type, SocketType::Unix)
    }

    /// Connect to the proxy described by a proxy configuration file.
    ///
    /// Reads the `proxy:` section's `interface_type` and the address list
    /// it selects, then connects to the first endpoint declared -- a UNIX
    /// socket for `SOCKET`, a TCP/IPv4 address for `IPV4`.
    ///
    /// Use this when the proxy's configuration is known but its socket
    /// path is not, which is the usual situation across packagings: NXP's
    /// rt-sdk-ara2 reads `/etc/rt-sdk-ara240/proxy_config.yaml` and the
    /// Kinara runtime meta-kinara packages reads `/etc/ara2.yaml`, and the
    /// two name different sockets.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ara2::Session;
    /// let session = Session::from_config("/etc/ara2.yaml".as_ref())?;
    /// # Ok::<(), ara2::Error>(())
    /// ```
    pub fn from_config(path: &std::path::Path) -> Result<Self, Error> {
        let config = crate::ProxyConfig::read(path)?;
        config
            .endpoints
            .first()
            .ok_or(Error::NoProxyEndpoint { pid: None })?
            .connect()
    }

    /// Connect to the ARA-2 proxy via a UNIX socket.
    ///
    /// # Arguments
    /// * `socket_path` - Path to the UNIX socket (e.g., `/var/run/proxy.sock`)
    pub fn create_via_unix_socket(socket_path: &str) -> Result<Self, Error> {
        let (lib, abi, dvapi_version, name) = crate::open_library()?;
        let vlib = VersionLib::open(abi, name)?;
        let mut ptr: *mut dv_session = std::ptr::null_mut();
        let socket_path = std::ffi::CString::new(socket_path)
            .map_err(|_| Error::NullPointer("Socket path contains null byte".to_owned()))?;
        let err = unsafe {
            lib.dv_session_create_via_unix_socket(socket_path.as_ptr() as *const c_char, &mut ptr)
        };

        if err != 0 {
            return Err(crate::status::status_error_from(&lib, err));
        }

        Ok(Self {
            inner: Arc::new(SessionInner {
                lib,
                ptr,
                socket_type: SocketType::Unix,
                abi,
                dvapi_version,
                vlib,
            }),
        })
    }

    /// Connect to the ARA-2 proxy via a TCP/IPv4 socket.
    ///
    /// # Arguments
    /// * `ip` - IPv4 address of the proxy host
    /// * `port` - TCP port number
    pub fn create_via_tcp_ipv4_socket(ip: Ipv4Addr, port: u16) -> Result<Self, Error> {
        let (lib, abi, dvapi_version, name) = crate::open_library()?;
        let vlib = VersionLib::open(abi, name)?;
        let mut ptr: *mut dv_session = std::ptr::null_mut();
        let ip_cstring = std::ffi::CString::new(ip.to_string())
            .map_err(|_| Error::NullPointer("Invalid IP address".to_owned()))?;
        let err = unsafe {
            lib.dv_session_create_via_tcp_ipv4_socket(ip_cstring.as_ptr(), port as i32, &mut ptr)
        };

        if err != 0 {
            return Err(crate::status::status_error_from(&lib, err));
        }

        Ok(Self {
            inner: Arc::new(SessionInner {
                lib,
                ptr,
                socket_type: SocketType::Tcp,
                abi,
                dvapi_version,
                vlib,
            }),
        })
    }

    /// The DVAPI generation of the `libaraclient` this session is using.
    ///
    /// Probed once when the library was opened. Struct layouts and the
    /// meaning of some endpoint states depend on it, so it is exposed for
    /// diagnostics and for callers that need to interpret a
    /// [`crate::State::Unknown`].
    pub fn abi(&self) -> Abi {
        self.inner.abi
    }

    /// The DVAPI version `dv_get_client_lib_version` reported, or `None`
    /// if the library would not report one.
    ///
    /// This is the interface version, not the version of the SDK the
    /// library was packaged in. See [`DvapiVersion`].
    pub fn dvapi_version(&self) -> Option<DvapiVersion> {
        self.inner.dvapi_version
    }

    /// Versions of the components currently running: proxy, firmware,
    /// client library and so on, as the proxy reports them.
    pub fn versions(&self) -> Result<Vec<ComponentVersion>, Error> {
        // SAFETY: `s` is this session's live handle; the out-pointers are
        // the locals `version_list` passes.
        self.version_list(|s, p, c| unsafe {
            self.inner.lib.dv_exchange_current_version_details(s, p, c)
        })
    }

    /// Every component version the proxy supports, which can list several
    /// versions of the same component (e.g. each firmware it can load).
    pub fn supported_versions(&self) -> Result<Vec<ComponentVersion>, Error> {
        // SAFETY: as in `versions`.
        self.version_list(|s, p, c| unsafe { self.inner.lib.dv_retrieve_version_details(s, p, c) })
    }

    fn version_list(
        &self,
        call: impl FnOnce(*mut dv_session, *mut *mut dv_product_version, *mut u8) -> dv_status_code_t,
    ) -> Result<Vec<ComponentVersion>, Error> {
        let mut products: *mut dv_product_version = std::ptr::null_mut();
        let mut count: u8 = 0;
        let err = call(self.inner.ptr, &mut products, &mut count);
        if err != 0 {
            return Err(self.inner.status_error(err));
        }
        if products.is_null() {
            return Ok(Vec::new());
        }
        // SAFETY: the library returned `count` entries at `products`.
        let list = unsafe { std::slice::from_raw_parts(products, count as usize) };
        let out = crate::version::component_versions(list);
        // SAFETY: `products` came from the call above and is freed once.
        unsafe { self.inner.lib.dv_free_version_details(products) };
        Ok(out)
    }

    /// Set the client library's log verbosity. DVAPI 1.3 and later.
    pub fn set_log_level(&self, level: crate::LogLevel) -> Result<(), Error> {
        let raw = level as u32;
        let inner = &self.inner;
        // SAFETY: the call takes a plain value and touches no handle.
        let code = match &inner.vlib {
            VersionLib::V1_3(v) if v.dv_client_set_log_level.is_ok() => unsafe {
                v.dv_client_set_log_level(raw)
            },
            VersionLib::V3_0(v) if v.dv_client_set_log_level.is_ok() => unsafe {
                v.dv_client_set_log_level(raw)
            },
            VersionLib::V1_1(_) | VersionLib::V1_3(_) | VersionLib::V3_0(_) => {
                return Err(Error::Unsupported {
                    feature: "client log level",
                    abi: inner.abi,
                });
            }
        };
        if code != 0 {
            return Err(inner.status_error(code));
        }
        Ok(())
    }

    /// List all available NPU endpoints connected to the proxy.
    ///
    /// The returned endpoints share ownership of the underlying C endpoint list
    /// buffer, which is automatically freed when all endpoints are dropped.
    pub fn list_endpoints(&self) -> Result<Vec<Endpoint>, Error> {
        let mut endpoints: *mut dv_endpoint = std::ptr::null_mut();
        let mut count: i32 = 0;
        let err = unsafe {
            self.inner
                .lib
                .dv_endpoint_get_list(self.inner.ptr, &mut endpoints, &mut count)
        };

        if err != 0 {
            return Err(self.inner.status_error(err));
        }

        // Wrap the endpoint list pointer so it is freed when all Endpoint
        // handles are dropped.
        let list = Arc::new(EndpointList {
            session: Arc::clone(&self.inner),
            ptr: endpoints,
        });

        let mut result = Vec::new();
        for i in 0..count {
            let endpoint = unsafe { endpoints.offset(i as isize) };
            result.push(Endpoint {
                session: Arc::clone(&self.inner),
                _list: Arc::clone(&list),
                ptr: endpoint,
            });
        }

        Ok(result)
    }

    /// Get the socket type used for this session.
    pub fn socket_type(&self) -> SocketType {
        self.inner.socket_type
    }

    /// Get the number of in-flight inference requests for this session.
    ///
    /// Returns the count of requests submitted via [`crate::Model::submit`] that
    /// the client library has not yet received a response for from the
    /// proxy. Useful for monitoring pipeline depth.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ara2::{Session, DEFAULT_TIMEOUT_MS};
    /// # let session = Session::connect()?;
    /// # let endpoints = session.list_endpoints()?;
    /// # let mut model = endpoints[0].load_model_from_file("m.dvm".as_ref())?;
    /// # model.allocate_tensors(None)?;
    /// let request = model.submit()?;
    /// assert_eq!(session.inflight_count()?, 1);
    ///
    /// let _stats = request.wait(DEFAULT_TIMEOUT_MS)?;
    /// assert_eq!(session.inflight_count()?, 0);
    /// # Ok::<(), ara2::Error>(())
    /// ```
    pub fn inflight_count(&self) -> Result<i32, Error> {
        let mut count: i32 = 0;
        let err = unsafe {
            self.inner
                .lib
                .dv_infer_get_inflight_count(self.inner.ptr, &mut count)
        };
        if err != 0 {
            return Err(self.inner.status_error(err));
        }
        Ok(count)
    }
}

/// Shared ownership of the C-allocated endpoint list buffer.
///
/// Freed via `dv_endpoint_free_group` when the last reference is dropped.
pub(crate) struct EndpointList {
    session: Arc<SessionInner>,
    ptr: *mut dv_endpoint,
}

// Safety: The pointer is only used for the free call on drop.
unsafe impl Send for EndpointList {}
unsafe impl Sync for EndpointList {}

impl Drop for EndpointList {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                self.session.lib.dv_endpoint_free_group(self.ptr);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Component;

    #[test]
    fn log_level_can_be_lowered() {
        let session = crate::tests::test_session();
        if session.abi() == Abi::V1_1 {
            assert!(matches!(
                session.set_log_level(crate::LogLevel::Warn),
                Err(Error::Unsupported { .. })
            ));
            return;
        }
        session.set_log_level(crate::LogLevel::Warn).unwrap();
        session.list_endpoints().unwrap()[0]
            .load_model_from_file(&crate::tests::test_model_path())
            .unwrap();
    }

    #[test]
    fn connect_uses_the_resolved_transport() {
        let endpoint = crate::proxy_endpoint().expect("should resolve a proxy endpoint");
        let session = Session::connect().expect("should connect to ARA-2 proxy");
        println!("connected to {endpoint:?}");
        match endpoint {
            crate::ProxyEndpoint::Unix(_) => {
                assert!(matches!(session.socket_type(), SocketType::Unix));
                assert!(session.supports_fd_passing());
            }
            crate::ProxyEndpoint::Tcp(..) => {
                assert!(matches!(session.socket_type(), SocketType::Tcp));
                assert!(!session.supports_fd_passing());
            }
        }
    }

    #[test]
    fn test_unix_socket_invalid_path() {
        let result = Session::create_via_unix_socket("/nonexistent/proxy.sock");
        assert!(result.is_err(), "should fail with invalid socket path");
    }

    #[test]
    fn failing_calls_name_their_status() {
        let session = crate::tests::test_session();
        let endpoints = session.list_endpoints().unwrap();
        let err = endpoints[0]
            .load_model_from_file("/nonexistent/model.dvm".as_ref())
            .expect_err("loading a missing file must fail");
        println!("load of a missing model: {err}");
        match err {
            Error::Ara2 { name, detail, .. } => {
                assert!(name.starts_with("DV_"), "{name}");
                if session.abi() != crate::Abi::V1_1 {
                    assert!(detail.is_some(), "1.3+ should give a detailed status");
                }
            }
            other => panic!("expected Error::Ara2, got {other:?}"),
        }
    }

    #[test]
    fn test_versions_report_running_components() {
        let session = crate::tests::test_session();
        let running = session
            .versions()
            .expect("should retrieve running versions");
        assert!(
            running.iter().any(|v| v.component == Component::Proxy),
            "{running:?}"
        );
        assert!(
            running.iter().any(|v| v.component == Component::ClientLib),
            "{running:?}"
        );
        let firmware: Vec<_> = running
            .iter()
            .filter(|v| v.component == Component::Firmware)
            .collect();
        assert!(
            firmware.len() <= 1,
            "running firmware should be one entry: {running:?}"
        );
        let supported = session
            .supported_versions()
            .expect("should retrieve supported versions");
        assert!(!supported.is_empty());
    }
}
