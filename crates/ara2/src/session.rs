use crate::{Endpoint, error::Error};
use ara2_sys::{araclient, dv_endpoint, dv_product_version, dv_session, dv_version};
use std::{collections::HashMap, ffi::c_char, net::Ipv4Addr, sync::Arc};

/// The Ara-2 proxy supports session connections over either a TCP or UNIX
/// named socket.
#[derive(Debug, Clone, Copy)]
pub enum SocketType {
    Tcp,
    Unix,
}

/// Internal session state holding the FFI handles.
/// This is wrapped in Arc to enable shared ownership across Session, Endpoint,
/// and Model.
pub(crate) struct SessionInner {
    pub(crate) lib: araclient,
    pub(crate) ptr: *mut dv_session,
    pub(crate) socket_type: SocketType,
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
/// let session = Session::create_via_unix_socket("/var/run/ara2.sock")?;
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
            .finish()
    }
}

impl Session {
    /// Connect to the ARA-2 proxy via a UNIX socket.
    ///
    /// # Arguments
    /// * `socket_path` - Path to the UNIX socket (e.g., `/var/run/ara2.sock`)
    pub fn create_via_unix_socket(socket_path: &str) -> Result<Self, Error> {
        let lib = crate::open_library()?;
        let mut ptr: *mut dv_session = std::ptr::null_mut();
        let socket_path = std::ffi::CString::new(socket_path)
            .map_err(|_| Error::NullPointer("Socket path contains null byte".to_owned()))?;
        let err = unsafe {
            lib.dv_session_create_via_unix_socket(socket_path.as_ptr() as *const c_char, &mut ptr)
        };

        if err != 0 {
            return Err(err.into());
        }

        Ok(Self {
            inner: Arc::new(SessionInner {
                lib,
                ptr,
                socket_type: SocketType::Unix,
            }),
        })
    }

    /// Connect to the ARA-2 proxy via a TCP/IPv4 socket.
    ///
    /// # Arguments
    /// * `ip` - IPv4 address of the proxy host
    /// * `port` - TCP port number
    pub fn create_via_tcp_ipv4_socket(ip: Ipv4Addr, port: u16) -> Result<Self, Error> {
        let lib = crate::open_library()?;
        let mut ptr: *mut dv_session = std::ptr::null_mut();
        let ip_cstring = std::ffi::CString::new(ip.to_string())
            .map_err(|_| Error::NullPointer("Invalid IP address".to_owned()))?;
        let err = unsafe {
            lib.dv_session_create_via_tcp_ipv4_socket(ip_cstring.as_ptr(), port as i32, &mut ptr)
        };

        if err != 0 {
            return Err(err.into());
        }

        Ok(Self {
            inner: Arc::new(SessionInner {
                lib,
                ptr,
                socket_type: SocketType::Tcp,
            }),
        })
    }

    /// Get version information for all components (proxy, firmware, drivers,
    /// etc.).
    pub fn versions(&self) -> Result<HashMap<String, String>, Error> {
        let mut products: *mut dv_product_version = std::ptr::null_mut();
        let mut count: u8 = 0;
        let err = unsafe {
            self.inner
                .lib
                .dv_retrieve_version_details(self.inner.ptr, &mut products, &mut count)
        };
        if err != 0 {
            return Err(err.into());
        }

        let mut versions = HashMap::new();
        for i in 0..count {
            let product = unsafe { products.offset(i as isize) };
            unsafe {
                match (*product).product_type {
                    ara2_sys::DV_PRODUCT_TYPE_PROXY => versions.insert(
                        "proxy".to_owned(),
                        version_to_string((*product).product_version),
                    ),
                    ara2_sys::DV_PRODUCT_TYPE_PCI_DRIVER => versions.insert(
                        "pci_driver".to_owned(),
                        version_to_string((*product).product_version),
                    ),
                    ara2_sys::DV_PRODUCT_TYPE_FIRMWARE => versions.insert(
                        "firmware".to_owned(),
                        version_to_string((*product).product_version),
                    ),
                    ara2_sys::DV_PRODUCT_TYPE_CNN_MODEL => versions.insert(
                        "cnn_model".to_owned(),
                        version_to_string((*product).product_version),
                    ),
                    ara2_sys::DV_PRODUCT_TYPE_LLM_MODEL => versions.insert(
                        "llm_model".to_owned(),
                        version_to_string((*product).product_version),
                    ),
                    ara2_sys::DV_PRODUCT_TYPE_CLIENT_LIB => versions.insert(
                        "client_lib".to_owned(),
                        version_to_string((*product).product_version),
                    ),
                    ara2_sys::DV_PRODUCT_TYPE_SYSAPI => versions.insert(
                        "sysapi".to_owned(),
                        version_to_string((*product).product_version),
                    ),
                    t => {
                        return Err(Error::UnknownProductType(t));
                    }
                };
            }
        }

        Ok(versions)
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
            return Err(err.into());
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

fn version_to_string(version: dv_version) -> String {
    format!(
        "{}.{}.{}.{}",
        version.major, version.minor, version.patch, version.patch_minor
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unix_socket_connect() {
        let session = Session::create_via_unix_socket(crate::DEFAULT_SOCKET)
            .expect("should connect to ARA-2 proxy");
        assert!(matches!(session.socket_type(), SocketType::Unix));
    }

    #[test]
    fn test_unix_socket_invalid_path() {
        let result = Session::create_via_unix_socket("/nonexistent/ara2.sock");
        assert!(result.is_err(), "should fail with invalid socket path");
    }

    #[test]
    fn test_versions_returns_expected_keys() {
        let session = crate::tests::test_session();
        let versions = session.versions().expect("should retrieve versions");

        // The proxy and client_lib versions should always be present
        assert!(versions.contains_key("proxy"), "missing proxy version");
        assert!(
            versions.contains_key("client_lib"),
            "missing client_lib version"
        );

        // All version strings should be non-empty
        for (key, value) in &versions {
            assert!(!value.is_empty(), "version for '{key}' should not be empty");
        }
    }
}
