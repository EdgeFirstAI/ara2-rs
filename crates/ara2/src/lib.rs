#![warn(missing_docs)]
//! Rust client library for the [Kinara](https://kinara.ai) ARA-2 neural
//! network accelerator.
//!
//! Provides session management, endpoint enumeration, model loading, and
//! inference (both synchronous and asynchronous) on NXP i.MX platforms
//! equipped with ARA-2 PCIe hardware.
//!
//! # Quick Start
//!
//! ```no_run
//! use ara2::{Session, DEFAULT_TIMEOUT_MS};
//! use edgefirst_tensor::{TensorMemory, TensorTrait as _};
//!
//! // Connect to the ARA-2 proxy service. Honors the `ARA2_SOCKET`
//! // environment variable, falling back to the first of `SOCKET_PATHS`
//! // that exists.
//! let session = Session::connect()?;
//! let endpoints = session.list_endpoints()?;
//!
//! // Load model and allocate DMA-backed tensors
//! let mut model = endpoints[0].load_model_from_file("model.dvm".as_ref())?;
//! model.allocate_tensors(Some(TensorMemory::DmaBuf))?;
//!
//! // Synchronous inference
//! let timing = model.run()?;
//! println!("NPU inference: {:?}", timing.run_time);
//! # Ok::<(), ara2::Error>(())
//! ```
//!
//! # Async Inference
//!
//! The [`Model::submit`] / [`InferRequest::wait`] API enables overlapping
//! CPU work with NPU execution — the building block for pipeline
//! parallelism:
//!
//! ```no_run
//! # use ara2::{Session, DEFAULT_TIMEOUT_MS};
//! # let session = Session::connect()?;
//! # let endpoints = session.list_endpoints()?;
//! # let mut model = endpoints[0].load_model_from_file("m.dvm".as_ref())?;
//! # model.allocate_tensors(None)?;
//! // Submit — returns immediately while NPU works
//! let request = model.submit()?;
//!
//! // CPU is free: preprocess next frame, run postprocessing, etc.
//!
//! // Wait for result
//! let timing = request.wait(DEFAULT_TIMEOUT_MS)?;
//! # Ok::<(), ara2::Error>(())
//! ```
//!
//! See [`InferRequest`] for details on ownership, thread safety, and
//! error handling.
//!
//! # Finding the proxy
//!
//! [`Session::connect`] covers the usual case. When the socket path is
//! not known ahead of time -- the two packagings in circulation configure
//! different ones -- there are two other ways in:
//!
//! ```no_run
//! use ara2::{Session, discover};
//!
//! // Ask the running proxies where they listen.
//! for proxy in discover()? {
//!     println!("pid {} at {:?}", proxy.pid, proxy.endpoints);
//! }
//!
//! // Or read it out of a proxy configuration file.
//! let session = Session::from_config("/etc/ara2.yaml".as_ref())?;
//! # Ok::<(), ara2::Error>(())
//! ```
//!
//! # DVAPI generations
//!
//! `libaraclient` ships in two interface generations whose struct layouts
//! differ, and both are supported. [`open_library`](crate) probes the
//! loaded library with `dv_get_client_lib_version` and records the result
//! as an [`Abi`], which [`Session::abi`] reports. This matters to callers
//! mainly through [`State`]: DVAPI 1.3 reuses two endpoint state values
//! for different names, so which of them can be reported depends on the
//! library loaded.

use ara2_sys::{araclient, dv_model_output_param, dv_model_output_param_1_3, dv_version};

mod discover;
pub mod dvm_metadata;
mod endpoint;
mod error;
mod model;
mod session;

pub use discover::{PROXY_COMM, Proxy, ProxyConfig, ProxyEndpoint, discover};
pub use dvm_metadata::{
    Ara2Info, CompilationInfo, DatasetInfo, DeploymentInfo, DvmMetadata, InputSpec, ModelInfo,
    OutputSpec, PpaMetrics, has_metadata, read_labels, read_labels_from_file, read_metadata,
    read_metadata_from_file,
};
pub use endpoint::{DramStatistics, Endpoint, EndpointStatistics, State};
pub use error::Error;
pub use model::{
    DEFAULT_TIMEOUT_MS, InferRequest, InputPreprocess, InputQuantization, InputSet, InputTensor,
    Model, ModelOutputType, ModelTiming, OutputQuantization, OutputSet, OutputTensor,
};
pub use session::{Session, SocketType};

/// Default socket path for the ARA-2 proxy service.
///
/// This matches `interface_socket_file` in NXP's rt-sdk-ara2
/// `proxy_config.yaml`, and is the first entry in [`SOCKET_PATHS`].
/// Override at runtime with the `ARA2_SOCKET` environment variable -- see
/// [`Session::connect`] -- or call [`Session::create_via_unix_socket`]
/// directly with an explicit path.
pub const DEFAULT_SOCKET: &str = "/var/run/proxy.sock";

/// Proxy socket paths tried, in order, when no `ARA2_SOCKET` is set.
///
/// The socket path is a packaging decision, like the library name in
/// [`LIBRARY_NAMES`], and the two packagings in circulation disagree:
/// NXP's rt-sdk-ara2 configures `/var/run/proxy.sock`, and the Kinara
/// runtime that [meta-kinara](https://github.com/EdgeFirstAI/meta-kinara)
/// packages configures `/var/run/ara2.sock`. Trying each lets one binary
/// serve both without a deployment having to supply the path.
pub const SOCKET_PATHS: &[&str] = &[DEFAULT_SOCKET, "/var/run/ara2.sock"];

/// The `ARA2_SOCKET` environment variable, which pins the proxy socket
/// path instead of searching [`SOCKET_PATHS`].
pub const SOCKET_OVERRIDE_ENV: &str = "ARA2_SOCKET";

/// Resolves the ARA-2 proxy socket path: `ARA2_SOCKET` if set, otherwise
/// the first entry in [`SOCKET_PATHS`] that exists, otherwise
/// [`DEFAULT_SOCKET`].
///
/// [`Session::connect`] resolves through this function and then makes a
/// single connection attempt. Selecting by existence rather than by
/// connecting is deliberate: a failed attempt is not free, as the DVAPI
/// 1.1 client keeps state across one and a later create on the same
/// handle can come back `DV_SESSION_INVALID_HANDLE`.
pub fn socket_path() -> String {
    if let Ok(path) = std::env::var(SOCKET_OVERRIDE_ENV) {
        return path;
    }
    SOCKET_PATHS
        .iter()
        .find(|path| std::path::Path::new(path).exists())
        .unwrap_or(&DEFAULT_SOCKET)
        .to_string()
}

/// The architecture-suffixed `libaraclient` name NXP's `imx-nxp-ara2`
/// package ships for this target, tried first in [`LIBRARY_NAMES`].
///
/// NXP's rt-sdk-ara2 packaging has no stable soname: a drop for i.MX
/// 95/8M Plus installs `libaraclient_aarch64.so`, and past drops have used
/// `libaraclient_x86_64.so` on x86_64 hosts and a soname-versioned
/// `libaraclient.so.1` elsewhere.
#[cfg(target_arch = "aarch64")]
const ARCH_LIBRARY_NAME: &str = "libaraclient_aarch64.so";
#[cfg(target_arch = "x86_64")]
const ARCH_LIBRARY_NAME: &str = "libaraclient_x86_64.so";

/// `libaraclient` names tried, in order, when opening the client library.
///
/// NXP has shipped this library under a different name in every SDK drop
/// so far (`libaraclient.so.1`, `libaraclient_aarch64.so`, ...), so a
/// single hardcoded name breaks the moment the packaging changes again.
/// Trying each candidate covers every scheme seen to date with one build.
///
/// On `aarch64` and `x86_64` the architecture-suffixed name NXP ships
/// comes first. Other targets get the unsuffixed names only, so the
/// crate still builds for platforms NXP has not published this scheme
/// for.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
pub const LIBRARY_NAMES: &[&str] = &[ARCH_LIBRARY_NAME, "libaraclient.so", "libaraclient.so.1"];
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
pub const LIBRARY_NAMES: &[&str] = &["libaraclient.so", "libaraclient.so.1"];

/// A DVAPI version, as reported by `dv_get_client_lib_version`.
///
/// This is the version of the `dvapi.h` interface a `libaraclient`
/// implements, which is what governs struct layout. It is not the version
/// of the SDK the library was packaged in -- NXP's rt-sdk-ara2 2.1.1 ships
/// DVAPI 1.3.2, and the Kinara runtime that
/// [meta-kinara](https://github.com/EdgeFirstAI/meta-kinara) packages as
/// 1.2.1 ships DVAPI 1.1.2.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct DvapiVersion {
    /// Major version. Both generations in circulation report `1`.
    pub major: u8,
    /// Minor version. This is what distinguishes the generations, and
    /// what [`Abi`] is keyed on.
    pub minor: u8,
    /// Patch version.
    pub patch: u8,
    /// Sub-patch version.
    pub patch_minor: u8,
}

impl std::fmt::Display for DvapiVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}.{}.{}.{}",
            self.major, self.minor, self.patch, self.patch_minor
        )
    }
}

/// The `dvapi.h` generation a loaded `libaraclient` implements.
///
/// Every layout difference between the generations is a field appended to
/// the end of a struct, so the older layout stays valid for reading fields
/// both share. Two things do not survive that way and are selected here:
/// the stride of an array the client indexes, and the meaning of endpoint
/// states 4 and 5, which 1.3 reuses for different names.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[non_exhaustive]
pub enum Abi {
    /// DVAPI 1.1.x, shipped by the Kinara ARA-2 runtime that
    /// [meta-kinara](https://github.com/EdgeFirstAI/meta-kinara) packages.
    V1_1,
    /// DVAPI 1.3.x, shipped by NXP's rt-sdk-ara2 as `imx-nxp-ara2`.
    V1_3,
}

impl std::fmt::Display for Abi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Abi::V1_1 => f.write_str("1.1"),
            Abi::V1_3 => f.write_str("1.3"),
        }
    }
}

impl Abi {
    /// Byte stride between elements of a `dv_model_output_param` array.
    ///
    /// The array belongs to the library, so its stride is the library's
    /// struct size rather than whichever one the bindings happen to name.
    /// 1.3 appended `has_nms_parent`, taking it from 112 to 120 bytes.
    pub(crate) const fn output_param_stride(self) -> usize {
        match self {
            Abi::V1_1 => size_of::<dv_model_output_param>(),
            Abi::V1_3 => size_of::<dv_model_output_param_1_3>(),
        }
    }

    /// Keyed on major and minor only. The one layout change seen so far
    /// landed on a minor bump, and a patch release that moved a field would
    /// be a packaging fault rather than something to accommodate here.
    fn from_version(version: DvapiVersion) -> Option<Self> {
        match (version.major, version.minor) {
            (1, 1) => Some(Abi::V1_1),
            (1, 3) => Some(Abi::V1_3),
            _ => None,
        }
    }
}

/// Environment variable that forces an [`Abi`] rather than probing for one.
///
/// Accepts `1.1` or `1.3`. It exists so a target carrying a DVAPI version
/// this build does not recognise can still be brought up without waiting on
/// a new release.
pub const ABI_OVERRIDE_ENV: &str = "ARA2_ABI";

fn abi_override() -> Result<Option<Abi>, error::Error> {
    let Ok(raw) = std::env::var(ABI_OVERRIDE_ENV) else {
        return Ok(None);
    };
    match raw.trim() {
        "1.1" => Ok(Some(Abi::V1_1)),
        "1.3" => Ok(Some(Abi::V1_3)),
        other => Err(error::Error::AbiOverrideInvalid(other.to_owned())),
    }
}

/// Reads the DVAPI version out of an opened library.
///
/// `dv_get_client_lib_version` takes no session, so this runs before
/// anything else the library can be asked to do -- which is the point, as
/// the answer decides how its structs may be read.
fn probe_version(lib: &araclient) -> Option<DvapiVersion> {
    let probe = lib.dv_get_client_lib_version.as_ref().ok()?;
    let mut raw = dv_version {
        major: 0,
        minor: 0,
        patch: 0,
        patch_minor: 0,
    };
    // SAFETY: `probe` was resolved from the loaded library, and `raw` is a
    // live, fully initialised `dv_version` for it to write through.
    if unsafe { probe(&mut raw) } != 0 {
        return None;
    }
    Some(DvapiVersion {
        major: raw.major,
        minor: raw.minor,
        patch: raw.patch,
        patch_minor: raw.patch_minor,
    })
}

pub(crate) fn open_library() -> Result<(araclient, Abi, Option<DvapiVersion>), error::Error> {
    let forced = abi_override()?;
    let mut last_err = None;
    let mut rejected: Option<(&'static str, Option<DvapiVersion>)> = None;

    for name in LIBRARY_NAMES {
        let lib = match unsafe { araclient::new(*name) } {
            Ok(lib) => lib,
            Err(e) => {
                last_err = Some(e);
                continue;
            }
        };

        let version = probe_version(&lib);
        if let Some(abi) = forced {
            return Ok((lib, abi, version));
        }
        match version.and_then(Abi::from_version) {
            Some(abi) => return Ok((lib, abi, version)),
            // Keep going rather than failing here: an image can carry more
            // than one libaraclient, and a later candidate may well be a
            // generation this build does know.
            None if rejected.is_none() => rejected = Some((*name, version)),
            None => {}
        }
    }

    // A library that loaded but reported an unusable version is a different
    // problem from none of them being installed, and has a different fix,
    // so say which happened.
    if let Some((name, version)) = rejected {
        return Err(error::Error::UnsupportedDvapi { name, version });
    }

    // Report every candidate, not just the last failure: the last name
    // tried is the least likely one to be the library the target actually
    // ships, so on its own it points diagnostics at the wrong file.
    Err(error::Error::LibraryNotFound {
        tried: LIBRARY_NAMES,
        source: last_err.expect("LIBRARY_NAMES is non-empty"),
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use crate::Session;

    /// Create a test session connected to the local ARA-2 proxy.
    ///
    /// Panics if the proxy is not running or the socket is not available.
    pub fn test_session() -> Session {
        Session::connect()
            .expect("Failed to connect to the ARA-2 proxy. Is rt-sdk-ara2.service running?")
    }

    /// Get the test model path from the ARA2_TEST_MODEL environment variable.
    ///
    /// Panics if the variable is not set.
    pub fn test_model_path() -> std::path::PathBuf {
        std::env::var("ARA2_TEST_MODEL")
            .expect("ARA2_TEST_MODEL env var must be set to a .dvm file path")
            .into()
    }
}
