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
//! // environment variable, falling back to `DEFAULT_SOCKET`.
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

use ara2_sys::araclient;

pub mod dvm_metadata;
mod endpoint;
mod error;
mod model;
mod session;

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
/// This matches `interface_socket_file` in NXP's rt-sdk-ara2 (SDK 2.1.1+)
/// `proxy_config.yaml`. Override at runtime with the `ARA2_SOCKET`
/// environment variable -- see [`Session::connect`] -- or call
/// [`Session::create_via_unix_socket`] directly with an explicit path.
pub const DEFAULT_SOCKET: &str = "/var/run/proxy.sock";

/// Resolves the ARA-2 proxy socket path: the `ARA2_SOCKET` environment
/// variable if set, otherwise [`DEFAULT_SOCKET`].
///
/// [`Session::connect`] uses this internally; it is exposed separately so
/// callers that need the path as a string (e.g. a CLI flag's default
/// value) can resolve it the same way.
pub fn socket_path() -> String {
    std::env::var("ARA2_SOCKET").unwrap_or_else(|_| DEFAULT_SOCKET.to_owned())
}

/// The architecture-suffixed `libaraclient` name NXP's `imx-nxp-ara2`
/// package ships for this target, tried before the unsuffixed names in
/// [`LIBRARY_NAMES`].
///
/// NXP's rt-sdk-ara2 packaging has no stable soname: a drop for i.MX
/// 95/8M Plus installs `libaraclient_aarch64.so`, and past drops have used
/// `libaraclient_x86_64.so` on x86_64 hosts and a soname-versioned
/// `libaraclient.so.1` elsewhere. Falling back on an unrecognised target
/// keeps the crate building for platforms NXP hasn't shipped this scheme
/// for yet, even though loading `libaraclient.so` will fail there at
/// runtime rather than at compile time.
#[cfg(target_arch = "aarch64")]
const ARCH_LIBRARY_NAME: &str = "libaraclient_aarch64.so";
#[cfg(target_arch = "x86_64")]
const ARCH_LIBRARY_NAME: &str = "libaraclient_x86_64.so";
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
const ARCH_LIBRARY_NAME: &str = "libaraclient.so";

/// `libaraclient` names tried, in order, by [`open_library`].
///
/// NXP has shipped this library under a different name in every SDK drop
/// so far (`libaraclient.so.1`, `libaraclient_aarch64.so`, ...), so a
/// single hardcoded name breaks the moment the packaging changes again.
/// Trying each candidate covers every scheme seen to date with one build.
const LIBRARY_NAMES: &[&str] = &[ARCH_LIBRARY_NAME, "libaraclient.so", "libaraclient.so.1"];

pub(crate) fn open_library() -> Result<araclient, error::Error> {
    let mut last_err = None;
    for name in LIBRARY_NAMES {
        match unsafe { araclient::new(*name) } {
            Ok(lib) => return Ok(lib),
            Err(e) => last_err = Some(e),
        }
    }
    Err(last_err.expect("LIBRARY_NAMES is non-empty").into())
}

#[cfg(test)]
pub(crate) mod tests {
    use crate::Session;

    /// Create a test session connected to the local ARA-2 proxy.
    ///
    /// Panics if the proxy is not running or the socket is not available.
    pub fn test_session() -> Session {
        Session::connect().expect("Failed to connect to ARA-2 proxy. Is ara2-proxy running?")
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
