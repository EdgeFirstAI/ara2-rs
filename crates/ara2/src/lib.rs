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
//! use ara2::{Session, DEFAULT_SOCKET, DEFAULT_TIMEOUT_MS};
//! use edgefirst_hal::tensor::{TensorMemory, TensorTrait as _};
//!
//! // Connect to the ARA-2 proxy service
//! let session = Session::create_via_unix_socket(DEFAULT_SOCKET)?;
//! let endpoints = session.list_endpoints()?;
//!
//! // Load model and allocate DMA-backed tensors
//! let mut model = endpoints[0].load_model_from_file("model.dvm".as_ref())?;
//! model.allocate_tensors(Some(TensorMemory::Dma))?;
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
//! # use ara2::{Session, DEFAULT_SOCKET, DEFAULT_TIMEOUT_MS};
//! # let session = Session::create_via_unix_socket(DEFAULT_SOCKET)?;
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
pub use endpoint::{DramStatistics, Endpoint, State};
pub use error::Error;
pub use model::{
    DEFAULT_TIMEOUT_MS, InferRequest, InputPreprocess, InputQuantization, InputTensor, Model,
    ModelOutputType, ModelTiming, OutputQuantization, OutputTensor,
};
pub use session::{Session, SocketType};

/// Default socket path for the ARA-2 proxy service.
pub const DEFAULT_SOCKET: &str = "/var/run/ara2.sock";

pub(crate) fn open_library() -> Result<araclient, error::Error> {
    unsafe { Ok(araclient::new("libaraclient.so.1")?) }
}

#[cfg(test)]
pub(crate) mod tests {
    use crate::Session;

    /// Create a test session connected to the local ARA-2 proxy.
    ///
    /// Panics if the proxy is not running or the socket is not available.
    pub fn test_session() -> Session {
        Session::create_via_unix_socket(crate::DEFAULT_SOCKET)
            .expect("Failed to connect to ARA-2 proxy. Is ara2-proxy running?")
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
