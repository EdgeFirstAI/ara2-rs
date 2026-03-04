use ara2_sys::araclient;

pub mod dvm_metadata;
mod endpoint;
mod error;
mod model;
mod session;

pub use dvm_metadata::{
    CompilationInfo, DatasetInfo, DeploymentInfo, DvmMetadata, InputSpec, ModelInfo, OutputSpec,
    PpaMetrics, has_metadata, read_labels, read_labels_from_file, read_metadata,
    read_metadata_from_file,
};
pub use endpoint::{DramStatistics, Endpoint, State};
pub use error::Error;
pub use model::{
    DEFAULT_TIMEOUT_MS, InputQuantization, InputTensor, Model, ModelOutputType, ModelTiming,
    OutputQuantization, OutputTensor,
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
