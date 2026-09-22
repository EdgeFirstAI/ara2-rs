use crate::DvapiVersion;
use ara2_sys::{DV_LAYER_OUTPUT_TYPE, dv_status_code};

/// `#[non_exhaustive]` because the variant set is not fixed: `Error::Codec`
/// exists only with the `codec` feature, and Cargo unifies features across the
/// whole graph, so an unrelated dependency turning that feature on would
/// otherwise make a downstream exhaustive `match` stop compiling. A wildcard
/// arm is required either way; declaring it makes that a stable contract
/// rather than something a sibling crate's feature choice can change.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// An I/O operation failed, typically reaching the proxy socket.
    Io(std::io::Error),
    /// Decoding an image failed. Requires the `codec` feature.
    #[cfg(feature = "codec")]
    Codec(edgefirst_codec::CodecError),
    /// A `libaraclient` was found but could not be opened.
    Library(libloading::Error),
    /// None of the [`crate::LIBRARY_NAMES`] candidates could be loaded.
    LibraryNotFound {
        /// Every candidate name attempted, in order.
        tried: &'static [&'static str],
        /// The failure from the last candidate attempted.
        source: libloading::Error,
    },
    /// A `libaraclient` loaded but implements a DVAPI generation this build
    /// has no layout for.
    UnsupportedDvapi {
        /// The candidate name that loaded.
        name: &'static str,
        /// The version reported, or `None` if the library would not report
        /// one at all.
        version: Option<DvapiVersion>,
    },
    /// [`crate::ABI_OVERRIDE_ENV`] was set to something other than a
    /// recognised [`crate::Abi`]. Carries the offending value.
    AbiOverrideInvalid(String),
    /// A proxy configuration file could not be read.
    ProxyConfigUnreadable {
        /// The file that could not be read.
        path: std::path::PathBuf,
        /// Why reading it failed.
        source: std::io::Error,
    },
    /// A proxy configuration file was read but could not be understood.
    ProxyConfigInvalid {
        /// The file that could not be parsed.
        path: std::path::PathBuf,
        /// What about it could not be understood.
        reason: String,
    },
    /// A configuration or a discovered proxy declared no endpoint that
    /// this client can connect to.
    NoProxyEndpoint {
        /// The process it came from, when it came from [`crate::discover`].
        pid: Option<u32>,
    },
    /// `/proc` could not be read, so running proxies cannot be found.
    ProcUnavailable {
        /// The path under `/proc` that could not be read.
        path: std::path::PathBuf,
        /// Why reading it failed.
        source: std::io::Error,
    },
    /// The client library returned a `dv_status_code_t` indicating failure.
    /// Codes are grouped by range: 100s hardware, 200s proxy, 300s client
    /// library, 400s proxy transport, 500s device.
    Ara2(dv_status_code),
    /// A `dv_product_type_t` outside the set this build recognises.
    UnknownProductType(i32),
    /// A `dv_layer_output_type_t` outside the set this build recognises.
    UnknownLayerOutputType(DV_LAYER_OUTPUT_TYPE),
    /// A tensor layout string the crate cannot map to an axis order.
    UnsupportedLayout(String),
    /// An element size with no matching tensor element type, in bytes.
    UnsupportedTypeSize(usize),
    /// The client library returned success alongside a null or empty
    /// buffer. Carries the call that did so.
    NullPointer(String),
    /// Allocating, mapping or registering a tensor failed.
    TensorError(edgefirst_tensor::Error),
    /// An image conversion or resize failed.
    ImageError(edgefirst_image::Error),
    /// A tensor's element count does not match the shape asked of it.
    ShapeError(ndarray::ShapeError),
    /// Reading the `.dvm` archive failed; it is a ZIP container.
    Zip(zip::result::ZipError),
    /// Parsing the JSON metadata inside a `.dvm` failed.
    Json(serde_json::Error),
    /// A quantization mode the crate has no dequantization path for.
    UnsupportedQmode(i32),
    /// A supplied tensor does not match the size the model declares.
    TensorSizeMismatch {
        /// Size the model declares, in bytes.
        expected: usize,
        /// Size actually supplied, in bytes.
        got: usize,
    },
    /// The NPU reported the inference itself as failed.
    InferenceFailed,
    /// An inference request had not completed before its timeout. Carries
    /// the request id, for correlation with proxy logs.
    InferenceNotCompleted(u32),
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

#[cfg(feature = "codec")]
impl From<edgefirst_codec::CodecError> for Error {
    fn from(e: edgefirst_codec::CodecError) -> Self {
        Error::Codec(e)
    }
}

impl From<libloading::Error> for Error {
    fn from(e: libloading::Error) -> Self {
        Error::Library(e)
    }
}

impl From<dv_status_code> for Error {
    fn from(e: dv_status_code) -> Self {
        Error::Ara2(e)
    }
}

impl From<edgefirst_tensor::Error> for Error {
    fn from(e: edgefirst_tensor::Error) -> Self {
        Error::TensorError(e)
    }
}

impl From<edgefirst_image::Error> for Error {
    fn from(e: edgefirst_image::Error) -> Self {
        Error::ImageError(e)
    }
}

impl From<ndarray::ShapeError> for Error {
    fn from(e: ndarray::ShapeError) -> Self {
        Error::ShapeError(e)
    }
}

impl From<zip::result::ZipError> for Error {
    fn from(e: zip::result::ZipError) -> Self {
        Error::Zip(e)
    }
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error::Json(e)
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Error::Io(e) => write!(f, "{e}"),
            #[cfg(feature = "codec")]
            Error::Codec(e) => write!(f, "{e}"),
            Error::Library(e) => write!(f, "{e}"),
            Error::LibraryNotFound { tried, source } => write!(
                f,
                "could not load the ARA-2 client library; tried {} (last error: {source})",
                tried.join(", ")
            ),
            Error::UnsupportedDvapi { name, version } => match version {
                Some(version) => write!(
                    f,
                    "{name} implements DVAPI {version}, which this build of ara2 has no \
                     struct layout for; set {} to 1.1 or 1.3 to force one",
                    crate::ABI_OVERRIDE_ENV
                ),
                None => write!(
                    f,
                    "{name} would not report a DVAPI version (no usable \
                     dv_get_client_lib_version); set {} to 1.1 or 1.3 to force one",
                    crate::ABI_OVERRIDE_ENV
                ),
            },
            Error::ProxyConfigUnreadable { path, source } => {
                write!(f, "could not read {}: {source}", path.display())
            }
            Error::ProxyConfigInvalid { path, reason } => {
                write!(
                    f,
                    "{} is not a usable proxy configuration: {reason}",
                    path.display()
                )
            }
            Error::NoProxyEndpoint { pid } => match pid {
                Some(pid) => write!(
                    f,
                    "the proxy at pid {pid} exposes no endpoint this client can reach"
                ),
                None => f.write_str("no endpoint this client can reach was declared"),
            },
            Error::ProcUnavailable { path, source } => write!(
                f,
                "could not read {} to look for a running proxy: {source}",
                path.display()
            ),
            Error::AbiOverrideInvalid(value) => write!(
                f,
                "{} is set to {value:?}; expected 1.1 or 1.3",
                crate::ABI_OVERRIDE_ENV
            ),
            Error::Ara2(e) => write!(f, "Ara2 error: {e:?}"),
            Error::UnknownProductType(e) => write!(f, "Unknown product type: {e:?}"),
            Error::UnknownLayerOutputType(e) => write!(f, "Unknown layer output type: {e:?}"),
            Error::UnsupportedLayout(e) => write!(f, "Unsupported layout: {e:?}"),
            Error::UnsupportedTypeSize(e) => write!(f, "Unsupported type size: {e:?}"),
            Error::NullPointer(e) => write!(f, "Null pointer error: {e}"),
            Error::TensorError(e) => write!(f, "Tensor error: {e:?}"),
            Error::ImageError(e) => write!(f, "Image error: {e:?}"),
            Error::ShapeError(e) => write!(f, "Shape error: {e:?}"),
            Error::Zip(e) => write!(f, "ZIP error: {e}"),
            Error::Json(e) => write!(f, "JSON error: {e}"),
            Error::UnsupportedQmode(q) => write!(
                f,
                "unsupported quantization mode: qmode={q} (only qmode 9 is supported)"
            ),
            Error::TensorSizeMismatch { expected, got } => write!(
                f,
                "tensor size mismatch: expected {expected} bytes, got {got} bytes"
            ),
            Error::InferenceFailed => write!(f, "asynchronous inference failed on the NPU"),
            Error::InferenceNotCompleted(status) => {
                write!(f, "inference not completed (status={status})")
            }
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            #[cfg(feature = "codec")]
            Error::Codec(e) => Some(e),
            Error::Library(e) => Some(e),
            Error::LibraryNotFound { source, .. } => Some(source),
            Error::ProxyConfigUnreadable { source, .. } => Some(source),
            Error::ProcUnavailable { source, .. } => Some(source),
            Error::TensorError(e) => Some(e),
            Error::ImageError(e) => Some(e),
            Error::ShapeError(e) => Some(e),
            Error::Zip(e) => Some(e),
            Error::Json(e) => Some(e),
            _ => None,
        }
    }
}
