use ara2_sys::{DV_ENDPOINT_STATE, DV_LAYER_OUTPUT_TYPE, dv_status_code};

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Image(image::ImageError),
    Library(libloading::Error),
    Ara2(dv_status_code),
    EndpointStateInvalid(DV_ENDPOINT_STATE),
    UnknownProductType(i32),
    UnknownLayerOutputType(DV_LAYER_OUTPUT_TYPE),
    UnsupportedLayout(String),
    UnsupportedTypeSize(usize),
    NullPointer(String),
    TensorError(edgefirst_hal::tensor::Error),
    ImageError(edgefirst_hal::image::Error),
    ShapeError(ndarray::ShapeError),
    Zip(zip::result::ZipError),
    Json(serde_json::Error),
    UnsupportedQmode(i32),
    TensorSizeMismatch { expected: usize, got: usize },
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

impl From<image::ImageError> for Error {
    fn from(e: image::ImageError) -> Self {
        Error::Image(e)
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

impl From<edgefirst_hal::tensor::Error> for Error {
    fn from(e: edgefirst_hal::tensor::Error) -> Self {
        Error::TensorError(e)
    }
}

impl From<edgefirst_hal::image::Error> for Error {
    fn from(e: edgefirst_hal::image::Error) -> Self {
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
            Error::Image(e) => write!(f, "{e}"),
            Error::Library(e) => write!(f, "{e}"),
            Error::Ara2(e) => write!(f, "Ara2 error: {e:?}"),
            Error::EndpointStateInvalid(e) => write!(f, "Invalid endpoint state: {e:?}"),
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
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            Error::Image(e) => Some(e),
            Error::Library(e) => Some(e),
            Error::TensorError(e) => Some(e),
            Error::ImageError(e) => Some(e),
            Error::ShapeError(e) => Some(e),
            Error::Zip(e) => Some(e),
            Error::Json(e) => Some(e),
            _ => None,
        }
    }
}
