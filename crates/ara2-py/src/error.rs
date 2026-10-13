// SPDX-License-Identifier: Apache-2.0
// Copyright © 2025 Au-Zone Technologies. All Rights Reserved.

use pyo3::prelude::*;

// Exception hierarchy: Ara2Error > specific error types
pyo3::create_exception!(
    edgefirst_ara2,
    Ara2Error,
    pyo3::exceptions::PyRuntimeError,
    "Base class for every error raised by this module."
);
pyo3::create_exception!(
    edgefirst_ara2,
    LibraryError,
    Ara2Error,
    "The libaraclient client library could not be loaded or is not a supported DVAPI generation."
);
pyo3::create_exception!(
    edgefirst_ara2,
    HardwareError,
    Ara2Error,
    "The NPU or an endpoint reported a fault."
);
pyo3::create_exception!(
    edgefirst_ara2,
    ProxyError,
    Ara2Error,
    "The proxy could not be reached, or its configuration could not be used."
);
pyo3::create_exception!(
    edgefirst_ara2,
    ModelError,
    Ara2Error,
    "A model failed to load, or an inference failed."
);
pyo3::create_exception!(
    edgefirst_ara2,
    TensorError,
    Ara2Error,
    "A tensor could not be allocated, mapped or registered."
);
pyo3::create_exception!(
    edgefirst_ara2,
    MetadataError,
    Ara2Error,
    "The metadata inside a .dvm archive could not be read."
);

/// Convert an `ara2::Error` into the appropriate Python exception.
///
/// Every exception carries `code`, `name`, `sub_code`, `component` and
/// `severity` attributes. `code` and `name` are set for a status error, the
/// other three when the library also gave a detailed status; the rest are
/// `None`.
pub fn to_py_err(err: ara2::Error) -> PyErr {
    let msg = err.to_string();
    let py_err = match &err {
        ara2::Error::Library(_)
        | ara2::Error::LibraryNotFound { .. }
        | ara2::Error::UnsupportedDvapi { .. }
        | ara2::Error::AbiOverrideInvalid(_)
        | ara2::Error::AbiOverrideRejected { .. } => LibraryError::new_err(msg),
        ara2::Error::Io(_) => ProxyError::new_err(msg),
        ara2::Error::ProxyConfigUnreadable { .. }
        | ara2::Error::ProxyConfigInvalid { .. }
        | ara2::Error::NoProxyEndpoint { .. }
        | ara2::Error::NoProxyFound { .. }
        | ara2::Error::InvalidSocketOverride(_)
        | ara2::Error::ProcUnavailable { .. } => ProxyError::new_err(msg),
        ara2::Error::Ara2 { code, .. } => {
            let code = *code;
            if code >= 500 {
                HardwareError::new_err(msg)
            } else if code >= 400 {
                ProxyError::new_err(msg)
            } else if code >= 300 {
                LibraryError::new_err(msg)
            } else if code >= 240 {
                ModelError::new_err(msg)
            } else if code >= 200 {
                ProxyError::new_err(msg)
            } else if code >= 100 {
                HardwareError::new_err(msg)
            } else {
                Ara2Error::new_err(msg)
            }
        }
        ara2::Error::FdPassingUnavailable => TensorError::new_err(msg),
        ara2::Error::Unsupported { .. } => Ara2Error::new_err(msg),
        ara2::Error::UnknownLayerOutputType(_)
        | ara2::Error::UnsupportedLayout(_)
        | ara2::Error::UnsupportedTypeSize(_) => ModelError::new_err(msg),
        ara2::Error::NullPointer(_) => Ara2Error::new_err(msg),
        ara2::Error::TensorError(_) | ara2::Error::ImageError(_) | ara2::Error::ShapeError(_) => {
            TensorError::new_err(msg)
        }
        ara2::Error::Zip(_) | ara2::Error::Json(_) => MetadataError::new_err(msg),
        ara2::Error::UnsupportedQmode(_) => Ara2Error::new_err(msg),
        // Every `ara2::Error` variant should have an explicit arm above; this
        // wildcard exists only because the enum is `#[non_exhaustive]`.
        #[allow(unreachable_patterns)]
        _ => Ara2Error::new_err(msg),
    };
    let (code, name, detail) = match &err {
        ara2::Error::Ara2 { code, name, detail } => (Some(*code), Some(name.as_str()), *detail),
        _ => (None, None, None),
    };
    Python::attach(|py| {
        let value = py_err.value(py);
        // Setting an attribute on a fresh exception instance cannot fail
        // short of memory exhaustion; the message carries the same
        // information either way.
        let _ = value.setattr("code", code);
        let _ = value.setattr("name", name);
        let _ = value.setattr("sub_code", detail.map(|d| d.sub_code));
        let _ = value.setattr("component", detail.map(|d| d.component.to_string()));
        let _ = value.setattr("severity", detail.map(|d| d.severity.to_string()));
    });
    py_err
}

/// Register all exception types on the Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("Ara2Error", m.py().get_type::<Ara2Error>())?;
    m.add("LibraryError", m.py().get_type::<LibraryError>())?;
    m.add("HardwareError", m.py().get_type::<HardwareError>())?;
    m.add("ProxyError", m.py().get_type::<ProxyError>())?;
    m.add("ModelError", m.py().get_type::<ModelError>())?;
    m.add("TensorError", m.py().get_type::<TensorError>())?;
    m.add("MetadataError", m.py().get_type::<MetadataError>())?;
    Ok(())
}
