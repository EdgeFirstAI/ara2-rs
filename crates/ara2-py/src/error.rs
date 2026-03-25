// SPDX-License-Identifier: Apache-2.0
// Copyright © 2025 Au-Zone Technologies. All Rights Reserved.

use pyo3::prelude::*;

// Exception hierarchy: Ara2Error > specific error types
pyo3::create_exception!(edgefirst_ara2, Ara2Error, pyo3::exceptions::PyRuntimeError);
pyo3::create_exception!(edgefirst_ara2, LibraryError, Ara2Error);
pyo3::create_exception!(edgefirst_ara2, HardwareError, Ara2Error);
pyo3::create_exception!(edgefirst_ara2, ProxyError, Ara2Error);
pyo3::create_exception!(edgefirst_ara2, ModelError, Ara2Error);
pyo3::create_exception!(edgefirst_ara2, TensorError, Ara2Error);
pyo3::create_exception!(edgefirst_ara2, MetadataError, Ara2Error);

/// Convert an `ara2::Error` into the appropriate Python exception.
pub fn to_py_err(err: ara2::Error) -> PyErr {
    let msg = err.to_string();
    match &err {
        ara2::Error::Library(_) => LibraryError::new_err(msg),
        ara2::Error::Io(_) => ProxyError::new_err(msg),
        ara2::Error::Ara2(code) => {
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
        ara2::Error::EndpointStateInvalid(_) | ara2::Error::UnknownProductType(_) => {
            HardwareError::new_err(msg)
        }
        ara2::Error::UnknownLayerOutputType(_)
        | ara2::Error::UnsupportedLayout(_)
        | ara2::Error::UnsupportedTypeSize(_) => ModelError::new_err(msg),
        ara2::Error::NullPointer(_) => Ara2Error::new_err(msg),
        ara2::Error::TensorError(_)
        | ara2::Error::ImageError(_)
        | ara2::Error::Image(_)
        | ara2::Error::ShapeError(_) => TensorError::new_err(msg),
        ara2::Error::Zip(_) | ara2::Error::Json(_) => MetadataError::new_err(msg),
        #[allow(unreachable_patterns)]
        _ => Ara2Error::new_err(msg),
    }
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
