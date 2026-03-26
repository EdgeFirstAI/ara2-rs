// SPDX-License-Identifier: Apache-2.0
// Copyright © 2025 Au-Zone Technologies. All Rights Reserved.

mod endpoint;
mod error;
pub mod metadata;
mod model;
mod session;
pub mod types;

use pyo3::prelude::*;

/// EdgeFirst ARA-2 Python Library
///
/// Python bindings for the ARA-2 neural accelerator client library.
///
/// Example:
///     >>> import edgefirst_ara2
///     >>> session = edgefirst_ara2.Session.create_via_unix_socket("/var/run/ara2.sock")
///     >>> endpoints = session.list_endpoints()
///     >>> model = endpoints[0].load_model("model.dvm")
///     >>> model.allocate_tensors("dma")
///     >>> timing = model.run()
#[pymodule(name = "edgefirst_ara2")]
fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Version
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // Constants
    m.add("DEFAULT_SOCKET", ara2::DEFAULT_SOCKET)?;

    // Exceptions
    error::register(m)?;

    // Core classes
    m.add_class::<session::Session>()?;
    m.add_class::<endpoint::Endpoint>()?;
    m.add_class::<model::Model>()?;

    // Type classes
    m.add_class::<types::State>()?;
    m.add_class::<types::DramStatistics>()?;
    m.add_class::<types::ModelTiming>()?;
    m.add_class::<types::InputQuantization>()?;
    m.add_class::<types::OutputQuantization>()?;
    m.add_class::<types::ModelOutputType>()?;
    m.add_class::<types::InputTensorInfo>()?;
    m.add_class::<types::OutputTensorInfo>()?;

    // Metadata classes
    m.add_class::<metadata::DvmMetadata>()?;
    m.add_class::<metadata::DatasetInfo>()?;
    m.add_class::<metadata::ModelInfo>()?;
    m.add_class::<metadata::DeploymentInfo>()?;
    m.add_class::<metadata::InputSpec>()?;
    m.add_class::<metadata::OutputSpec>()?;
    m.add_class::<metadata::CompilationInfo>()?;
    m.add_class::<metadata::PpaMetrics>()?;

    // Metadata functions
    m.add_function(wrap_pyfunction!(metadata::read_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(metadata::read_labels, m)?)?;
    m.add_function(wrap_pyfunction!(metadata::has_metadata, m)?)?;

    Ok(())
}
