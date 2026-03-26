// SPDX-License-Identifier: Apache-2.0
// Copyright © 2025 Au-Zone Technologies. All Rights Reserved.

use crate::error::to_py_err;
use crate::model::Model;
use crate::types::{DramStatistics, State};
use pyo3::prelude::*;
use std::path::PathBuf;

/// ARA-2 accelerator endpoint.
///
/// An endpoint represents a single ARA-2 accelerator device that can load
/// and execute neural network models.
#[pyclass(module = "edgefirst_ara2")]
pub struct Endpoint(pub(crate) ara2::Endpoint);

#[pymethods]
impl Endpoint {
    /// Check the current status/state of the endpoint.
    ///
    /// Returns:
    ///     State: Current endpoint state
    fn check_status(&self) -> PyResult<State> {
        let state = self.0.check_status().map_err(to_py_err)?;
        Ok(State::from(state))
    }

    /// Get DRAM statistics for the endpoint.
    ///
    /// Returns:
    ///     DramStatistics: Memory usage information
    fn dram_statistics(&self) -> PyResult<DramStatistics> {
        let stats = self.0.dram_statistics().map_err(to_py_err)?;
        Ok(DramStatistics::from(stats))
    }

    /// Load a neural network model from a file.
    ///
    /// Args:
    ///     model_path: Path to the compiled model file (.dvm)
    ///
    /// Returns:
    ///     Model: Loaded model ready for inference
    fn load_model(&self, model_path: PathBuf) -> PyResult<Model> {
        let model = self
            .0
            .load_model_from_file(&model_path)
            .map_err(to_py_err)?;
        Ok(Model::new(model))
    }

    fn __repr__(&self) -> String {
        "Endpoint()".to_string()
    }
}
