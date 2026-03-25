// SPDX-License-Identifier: Apache-2.0
// Copyright © 2025 Au-Zone Technologies. All Rights Reserved.

use crate::error::to_py_err;
use pyo3::prelude::*;
use std::path::PathBuf;

/// Read EdgeFirst metadata from a DVM model file.
///
/// Args:
///     path: Path to the .dvm model file (str or os.PathLike)
///
/// Returns:
///     DvmMetadata or None if the file has no embedded metadata
#[pyfunction]
pub fn read_metadata(path: PathBuf) -> PyResult<Option<DvmMetadata>> {
    let meta = ara2::read_metadata_from_file(&path).map_err(to_py_err)?;
    Ok(meta.map(DvmMetadata))
}

/// Read class labels from a DVM model file.
///
/// Args:
///     path: Path to the .dvm model file (str or os.PathLike)
///
/// Returns:
///     list[str]: Class label strings (empty if not present)
#[pyfunction]
pub fn read_labels(path: PathBuf) -> PyResult<Vec<String>> {
    ara2::read_labels_from_file(&path).map_err(to_py_err)
}

/// Check if a DVM file has embedded metadata.
///
/// Args:
///     path: Path to the .dvm model file (str or os.PathLike)
///
/// Returns:
///     bool: True if the file contains EdgeFirst metadata
#[pyfunction]
pub fn has_metadata(path: PathBuf) -> PyResult<bool> {
    let data =
        std::fs::read(&path).map_err(|e| crate::error::MetadataError::new_err(e.to_string()))?;
    Ok(ara2::has_metadata(&data))
}

/// EdgeFirst metadata embedded in a DVM model file.
#[pyclass(module = "edgefirst_ara2")]
pub struct DvmMetadata(ara2::DvmMetadata);

#[pymethods]
impl DvmMetadata {
    /// Model task type (e.g., "detect", "segment", "classify").
    #[getter]
    fn task(&self) -> Option<&str> {
        self.0.task()
    }

    /// Class labels from the dataset.
    #[getter]
    fn classes(&self) -> Vec<String> {
        self.0.classes().to_vec()
    }

    /// Dataset information.
    #[getter]
    fn dataset(&self) -> Option<DatasetInfo> {
        self.0.dataset.clone().map(DatasetInfo)
    }

    /// Input specification.
    #[getter]
    fn input(&self) -> Option<InputSpec> {
        self.0.input.clone().map(InputSpec)
    }

    /// Model information.
    #[getter]
    fn model(&self) -> Option<ModelInfo> {
        self.0.model.clone().map(ModelInfo)
    }

    /// Deployment information.
    #[getter]
    fn deployment(&self) -> Option<DeploymentInfo> {
        self.0.deployment.clone().map(DeploymentInfo)
    }

    /// Compilation information.
    #[getter]
    fn compilation(&self) -> Option<CompilationInfo> {
        self.0.compilation.clone().map(CompilationInfo)
    }

    /// Decoder version string (e.g., "yolov8").
    #[getter]
    fn decoder_version(&self) -> Option<&str> {
        self.0.decoder_version.as_deref()
    }

    /// NMS type (e.g., "class_agnostic").
    #[getter]
    fn nms(&self) -> Option<&str> {
        self.0.nms.as_deref()
    }

    /// Output specifications.
    #[getter]
    fn outputs(&self) -> Vec<OutputSpec> {
        self.0.outputs.iter().cloned().map(OutputSpec).collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "DvmMetadata(task={:?}, classes={}, outputs={})",
            self.0.task(),
            self.0.classes().len(),
            self.0.outputs.len()
        )
    }
}

/// Dataset information from model metadata.
#[pyclass(module = "edgefirst_ara2")]
pub struct DatasetInfo(ara2::DatasetInfo);

#[pymethods]
impl DatasetInfo {
    #[getter]
    fn classes(&self) -> Vec<String> {
        self.0.classes.clone()
    }

    #[getter]
    fn id(&self) -> Option<&str> {
        self.0.id.as_deref()
    }

    #[getter]
    fn name(&self) -> Option<&str> {
        self.0.name.as_deref()
    }

    fn __repr__(&self) -> String {
        format!(
            "DatasetInfo(classes={}, name={:?})",
            self.0.classes.len(),
            self.0.name
        )
    }
}

/// Model information from metadata.
#[pyclass(module = "edgefirst_ara2")]
pub struct ModelInfo(ara2::ModelInfo);

#[pymethods]
impl ModelInfo {
    #[getter]
    fn model_task(&self) -> Option<&str> {
        self.0.model_task.as_deref()
    }

    #[getter]
    fn model_size(&self) -> Option<&str> {
        self.0.model_size.as_deref()
    }

    #[getter]
    fn model_version(&self) -> Option<&str> {
        self.0.model_version.as_deref()
    }

    #[getter]
    fn detection(&self) -> bool {
        self.0.detection
    }

    #[getter]
    fn segmentation(&self) -> bool {
        self.0.segmentation
    }

    fn __repr__(&self) -> String {
        format!(
            "ModelInfo(task={:?}, size={:?})",
            self.0.model_task, self.0.model_size
        )
    }
}

/// Deployment information from metadata.
#[pyclass(module = "edgefirst_ara2")]
pub struct DeploymentInfo(ara2::DeploymentInfo);

#[pymethods]
impl DeploymentInfo {
    #[getter]
    fn model_name(&self) -> Option<&str> {
        self.0.model_name.as_deref()
    }

    #[getter]
    fn name(&self) -> Option<&str> {
        self.0.name.as_deref()
    }

    #[getter]
    fn author(&self) -> Option<&str> {
        self.0.author.as_deref()
    }

    #[getter]
    fn description(&self) -> Option<&str> {
        self.0.description.as_deref()
    }

    fn __repr__(&self) -> String {
        format!(
            "DeploymentInfo(name={:?}, author={:?})",
            self.0.name, self.0.author
        )
    }
}

/// Input specification from metadata.
#[pyclass(module = "edgefirst_ara2")]
pub struct InputSpec(ara2::InputSpec);

#[pymethods]
impl InputSpec {
    #[getter]
    fn size(&self) -> Option<&str> {
        self.0.size.as_deref()
    }

    #[getter]
    fn input_channels(&self) -> Option<u32> {
        self.0.input_channels
    }

    #[getter]
    fn output_channels(&self) -> Option<u32> {
        self.0.output_channels
    }

    #[getter]
    fn cameraadaptor(&self) -> Option<&str> {
        self.0.cameraadaptor.as_deref()
    }

    /// Parse the size string into (width, height).
    fn dimensions(&self) -> Option<(u32, u32)> {
        self.0.dimensions()
    }

    fn __repr__(&self) -> String {
        format!(
            "InputSpec(size={:?}, channels={:?})",
            self.0.size, self.0.input_channels
        )
    }
}

/// Output tensor specification from metadata.
#[pyclass(module = "edgefirst_ara2")]
pub struct OutputSpec(ara2::OutputSpec);

#[pymethods]
impl OutputSpec {
    #[getter]
    fn index(&self) -> Option<u32> {
        self.0.index
    }

    #[getter]
    fn name(&self) -> Option<&str> {
        self.0.name.as_deref()
    }

    #[getter]
    fn output_type(&self) -> Option<&str> {
        self.0.output_type.as_deref()
    }

    #[getter]
    fn decoder(&self) -> Option<&str> {
        self.0.decoder.as_deref()
    }

    #[getter]
    fn decode(&self) -> bool {
        self.0.decode
    }

    #[getter]
    fn dtype(&self) -> Option<&str> {
        self.0.dtype.as_deref()
    }

    #[getter]
    fn shape(&self) -> Vec<i64> {
        self.0.shape.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "OutputSpec(name={:?}, type={:?}, shape={:?})",
            self.0.name, self.0.output_type, self.0.shape
        )
    }
}

/// Compilation information from metadata.
#[pyclass(module = "edgefirst_ara2")]
pub struct CompilationInfo(ara2::CompilationInfo);

#[pymethods]
impl CompilationInfo {
    #[getter]
    fn target(&self) -> Option<&str> {
        self.0.target.as_deref()
    }

    #[getter]
    fn format(&self) -> Option<&str> {
        self.0.format.as_deref()
    }

    #[getter]
    fn ppa(&self) -> Option<PpaMetrics> {
        self.0.ppa.clone().map(PpaMetrics)
    }

    fn __repr__(&self) -> String {
        format!(
            "CompilationInfo(target={:?}, format={:?})",
            self.0.target, self.0.format
        )
    }
}

/// Performance, power, and area metrics from compilation.
#[pyclass(module = "edgefirst_ara2")]
pub struct PpaMetrics(ara2::PpaMetrics);

#[pymethods]
impl PpaMetrics {
    /// Inferences per second.
    #[getter]
    fn ips(&self) -> Option<f64> {
        self.0.ips
    }

    /// Power consumption in milliwatts.
    #[getter]
    fn power_mw(&self) -> Option<f64> {
        self.0.power_mw
    }

    /// Execution cycles.
    #[getter]
    fn cycles(&self) -> Option<u64> {
        self.0.cycles
    }

    /// DDR bandwidth in MB/s.
    #[getter]
    fn ddr_bw_mbps(&self) -> Option<f64> {
        self.0.ddr_bw_mbps
    }

    fn __repr__(&self) -> String {
        format!(
            "PpaMetrics(ips={:?}, power_mw={:?})",
            self.0.ips, self.0.power_mw
        )
    }
}
