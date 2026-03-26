// SPDX-License-Identifier: Apache-2.0
// Copyright © 2025 Au-Zone Technologies. All Rights Reserved.

use pyo3::prelude::*;

/// Endpoint state enum.
#[pyclass(module = "edgefirst_ara2", eq)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum State {
    Init,
    Idle,
    Active,
    ActiveSlow,
    ActiveBoosted,
    ThermalInactive,
    ThermalUnknown,
    Inactive,
    Fault,
}

impl From<ara2::State> for State {
    fn from(state: ara2::State) -> Self {
        match state {
            ara2::State::Init => State::Init,
            ara2::State::Idle => State::Idle,
            ara2::State::Active => State::Active,
            ara2::State::ActiveSlow => State::ActiveSlow,
            ara2::State::ActiveBoosted => State::ActiveBoosted,
            ara2::State::ThermalInactive => State::ThermalInactive,
            ara2::State::ThermalUnknown => State::ThermalUnknown,
            ara2::State::Inactive => State::Inactive,
            ara2::State::Fault => State::Fault,
        }
    }
}

#[pymethods]
impl State {
    fn __str__(&self) -> String {
        format!("{:?}", self)
    }

    fn __repr__(&self) -> String {
        format!("State.{:?}", self)
    }
}

/// DRAM statistics for an endpoint.
#[pyclass(module = "edgefirst_ara2", get_all)]
#[derive(Clone, Debug)]
pub struct DramStatistics {
    pub dram_size: u64,
    pub dram_occupancy_size: u64,
    pub free_size: u64,
    pub reserved_occupancy_size: u64,
    pub model_occupancy_size: u64,
    pub tensor_occupancy_size: u64,
}

impl From<ara2::DramStatistics> for DramStatistics {
    fn from(s: ara2::DramStatistics) -> Self {
        DramStatistics {
            dram_size: s.dram_size,
            dram_occupancy_size: s.dram_occupancy_size,
            free_size: s.free_size,
            reserved_occupancy_size: s.reserved_occupancy_size,
            model_occupancy_size: s.model_occupancy_size,
            tensor_occupancy_size: s.tensor_occupancy_size,
        }
    }
}

#[pymethods]
impl DramStatistics {
    fn __repr__(&self) -> String {
        format!(
            "DramStatistics(dram_size={}, free_size={}, model_occupancy_size={})",
            self.dram_size, self.free_size, self.model_occupancy_size
        )
    }
}

/// Timing information from a model inference run.
#[pyclass(module = "edgefirst_ara2", get_all)]
#[derive(Clone, Copy, Debug)]
pub struct ModelTiming {
    pub run_time_us: u64,
    pub input_time_us: u64,
    pub output_time_us: u64,
}

impl From<ara2::ModelTiming> for ModelTiming {
    fn from(t: ara2::ModelTiming) -> Self {
        ModelTiming {
            run_time_us: t.run_time.as_micros() as u64,
            input_time_us: t.input_time.as_micros() as u64,
            output_time_us: t.output_time.as_micros() as u64,
        }
    }
}

#[pymethods]
impl ModelTiming {
    fn __repr__(&self) -> String {
        format!(
            "ModelTiming(run_time_us={}, input_time_us={}, output_time_us={})",
            self.run_time_us, self.input_time_us, self.output_time_us
        )
    }
}

/// Input tensor quantization parameters.
#[pyclass(module = "edgefirst_ara2", get_all)]
#[derive(Clone, Debug)]
pub struct InputQuantization {
    pub qn: f32,
    pub scale: f32,
    pub mean: f32,
    pub is_signed: bool,
}

impl From<ara2::InputQuantization> for InputQuantization {
    fn from(q: ara2::InputQuantization) -> Self {
        InputQuantization {
            qn: q.qn,
            scale: q.scale,
            mean: q.mean,
            is_signed: q.is_signed,
        }
    }
}

#[pymethods]
impl InputQuantization {
    fn __repr__(&self) -> String {
        format!(
            "InputQuantization(qn={}, scale={}, mean={}, is_signed={})",
            self.qn, self.scale, self.mean, self.is_signed
        )
    }
}

/// Output tensor quantization parameters.
#[pyclass(module = "edgefirst_ara2", get_all)]
#[derive(Clone, Debug)]
pub struct OutputQuantization {
    pub qn: f32,
    pub scale: f32,
    pub offset: i32,
    pub is_signed: bool,
}

impl From<ara2::OutputQuantization> for OutputQuantization {
    fn from(q: ara2::OutputQuantization) -> Self {
        OutputQuantization {
            qn: q.qn,
            scale: q.scale,
            offset: q.offset,
            is_signed: q.is_signed,
        }
    }
}

#[pymethods]
impl OutputQuantization {
    fn __repr__(&self) -> String {
        format!(
            "OutputQuantization(qn={}, scale={}, offset={}, is_signed={})",
            self.qn, self.scale, self.offset, self.is_signed
        )
    }
}

/// The type of output produced by a model layer.
#[pyclass(module = "edgefirst_ara2", eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ModelOutputType {
    Classification = 0,
    Detection = 1,
    SemanticSegmentation = 2,
    Raw = 3,
}

impl From<&ara2::ModelOutputType> for ModelOutputType {
    fn from(t: &ara2::ModelOutputType) -> Self {
        match t {
            ara2::ModelOutputType::Classification => ModelOutputType::Classification,
            ara2::ModelOutputType::Detection => ModelOutputType::Detection,
            ara2::ModelOutputType::SemanticSegmentation => ModelOutputType::SemanticSegmentation,
            ara2::ModelOutputType::Raw => ModelOutputType::Raw,
        }
    }
}

#[pymethods]
impl ModelOutputType {
    fn __str__(&self) -> String {
        format!("{:?}", self)
    }

    fn __repr__(&self) -> String {
        format!("ModelOutputType.{:?}", self)
    }
}

/// Detailed information about an input tensor.
#[pyclass(module = "edgefirst_ara2", get_all)]
#[derive(Clone, Debug)]
pub struct InputTensorInfo {
    pub layer_id: i32,
    pub blob_id: i32,
    pub layer_name: String,
    pub blob_name: String,
    pub layer_type: String,
    pub layout: String,
    pub size: usize,
    pub width: usize,
    pub height: usize,
    pub nch: usize,
    pub bpp: usize,
    pub batch_size: usize,
    pub quant: InputQuantization,
}

impl From<ara2::InputTensor> for InputTensorInfo {
    fn from(t: ara2::InputTensor) -> Self {
        InputTensorInfo {
            layer_id: t.layer_id,
            blob_id: t.blob_id,
            layer_name: t.layer_name,
            blob_name: t.blob_name,
            layer_type: t.layer_type,
            layout: t.layout,
            size: t.size,
            width: t.width,
            height: t.height,
            nch: t.nch,
            bpp: t.bpp,
            batch_size: t.batch_size,
            quant: t.quant.into(),
        }
    }
}

#[pymethods]
impl InputTensorInfo {
    fn __repr__(&self) -> String {
        format!(
            "InputTensorInfo(layer_name='{}', size={}, shape=({}, {}, {}), bpp={})",
            self.layer_name, self.size, self.nch, self.height, self.width, self.bpp
        )
    }
}

/// Detailed information about an output tensor.
#[pyclass(module = "edgefirst_ara2", get_all)]
#[derive(Clone, Debug)]
pub struct OutputTensorInfo {
    pub layer_id: i32,
    pub blob_id: i32,
    pub fused_parent_id: i32,
    pub layer_name: String,
    pub blob_name: String,
    pub layer_fused_parent_name: String,
    pub layer_type: String,
    pub layout: String,
    pub size: usize,
    pub width: usize,
    pub height: usize,
    pub nch: usize,
    pub bpp: usize,
    pub num_classes: usize,
    pub layer_output_type: ModelOutputType,
    pub max_dynamic_id: i32,
    pub quant: OutputQuantization,
}

impl From<ara2::OutputTensor> for OutputTensorInfo {
    fn from(t: ara2::OutputTensor) -> Self {
        OutputTensorInfo {
            layer_id: t.layer_id,
            blob_id: t.blob_id,
            fused_parent_id: t.fused_parent_id,
            layer_name: t.layer_name,
            blob_name: t.blob_name,
            layer_fused_parent_name: t.layer_fused_parent_name,
            layer_type: t.layer_type,
            layout: t.layout,
            size: t.size,
            width: t.width,
            height: t.height,
            nch: t.nch,
            bpp: t.bpp,
            num_classes: t.num_classes,
            layer_output_type: ModelOutputType::from(&t.layer_output_type),
            max_dynamic_id: t.max_dynamic_id,
            quant: t.quant.into(),
        }
    }
}

#[pymethods]
impl OutputTensorInfo {
    fn __repr__(&self) -> String {
        format!(
            "OutputTensorInfo(layer_name='{}', size={}, shape=({}, {}, {}), bpp={}, type={:?})",
            self.layer_name,
            self.size,
            self.nch,
            self.height,
            self.width,
            self.bpp,
            self.layer_output_type
        )
    }
}
