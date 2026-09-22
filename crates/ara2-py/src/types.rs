// SPDX-License-Identifier: Apache-2.0
// Copyright © 2025 Au-Zone Technologies. All Rights Reserved.

use pyo3::prelude::*;

/// The DVAPI generation a loaded ``libaraclient`` implements.
#[pyclass(module = "edgefirst_ara2", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Abi {
    V1_1,
    V1_3,
}

impl From<ara2::Abi> for Abi {
    fn from(abi: ara2::Abi) -> Self {
        match abi {
            ara2::Abi::V1_1 => Abi::V1_1,
            ara2::Abi::V1_3 => Abi::V1_3,
            // ara2::Abi is #[non_exhaustive]; a generation added upstream
            // reaches Python as the newest one this build knows.
            _ => Abi::V1_3,
        }
    }
}

#[pymethods]
impl Abi {
    fn __str__(&self) -> &'static str {
        match self {
            Abi::V1_1 => "1.1",
            Abi::V1_3 => "1.3",
        }
    }

    fn __repr__(&self) -> String {
        format!("Abi.{:?}", self)
    }
}

/// Endpoint state enum.
///
/// The members are the union of both DVAPI generations: 1.3 reuses values
/// 4 and 5 for names 1.1 does not have, so which of ``ActiveBoosted`` /
/// ``ThermalActiveSlow`` (and ``ThermalInactive`` / ``FailSafe``) you see
/// depends on the library loaded. ``Session.abi`` reports which that is.
#[pyclass(module = "edgefirst_ara2", eq, from_py_object)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum State {
    Init,
    Idle,
    Active,
    ActiveSlow,
    ActiveBoosted,
    ThermalInactive,
    ThermalActiveSlow,
    FailSafe,
    ThermalUnknown,
    Inactive,
    Fault,
    Unknown,
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
            ara2::State::ThermalActiveSlow => State::ThermalActiveSlow,
            ara2::State::FailSafe => State::FailSafe,
            ara2::State::ThermalUnknown => State::ThermalUnknown,
            ara2::State::Inactive => State::Inactive,
            ara2::State::Fault => State::Fault,
            // A Python enum member cannot carry the raw value, so log it
            // rather than discard it -- the number is the only thing that
            // makes an unknown state actionable.
            ara2::State::Unknown(raw, abi) => {
                log::warn!("endpoint reported state {raw}, undefined in DVAPI {abi}");
                State::Unknown
            }
            _ => State::Unknown,
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
#[pyclass(module = "edgefirst_ara2", get_all, from_py_object)]
#[derive(Clone, Debug)]
pub struct DramStatistics {
    /// Total DRAM capacity in bytes.
    pub dram_size: u64,
    /// Total occupied DRAM in bytes.
    pub dram_occupancy_size: u64,
    /// Free DRAM in bytes.
    pub free_size: u64,
    /// Reserved DRAM in bytes.
    pub reserved_occupancy_size: u64,
    /// DRAM occupied by loaded models in bytes.
    pub model_occupancy_size: u64,
    /// DRAM occupied by tensor buffers in bytes.
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
#[pyclass(module = "edgefirst_ara2", get_all, from_py_object)]
#[derive(Clone, Copy, Debug)]
pub struct ModelTiming {
    /// NPU inference execution time in microseconds.
    pub run_time_us: u64,
    /// Input DMA transfer time in microseconds.
    pub input_time_us: u64,
    /// Output DMA transfer time in microseconds.
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
///
/// For qmode 9 models (the production default), ``qn`` is the per-tensor
/// scale and ``offset`` is the integer zero-point. Per-channel image
/// preprocessing lives on :class:`InputPreprocess` instead.
#[pyclass(module = "edgefirst_ara2", get_all, from_py_object)]
#[derive(Clone, Debug)]
pub struct InputQuantization {
    /// Per-tensor quantization scale.
    pub qn: f32,
    /// Integer zero-point offset.
    pub offset: i32,
    /// True if the tensor uses signed int8, False for uint8.
    pub is_signed: bool,
    /// Kinara quantization mode (9 = asymmetric, production default).
    pub qmode: i32,
}

impl From<ara2::InputQuantization> for InputQuantization {
    fn from(q: ara2::InputQuantization) -> Self {
        InputQuantization {
            qn: q.qn,
            offset: q.offset,
            is_signed: q.is_signed,
            qmode: q.qmode,
        }
    }
}

#[pymethods]
impl InputQuantization {
    fn __repr__(&self) -> String {
        format!(
            "InputQuantization(qn={}, offset={}, is_signed={}, qmode={})",
            self.qn, self.offset, self.is_signed, self.qmode
        )
    }
}

/// Image preprocessing parameters for an input tensor.
///
/// Describes how float image data is expected to be normalized before
/// quantization: ``(pixel - mean[c]) * scale[c]``.
#[pyclass(module = "edgefirst_ara2", get_all, from_py_object)]
#[derive(Clone, Debug)]
pub struct InputPreprocess {
    /// Per-channel mean subtracted during preprocessing.
    pub mean: (f32, f32, f32),
    /// Per-channel scale applied after mean subtraction.
    pub scale: (f32, f32, f32),
    /// True if BGR inputs must be swapped to RGB before normalization.
    pub bgr_to_rgb: bool,
    /// True if the source image should be letterboxed to preserve aspect
    /// ratio.
    pub aspect_resize: bool,
    /// True if the input should be horizontally mirrored.
    pub mirror: bool,
    /// True if the input should be center-cropped to the model's input
    /// size.
    pub center_crop: bool,
}

impl From<ara2::InputPreprocess> for InputPreprocess {
    fn from(p: ara2::InputPreprocess) -> Self {
        InputPreprocess {
            mean: (p.mean[0], p.mean[1], p.mean[2]),
            scale: (p.scale[0], p.scale[1], p.scale[2]),
            bgr_to_rgb: p.bgr_to_rgb,
            aspect_resize: p.aspect_resize,
            mirror: p.mirror,
            center_crop: p.center_crop,
        }
    }
}

#[pymethods]
impl InputPreprocess {
    fn __repr__(&self) -> String {
        format!(
            "InputPreprocess(mean={:?}, scale={:?}, bgr_to_rgb={}, \
             aspect_resize={}, mirror={}, center_crop={})",
            self.mean,
            self.scale,
            self.bgr_to_rgb,
            self.aspect_resize,
            self.mirror,
            self.center_crop
        )
    }
}

/// Output tensor quantization parameters (qmode 9 semantics).
#[pyclass(module = "edgefirst_ara2", get_all, from_py_object)]
#[derive(Clone, Debug)]
pub struct OutputQuantization {
    /// Per-tensor quantization scale. ``float_val = (raw - offset) * qn``.
    pub qn: f32,
    /// Integer zero-point offset.
    pub offset: i32,
    /// True if the tensor uses signed int8, False for uint8.
    pub is_signed: bool,
}

impl From<ara2::OutputQuantization> for OutputQuantization {
    fn from(q: ara2::OutputQuantization) -> Self {
        OutputQuantization {
            qn: q.qn,
            offset: q.offset,
            is_signed: q.is_signed,
        }
    }
}

#[pymethods]
impl OutputQuantization {
    fn __repr__(&self) -> String {
        format!(
            "OutputQuantization(qn={}, offset={}, is_signed={})",
            self.qn, self.offset, self.is_signed
        )
    }
}

/// The type of output produced by a model layer.
#[pyclass(module = "edgefirst_ara2", eq, eq_int, from_py_object)]
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
#[pyclass(module = "edgefirst_ara2", get_all, from_py_object)]
#[derive(Clone, Debug)]
pub struct InputTensorInfo {
    /// Model layer this tensor belongs to.
    pub layer_id: i32,
    /// Blob index within the layer.
    pub blob_id: i32,
    /// Name of the layer, as compiled into the model.
    pub layer_name: String,
    /// Name of the blob within the layer.
    pub blob_name: String,
    /// Layer type reported by the compiler (e.g. "Convolution").
    pub layer_type: String,
    /// Data layout string (e.g., "NCHW").
    pub layout: String,
    /// Total size in bytes.
    pub size: usize,
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// Number of channels.
    pub nch: usize,
    /// Bytes per element.
    pub bpp: usize,
    /// Number of batches this tensor holds.
    pub batch_size: usize,
    /// Quantization parameters for converting to and from float.
    pub quant: InputQuantization,
    /// Image preprocessing parameters (per-channel mean/scale, BGR swap,
    /// etc.).
    pub preprocess: InputPreprocess,
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
            preprocess: t.preprocess.into(),
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
#[pyclass(module = "edgefirst_ara2", get_all, from_py_object)]
#[derive(Clone, Debug)]
pub struct OutputTensorInfo {
    /// Model layer this tensor belongs to.
    pub layer_id: i32,
    /// Blob index within the layer.
    pub blob_id: i32,
    /// Layer this one was fused into, or -1 when it was not fused.
    pub fused_parent_id: i32,
    /// Name of the layer, as compiled into the model.
    pub layer_name: String,
    /// Name of the blob within the layer.
    pub blob_name: String,
    /// Name of the fused parent layer, empty when it was not fused.
    pub layer_fused_parent_name: String,
    /// Layer type reported by the compiler (e.g. "Convolution").
    pub layer_type: String,
    /// Data layout string (e.g., "NCHW").
    pub layout: String,
    /// Total size in bytes.
    pub size: usize,
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// Number of channels.
    pub nch: usize,
    /// Bytes per element.
    pub bpp: usize,
    /// Number of classes the model was trained on.
    pub num_classes: usize,
    /// What kind of output this layer produces.
    pub layer_output_type: ModelOutputType,
    /// Highest batch id this tensor supports.
    pub max_dynamic_id: i32,
    /// Quantization parameters for converting to and from float.
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
