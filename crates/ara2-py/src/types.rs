// SPDX-License-Identifier: Apache-2.0
// Copyright © 2025 Au-Zone Technologies. All Rights Reserved.

use std::sync::Once;

use pyo3::prelude::*;

/// The DVAPI generation a loaded ``libaraclient`` implements.
#[pyclass(module = "edgefirst_ara2", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Abi {
    V1_1,
    V1_3,
    V3_0,
}

/// A component and the version the proxy reports for it.
#[pyclass(module = "edgefirst_ara2", frozen, get_all, from_py_object)]
#[derive(Clone)]
pub struct ComponentVersion {
    /// Component name, e.g. "proxy", "firmware", "client_lib",
    /// or "product_type_N" for one this build does not name.
    pub component: String,
    /// Version as "major.minor.patch.patch_minor".
    pub version: String,
}

impl From<ara2::ComponentVersion> for ComponentVersion {
    fn from(v: ara2::ComponentVersion) -> Self {
        ComponentVersion {
            component: v.component.to_string(),
            version: v.version.to_string(),
        }
    }
}

#[pymethods]
impl ComponentVersion {
    fn __repr__(&self) -> String {
        format!(
            "ComponentVersion(component={:?}, version={:?})",
            self.component, self.version
        )
    }
}

impl From<ara2::Abi> for Abi {
    fn from(abi: ara2::Abi) -> Self {
        match abi {
            ara2::Abi::V1_1 => Abi::V1_1,
            ara2::Abi::V1_3 => Abi::V1_3,
            ara2::Abi::V3_0 => Abi::V3_0,
            // ara2::Abi is #[non_exhaustive]; a generation added upstream
            // reaches Python as the newest one this build knows.
            _ => Abi::V3_0,
        }
    }
}

#[pymethods]
impl Abi {
    fn __str__(&self) -> &'static str {
        match self {
            Abi::V1_1 => "1.1",
            Abi::V1_3 => "1.3",
            Abi::V3_0 => "3.0",
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
    /// Largest contiguous free block in bytes; ``None`` before DVAPI 3.0.
    pub largest_contiguous_free: Option<u64>,
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
            largest_contiguous_free: s.largest_contiguous_free,
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
    /// NPU inference execution time in seconds.
    pub run_time: f64,
    /// Input DMA transfer time in seconds.
    pub input_time: f64,
    /// Output DMA transfer time in seconds.
    pub output_time: f64,
}

/// An optional value as Python would print it.
fn opt<T: std::fmt::Debug>(v: &Option<T>) -> String {
    match v {
        Some(v) => format!("{v:?}"),
        None => "None".to_string(),
    }
}

impl From<ara2::ModelTiming> for ModelTiming {
    fn from(t: ara2::ModelTiming) -> Self {
        ModelTiming {
            run_time: t.run_time.as_secs_f64(),
            input_time: t.input_time.as_secs_f64(),
            output_time: t.output_time.as_secs_f64(),
        }
    }
}

#[pymethods]
impl ModelTiming {
    fn __repr__(&self) -> String {
        format!(
            "ModelTiming(run_time={}, input_time={}, output_time={})",
            self.run_time, self.input_time, self.output_time
        )
    }
}

/// Clock frequencies in MHz.
#[pyclass(module = "edgefirst_ara2", frozen, get_all, from_py_object)]
#[derive(Clone, Copy, Debug)]
pub struct Clocks {
    /// System core clock.
    pub sys_mhz: i32,
    /// DRAM clock.
    pub dram_mhz: i32,
    /// Neural processor clock; ``None`` where the library does not report it.
    pub nnp_mhz: Option<i32>,
    /// SBP clock; ``None`` where the library does not report it.
    pub sbp_mhz: Option<i32>,
}

impl From<ara2::Clocks> for Clocks {
    fn from(c: ara2::Clocks) -> Self {
        Clocks {
            sys_mhz: c.sys_mhz,
            dram_mhz: c.dram_mhz,
            nnp_mhz: c.nnp_mhz,
            sbp_mhz: c.sbp_mhz,
        }
    }
}

#[pymethods]
impl Clocks {
    fn __repr__(&self) -> String {
        format!(
            "Clocks(sys_mhz={}, dram_mhz={}, nnp_mhz={}, sbp_mhz={})",
            self.sys_mhz,
            self.dram_mhz,
            opt(&self.nnp_mhz),
            opt(&self.sbp_mhz)
        )
    }
}

fn monotonic_secs(t: Option<ara2::MonotonicTime>) -> Option<f64> {
    t.map(|t| t.as_duration().as_secs_f64())
}

/// Everything the runtime reports about one completed inference.
///
/// The ``*_start`` timestamps are seconds on ``CLOCK_MONOTONIC``, the clock
/// ``time.monotonic()`` reads. They are ``None`` on DVAPI 1.x, whose runtime
/// stamps with the wall clock.
#[pyclass(module = "edgefirst_ara2", frozen, get_all, from_py_object)]
#[derive(Clone, Copy, Debug)]
pub struct InferenceStats {
    /// Execution and transfer durations.
    pub timing: ModelTiming,
    /// Clocks the inference ran at.
    pub clocks: Clocks,
    /// Hardware cycles, including floating-point work.
    pub hw_cycles: u32,
    /// Hardware cycles spent on floating-point work.
    pub fp_cycles: u32,
    /// When the input transfer started.
    pub input_start: Option<f64>,
    /// When the inference entered the NNP queue.
    pub inference_start: Option<f64>,
    /// When the output transfer started.
    pub output_start: Option<f64>,
    /// Inference retries the proxy performed across this session.
    pub retries_session_total: u32,
    /// Retries of this inference.
    pub retries_this_inference: u32,
}

impl From<ara2::InferenceStats> for InferenceStats {
    fn from(s: ara2::InferenceStats) -> Self {
        InferenceStats {
            timing: s.timing.into(),
            clocks: s.clocks.into(),
            hw_cycles: s.hw_cycles,
            fp_cycles: s.fp_cycles,
            input_start: monotonic_secs(s.input_start),
            inference_start: monotonic_secs(s.inference_start),
            output_start: monotonic_secs(s.output_start),
            retries_session_total: s.retries.session_total,
            retries_this_inference: s.retries.this_inference,
        }
    }
}

#[pymethods]
impl InferenceStats {
    fn __repr__(&self) -> String {
        format!(
            "InferenceStats(run_time={}, hw_cycles={}, fp_cycles={}, \
             retries_this_inference={})",
            self.timing.run_time, self.hw_cycles, self.fp_cycles, self.retries_this_inference
        )
    }
}

/// Verbosity of the client library's own logging.
#[pyclass(module = "edgefirst_ara2", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
    Critical,
    Off,
}

impl From<LogLevel> for ara2::LogLevel {
    fn from(l: LogLevel) -> Self {
        match l {
            LogLevel::Trace => ara2::LogLevel::Trace,
            LogLevel::Debug => ara2::LogLevel::Debug,
            LogLevel::Info => ara2::LogLevel::Info,
            LogLevel::Warn => ara2::LogLevel::Warn,
            LogLevel::Error => ara2::LogLevel::Error,
            LogLevel::Critical => ara2::LogLevel::Critical,
            LogLevel::Off => ara2::LogLevel::Off,
        }
    }
}

#[pymethods]
impl LogLevel {
    fn __str__(&self) -> String {
        format!("{:?}", self)
    }

    fn __repr__(&self) -> String {
        format!("LogLevel.{:?}", self)
    }
}

/// The kind of model, as the runtime reports it.
///
/// ``Other`` is a value this build does not know for the loaded version;
/// the first such raw value is logged at warning level.
#[pyclass(module = "edgefirst_ara2", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ModelType {
    Ara1Cnn,
    Ara2Cnn,
    Ara2Llm,
    Ara2LlmDynV2,
    McsLlm,
    Other,
}

impl From<ara2::ModelType> for ModelType {
    fn from(t: ara2::ModelType) -> Self {
        match t {
            ara2::ModelType::Ara1Cnn => ModelType::Ara1Cnn,
            ara2::ModelType::Ara2Cnn => ModelType::Ara2Cnn,
            ara2::ModelType::Ara2Llm => ModelType::Ara2Llm,
            ara2::ModelType::Ara2LlmDynV2 => ModelType::Ara2LlmDynV2,
            ara2::ModelType::McsLlm => ModelType::McsLlm,
            ara2::ModelType::Other(raw) => {
                static WARNED: Once = Once::new();
                WARNED.call_once(|| {
                    log::warn!("model reported type {raw}, which this version does not define")
                });
                ModelType::Other
            }
            _ => ModelType::Other,
        }
    }
}

#[pymethods]
impl ModelType {
    fn __str__(&self) -> String {
        format!("{:?}", self)
    }

    fn __repr__(&self) -> String {
        format!("ModelType.{:?}", self)
    }
}

/// The host bus an endpoint is attached through.
///
/// ``Other`` is a host interface value this build does not recognise; the
/// first such raw value is logged at warning level.
#[pyclass(module = "edgefirst_ara2", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InterfaceKind {
    Pcie,
    Usb,
    Other,
}

impl From<ara2::InterfaceKind> for InterfaceKind {
    fn from(k: ara2::InterfaceKind) -> Self {
        match k {
            ara2::InterfaceKind::Pcie => InterfaceKind::Pcie,
            ara2::InterfaceKind::Usb => InterfaceKind::Usb,
            ara2::InterfaceKind::Other(raw) => {
                static WARNED: Once = Once::new();
                WARNED.call_once(|| {
                    log::warn!(
                        "endpoint reported host interface {raw}, which this build does not \
                         recognise"
                    )
                });
                InterfaceKind::Other
            }
            _ => InterfaceKind::Other,
        }
    }
}

#[pymethods]
impl InterfaceKind {
    fn __str__(&self) -> String {
        format!("{:?}", self)
    }

    fn __repr__(&self) -> String {
        format!("InterfaceKind.{:?}", self)
    }
}

/// The ARA-2 chip an endpoint is built on.
#[pyclass(module = "edgefirst_ara2", frozen, get_all, from_py_object)]
#[derive(Clone, Debug)]
pub struct ChipInfo {
    /// Chip name, such as ``ARA-2``; ``None`` when the library reports none.
    pub id: Option<String>,
    /// Silicon revision; ``None`` when the library reports none.
    pub revision: Option<String>,
    /// Number of control processors.
    pub control_processors: i32,
    /// Number of neural processors.
    pub neural_processors: i32,
    /// L2 memory size in bytes.
    pub l2_bytes: u32,
}

impl From<ara2::ChipInfo> for ChipInfo {
    fn from(c: ara2::ChipInfo) -> Self {
        ChipInfo {
            id: c.id,
            revision: c.revision,
            control_processors: c.control_processors,
            neural_processors: c.neural_processors,
            l2_bytes: c.l2_bytes,
        }
    }
}

#[pymethods]
impl ChipInfo {
    fn __repr__(&self) -> String {
        format!(
            "ChipInfo(id={}, revision={}, control_processors={}, neural_processors={}, l2_bytes={})",
            opt(&self.id),
            opt(&self.revision),
            self.control_processors,
            self.neural_processors,
            self.l2_bytes
        )
    }
}

/// The DRAM attached to an endpoint.
#[pyclass(module = "edgefirst_ara2", frozen, get_all, from_py_object)]
#[derive(Clone, Debug)]
pub struct DramInfo {
    /// Manufacturer name, when the runtime knows it.
    pub vendor_name: Option<String>,
}

impl From<ara2::DramInfo> for DramInfo {
    fn from(d: ara2::DramInfo) -> Self {
        DramInfo {
            vendor_name: d.vendor_name,
        }
    }
}

#[pymethods]
impl DramInfo {
    fn __repr__(&self) -> String {
        format!("DramInfo(vendor_name={})", opt(&self.vendor_name))
    }
}

/// Where an endpoint sits on the host.
#[pyclass(module = "edgefirst_ara2", frozen, get_all, from_py_object)]
#[derive(Clone, Debug)]
pub struct Interface {
    /// The host bus type.
    pub kind: InterfaceKind,
    /// Bus number.
    pub bus: i32,
    /// Device number.
    pub device: i32,
    /// Port number.
    pub port: i32,
    /// The device's sysfs directory; reported for PCIe only.
    pub sysfs_path: Option<std::path::PathBuf>,
}

impl From<ara2::Interface> for Interface {
    fn from(i: ara2::Interface) -> Self {
        Interface {
            kind: i.kind.into(),
            bus: i.bus,
            device: i.device,
            port: i.port,
            sysfs_path: i.sysfs_path,
        }
    }
}

#[pymethods]
impl Interface {
    fn __repr__(&self) -> String {
        format!(
            "Interface(kind={:?}, bus={}, device={}, port={})",
            self.kind, self.bus, self.device, self.port
        )
    }
}

/// Static information about an endpoint's hardware.
#[pyclass(module = "edgefirst_ara2", frozen, get_all, from_py_object)]
#[derive(Clone, Debug)]
pub struct EndpointInfo {
    /// PCI-style device identifier.
    pub device_id: u32,
    /// Vendor identifier.
    pub vendor_id: u32,
    /// The ARA-2 chip, when the runtime reports it.
    pub chip: Option<ChipInfo>,
    /// The attached DRAM, when the runtime reports it.
    pub dram: Option<DramInfo>,
    /// How the endpoint is attached to the host, when the runtime reports it.
    pub interface: Option<Interface>,
}

impl From<ara2::EndpointInfo> for EndpointInfo {
    fn from(i: ara2::EndpointInfo) -> Self {
        EndpointInfo {
            device_id: i.device_id,
            vendor_id: i.vendor_id,
            chip: i.chip.map(Into::into),
            dram: i.dram.map(Into::into),
            interface: i.interface.map(Into::into),
        }
    }
}

#[pymethods]
impl EndpointInfo {
    fn __repr__(&self) -> String {
        format!(
            "EndpointInfo(device_id={:#x}, vendor_id={:#x})",
            self.device_id, self.vendor_id
        )
    }
}

/// Live operational statistics for an endpoint.
#[pyclass(module = "edgefirst_ara2", frozen, get_all, from_py_object)]
#[derive(Clone, Debug)]
pub struct EndpointStatistics {
    /// Current operational state of the endpoint.
    pub state: State,
    /// Endpoint clocks.
    pub clocks: Clocks,
    /// Average core voltage across the measurement points, in volts.
    /// DVAPI 1.3 and later; the DVAPI 1.1 runtime (Kinara runtime 1.2.1)
    /// does not measure it and reports 0.0.
    pub core_voltage_v: f32,
    /// Average endpoint temperature across the measurement points, in
    /// degrees Celsius.
    pub temp_c: f32,
    /// Soft resets the endpoint has performed.
    pub soft_reset_count: u32,
    /// DRAM usage statistics.
    pub dram: DramStatistics,
}

impl From<ara2::EndpointStatistics> for EndpointStatistics {
    fn from(s: ara2::EndpointStatistics) -> Self {
        EndpointStatistics {
            state: s.state.into(),
            clocks: s.clocks.into(),
            core_voltage_v: s.core_voltage_v,
            temp_c: s.temp_c,
            soft_reset_count: s.soft_reset_count,
            dram: s.dram.into(),
        }
    }
}

#[pymethods]
impl EndpointStatistics {
    fn __repr__(&self) -> String {
        format!(
            "EndpointStatistics(state={:?}, temp_c={}, soft_reset_count={})",
            self.state, self.temp_c, self.soft_reset_count
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
