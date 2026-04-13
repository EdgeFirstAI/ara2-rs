// SPDX-License-Identifier: Apache-2.0
// Copyright © 2025 Au-Zone Technologies. All Rights Reserved.

use crate::error::{self, to_py_err};
use crate::types::{
    InputPreprocess, InputQuantization, InputTensorInfo, ModelTiming, OutputQuantization,
    OutputTensorInfo,
};
use edgefirst_hal::tensor::{TensorMapTrait as _, TensorMemory, TensorTrait as _};
use numpy::IntoPyArray as _;
use numpy::ndarray::{ArrayD, IxDyn};
use pyo3::prelude::*;
use std::os::fd::IntoRawFd as _;

/// Convert a tensor error to a Python TensorError exception.
fn tensor_err(e: impl std::fmt::Display) -> PyErr {
    error::TensorError::new_err(e.to_string())
}

/// Neural network model loaded on an endpoint.
///
/// The Model class provides methods to run inference on a loaded neural network.
/// Input and output tensors are accessed via zero-based indices.
///
/// Supports the context manager protocol. The model is unloaded from the
/// NPU when the last Python reference is garbage-collected (via Rust Drop).
/// Using ``with`` ensures references are released promptly on scope exit.
///
/// Typical workflow::
///
///     with endpoint.load_model("model.dvm") as model:
///         model.allocate_tensors()
///         model.set_input_tensor(0, input_data)
///         timing = model.run()
///         output = model.get_output_tensor(0)
#[pyclass(module = "edgefirst_ara2", unsendable)]
pub struct Model {
    inner: Option<ara2::Model>,
    tensors_allocated: bool,
}

impl Model {
    pub fn new(inner: ara2::Model) -> Self {
        Model {
            inner: Some(inner),
            tensors_allocated: false,
        }
    }

    fn inner_ref(&self) -> PyResult<&ara2::Model> {
        self.inner
            .as_ref()
            .ok_or_else(|| error::Ara2Error::new_err("model is closed"))
    }

    fn inner_mut(&mut self) -> PyResult<&mut ara2::Model> {
        self.inner
            .as_mut()
            .ok_or_else(|| error::Ara2Error::new_err("model is closed"))
    }

    fn check_allocated(&self) -> PyResult<()> {
        if !self.tensors_allocated {
            return Err(error::TensorError::new_err(
                "tensors not allocated: call allocate_tensors() first",
            ));
        }
        Ok(())
    }

    fn check_input_index(&self, index: usize) -> PyResult<()> {
        let m = self.inner_ref()?;
        if index >= m.n_inputs() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "input index {index} out of range (model has {} inputs)",
                m.n_inputs()
            )));
        }
        Ok(())
    }

    fn check_output_index(&self, index: usize) -> PyResult<()> {
        let m = self.inner_ref()?;
        if index >= m.n_outputs() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "output index {index} out of range (model has {} outputs)",
                m.n_outputs()
            )));
        }
        Ok(())
    }
}

#[pymethods]
impl Model {
    // ========================================================================
    // Lifecycle
    // ========================================================================

    /// Allocate input and output tensors for this model.
    ///
    /// Must be called before ``run()``, ``set_input_tensor()``, or any
    /// tensor accessor method.
    ///
    /// Args:
    ///     memory: Memory type for tensor allocation. One of ``"dma"``,
    ///             ``"shm"``, ``"mem"``, or ``None`` for auto-selection
    ///             (tries DMA first). Use ``"dma"`` for zero-copy workflows
    ///             with edgefirst-hal.
    #[pyo3(signature = (memory=None))]
    fn allocate_tensors(&mut self, memory: Option<&str>) -> PyResult<()> {
        let mem = match memory {
            Some("dma") => Some(TensorMemory::Dma),
            Some("shm") => Some(TensorMemory::Shm),
            Some("mem") => Some(TensorMemory::Mem),
            None => None,
            Some(other) => {
                return Err(error::Ara2Error::new_err(format!(
                    "unknown memory type '{other}': expected 'dma', 'shm', 'mem', or None"
                )));
            }
        };
        self.inner_mut()?.allocate_tensors(mem).map_err(to_py_err)?;
        self.tensors_allocated = true;
        Ok(())
    }

    /// Set the inference timeout in milliseconds.
    ///
    /// Args:
    ///     timeout_ms: Timeout in milliseconds (default: 1000)
    fn set_timeout_ms(&mut self, timeout_ms: i32) -> PyResult<()> {
        self.inner_mut()?.set_timeout_ms(timeout_ms);
        Ok(())
    }

    /// Run inference on the model.
    ///
    /// Tensors must be allocated via ``allocate_tensors()`` before calling
    /// this method.
    ///
    /// Returns:
    ///     ModelTiming: Timing information for the inference run
    ///
    /// Raises:
    ///     TensorError: If tensors have not been allocated
    fn run(&mut self) -> PyResult<ModelTiming> {
        self.check_allocated()?;
        let timing = self.inner_mut()?.run().map_err(to_py_err)?;
        Ok(ModelTiming::from(timing))
    }

    // ========================================================================
    // Tensor I/O (numpy)
    // ========================================================================

    /// Copy numpy array data into an input tensor.
    ///
    /// Args:
    ///     index: Input tensor index (0-based)
    ///     data: numpy array whose total byte count matches the tensor
    ///           size. Any dtype (uint8, int8, uint16, int16, float32,
    ///           etc.) is accepted; the underlying buffer is copied
    ///           verbatim into the tensor, so the dtype must already
    ///           match what the model expects.
    ///
    /// Raises:
    ///     IndexError: If index is out of range
    ///     TensorError: If tensors are not allocated or sizes don't match
    fn set_input_tensor(&mut self, index: usize, data: &Bound<'_, pyo3::PyAny>) -> PyResult<()> {
        self.check_allocated()?;
        self.check_input_index(index)?;

        // Accept any numpy array by calling its ``tobytes()`` method. This
        // materializes a fresh C-contiguous bytes object — NumPy always
        // allocates a new buffer, even when the source is already
        // contiguous — so this path costs one extra copy beyond the
        // ``copy_from_slice`` into the tensor below. We accept that cost
        // here for simplicity and dtype-uniformity: a zero-copy buffer
        // protocol path would need to handle every possible element
        // format (uint8, int8, uint16, int16, float32, …) and
        // stride/alignment combination, whereas ``tobytes()`` collapses
        // all of that into a single well-defined byte stream.
        let bytes_obj = data.call_method0("tobytes")?;
        let bytes: &[u8] = bytes_obj.downcast::<pyo3::types::PyBytes>()?.as_bytes();

        let m = self.inner_mut()?;
        let tensor = m.input_tensor(index);
        let mut map = tensor.map().map_err(tensor_err)?;
        let dest = map.as_mut_slice();

        if bytes.len() != dest.len() {
            return Err(error::TensorError::new_err(format!(
                "input data size {} bytes does not match tensor size {} bytes",
                bytes.len(),
                dest.len()
            )));
        }

        dest.copy_from_slice(bytes);
        Ok(())
    }

    /// Get output tensor data as a typed numpy array.
    ///
    /// The dtype is derived from the tensor's quantization metadata:
    ///
    /// - ``bpp == 1`` and ``is_signed`` → ``int8``
    /// - ``bpp == 1`` and not signed → ``uint8``
    /// - ``bpp == 2`` and ``is_signed`` → ``int16``
    /// - ``bpp == 2`` and not signed → ``uint16``
    /// - ``bpp == 4`` → ``float32``
    ///
    /// The returned array is reshaped to the tensor's declared shape
    /// ``(channels, height, width)``; callers that need the legacy flat
    /// layout can call ``.ravel()``.
    ///
    /// Args:
    ///     index: Output tensor index (0-based)
    ///
    /// Returns:
    ///     numpy.ndarray: Shaped, typed view of the output tensor data
    ///
    /// Raises:
    ///     IndexError: If index is out of range
    ///     TensorError: If tensors are not allocated or have an unsupported bpp
    fn get_output_tensor<'py>(&self, py: Python<'py>, index: usize) -> PyResult<PyObject> {
        self.check_allocated()?;
        self.check_output_index(index)?;

        let m = self.inner_ref()?;
        let tensor = m.output_tensor(index);
        let map = tensor.map().map_err(tensor_err)?;
        let bytes = map.as_slice();

        let quant = m.output_quants(index).map_err(to_py_err)?;
        let bpp = m.output_bpp(index);
        let [c, h, w] = m.output_shape(index);
        let shape = IxDyn(&[c, h, w]);

        let obj: PyObject = match (bpp, quant.is_signed) {
            (1, false) => {
                let data: Vec<u8> = bytes.to_vec();
                ArrayD::<u8>::from_shape_vec(shape, data)
                    .map_err(tensor_err)?
                    .into_pyarray(py)
                    .into_any()
                    .unbind()
            }
            (1, true) => {
                let data: Vec<i8> = bytes.iter().map(|&b| b as i8).collect();
                ArrayD::<i8>::from_shape_vec(shape, data)
                    .map_err(tensor_err)?
                    .into_pyarray(py)
                    .into_any()
                    .unbind()
            }
            (2, false) => {
                let data: Vec<u16> = bytes
                    .chunks_exact(2)
                    .map(|ch| u16::from_le_bytes([ch[0], ch[1]]))
                    .collect();
                ArrayD::<u16>::from_shape_vec(shape, data)
                    .map_err(tensor_err)?
                    .into_pyarray(py)
                    .into_any()
                    .unbind()
            }
            (2, true) => {
                let data: Vec<i16> = bytes
                    .chunks_exact(2)
                    .map(|ch| i16::from_le_bytes([ch[0], ch[1]]))
                    .collect();
                ArrayD::<i16>::from_shape_vec(shape, data)
                    .map_err(tensor_err)?
                    .into_pyarray(py)
                    .into_any()
                    .unbind()
            }
            (4, _) => {
                let data: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|ch| f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]))
                    .collect();
                ArrayD::<f32>::from_shape_vec(shape, data)
                    .map_err(tensor_err)?
                    .into_pyarray(py)
                    .into_any()
                    .unbind()
            }
            (other, _) => {
                return Err(error::TensorError::new_err(format!(
                    "unsupported output bpp={other} for index {index}"
                )));
            }
        };

        Ok(obj)
    }

    /// Dequantize an output tensor to ``float32``.
    ///
    /// Uses the model's quantization mode (read from input 0) and the
    /// output tensor's ``(qn, offset)`` pair. Currently only qmode 9
    /// (asymmetric) is supported; other modes raise ``Ara2Error``.
    /// The returned array has the tensor's declared ``(C, H, W)`` shape.
    ///
    /// Args:
    ///     index: Output tensor index (0-based)
    ///
    /// Returns:
    ///     numpy.ndarray: ``float32`` array reshaped to ``(C, H, W)``
    ///
    /// Raises:
    ///     IndexError: If index is out of range
    ///     TensorError: If tensors are not allocated
    ///     Ara2Error: If the model uses an unsupported qmode
    fn dequantize<'py>(&self, py: Python<'py>, index: usize) -> PyResult<PyObject> {
        self.check_allocated()?;
        self.check_output_index(index)?;

        let m = self.inner_ref()?;
        let quant = m.output_quants(index).map_err(to_py_err)?;
        let qmode = m.input_quants(0).qmode;
        let (scale, offset) = quant.effective(qmode).map_err(to_py_err)?;

        let output = m.output_tensor(index);
        let map = output.map().map_err(tensor_err)?;
        let bytes = map.as_slice();

        let bpp = m.output_bpp(index);
        let [c, h, w] = m.output_shape(index);
        let shape = IxDyn(&[c, h, w]);

        let data: Vec<f32> = match (bpp, quant.is_signed) {
            (1, false) => bytes
                .iter()
                .map(|&b| (b as i32 - offset) as f32 * scale)
                .collect(),
            (1, true) => bytes
                .iter()
                .map(|&b| (b as i8 as i32 - offset) as f32 * scale)
                .collect(),
            (2, false) => bytes
                .chunks_exact(2)
                .map(|ch| {
                    let v = u16::from_le_bytes([ch[0], ch[1]]) as i32;
                    (v - offset) as f32 * scale
                })
                .collect(),
            (2, true) => bytes
                .chunks_exact(2)
                .map(|ch| {
                    let v = i16::from_le_bytes([ch[0], ch[1]]) as i32;
                    (v - offset) as f32 * scale
                })
                .collect(),
            (other, _) => {
                return Err(error::TensorError::new_err(format!(
                    "unsupported output bpp={other} for dequantize on index {index}"
                )));
            }
        };

        let arr = ArrayD::<f32>::from_shape_vec(shape, data).map_err(tensor_err)?;
        Ok(arr.into_pyarray(py).into_any().unbind())
    }

    // ========================================================================
    // DMA-BUF Zero-Copy Access
    // ========================================================================

    /// Get a cloned DMA-BUF file descriptor for an input tensor.
    ///
    /// The returned FD is owned by the caller. Pass it to
    /// ``edgefirst_hal.import_image()`` for zero-copy GPU preprocessing.
    /// The ``import_image`` function duplicates the FD internally, so you
    /// should close the returned FD with ``os.close()`` when done, or let
    /// ``import_image`` manage it.
    ///
    /// Args:
    ///     index: Input tensor index (0-based)
    ///
    /// Returns:
    ///     int: File descriptor for the input tensor's DMA-BUF
    ///
    /// Raises:
    ///     IndexError: If index is out of range
    ///     TensorError: If tensors are not allocated or use system memory
    fn input_tensor_fd(&mut self, index: usize) -> PyResult<i32> {
        self.check_allocated()?;
        self.check_input_index(index)?;

        let m = self.inner_mut()?;
        let tensor = m.input_tensor(index);
        let fd = tensor.clone_fd().map_err(tensor_err)?;
        Ok(fd.into_raw_fd())
    }

    /// Get a cloned DMA-BUF file descriptor for an output tensor.
    ///
    /// The returned FD is owned by the caller. Pass it to
    /// ``edgefirst_hal.import_image()`` for zero-copy GPU post-processing.
    /// Close with ``os.close()`` when done.
    ///
    /// Args:
    ///     index: Output tensor index (0-based)
    ///
    /// Returns:
    ///     int: File descriptor for the output tensor's DMA-BUF
    ///
    /// Raises:
    ///     IndexError: If index is out of range
    ///     TensorError: If tensors are not allocated or use system memory
    fn output_tensor_fd(&self, index: usize) -> PyResult<i32> {
        self.check_allocated()?;
        self.check_output_index(index)?;

        let m = self.inner_ref()?;
        let tensor = m.output_tensor(index);
        let fd = tensor.clone_fd().map_err(tensor_err)?;
        Ok(fd.into_raw_fd())
    }

    /// Get the memory type of an input tensor.
    ///
    /// Args:
    ///     index: Input tensor index (0-based)
    ///
    /// Returns:
    ///     str: ``"dma"``, ``"shm"``, or ``"mem"``
    ///
    /// Raises:
    ///     IndexError: If index is out of range
    ///     TensorError: If tensors are not allocated
    fn input_tensor_memory(&mut self, index: usize) -> PyResult<&str> {
        self.check_allocated()?;
        self.check_input_index(index)?;
        let m = self.inner_mut()?;
        Ok(memory_type_str(m.input_tensor(index).memory()))
    }

    /// Get the memory type of an output tensor.
    ///
    /// Args:
    ///     index: Output tensor index (0-based)
    ///
    /// Returns:
    ///     str: ``"dma"``, ``"shm"``, or ``"mem"``
    ///
    /// Raises:
    ///     IndexError: If index is out of range
    ///     TensorError: If tensors are not allocated
    fn output_tensor_memory(&self, index: usize) -> PyResult<&str> {
        self.check_allocated()?;
        self.check_output_index(index)?;
        let m = self.inner_ref()?;
        Ok(memory_type_str(m.output_tensor(index).memory()))
    }

    // ========================================================================
    // Introspection
    // ========================================================================

    /// Number of input tensors.
    #[getter]
    fn n_inputs(&self) -> PyResult<usize> {
        Ok(self.inner_ref()?.n_inputs())
    }

    /// Number of output tensors.
    #[getter]
    fn n_outputs(&self) -> PyResult<usize> {
        Ok(self.inner_ref()?.n_outputs())
    }

    /// Get the shape of an input tensor as (channels, height, width).
    ///
    /// Raises:
    ///     IndexError: If index is out of range
    fn input_shape(&self, index: usize) -> PyResult<(usize, usize, usize)> {
        self.check_input_index(index)?;
        let s = self.inner_ref()?.input_shape(index);
        Ok((s[0], s[1], s[2]))
    }

    /// Get the shape of an output tensor as (channels, height, width).
    ///
    /// Raises:
    ///     IndexError: If index is out of range
    fn output_shape(&self, index: usize) -> PyResult<(usize, usize, usize)> {
        self.check_output_index(index)?;
        let s = self.inner_ref()?.output_shape(index);
        Ok((s[0], s[1], s[2]))
    }

    /// Get the size in bytes of an input tensor.
    ///
    /// Raises:
    ///     IndexError: If index is out of range
    fn input_size(&self, index: usize) -> PyResult<usize> {
        self.check_input_index(index)?;
        Ok(self.inner_ref()?.input_size(index))
    }

    /// Get the size in bytes of an output tensor.
    ///
    /// Raises:
    ///     IndexError: If index is out of range
    fn output_size(&self, index: usize) -> PyResult<usize> {
        self.check_output_index(index)?;
        Ok(self.inner_ref()?.output_size(index))
    }

    /// Get the bytes per element for an input tensor.
    ///
    /// Raises:
    ///     IndexError: If index is out of range
    fn input_bpp(&self, index: usize) -> PyResult<usize> {
        self.check_input_index(index)?;
        Ok(self.inner_ref()?.input_bpp(index))
    }

    /// Get the bytes per element for an output tensor.
    ///
    /// Raises:
    ///     IndexError: If index is out of range
    fn output_bpp(&self, index: usize) -> PyResult<usize> {
        self.check_output_index(index)?;
        Ok(self.inner_ref()?.output_bpp(index))
    }

    /// Get detailed information about an input tensor.
    ///
    /// Raises:
    ///     IndexError: If index is out of range
    fn input_info(&self, index: usize) -> PyResult<InputTensorInfo> {
        self.check_input_index(index)?;
        Ok(InputTensorInfo::from(self.inner_ref()?.input_info(index)))
    }

    /// Get detailed information about an output tensor.
    ///
    /// Raises:
    ///     IndexError: If index is out of range
    fn output_info(&self, index: usize) -> PyResult<OutputTensorInfo> {
        self.check_output_index(index)?;
        let info = self.inner_ref()?.output_info(index).map_err(to_py_err)?;
        Ok(OutputTensorInfo::from(info))
    }

    /// Get quantization parameters for an input tensor.
    ///
    /// Raises:
    ///     IndexError: If index is out of range
    fn input_quants(&self, index: usize) -> PyResult<InputQuantization> {
        self.check_input_index(index)?;
        Ok(InputQuantization::from(
            self.inner_ref()?.input_quants(index),
        ))
    }

    /// Get preprocessing parameters for an input tensor.
    ///
    /// Returns the per-channel ``mean`` and ``scale`` used to normalize
    /// float input data before quantization, plus layout-affecting flags
    /// (``bgr_to_rgb``, ``aspect_resize``, ``mirror``, ``center_crop``).
    ///
    /// Raises:
    ///     IndexError: If index is out of range
    fn input_preprocess(&self, index: usize) -> PyResult<InputPreprocess> {
        self.check_input_index(index)?;
        Ok(InputPreprocess::from(
            self.inner_ref()?.input_preprocess(index),
        ))
    }

    /// Get quantization parameters for an output tensor.
    ///
    /// Raises:
    ///     IndexError: If index is out of range
    fn output_quants(&self, index: usize) -> PyResult<OutputQuantization> {
        self.check_output_index(index)?;
        let q = self.inner_ref()?.output_quants(index).map_err(to_py_err)?;
        Ok(OutputQuantization::from(q))
    }

    /// Unload the model and release its resources.
    ///
    /// After calling ``close()``, any further method call raises
    /// ``Ara2Error``. Safe to call multiple times.
    fn close(&mut self) {
        self.inner = None;
        self.tensors_allocated = false;
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            Some(m) => format!(
                "Model(n_inputs={}, n_outputs={})",
                m.n_inputs(),
                m.n_outputs()
            ),
            None => "Model(closed)".to_string(),
        }
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    #[allow(unused_variables)]
    fn __exit__(
        &mut self,
        exc_type: Option<&Bound<'_, pyo3::PyAny>>,
        exc_val: Option<&Bound<'_, pyo3::PyAny>>,
        exc_tb: Option<&Bound<'_, pyo3::PyAny>>,
    ) -> bool {
        self.close();
        false
    }
}

fn memory_type_str(memory: TensorMemory) -> &'static str {
    match memory {
        TensorMemory::Dma => "dma",
        TensorMemory::Shm => "shm",
        // Pbo (OpenGL pixel buffer) is a GPU-backed fallback for system memory;
        // expose it as "mem" since Python users only care about DMA-BUF capability.
        TensorMemory::Mem | TensorMemory::Pbo => "mem",
    }
}
