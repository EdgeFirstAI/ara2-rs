// SPDX-License-Identifier: Apache-2.0
// Copyright © 2025 Au-Zone Technologies. All Rights Reserved.

use crate::error::{self, to_py_err};
use crate::types::{
    InputQuantization, InputTensorInfo, ModelTiming, OutputQuantization, OutputTensorInfo,
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
/// Typical workflow::
///
///     model = endpoint.load_model("model.dvm")
///     model.allocate_tensors()
///     model.set_input_tensor(0, input_data)
///     timing = model.run()
///     output = model.get_output_tensor(0)
#[pyclass(module = "edgefirst_ara2", unsendable)]
pub struct Model {
    inner: ara2::Model,
    tensors_allocated: bool,
}

impl Model {
    pub fn new(inner: ara2::Model) -> Self {
        Model {
            inner,
            tensors_allocated: false,
        }
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
        if index >= self.inner.n_inputs() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "input index {index} out of range (model has {} inputs)",
                self.inner.n_inputs()
            )));
        }
        Ok(())
    }

    fn check_output_index(&self, index: usize) -> PyResult<()> {
        if index >= self.inner.n_outputs() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "output index {index} out of range (model has {} outputs)",
                self.inner.n_outputs()
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
        self.inner.allocate_tensors(mem).map_err(to_py_err)?;
        self.tensors_allocated = true;
        Ok(())
    }

    /// Set the inference timeout in milliseconds.
    ///
    /// Args:
    ///     timeout_ms: Timeout in milliseconds (default: 1000)
    fn set_timeout_ms(&mut self, timeout_ms: i32) {
        self.inner.set_timeout_ms(timeout_ms);
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
        let timing = self.inner.run().map_err(to_py_err)?;
        Ok(ModelTiming::from(timing))
    }

    // ========================================================================
    // Tensor I/O (numpy)
    // ========================================================================

    /// Copy numpy array data into an input tensor.
    ///
    /// Args:
    ///     index: Input tensor index (0-based)
    ///     data: numpy ``uint8`` array with data to copy. The total byte
    ///           count must match the tensor size.
    ///
    /// Raises:
    ///     IndexError: If index is out of range
    ///     TensorError: If tensors are not allocated or sizes don't match
    fn set_input_tensor(&mut self, index: usize, data: &Bound<'_, pyo3::PyAny>) -> PyResult<()> {
        self.check_allocated()?;
        self.check_input_index(index)?;

        let arr: numpy::PyReadonlyArrayDyn<'_, u8> = data.extract()?;
        let src = arr.as_slice().map_err(tensor_err)?;

        let tensor = self.inner.input_tensor(index);
        let mut map = tensor.map().map_err(tensor_err)?;
        let dest = map.as_mut_slice();

        if src.len() != dest.len() {
            return Err(error::TensorError::new_err(format!(
                "input data size {} does not match tensor size {}",
                src.len(),
                dest.len()
            )));
        }

        dest.copy_from_slice(src);
        Ok(())
    }

    /// Get output tensor data as a numpy ``uint8`` array.
    ///
    /// Args:
    ///     index: Output tensor index (0-based)
    ///
    /// Returns:
    ///     numpy.ndarray: Flat ``uint8`` array containing the raw output bytes
    ///
    /// Raises:
    ///     IndexError: If index is out of range
    ///     TensorError: If tensors are not allocated
    fn get_output_tensor<'py>(&self, py: Python<'py>, index: usize) -> PyResult<PyObject> {
        self.check_allocated()?;
        self.check_output_index(index)?;

        let tensor = self.inner.output_tensor(index);
        let map = tensor.map().map_err(tensor_err)?;
        let data = map.as_slice().to_vec();
        let arr = ArrayD::<u8>::from_shape_vec(IxDyn(&[data.len()]), data).map_err(tensor_err)?;
        Ok(arr.into_pyarray(py).into_any().unbind())
    }

    /// Dequantize an output tensor to ``float32``.
    ///
    /// Args:
    ///     index: Output tensor index (0-based)
    ///
    /// Returns:
    ///     numpy.ndarray: Flat ``float32`` array with dequantized values
    ///
    /// Raises:
    ///     IndexError: If index is out of range
    ///     TensorError: If tensors are not allocated or quantization is invalid
    fn dequantize<'py>(&self, py: Python<'py>, index: usize) -> PyResult<PyObject> {
        self.check_allocated()?;
        self.check_output_index(index)?;

        let output = self.inner.output_tensor(index);
        let quant = self.inner.output_quants(index).map_err(to_py_err)?;

        if quant.qn == 0.0 {
            return Err(error::TensorError::new_err(
                "output tensor quantization scale (qn) is zero; cannot dequantize",
            ));
        }

        let map = output.map().map_err(tensor_err)?;
        let qn = 1.0 / quant.qn;

        let f32_data: Vec<f32> = if quant.is_signed {
            map.as_slice()
                .iter()
                .map(|&x| x as i8 as f32 * qn)
                .collect()
        } else {
            map.as_slice().iter().map(|&x| x as f32 * qn).collect()
        };

        let arr = ArrayD::<f32>::from_shape_vec(IxDyn(&[f32_data.len()]), f32_data)
            .map_err(tensor_err)?;
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

        let tensor = self.inner.input_tensor(index);
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

        let tensor = self.inner.output_tensor(index);
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
        Ok(memory_type_str(self.inner.input_tensor(index).memory()))
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
        Ok(memory_type_str(self.inner.output_tensor(index).memory()))
    }

    // ========================================================================
    // Introspection
    // ========================================================================

    /// Number of input tensors.
    #[getter]
    fn n_inputs(&self) -> usize {
        self.inner.n_inputs()
    }

    /// Number of output tensors.
    #[getter]
    fn n_outputs(&self) -> usize {
        self.inner.n_outputs()
    }

    /// Get the shape of an input tensor as (channels, height, width).
    ///
    /// Raises:
    ///     IndexError: If index is out of range
    fn input_shape(&self, index: usize) -> PyResult<(usize, usize, usize)> {
        self.check_input_index(index)?;
        let s = self.inner.input_shape(index);
        Ok((s[0], s[1], s[2]))
    }

    /// Get the shape of an output tensor as (channels, height, width).
    ///
    /// Raises:
    ///     IndexError: If index is out of range
    fn output_shape(&self, index: usize) -> PyResult<(usize, usize, usize)> {
        self.check_output_index(index)?;
        let s = self.inner.output_shape(index);
        Ok((s[0], s[1], s[2]))
    }

    /// Get the size in bytes of an input tensor.
    ///
    /// Raises:
    ///     IndexError: If index is out of range
    fn input_size(&self, index: usize) -> PyResult<usize> {
        self.check_input_index(index)?;
        Ok(self.inner.input_size(index))
    }

    /// Get the size in bytes of an output tensor.
    ///
    /// Raises:
    ///     IndexError: If index is out of range
    fn output_size(&self, index: usize) -> PyResult<usize> {
        self.check_output_index(index)?;
        Ok(self.inner.output_size(index))
    }

    /// Get the bytes per element for an input tensor.
    ///
    /// Raises:
    ///     IndexError: If index is out of range
    fn input_bpp(&self, index: usize) -> PyResult<usize> {
        self.check_input_index(index)?;
        Ok(self.inner.input_bpp(index))
    }

    /// Get the bytes per element for an output tensor.
    ///
    /// Raises:
    ///     IndexError: If index is out of range
    fn output_bpp(&self, index: usize) -> PyResult<usize> {
        self.check_output_index(index)?;
        Ok(self.inner.output_bpp(index))
    }

    /// Get detailed information about an input tensor.
    ///
    /// Raises:
    ///     IndexError: If index is out of range
    fn input_info(&self, index: usize) -> PyResult<InputTensorInfo> {
        self.check_input_index(index)?;
        Ok(InputTensorInfo::from(self.inner.input_info(index)))
    }

    /// Get detailed information about an output tensor.
    ///
    /// Raises:
    ///     IndexError: If index is out of range
    fn output_info(&self, index: usize) -> PyResult<OutputTensorInfo> {
        self.check_output_index(index)?;
        let info = self.inner.output_info(index).map_err(to_py_err)?;
        Ok(OutputTensorInfo::from(info))
    }

    /// Get quantization parameters for an input tensor.
    ///
    /// Raises:
    ///     IndexError: If index is out of range
    fn input_quants(&self, index: usize) -> PyResult<InputQuantization> {
        self.check_input_index(index)?;
        Ok(InputQuantization::from(self.inner.input_quants(index)))
    }

    /// Get quantization parameters for an output tensor.
    ///
    /// Raises:
    ///     IndexError: If index is out of range
    fn output_quants(&self, index: usize) -> PyResult<OutputQuantization> {
        self.check_output_index(index)?;
        let q = self.inner.output_quants(index).map_err(to_py_err)?;
        Ok(OutputQuantization::from(q))
    }

    fn __repr__(&self) -> String {
        format!(
            "Model(n_inputs={}, n_outputs={})",
            self.inner.n_inputs(),
            self.inner.n_outputs()
        )
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    #[allow(unused_variables)]
    fn __exit__(
        &self,
        exc_type: Option<&Bound<'_, pyo3::PyAny>>,
        exc_val: Option<&Bound<'_, pyo3::PyAny>>,
        exc_tb: Option<&Bound<'_, pyo3::PyAny>>,
    ) -> bool {
        false
    }
}

fn memory_type_str(memory: TensorMemory) -> &'static str {
    match memory {
        TensorMemory::Dma => "dma",
        TensorMemory::Shm => "shm",
        TensorMemory::Mem => "mem",
        TensorMemory::Pbo => "pbo",
    }
}
