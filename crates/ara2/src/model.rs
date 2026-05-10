use crate::{Error, session::SessionInner};
use ara2_sys::{
    DV_INFERENCE_STATUS_DV_INFERENCE_STATUS_COMPLETED,
    DV_INFERENCE_STATUS_DV_INFERENCE_STATUS_FAILED, DV_INFERENCE_STATUS_DV_INFERENCE_STATUS_QUEUED,
    DV_INFERENCE_STATUS_DV_INFERENCE_STATUS_RUNNING, DV_LAYER_OUTPUT_TYPE, dv_blob, dv_endpoint,
    dv_infer_request, dv_model, dv_shm_descriptor,
};
use edgefirst_hal::tensor::{Tensor, TensorMap, TensorMapTrait as _, TensorMemory, TensorTrait};
use log::debug;
use ndarray::parallel::prelude::{
    IndexedParallelIterator as _, IntoParallelRefMutIterator as _, ParallelIterator as _,
};
use std::{ops::Add, os::fd::AsRawFd, sync::Arc, time::Duration};

/// Timing statistics from a model inference run.
#[derive(Clone, Copy, Debug)]
pub struct ModelTiming {
    pub run_time: Duration,
    pub input_time: Duration,
    pub output_time: Duration,
}

impl Default for ModelTiming {
    fn default() -> Self {
        ModelTiming {
            run_time: Duration::ZERO,
            input_time: Duration::ZERO,
            output_time: Duration::ZERO,
        }
    }
}

impl Add for ModelTiming {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        ModelTiming {
            run_time: self.run_time + other.run_time,
            input_time: self.input_time + other.input_time,
            output_time: self.output_time + other.output_time,
        }
    }
}

/// Default inference timeout in milliseconds.
pub const DEFAULT_TIMEOUT_MS: i32 = 1000;

/// A neural network model loaded on an ARA-2 endpoint.
///
/// Models own their loaded resources and are NOT cloneable. When dropped,
/// the model is automatically unloaded from the NPU.
///
/// The model keeps the session alive through reference counting, so it's
/// safe to drop the original `Session` and `Endpoint` handles after loading.
///
/// # Inference Modes
///
/// Two inference modes are available:
///
/// - **Synchronous** — [`run()`](Self::run) blocks until the NPU finishes
///   and returns timing. Simplest path for single-inference workloads.
/// - **Asynchronous** — [`submit()`](Self::submit) returns an
///   [`InferRequest`] handle immediately, allowing the CPU to do other work
///   (e.g., preprocessing the next frame) while the NPU executes. Call
///   [`InferRequest::wait()`] to retrieve the result.
///
/// # Example
///
/// ```no_run
/// use ara2::{Session, DEFAULT_SOCKET, DEFAULT_TIMEOUT_MS};
/// use edgefirst_hal::tensor::TensorMemory;
///
/// let session = Session::create_via_unix_socket(DEFAULT_SOCKET)?;
/// let endpoints = session.list_endpoints()?;
/// let mut model = endpoints[0].load_model_from_file("model.dvm".as_ref())?;
/// model.allocate_tensors(Some(TensorMemory::Dma))?;
///
/// // Synchronous inference
/// let timing = model.run()?;
///
/// // Asynchronous inference — overlap CPU work with NPU execution
/// let request = model.submit()?;
/// // ... do CPU work here while the NPU is busy ...
/// let timing = request.wait(DEFAULT_TIMEOUT_MS)?;
/// # Ok::<(), ara2::Error>(())
/// ```
pub struct Model {
    session: Arc<SessionInner>,
    endpoint_ptr: *mut dv_endpoint,
    ptr: *mut dv_model,
    inputs: Vec<(Tensor<u8>, *mut dv_shm_descriptor)>,
    outputs: Vec<(Tensor<u8>, *mut dv_shm_descriptor)>,
    timeout_ms: i32,
}

impl std::fmt::Debug for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Model")
            .field("n_inputs", &self.n_inputs())
            .field("n_outputs", &self.n_outputs())
            .field("timeout_ms", &self.timeout_ms)
            .finish()
    }
}

// Safety: Model contains Arc (Send+Sync) and raw pointers used only for FFI.
// The FFI calls go through the session's library handle.
// Model should be Send to allow moving between threads, but not Sync
// as inference operations are not thread-safe.
unsafe impl Send for Model {}

/// Blob descriptors paired with map guards that keep memory mappings alive.
type BlobsWithGuards = (Vec<dv_blob>, Vec<dv_blob>, Vec<TensorMap<u8>>);

impl Model {
    /// Create a new Model instance.
    pub(crate) fn new(
        session: Arc<SessionInner>,
        endpoint_ptr: *mut dv_endpoint,
        ptr: *mut dv_model,
    ) -> Self {
        Model {
            session,
            endpoint_ptr,
            ptr,
            inputs: vec![],
            outputs: vec![],
            timeout_ms: DEFAULT_TIMEOUT_MS,
        }
    }

    /// Set the inference timeout in milliseconds.
    ///
    /// The default is [`DEFAULT_TIMEOUT_MS`] (1000ms). Increase this for
    /// large models that take longer to execute.
    pub fn set_timeout_ms(&mut self, timeout_ms: i32) {
        self.timeout_ms = timeout_ms;
    }

    /// Build blob descriptors from the pre-allocated tensor buffers.
    ///
    /// Returns `(input_blobs, output_blobs, map_guards)`. For SHM/DMA
    /// tensors the blob references the registered descriptor; for plain
    /// memory it references the mapped pointer directly. The returned
    /// `map_guards` vector keeps the memory mappings alive — callers must
    /// hold it until the FFI call using the blobs has completed.
    fn build_blobs(&self) -> Result<BlobsWithGuards, Error> {
        let mut map_guards: Vec<TensorMap<u8>> = Vec::new();

        let input_blobs = self
            .inputs
            .iter()
            .map(|tensor| {
                let blob = if tensor.1.is_null() {
                    let map = tensor.0.map()?;
                    let ptr = map.as_ptr() as *mut std::ffi::c_void;
                    map_guards.push(map);
                    (ptr, ara2_sys::DV_BLOB_TYPE_DV_BLOB_TYPE_RAW_POINTER)
                } else {
                    (
                        tensor.1 as *mut std::ffi::c_void,
                        ara2_sys::DV_BLOB_TYPE_DV_BLOB_TYPE_SHM_DESCRIPTOR,
                    )
                };

                Ok(dv_blob {
                    handle: blob.0,
                    offset: 0,
                    size: tensor.0.size() as u64,
                    blob_type: blob.1,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;

        let output_blobs = self
            .outputs
            .iter()
            .map(|tensor| {
                let blob = if tensor.1.is_null() {
                    let map = tensor.0.map()?;
                    let ptr = map.as_ptr() as *mut std::ffi::c_void;
                    map_guards.push(map);
                    (ptr, ara2_sys::DV_BLOB_TYPE_DV_BLOB_TYPE_RAW_POINTER)
                } else {
                    (
                        tensor.1 as *mut std::ffi::c_void,
                        ara2_sys::DV_BLOB_TYPE_DV_BLOB_TYPE_SHM_DESCRIPTOR,
                    )
                };

                Ok(dv_blob {
                    handle: blob.0,
                    offset: 0,
                    size: tensor.0.size() as u64,
                    blob_type: blob.1,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;

        Ok((input_blobs, output_blobs, map_guards))
    }

    /// Run inference synchronously on the model using the pre-allocated
    /// tensors.
    ///
    /// Blocks until the NPU finishes and returns timing statistics. For
    /// workloads where you want to overlap CPU preprocessing with NPU
    /// execution, use [`submit()`](Self::submit) instead.
    ///
    /// Call [`allocate_tensors`](Self::allocate_tensors) before calling
    /// this method to set up input and output buffers.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ara2::{Session, DEFAULT_SOCKET};
    /// # let session = Session::create_via_unix_socket(DEFAULT_SOCKET)?;
    /// # let endpoints = session.list_endpoints()?;
    /// # let mut model = endpoints[0].load_model_from_file("m.dvm".as_ref())?;
    /// # model.allocate_tensors(None)?;
    /// let timing = model.run()?;
    /// println!("NPU: {:?}, DMA in: {:?}, DMA out: {:?}",
    ///     timing.run_time, timing.input_time, timing.output_time);
    /// # Ok::<(), ara2::Error>(())
    /// ```
    pub fn run(&mut self) -> Result<ModelTiming, Error> {
        // `_map_guards` keeps memory mappings alive through the FFI call.
        let (mut input_blobs, mut output_blobs, _map_guards) = self.build_blobs()?;

        let mut request: *mut dv_infer_request = std::ptr::null_mut();

        let err = unsafe {
            self.session.lib.dv_infer_sync(
                self.session.ptr,
                self.endpoint_ptr,
                self.ptr,
                input_blobs.as_mut_ptr(),
                output_blobs.as_mut_ptr(),
                self.timeout_ms,
                true,
                &mut request,
            )
        };

        if err != 0 {
            return Err(err.into());
        }

        let timing = extract_timing(request)?;

        let err = unsafe { self.session.lib.dv_infer_free(request) };

        if err != 0 {
            return Err(err.into());
        }

        Ok(timing)
    }

    /// Submit inference asynchronously.
    ///
    /// Returns an [`InferRequest`] handle immediately without waiting for
    /// the NPU to finish. Call [`InferRequest::wait`] to block until the
    /// result is ready. This enables overlapping CPU preprocessing with
    /// NPU execution — the key building block for pipeline parallelism.
    ///
    /// # Safety contract
    ///
    /// The caller **must not** drop or reallocate this model's tensors
    /// (via [`allocate_tensors`](Self::allocate_tensors)) while the
    /// returned `InferRequest` is still pending. The NPU reads from and
    /// writes to the tensor buffers asynchronously.
    ///
    /// Call [`allocate_tensors`](Self::allocate_tensors) before calling
    /// this method to set up input and output buffers.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ara2::{Session, DEFAULT_SOCKET, DEFAULT_TIMEOUT_MS};
    /// # let session = Session::create_via_unix_socket(DEFAULT_SOCKET)?;
    /// # let endpoints = session.list_endpoints()?;
    /// # let mut model = endpoints[0].load_model_from_file("m.dvm".as_ref())?;
    /// # model.allocate_tensors(None)?;
    /// // Submit inference — returns immediately
    /// let request = model.submit()?;
    ///
    /// // CPU is free to do other work while the NPU executes
    /// println!("Request #{} submitted", request.request_id()?);
    ///
    /// // Block until the NPU finishes (up to 1000ms)
    /// let timing = request.wait(DEFAULT_TIMEOUT_MS)?;
    /// println!("NPU inference: {:?}", timing.run_time);
    /// # Ok::<(), ara2::Error>(())
    /// ```
    pub fn submit(&mut self) -> Result<InferRequest, Error> {
        // `_map_guards` keeps memory mappings alive through the FFI call.
        // For async inference the NPU copies blob data during submission,
        // so guards only need to survive the `dv_infer_async` call itself.
        let (mut input_blobs, mut output_blobs, _map_guards) = self.build_blobs()?;

        let mut request: *mut dv_infer_request = std::ptr::null_mut();

        let err = unsafe {
            self.session.lib.dv_infer_async(
                self.session.ptr,
                self.endpoint_ptr,
                self.ptr,
                input_blobs.as_mut_ptr(),
                output_blobs.as_mut_ptr(),
                true,
                &mut request,
            )
        };

        if err != 0 {
            return Err(err.into());
        }

        Ok(InferRequest {
            session: Arc::clone(&self.session),
            ptr: request,
        })
    }

    /// Get a mutable reference to an input tensor.
    pub fn input_tensor(&mut self, idx: usize) -> &mut Tensor<u8> {
        &mut self.inputs[idx].0
    }

    /// Get a reference to an output tensor.
    pub fn output_tensor(&self, idx: usize) -> &Tensor<u8> {
        &self.outputs[idx].0
    }

    /// Allocate input and output tensors for this model.
    ///
    /// # Arguments
    /// * `memory` - The memory type to use for tensors. Use `TensorMemory::Dma`
    ///   or `TensorMemory::Shm` for zero-copy inference.
    ///
    /// Input tensors are allocated with CHW shape `[channels, height, width]`
    /// to enable zero-copy preprocessing with HAL's `TensorImageRef`.
    /// Output tensors use flat shape since they may have padding or
    /// different element sizes.
    pub fn allocate_tensors(&mut self, memory: Option<TensorMemory>) -> Result<(), Error> {
        self.inputs = (0..self.n_inputs())
            .map(|i| {
                // Allocate with CHW shape for compatibility with HAL TensorImageRef
                let shape = self.input_shape(i);
                let tensor = Tensor::<u8>::new(&shape, memory, None)?;
                match tensor.memory() {
                    TensorMemory::Shm | TensorMemory::Dma => {
                        let desc = self.shmfd_register(&tensor)?;
                        Ok((tensor, desc))
                    }
                    _ => Ok((tensor, std::ptr::null_mut())),
                }
            })
            .collect::<Result<Vec<_>, Error>>()?;
        self.outputs = (0..self.n_outputs())
            .map(|i| {
                // Use the actual size from the model, not computed from shape
                let size = self.output_size(i);
                let tensor = Tensor::<u8>::new(&[size], memory, None)?;
                match tensor.memory() {
                    TensorMemory::Shm | TensorMemory::Dma => {
                        let desc = self.shmfd_register(&tensor)?;
                        Ok((tensor, desc))
                    }
                    _ => Ok((tensor, std::ptr::null_mut())),
                }
            })
            .collect::<Result<Vec<_>, Error>>()?;

        Ok(())
    }

    /// Register a tensor's file descriptor with the proxy for zero-copy
    /// transfers.
    fn shmfd_register(&self, tensor: &Tensor<u8>) -> Result<*mut dv_shm_descriptor, Error> {
        let mut desc: *mut dv_shm_descriptor = std::ptr::null_mut();
        match tensor.memory() {
            TensorMemory::Shm | TensorMemory::Dma => {}
            other => {
                return Err(Error::UnsupportedLayout(format!(
                    "shmfd_register only supports Shm or Dma memory, got {other:?}"
                )));
            }
        }
        let fd = tensor
            .clone_fd()
            .map_err(|e| Error::UnsupportedLayout(format!("clone_fd failed: {e}")))?;

        let result = unsafe {
            self.session.lib.dv_shmfd_register(
                self.session.ptr,
                fd.as_raw_fd(),
                tensor.size() as u32,
                0,
                0,
                &mut desc,
            )
        };

        if result != 0 {
            return Err(result.into());
        }

        if desc.is_null() {
            return Err(Error::NullPointer(
                "shmfd_register returned null".to_string(),
            ));
        }

        debug!(
            "shmfd_register {:?} fd {:?} => {:?}",
            tensor.memory(),
            fd,
            unsafe { *desc }
        );

        Ok(desc)
    }

    /// Get the number of input tensors for this model.
    pub fn n_inputs(&self) -> usize {
        unsafe { (*self.ptr).num_inputs as usize }
    }

    /// Get the shape of an input tensor.
    pub fn input_shape(&self, idx: usize) -> [usize; 3] {
        let input = unsafe { (*self.ptr).input_param.add(idx) };
        unsafe {
            [
                (*input).nch as usize,
                (*input).height as usize,
                (*input).width as usize,
            ]
        }
    }

    /// Get the quantization parameters for an input tensor.
    pub fn input_quants(&self, idx: usize) -> InputQuantization {
        self.input_info(idx).quant
    }

    /// Get the preprocessing parameters for an input tensor.
    pub fn input_preprocess(&self, idx: usize) -> InputPreprocess {
        self.input_info(idx).preprocess
    }

    /// Get detailed information about an input tensor.
    pub fn input_info(&self, idx: usize) -> InputTensor {
        let input = unsafe { (*self.ptr).input_param.add(idx) };

        let params = unsafe { (*input).preprocess_param };

        // Safe per-channel mean/scale arrays: the C FFI exposes these as
        // pointers that may be null when nch > 3. Compute once under a
        // single null guard and reuse for both the debug log and the
        // struct construction so the log cannot UB on a null pointer.
        let (mean_arr, scale_arr) = unsafe {
            if (*input).nch as usize <= 3 && !(*params).mean.is_null() && !(*params).scale.is_null()
            {
                let m = std::slice::from_raw_parts((*params).mean, 3);
                let s = std::slice::from_raw_parts((*params).scale, 3);
                ([m[0], m[1], m[2]], [s[0], s[1], s[2]])
            } else {
                ([0.0; 3], [1.0; 3])
            }
        };

        unsafe {
            debug!(
                "input_{} qn: {:?} scale: {:?} mean: {:?} is_signed: {:?}",
                idx,
                (*params).qn,
                scale_arr,
                mean_arr,
                (*params).is_signed
            )
        };

        unsafe {
            InputTensor {
                layer_id: (*input).layer_id,
                blob_id: (*input).blob_id,
                layer_name: std::ffi::CStr::from_ptr((*input).layer_name)
                    .to_string_lossy()
                    .into_owned(),
                blob_name: std::ffi::CStr::from_ptr((*input).blob_name)
                    .to_string_lossy()
                    .into_owned(),
                layer_type: std::ffi::CStr::from_ptr((*input).layer_type)
                    .to_string_lossy()
                    .into_owned(),
                layout: std::ffi::CStr::from_ptr((*input).layout)
                    .to_string_lossy()
                    .into_owned(),
                size: (*input).size as usize,
                width: (*input).width as usize,
                height: (*input).height as usize,
                nch: (*input).nch as usize,
                bpp: (*input).bpp as usize,
                batch_size: (*input).batch_size as usize,
                quant: InputQuantization {
                    qn: (*params).qn,
                    offset: (*params).offset,
                    is_signed: (*params).is_signed,
                    qmode: (*params).qmode,
                },
                preprocess: InputPreprocess {
                    mean: mean_arr,
                    scale: scale_arr,
                    bgr_to_rgb: (*params).bgr_to_rgb,
                    aspect_resize: (*params).aspect_resize,
                    mirror: (*params).mirror,
                    center_crop: (*params).center_crop,
                },
            }
        }
    }

    /// Get the number of output tensors for this model.
    pub fn n_outputs(&self) -> usize {
        unsafe { (*self.ptr).num_outputs as usize }
    }

    /// Get the shape of an output tensor.
    pub fn output_shape(&self, idx: usize) -> [usize; 3] {
        let output = unsafe { (*self.ptr).output_param.add(idx) };
        unsafe {
            [
                (*output).nch as usize,
                (*output).height as usize,
                (*output).width as usize,
            ]
        }
    }

    /// Get the size in bytes of an output tensor (as specified by the model).
    pub fn output_size(&self, idx: usize) -> usize {
        let output = unsafe { (*self.ptr).output_param.add(idx) };
        unsafe { (*output).size as usize }
    }

    /// Get the size in bytes of an input tensor (as specified by the model).
    pub fn input_size(&self, idx: usize) -> usize {
        let input = unsafe { (*self.ptr).input_param.add(idx) };
        unsafe { (*input).size as usize }
    }

    /// Get the bytes per pixel (element size) for an output tensor.
    /// Returns 1 for int8/uint8, 2 for int16/uint16, 4 for float32, etc.
    pub fn output_bpp(&self, idx: usize) -> usize {
        let output = unsafe { (*self.ptr).output_param.add(idx) };
        unsafe { (*output).bpp as usize }
    }

    /// Get the bytes per pixel (element size) for an input tensor.
    pub fn input_bpp(&self, idx: usize) -> usize {
        let input = unsafe { (*self.ptr).input_param.add(idx) };
        unsafe { (*input).bpp as usize }
    }

    /// Dequantize an output tensor to f32.
    ///
    /// Reads the quantized output at `idx` and writes dequantized f32 values
    /// into `tensor`. The tensor must be pre-allocated with `C * H * W`
    /// f32 elements.
    ///
    /// Supported tensor element sizes: 1 byte (uint8/int8) and 2 bytes
    /// (uint16/int16). Other bpp values return
    /// [`Error::UnsupportedTypeSize`].
    ///
    /// The dequantization formula is `(raw - offset) * scale`, where the
    /// effective scale and offset are derived from the model's `qmode` via
    /// [`OutputQuantization::effective`]. Currently only qmode 9 is
    /// supported; other modes return [`Error::UnsupportedQmode`].
    pub fn dequantize(&self, idx: usize, tensor: &Tensor<f32>) -> Result<(), Error> {
        let output = self.output_tensor(idx);
        let quant = self.output_quants(idx)?;
        let qmode = self.input_quants(0).qmode;
        let (scale, offset) = quant.effective(qmode)?;
        let bpp = self.output_bpp(idx);

        let mut tensor_map = tensor.map()?;
        let output_map = output.map()?;
        let src = output_map.as_slice();
        let dst = tensor_map.as_mut_slice();

        // Reject unsupported element sizes early so the length check below
        // has a well-defined expected byte count. Only 1- and 2-byte
        // elements are implemented.
        if !matches!(bpp, 1 | 2) {
            return Err(Error::UnsupportedTypeSize(bpp));
        }

        // Up-front length check: every branch below assumes
        // src.len() == dst.len() * bpp. Without this check a mismatch
        // would panic inside the parallel loop on out-of-bounds indexing.
        let expected = dst
            .len()
            .checked_mul(bpp)
            .ok_or(Error::TensorSizeMismatch {
                expected: usize::MAX,
                got: src.len(),
            })?;
        if src.len() != expected {
            return Err(Error::TensorSizeMismatch {
                expected,
                got: src.len(),
            });
        }

        match (bpp, quant.is_signed) {
            (1, false) => {
                dst.par_iter_mut().enumerate().for_each(|(i, out)| {
                    *out = dequant_u8(src[i], offset, scale);
                });
            }
            (1, true) => {
                dst.par_iter_mut().enumerate().for_each(|(i, out)| {
                    *out = dequant_i8(src[i], offset, scale);
                });
            }
            (2, false) => {
                dst.par_iter_mut().enumerate().for_each(|(i, out)| {
                    *out = dequant_u16_le(src[2 * i], src[2 * i + 1], offset, scale);
                });
            }
            (2, true) => {
                dst.par_iter_mut().enumerate().for_each(|(i, out)| {
                    *out = dequant_i16_le(src[2 * i], src[2 * i + 1], offset, scale);
                });
            }
            // bpp already validated above.
            _ => unreachable!(),
        }

        Ok(())
    }

    /// Get the quantization parameters for an output tensor.
    pub fn output_quants(&self, idx: usize) -> Result<OutputQuantization, Error> {
        Ok(self.output_info(idx)?.quant)
    }

    /// Get detailed information about an output tensor.
    pub fn output_info(&self, idx: usize) -> Result<OutputTensor, Error> {
        let output = unsafe { (*self.ptr).output_param.add(idx) };

        let params = unsafe { (*output).postprocess_param };
        unsafe {
            debug!(
                "output_{} qn: {:?} scale: {:?} offset: {:?} is_signed: {:?}",
                idx,
                (*params).qn,
                (*params).output_scale,
                (*params).offset,
                (*params).is_signed
            )
        };

        unsafe {
            Ok(OutputTensor {
                layer_id: (*output).layer_id,
                blob_id: (*output).blob_id,
                fused_parent_id: (*output).fused_parent_id,
                layer_name: std::ffi::CStr::from_ptr((*output).layer_name)
                    .to_string_lossy()
                    .into_owned(),
                blob_name: std::ffi::CStr::from_ptr((*output).blob_name)
                    .to_string_lossy()
                    .into_owned(),
                layer_fused_parent_name: std::ffi::CStr::from_ptr(
                    (*output).layer_fused_parent_name,
                )
                .to_string_lossy()
                .into_owned(),
                layer_type: std::ffi::CStr::from_ptr((*output).layer_type)
                    .to_string_lossy()
                    .into_owned(),
                layout: std::ffi::CStr::from_ptr((*output).layout)
                    .to_string_lossy()
                    .into_owned(),
                size: (*output).size as usize,
                width: (*output).width as usize,
                height: (*output).height as usize,
                nch: (*output).nch as usize,
                bpp: (*output).bpp as usize,
                num_classes: (*output).num_classes as usize,
                layer_output_type: (*output).layer_output_type.try_into()?,
                max_dynamic_id: (*output).max_dynamic_id,
                quant: OutputQuantization {
                    qn: (*params).qn,
                    offset: (*params).offset,
                    is_signed: (*params).is_signed,
                },
            })
        }
    }
}

/// Extract [`ModelTiming`] from a completed inference request.
///
/// # Safety
///
/// The `request` pointer must be non-null and point to a valid, completed
/// `dv_infer_request`.
///
/// # Errors
///
/// Returns [`Error::NullPointer`] if the `stats` field is null, which
/// indicates the C library did not populate timing data (e.g., statistics
/// were not enabled).
fn extract_timing(request: *mut dv_infer_request) -> Result<ModelTiming, Error> {
    unsafe {
        let timing_ptr = (*request).stats;
        if timing_ptr.is_null() {
            return Err(Error::NullPointer(
                "dv_infer_request.stats is null — timing data unavailable".to_owned(),
            ));
        }
        Ok(ModelTiming {
            run_time: Duration::from_micros((*timing_ptr).inference_execution_time as u64),
            input_time: Duration::from_micros((*timing_ptr).input_transfer_time as u64),
            output_time: Duration::from_micros((*timing_ptr).output_transfer_time as u64),
        })
    }
}

/// A pending asynchronous inference request.
///
/// Created by [`Model::submit`]. Call [`wait`](Self::wait) to block until
/// the NPU finishes and retrieve timing statistics.
///
/// The underlying C request is freed automatically on drop via
/// `dv_infer_free`. If the request has not completed when it is dropped,
/// the C library cancels it.
///
/// # Thread safety
///
/// `InferRequest` is [`Send`] — it can be moved to another thread and
/// waited on there. This enables patterns like submitting from a
/// preprocessing thread and waiting from a postprocessing thread.
///
/// # Ownership
///
/// The `InferRequest` borrows the model's tensor buffers. The caller must
/// keep the originating [`Model`] alive (and must not call
/// [`allocate_tensors`](Model::allocate_tensors)) until this request is
/// consumed by [`wait`](Self::wait) or dropped.
///
/// # Example
///
/// ```no_run
/// # use ara2::{Session, DEFAULT_SOCKET, DEFAULT_TIMEOUT_MS};
/// # let session = Session::create_via_unix_socket(DEFAULT_SOCKET)?;
/// # let endpoints = session.list_endpoints()?;
/// # let mut model = endpoints[0].load_model_from_file("m.dvm".as_ref())?;
/// # model.allocate_tensors(None)?;
/// let request = model.submit()?;
/// let id = request.request_id()?;
///
/// // Do preprocessing for the next frame while NPU is busy...
///
/// let timing = request.wait(DEFAULT_TIMEOUT_MS)?;
/// println!("Request {id}: NPU={:?}", timing.run_time);
/// # Ok::<(), ara2::Error>(())
/// ```
pub struct InferRequest {
    session: Arc<SessionInner>,
    ptr: *mut dv_infer_request,
}

// Safety: The C library is internally synchronized for inference
// request operations. The pointer is only used through the library
// handle.
unsafe impl Send for InferRequest {}

impl InferRequest {
    /// Block until this inference request completes and return timing.
    ///
    /// Consumes the request handle — the underlying C object is freed
    /// after extracting timing information. Calling `wait` a second time
    /// is not possible since `self` is moved.
    ///
    /// # Cancellation on error
    ///
    /// Because `wait` takes `self` by value, any error (including timeout)
    /// drops the `InferRequest`, which calls `dv_infer_free` and cancels
    /// the in-flight request. If you need longer wait times, increase
    /// `timeout_ms` rather than retrying — the timeout applies per poll
    /// iteration, not cumulatively.
    ///
    /// # Arguments
    ///
    /// * `timeout_ms` - Maximum time in milliseconds to wait per poll
    ///   iteration. Use [`DEFAULT_TIMEOUT_MS`] for the default (1000 ms).
    ///   Increase for large models that take longer to execute.
    ///
    /// # Errors
    ///
    /// - [`Error::Ara2`] — the wait call itself failed (e.g., timeout
    ///   expired). The request is cancelled on drop.
    /// - [`Error::InferenceFailed`] — the NPU reported a failure
    /// - [`Error::InferenceNotCompleted`] — the request reached an
    ///   unexpected terminal state (e.g., UNKNOWN)
    /// - [`Error::NullPointer`] — the C API returned a null pointer
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ara2::{Session, DEFAULT_SOCKET, DEFAULT_TIMEOUT_MS};
    /// # let session = Session::create_via_unix_socket(DEFAULT_SOCKET)?;
    /// # let endpoints = session.list_endpoints()?;
    /// # let mut model = endpoints[0].load_model_from_file("m.dvm".as_ref())?;
    /// # model.allocate_tensors(None)?;
    /// let request = model.submit()?;
    /// let timing = request.wait(DEFAULT_TIMEOUT_MS)?;
    /// println!("Inference took {:?}", timing.run_time);
    /// # Ok::<(), ara2::Error>(())
    /// ```
    pub fn wait(self, timeout_ms: i32) -> Result<ModelTiming, Error> {
        let mut completed: *mut dv_infer_request = std::ptr::null_mut();
        let mut list = [self.ptr];

        // Loop until the request reaches a terminal state (COMPLETED,
        // FAILED, or UNKNOWN). The C API's `dv_infer_wait_for_completion`
        // returns whenever a request's status changes, which may be an
        // intermediate transition (e.g., QUEUED → RUNNING). We re-call
        // with the same timeout each iteration — the proxy tracks
        // elapsed time internally.
        loop {
            let err = unsafe {
                self.session.lib.dv_infer_wait_for_completion(
                    self.session.ptr,
                    list.as_mut_ptr(),
                    1,
                    timeout_ms,
                    &mut completed,
                )
            };

            if err != 0 {
                return Err(err.into());
            }

            // Use the `completed` out-parameter to read status — this is
            // the request whose state actually changed, as documented by
            // the C API.
            if completed.is_null() {
                return Err(Error::NullPointer(
                    "dv_infer_wait_for_completion returned null completed pointer".to_owned(),
                ));
            }

            let status = unsafe { (*completed).status };
            match status {
                DV_INFERENCE_STATUS_DV_INFERENCE_STATUS_COMPLETED => {
                    return extract_timing(completed);
                    // Drop runs here, calling dv_infer_free
                }
                DV_INFERENCE_STATUS_DV_INFERENCE_STATUS_FAILED => {
                    return Err(Error::InferenceFailed);
                }
                DV_INFERENCE_STATUS_DV_INFERENCE_STATUS_QUEUED
                | DV_INFERENCE_STATUS_DV_INFERENCE_STATUS_RUNNING => {
                    // Intermediate status — loop and wait for terminal state.
                }
                _ => {
                    // UNKNOWN or any unexpected status — treat as terminal error
                    // to avoid infinite loops.
                    return Err(Error::InferenceNotCompleted(status));
                }
            }
        }
    }

    /// Get the proxy-assigned request ID for log correlation.
    ///
    /// Each submitted inference request is assigned a unique ID by the
    /// proxy service. This can be used to correlate client-side events
    /// with proxy-side logs (e.g., `journalctl -u ara2`).
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ara2::{Session, DEFAULT_SOCKET};
    /// # let session = Session::create_via_unix_socket(DEFAULT_SOCKET)?;
    /// # let endpoints = session.list_endpoints()?;
    /// # let mut model = endpoints[0].load_model_from_file("m.dvm".as_ref())?;
    /// # model.allocate_tensors(None)?;
    /// let request = model.submit()?;
    /// println!("Submitted request #{}", request.request_id()?);
    /// # Ok::<(), ara2::Error>(())
    /// ```
    pub fn request_id(&self) -> Result<u64, Error> {
        let mut req_id: u64 = 0;
        let err = unsafe { self.session.lib.dv_infer_get_req_id(self.ptr, &mut req_id) };
        if err != 0 {
            return Err(err.into());
        }
        Ok(req_id)
    }
}

impl Drop for InferRequest {
    fn drop(&mut self) {
        unsafe {
            self.session.lib.dv_infer_free(self.ptr);
        }
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe {
            self.session.lib.dv_model_unload(self.ptr);
        }
    }
}

/// Detailed information about an input tensor from the model.
#[derive(Debug)]
pub struct InputTensor {
    /// Layer identifier within the model graph.
    pub layer_id: i32,
    /// Blob identifier within the layer.
    pub blob_id: i32,
    /// Layer name from the model.
    pub layer_name: String,
    /// Blob name from the model.
    pub blob_name: String,
    /// Layer type (e.g., "Input").
    pub layer_type: String,
    /// Data layout string (e.g., "NCHW", "NHWC").
    pub layout: String,
    /// Total size in bytes.
    pub size: usize,
    /// Width in pixels/elements.
    pub width: usize,
    /// Height in pixels/elements.
    pub height: usize,
    /// Number of channels.
    pub nch: usize,
    /// Bytes per pixel/element.
    pub bpp: usize,
    /// Batch size.
    pub batch_size: usize,
    /// Quantization parameters.
    pub quant: InputQuantization,
    /// Image preprocessing parameters.
    pub preprocess: InputPreprocess,
}

/// Detailed information about an output tensor from the model.
#[derive(Debug)]
pub struct OutputTensor {
    /// Layer identifier within the model graph.
    pub layer_id: i32,
    /// Blob identifier within the layer.
    pub blob_id: i32,
    /// Fused parent layer identifier.
    pub fused_parent_id: i32,
    /// Layer name from the model.
    pub layer_name: String,
    /// Blob name from the model.
    pub blob_name: String,
    /// Fused parent layer name.
    pub layer_fused_parent_name: String,
    /// Layer type (e.g., "Convolution", "Pooling").
    pub layer_type: String,
    /// Data layout string (e.g., "NCHW", "NHWC").
    pub layout: String,
    /// Total size in bytes.
    pub size: usize,
    /// Width in elements.
    pub width: usize,
    /// Height in elements.
    pub height: usize,
    /// Number of channels.
    pub nch: usize,
    /// Bytes per element (1 for int8/uint8, 2 for int16, 4 for float32).
    pub bpp: usize,
    /// Number of classification classes (if applicable).
    pub num_classes: usize,
    /// The type of output produced by this layer.
    pub layer_output_type: ModelOutputType,
    /// Maximum dynamic identifier.
    pub max_dynamic_id: i32,
    /// Quantization parameters.
    pub quant: OutputQuantization,
}

/// Input tensor quantization parameters.
///
/// For all supported models today (qmode 9, asymmetric), `qn` is the
/// per-tensor scale factor and `offset` is the integer zero-point.
/// Per-channel preprocessing (mean/std normalization, BGR-to-RGB) lives
/// on [`InputPreprocess`], queried separately.
#[derive(Debug, Copy, Clone)]
pub struct InputQuantization {
    /// Per-tensor quantization scale (qmode 9: direct scale; other qmodes: 1/scale).
    pub qn: f32,
    /// Integer zero-point (asymmetric quantization).
    pub offset: i32,
    /// Whether the tensor uses signed storage (`i8`/`i16` vs `u8`/`u16`).
    pub is_signed: bool,
    /// Kinara quantization mode. `9` = asymmetric (the only production-tested mode).
    pub qmode: i32,
}

/// Image preprocessing parameters for an input tensor.
///
/// These describe how the compiler expects float image data to be
/// normalized before quantization: `(pixel - mean[c]) * scale[c]`. They
/// are independent of the quantization parameters on [`InputQuantization`].
#[derive(Debug, Copy, Clone)]
pub struct InputPreprocess {
    /// Per-channel normalization mean (RGB ordering after any BGR→RGB swap).
    pub mean: [f32; 3],
    /// Per-channel normalization scale.
    pub scale: [f32; 3],
    /// Whether the compiler inserted a BGR→RGB swap before normalization.
    pub bgr_to_rgb: bool,
    /// Whether aspect-ratio preserving resize is applied.
    pub aspect_resize: bool,
    /// Whether horizontal mirroring is applied.
    pub mirror: bool,
    /// Whether center-cropping is applied.
    pub center_crop: bool,
}

/// Output tensor quantization parameters (qmode 9 semantics).
#[derive(Debug, Clone, Copy)]
pub struct OutputQuantization {
    /// Per-tensor quantization scale.
    pub qn: f32,
    /// Integer zero-point.
    pub offset: i32,
    /// Whether the tensor uses signed storage.
    pub is_signed: bool,
}

impl OutputQuantization {
    /// Return the normalized `(scale, offset)` pair for dequantization.
    ///
    /// The Kinara DVM format documents different dequantization formulas
    /// per quantization mode (see `dv_model_input_preprocess_param` docs
    /// in the C header). This helper normalizes them:
    ///
    /// - **qmode 9** (asymmetric, production default): returns `(qn, offset)`
    ///   directly. Dequantized value = `(raw - offset) * scale`.
    /// - **Other qmodes**: returns [`Error::UnsupportedQmode`]. Support
    ///   will be added only when a testable model using that qmode is
    ///   available.
    pub fn effective(&self, qmode: i32) -> Result<(f32, i32), Error> {
        match qmode {
            9 => Ok((self.qn, self.offset)),
            other => Err(Error::UnsupportedQmode(other)),
        }
    }
}

/// Dequantize a single unsigned-byte quantized value.
#[inline]
fn dequant_u8(raw: u8, offset: i32, scale: f32) -> f32 {
    (raw as i32 - offset) as f32 * scale
}

/// Dequantize a single signed-byte quantized value.
///
/// The raw buffer is `&[u8]` at the C FFI boundary; for signed outputs
/// we reinterpret each byte as `i8` before the offset subtraction.
#[inline]
fn dequant_i8(raw: u8, offset: i32, scale: f32) -> f32 {
    (raw as i8 as i32 - offset) as f32 * scale
}

/// Dequantize a single unsigned-16-bit quantized value from two
/// little-endian bytes.
#[inline]
fn dequant_u16_le(lo: u8, hi: u8, offset: i32, scale: f32) -> f32 {
    (u16::from_le_bytes([lo, hi]) as i32 - offset) as f32 * scale
}

/// Dequantize a single signed-16-bit quantized value from two
/// little-endian bytes.
#[inline]
fn dequant_i16_le(lo: u8, hi: u8, offset: i32, scale: f32) -> f32 {
    (i16::from_le_bytes([lo, hi]) as i32 - offset) as f32 * scale
}

/// The type of output produced by a model layer.
#[derive(Debug)]
pub enum ModelOutputType {
    Classification = 0,
    Detection = 1,
    SemanticSegmentation = 2,
    Raw = 3,
}

impl TryFrom<DV_LAYER_OUTPUT_TYPE> for ModelOutputType {
    type Error = Error;

    fn try_from(value: DV_LAYER_OUTPUT_TYPE) -> Result<Self, Error> {
        match value {
            ara2_sys::DV_LAYER_OUTPUT_TYPE_DV_LAYER_OUTPUT_TYPE_CLASSIFICATION => {
                Ok(ModelOutputType::Classification)
            }
            ara2_sys::DV_LAYER_OUTPUT_TYPE_DV_LAYER_OUTPUT_TYPE_DETECTION => {
                Ok(ModelOutputType::Detection)
            }
            ara2_sys::DV_LAYER_OUTPUT_TYPE_DV_LAYER_OUTPUT_TYPE_SEMANTIC_SEGMENTATION => {
                Ok(ModelOutputType::SemanticSegmentation)
            }
            ara2_sys::DV_LAYER_OUTPUT_TYPE_DV_LAYER_OUTPUT_TYPE_RAW => Ok(ModelOutputType::Raw),
            _ => Err(Error::UnknownLayerOutputType(value)),
        }
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_load_model_from_file() {
        let session = crate::tests::test_session();
        let endpoints = session.list_endpoints().unwrap();
        let endpoint = &endpoints[0];
        let model_path = crate::tests::test_model_path();

        let model = endpoint
            .load_model_from_file(&model_path)
            .expect("should load model from file");

        assert!(model.n_inputs() > 0, "model should have at least one input");
        assert!(
            model.n_outputs() > 0,
            "model should have at least one output"
        );
    }

    #[test]
    fn test_model_tensor_shapes() {
        let session = crate::tests::test_session();
        let endpoints = session.list_endpoints().unwrap();
        let endpoint = &endpoints[0];
        let model_path = crate::tests::test_model_path();

        let model = endpoint.load_model_from_file(&model_path).unwrap();

        for i in 0..model.n_inputs() {
            let shape = model.input_shape(i);
            assert!(
                shape.iter().all(|&d| d > 0),
                "input {i} shape dimensions should all be > 0: {shape:?}"
            );
            assert!(model.input_size(i) > 0, "input {i} size should be > 0");
        }

        for i in 0..model.n_outputs() {
            let shape = model.output_shape(i);
            assert!(
                shape.iter().all(|&d| d > 0),
                "output {i} shape dimensions should all be > 0: {shape:?}"
            );
            assert!(model.output_size(i) > 0, "output {i} size should be > 0");
        }
    }

    #[test]
    fn test_model_allocate_and_run() {
        let session = crate::tests::test_session();
        let endpoints = session.list_endpoints().unwrap();
        let endpoint = &endpoints[0];
        let model_path = crate::tests::test_model_path();

        let mut model = endpoint.load_model_from_file(&model_path).unwrap();
        model
            .allocate_tensors(None)
            .expect("should allocate tensors");

        let timing = model.run().expect("inference should succeed");
        assert!(timing.run_time.as_micros() > 0, "run_time should be > 0");
    }

    #[test]
    fn test_model_submit_and_wait() {
        let session = crate::tests::test_session();
        let endpoints = session.list_endpoints().unwrap();
        let endpoint = &endpoints[0];
        let model_path = crate::tests::test_model_path();

        let mut model = endpoint.load_model_from_file(&model_path).unwrap();
        model
            .allocate_tensors(None)
            .expect("should allocate tensors");

        // Submit async inference
        let request = model.submit().expect("submit should succeed");
        // Just verify request_id() succeeds — IDs are proxy-assigned and
        // may start at 0.
        let _req_id = request.request_id().expect("should get request id");

        // Wait for completion
        let timing = request
            .wait(super::DEFAULT_TIMEOUT_MS)
            .expect("wait should succeed");
        assert!(timing.run_time.as_micros() > 0, "run_time should be > 0");

        // No in-flight requests after wait
        let count = session.inflight_count().expect("should get inflight count");
        assert_eq!(count, 0, "inflight count should be 0 after wait");
    }

    #[test]
    fn test_model_submit_drop_without_wait() {
        let session = crate::tests::test_session();
        let endpoints = session.list_endpoints().unwrap();
        let endpoint = &endpoints[0];
        let model_path = crate::tests::test_model_path();

        let mut model = endpoint.load_model_from_file(&model_path).unwrap();
        model.allocate_tensors(None).unwrap();

        // Submit and drop without waiting — should not panic or leak
        let request = model.submit().expect("submit should succeed");
        drop(request);

        // Verify we can still run inference after dropping a request
        let timing = model.run().expect("inference should succeed after drop");
        assert!(timing.run_time.as_micros() > 0);
    }
}

#[cfg(test)]
mod quant_tests {
    use super::*;

    #[test]
    fn input_quant_exposes_qmode_and_offset() {
        let q = InputQuantization {
            qn: 0.0039,
            offset: 128,
            is_signed: false,
            qmode: 9,
        };
        assert_eq!(q.qmode, 9);
        assert_eq!(q.offset, 128);
    }

    #[test]
    fn input_preprocess_exposes_per_channel_arrays() {
        let p = InputPreprocess {
            mean: [127.5, 127.5, 127.5],
            scale: [0.00784, 0.00784, 0.00784],
            bgr_to_rgb: false,
            aspect_resize: true,
            mirror: false,
            center_crop: false,
        };
        assert_eq!(p.mean[0], 127.5);
        assert_eq!(p.scale.len(), 3);
    }

    #[test]
    fn effective_qmode9_returns_direct_scale_and_offset() {
        let q = OutputQuantization {
            qn: 0.0039,
            offset: 128,
            is_signed: false,
        };
        let (scale, offset) = q.effective(9).unwrap();
        assert!((scale - 0.0039).abs() < 1e-9);
        assert_eq!(offset, 128);
    }

    #[test]
    fn effective_rejects_unsupported_qmode() {
        let q = OutputQuantization {
            qn: 256.0,
            offset: 0,
            is_signed: false,
        };
        let err = q.effective(0).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("qmode"));
        assert!(msg.contains("9"));
    }

    #[test]
    fn dequantize_formula_on_qmode9_sample_values() {
        // (raw - offset) * scale
        // raw=255, offset=128, scale=0.0039 → (127)*0.0039 ≈ 0.4953
        let q = OutputQuantization {
            qn: 0.0039,
            offset: 128,
            is_signed: false,
        };
        let (scale, offset) = q.effective(9).unwrap();
        let dequant = |raw: u8| -> f32 { (raw as i32 - offset) as f32 * scale };
        assert!((dequant(255) - 0.4953).abs() < 1e-3);
        assert!((dequant(128) - 0.0).abs() < 1e-6);
        assert!((dequant(0) - (-0.4992)).abs() < 1e-3);
    }

    #[test]
    fn dequant_u8_formula() {
        // (raw - offset) * scale with offset=128, scale=0.0039
        assert!((dequant_u8(255, 128, 0.0039) - 0.4953).abs() < 1e-3);
        assert!((dequant_u8(128, 128, 0.0039) - 0.0).abs() < 1e-6);
        assert!((dequant_u8(0, 128, 0.0039) - (-0.4992)).abs() < 1e-3);
    }

    #[test]
    fn dequant_i8_formula() {
        // raw bytes are interpreted as i8 first
        // 0xFF → -1, 0x00 → 0, 0x7F → 127, 0x80 → -128
        assert!((dequant_i8(0xFF, 0, 0.0039) - (-0.0039)).abs() < 1e-6);
        assert!((dequant_i8(0x00, 0, 0.0039) - 0.0).abs() < 1e-6);
        assert!((dequant_i8(0x7F, 0, 0.0039) - (127.0 * 0.0039)).abs() < 1e-6);
        assert!((dequant_i8(0x80, 0, 0.0039) - (-128.0 * 0.0039)).abs() < 1e-6);
    }

    #[test]
    fn dequant_u16_le_formula() {
        // 0x0100 little-endian = 256; (256 - 0) * 1.0 = 256.0
        assert!((dequant_u16_le(0x00, 0x01, 0, 1.0) - 256.0).abs() < 1e-6);
        // 0xFFFF = 65535
        assert!((dequant_u16_le(0xFF, 0xFF, 0, 0.0001) - 6.5535).abs() < 1e-3);
        // with offset
        assert!((dequant_u16_le(0x0A, 0x00, 5, 1.0) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn dequant_i16_le_formula() {
        // 0xFFFF LE = -1 as i16
        assert!((dequant_i16_le(0xFF, 0xFF, 0, 1.0) - (-1.0)).abs() < 1e-6);
        // 0x8000 LE = -32768
        assert!((dequant_i16_le(0x00, 0x80, 0, 1.0) - (-32768.0)).abs() < 1e-3);
        // 0x7F7F LE = 32639
        assert!((dequant_i16_le(0x7F, 0x7F, 0, 1.0) - 32639.0).abs() < 1e-3);
    }

    #[test]
    fn tensor_size_mismatch_error_displays_byte_counts() {
        let err = Error::TensorSizeMismatch {
            expected: 1024,
            got: 512,
        };
        let msg = format!("{err}");
        assert!(msg.contains("1024"));
        assert!(msg.contains("512"));
        assert!(msg.contains("mismatch"));
    }
}
