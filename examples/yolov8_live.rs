// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! # YOLOv8 Live Camera Inference on ARA-2 NPU
//!
//! Minimal serial example: captures NV12 frames from a camera via libcamera,
//! runs YOLOv8 detection + instance segmentation on the ARA-2 NPU, and
//! displays results in a native Wayland window.
//!
//! The entire path from camera sensor to display uses zero-copy DMA-BUF
//! buffers.  The pipeline is intentionally single-threaded and synchronous.
//!
//! Display uses the `zwp_linux_dmabuf_v1` Wayland protocol to submit the
//! RGBA canvas DMA-BUF directly to the compositor — no EGL or OpenGL.
//!
//! ## Pipeline
//!
//! ```text
//! libcamera (NV12 DMA-BUF)
//!   -> HAL import (PlaneDescriptor, cached by buffer index)
//!   -> HAL convert (NV12 -> PlanarRGB letterbox)
//!   -> ARA-2 NPU inference
//!   -> HAL draw_masks (decode + composite -> RGBA canvas)
//!   -> Wayland display (DMA-BUF -> wl_buffer -> compositor)
//! ```
//!
//! ## Usage
//!
//! ```text
//! yolov8_live <model.dvm> [--width 1920] [--height 1080]
//!     [--camera-name NAME] [--threshold 0.50] [--iou 0.45]
//!     [--socket /var/run/ara2.sock]
//! ```
//!
//! ## Requirements
//!
//! - libcamera (camera capture)
//! - ARA-2 proxy service running
//! - Wayland compositor (Weston) with `zwp_linux_dmabuf_v1` support

use ara2::{Session, dvm_metadata};
use edgefirst_hal::{
    decoder::{
        DecoderBuilder,
        configs::{self, DecoderType, QuantTuple},
    },
    image::{
        ColorMode, Crop, Flip, ImageProcessor, ImageProcessorTrait as _, MaskOverlay, Rect,
        Rotation,
    },
    tensor::{DType, PixelFormat, PlaneDescriptor, TensorDyn, TensorMemory, TensorTrait as _},
};
use libcamera::{
    camera::CameraConfigurationStatus,
    camera_manager::CameraManager,
    framebuffer::AsFrameBuffer,
    framebuffer_allocator::{FrameBuffer, FrameBufferAllocator},
    request::ReuseFlag,
    stream::StreamRole,
};
use std::collections::HashMap;
use std::os::fd::{AsFd as _, AsRawFd, BorrowedFd};
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use wayland_client::protocol::{wl_buffer, wl_callback, wl_compositor, wl_registry, wl_surface};
use wayland_client::{Connection, Dispatch, EventQueue, QueueHandle, delegate_noop};
use wayland_protocols::xdg::shell::client::{xdg_surface, xdg_toplevel, xdg_wm_base};
use wayland_protocols::wp::linux_dmabuf::zv1::client::{
    zwp_linux_buffer_params_v1, zwp_linux_dmabuf_v1,
};

// ── Wayland display (direct DMA-BUF submission, no EGL/GL) ──────────────────

/// DRM fourcc code for `ABGR8888` (`0x34324241`).
///
/// This is the Wayland/DRM equivalent of HAL's [`PixelFormat::Rgba`] layout:
/// each pixel is stored as R, G, B, A bytes in memory, which DRM interprets
/// as AB-GR in little-endian register order.
const DRM_FORMAT_ABGR8888: u32 = 0x34324241;

/// Mutable state for the Wayland event loop, passed as the `Dispatch` target.
///
/// Globals are populated during the initial registry roundtrip and remain
/// `Some` for the lifetime of the display.
struct DisplayState {
    /// Wayland compositor global — used to create surfaces.
    compositor: Option<wl_compositor::WlCompositor>,
    /// XDG shell global — used to create desktop windows (toplevels).
    wm_base: Option<xdg_wm_base::XdgWmBase>,
    /// `zwp_linux_dmabuf_v1` global — used to import DMA-BUF fds as `wl_buffer`s.
    dmabuf: Option<zwp_linux_dmabuf_v1::ZwpLinuxDmabufV1>,
    /// The single application surface (created once after globals are bound).
    surface: Option<wl_surface::WlSurface>,
    /// Set to `true` after the initial `xdg_surface::configure` handshake completes.
    configured: bool,
    /// Set to `true` when the compositor sends `xdg_toplevel::Close` (e.g. user closes the window).
    closed: bool,
    /// Frame-pacing flag: `true` when the compositor is ready to accept the next frame.
    /// Reset to `false` after submitting a buffer, set to `true` by the `wl_callback::Done` event.
    frame_done: bool,
    /// Window width in pixels (signed to match Wayland protocol types).
    width: i32,
    /// Window height in pixels.
    height: i32,
    /// Maps DMA-BUF fd values to their imported `wl_buffer`. Since the HAL canvas
    /// uses a fixed set of fds, each fd is imported at most once and then reused.
    buffer_cache: HashMap<i32, wl_buffer::WlBuffer>,
}

/// Wayland display connection for zero-copy DMA-BUF presentation.
///
/// Wraps the Wayland client connection, event queue, and display state needed
/// to submit DMA-BUF frames directly to the compositor without EGL or OpenGL.
/// The window is created via XDG shell and buffers are imported through the
/// `zwp_linux_dmabuf_v1` protocol.
struct WaylandDisplay {
    conn: Connection,
    queue: EventQueue<DisplayState>,
    state: DisplayState,
}

impl WaylandDisplay {
    /// Create a new Wayland window and bind the required protocol globals.
    ///
    /// Connects to the Wayland compositor via `$WAYLAND_DISPLAY`, performs a
    /// registry roundtrip to bind `wl_compositor`, `xdg_wm_base`, and
    /// `zwp_linux_dmabuf_v1`, then creates an XDG toplevel surface and waits
    /// for the initial configure event.
    ///
    /// Returns an error if any required global is missing (e.g. the compositor
    /// does not support the DMA-BUF protocol).
    fn new(width: usize, height: usize, title: &str) -> Result<Self, String> {
        let conn = Connection::connect_to_env()
            .map_err(|e| format!("No Wayland compositor: {e}"))?;

        let mut state = DisplayState {
            compositor: None,
            wm_base: None,
            dmabuf: None,
            surface: None,
            configured: false,
            closed: false,
            frame_done: true,
            width: width as i32,
            height: height as i32,
            buffer_cache: HashMap::new(),
        };

        let mut queue = conn.new_event_queue();
        let qh = queue.handle();
        let display = conn.display();
        display.get_registry(&qh, ());

        // Roundtrip to bind globals
        queue.roundtrip(&mut state).map_err(|e| format!("roundtrip: {e}"))?;

        if state.compositor.is_none() {
            return Err("Missing wl_compositor".into());
        }
        if state.wm_base.is_none() {
            return Err("Missing xdg_wm_base".into());
        }
        if state.dmabuf.is_none() {
            return Err("Missing zwp_linux_dmabuf_v1".into());
        }

        // Create surface + xdg shell window
        let surface = state.compositor.as_ref().unwrap().create_surface(&qh, ());
        let xdg_surface = state.wm_base.as_ref().unwrap().get_xdg_surface(&surface, &qh, ());
        let toplevel = xdg_surface.get_toplevel(&qh, ());
        toplevel.set_title(title.to_string());
        toplevel.set_app_id("ara2-demo".to_string());
        surface.commit();

        state.surface = Some(surface);

        // Wait for configure
        while !state.configured {
            queue.blocking_dispatch(&mut state)
                .map_err(|e| format!("dispatch: {e}"))?;
        }

        eprintln!("display: {}x{} wayland dmabuf", width, height);

        Ok(Self { conn, queue, state })
    }

    /// Submit a DMA-BUF frame to the compositor for display.
    ///
    /// Uses `wl_surface.frame` callbacks for pacing: a frame is only submitted
    /// when [`DisplayState::frame_done`] is `true`, meaning the compositor has
    /// signaled readiness for a new buffer. If the compositor is still busy
    /// with the previous frame, this call is a no-op (frame is skipped) and
    /// returns `true`.
    ///
    /// The `wl_buffer` for each DMA-BUF fd is cached in
    /// [`DisplayState::buffer_cache`] so the `zwp_linux_buffer_params_v1`
    /// import is performed only once per unique fd.
    ///
    /// Returns `false` if the window has been closed, `true` otherwise.
    fn render_dmabuf(&mut self, fd: i32) -> bool {
        // Process pending events (close, frame callback, ping)
        self.queue.dispatch_pending(&mut self.state).ok();

        if self.state.closed {
            return false;
        }

        // Only submit when the compositor is ready for a new frame
        if !self.state.frame_done {
            // Compositor hasn't consumed the previous frame yet — skip
            self.conn.flush().ok();
            return true;
        }

        let qh = self.queue.handle();
        let w = self.state.width;
        let h = self.state.height;

        // Get or create wl_buffer for this DMA-BUF fd
        if !self.state.buffer_cache.contains_key(&fd) {
            let dmabuf = self.state.dmabuf.as_ref().unwrap();
            let params = dmabuf.create_params(&qh, ());
            // SAFETY: fd is a valid DMA-BUF fd owned by the HAL canvas tensor
            let borrowed = unsafe { BorrowedFd::borrow_raw(fd) };
            params.add(borrowed, 0, 0, (w * 4) as u32, 0, 0);
            let flags = zwp_linux_buffer_params_v1::Flags::empty();
            let buffer = params.create_immed(w, h, DRM_FORMAT_ABGR8888, flags, &qh, ());
            params.destroy();
            self.state.buffer_cache.insert(fd, buffer);
        }
        let buffer = &self.state.buffer_cache[&fd];

        let surface = self.state.surface.as_ref().unwrap();
        surface.attach(Some(buffer), 0, 0);
        surface.damage_buffer(0, 0, w, h);

        // Request frame callback — paces us to the compositor's refresh
        surface.frame(&qh, ());
        self.state.frame_done = false;

        surface.commit();
        self.conn.flush().ok();
        true
    }

    /// Check whether the window is still open.
    ///
    /// Dispatches any pending Wayland events (which may set `closed`) and
    /// returns `true` if the window has not been closed by the user or compositor.
    fn is_open(&mut self) -> bool {
        self.queue.dispatch_pending(&mut self.state).ok();
        self.conn.flush().ok();
        !self.state.closed
    }
}

// ── Wayland protocol dispatch ───────────────────────────────────────────────

/// Handles `wl_registry::Global` events to bind compositor, XDG shell, and DMA-BUF globals.
impl Dispatch<wl_registry::WlRegistry, ()> for DisplayState {
    fn event(
        state: &mut Self,
        registry: &wl_registry::WlRegistry,
        event: wl_registry::Event,
        _: &(),
        _: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        if let wl_registry::Event::Global { name, interface, version } = event {
            match interface.as_str() {
                "wl_compositor" => {
                    state.compositor =
                        Some(registry.bind(name, version.min(4), qh, ()));
                }
                "xdg_wm_base" => {
                    state.wm_base =
                        Some(registry.bind(name, version.min(1), qh, ()));
                }
                "zwp_linux_dmabuf_v1" => {
                    state.dmabuf =
                        Some(registry.bind(name, version.min(3), qh, ()));
                }
                _ => {}
            }
        }
    }
}

/// Responds to compositor ping events with a pong to keep the connection alive.
impl Dispatch<xdg_wm_base::XdgWmBase, ()> for DisplayState {
    fn event(
        _: &mut Self,
        wm_base: &xdg_wm_base::XdgWmBase,
        event: xdg_wm_base::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let xdg_wm_base::Event::Ping { serial } = event {
            wm_base.pong(serial);
        }
    }
}

/// Acknowledges XDG surface configure events and marks the surface as ready.
impl Dispatch<xdg_surface::XdgSurface, ()> for DisplayState {
    fn event(
        state: &mut Self,
        xdg_surface: &xdg_surface::XdgSurface,
        event: xdg_surface::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let xdg_surface::Event::Configure { serial } = event {
            xdg_surface.ack_configure(serial);
            state.configured = true;
        }
    }
}

/// Handles the toplevel close event (user closes the window).
impl Dispatch<xdg_toplevel::XdgToplevel, ()> for DisplayState {
    fn event(
        state: &mut Self,
        _: &xdg_toplevel::XdgToplevel,
        event: xdg_toplevel::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let xdg_toplevel::Event::Close = event {
            state.closed = true;
        }
    }
}

/// Handles `wl_surface.frame` callback completion, signaling readiness for the next frame.
impl Dispatch<wl_callback::WlCallback, ()> for DisplayState {
    fn event(
        state: &mut Self,
        _: &wl_callback::WlCallback,
        event: wl_callback::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let wl_callback::Event::Done { .. } = event {
            state.frame_done = true;
        }
    }
}

// No-op dispatchers for objects we don't handle events on
delegate_noop!(DisplayState: ignore wl_compositor::WlCompositor);
delegate_noop!(DisplayState: ignore wl_surface::WlSurface);
delegate_noop!(DisplayState: ignore wl_buffer::WlBuffer);
delegate_noop!(DisplayState: ignore zwp_linux_dmabuf_v1::ZwpLinuxDmabufV1);
delegate_noop!(DisplayState: ignore zwp_linux_buffer_params_v1::ZwpLinuxBufferParamsV1);

// ── DMA-BUF tensor cache ────────────────────────────────────────────────────

/// Cache HAL tensors by libcamera buffer index.
///
/// Unlike GStreamer (which recycles fds), libcamera's `FrameBufferAllocator`
/// pre-allocates stable buffers.  We cache by the request cookie (buffer
/// index) so `import_image` is called only once per slot.
struct FrameCache {
    entries: Vec<Option<TensorDyn>>,
}

impl FrameCache {
    fn new(capacity: usize) -> Self {
        Self {
            entries: (0..capacity).map(|_| None).collect(),
        }
    }

    /// Return the cached HAL tensor for buffer `index`, importing it on first use.
    ///
    /// On the first call for a given `index`, the DMA-BUF fd(s) from the
    /// libcamera `framebuffer` are imported into a HAL tensor via
    /// [`ImageProcessor::import_image`]. The fd is duplicated internally, so
    /// the libcamera buffer can be requeued immediately after this call.
    ///
    /// - `index` — libcamera request cookie (buffer slot index).
    /// - `processor` — HAL image processor used to import the DMA-BUF.
    /// - `framebuffer` — libcamera frame buffer containing the DMA-BUF plane fd(s).
    /// - `width`, `height` — frame dimensions in pixels.
    /// - `format` — pixel format of the captured frame (e.g. `Nv12`, `Yuyv`).
    ///
    /// # Safety
    ///
    /// The libcamera `framebuffer` fds must be valid for the duration of this
    /// call. After import, the cached tensor holds its own duplicated fd.
    fn get_or_import(
        &mut self,
        index: usize,
        processor: &ImageProcessor,
        framebuffer: &FrameBuffer,
        width: usize,
        height: usize,
        format: PixelFormat,
    ) -> Result<&TensorDyn, Box<dyn std::error::Error>> {
        if self.entries[index].is_none() {
            let planes = framebuffer.planes();
            let fd0 = planes.get(0).expect("buffer requires at least one plane").fd();
            // SAFETY: fds come from libcamera FrameBuffer, valid while request is alive.
            // PlaneDescriptor::new() dups the fd so the cached tensor is independent.
            let primary =
                PlaneDescriptor::new(unsafe { BorrowedFd::borrow_raw(fd0) })?;

            // Semi-planar formats (NV12) may have a separate chroma plane
            let chroma = if planes.len() >= 2 && format == PixelFormat::Nv12 {
                let p1 = planes.get(1).unwrap();
                let mut desc =
                    PlaneDescriptor::new(unsafe { BorrowedFd::borrow_raw(p1.fd()) })?;
                if let Some(off) = p1.offset() {
                    if off > 0 {
                        desc = desc.with_offset(off);
                    }
                }
                Some(desc)
            } else {
                None
            };

            let tensor =
                processor.import_image(primary, chroma, width, height, format, DType::U8)?;
            self.entries[index] = Some(tensor);
        }
        Ok(self.entries[index].as_ref().unwrap())
    }
}

// ── Arguments ───────────────────────────────────────────────────────────────

/// Command-line arguments for the live inference demo.
struct Args {
    /// Path to the compiled DVM model file (e.g. `yolov8n-seg.dvm`).
    model: PathBuf,
    /// Minimum confidence score for detections (default: 0.50).
    threshold: f32,
    /// IoU threshold for non-maximum suppression (default: 0.45).
    iou: f32,
    /// Capture width in pixels (default: 1920).
    width: usize,
    /// Capture height in pixels (default: 1080).
    height: usize,
    /// Optional libcamera camera ID string. When `None`, the first available camera is used.
    camera_name: Option<String>,
    /// Unix socket path for the ARA-2 proxy service.
    socket: String,
    /// Camera pixel format name: `"nv12"` or `"yuyv"` (default: `"nv12"`).
    format: String,
}

/// Parse command-line arguments into [`Args`].
///
/// # Panics
///
/// Exits the process with a usage message if no model path is provided,
/// or if a flag value cannot be parsed to its expected type.
fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "Usage: {} <model.dvm> [--threshold N] [--iou N] \
             [--width N] [--height N] [--camera-name NAME] [--format nv12|yuyv] \
             [--socket PATH]",
            args[0]
        );
        std::process::exit(1);
    }
    let mut threshold = 0.50;
    let mut iou = 0.45;
    let mut width = 1920;
    let mut height = 1080;
    let mut camera_name = None;
    let mut socket = ara2::DEFAULT_SOCKET.to_string();
    let mut format = "nv12".to_string();
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            flag @ ("--threshold" | "--iou" | "--width" | "--height" | "--camera-name"
            | "--socket" | "--format") => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: {flag} requires a value");
                    std::process::exit(1);
                }
                match flag {
                    "--threshold" => {
                        threshold = args[i].parse().expect("invalid --threshold value")
                    }
                    "--iou" => iou = args[i].parse().expect("invalid --iou value"),
                    "--width" => width = args[i].parse().expect("invalid --width value"),
                    "--height" => height = args[i].parse().expect("invalid --height value"),
                    "--camera-name" => camera_name = Some(args[i].clone()),
                    "--socket" => socket = args[i].clone(),
                    "--format" => format = args[i].to_lowercase(),
                    _ => unreachable!(),
                }
            }
            other => eprintln!("Unknown argument: {other}"),
        }
        i += 1;
    }
    Args {
        model: args[1].clone().into(),
        threshold,
        iou,
        width,
        height,
        camera_name,
        socket,
        format,
    }
}

// ── COCO labels (fallback) ──────────────────────────────────────────────────

/// Default COCO-80 class labels, used when the DVM model does not embed its own label list.
const COCO: &[&str] = &[
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
];

// ── Output identification ───────────────────────────────────────────────────

/// The type of YOLOv8 task, determined from DVM metadata or output shape inspection.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Task {
    /// Object detection only (bounding boxes + class scores, 2 outputs).
    Detect,
    /// Instance segmentation (bounding boxes + class scores + mask coefficients + prototypes, 4 outputs).
    Segment,
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Normalize an ARA-2 output shape for the HAL decoder.
///
/// ARA-2 reports 3D shapes as `[nch, height, width]`. This function strips
/// trailing dimensions of size 1 (which are padding artifacts from the
/// compiler) and prepends a batch dimension of 1 to match the NCHW layout
/// that the HAL decoder expects.
///
/// # Examples
///
/// ```text
/// [4, 8400, 1] -> strip trailing 1 -> [4, 8400] -> prepend batch -> [1, 4, 8400]
/// [80, 8400, 1] -> strip trailing 1 -> [80, 8400] -> prepend batch -> [1, 80, 8400]
/// [32, 160, 160] -> no trailing 1s  -> [32, 160, 160] -> prepend batch -> [1, 32, 160, 160]
/// ```
fn normalize_shape(raw: [usize; 3]) -> Vec<usize> {
    let mut shape: Vec<usize> = raw.to_vec();
    while shape.len() > 1 && shape.last() == Some(&1) {
        shape.pop();
    }
    shape.insert(0, 1);
    shape
}

/// Identify the boxes and scores output indices for a detection model.
///
/// Matches outputs by shape:
/// - **Boxes**: 3D shape where `shape[1] == 4` (i.e. `[1, 4, N]`).
/// - **Scores**: the first remaining 3D tensor (i.e. `[1, num_classes, N]`).
///
/// Returns `(boxes_index, scores_index)`.
fn identify_det_outputs(shapes: &[Vec<usize>]) -> Result<(usize, usize), String> {
    if shapes.len() < 2 {
        return Err(format!("detection needs >= 2 outputs, got {}", shapes.len()));
    }
    let (mut boxes, mut scores) = (None, None);
    for (i, s) in shapes.iter().enumerate() {
        if s.len() == 3 && s[1] == 4 {
            boxes = Some(i);
        } else if s.len() == 3 && scores.is_none() {
            scores = Some(i);
        }
    }
    Ok((
        boxes.ok_or("cannot identify boxes output (shape[1] == 4)")?,
        scores.ok_or("cannot identify scores output")?,
    ))
}

/// Identify the four output indices for a segmentation model.
///
/// YOLOv8-seg produces four outputs whose roles are inferred by shape:
/// - **Protos**: the only 4D tensor (`[1, mask_dim, H, W]`).
/// - **Boxes**: 3D with `shape[1] == 4` (`[1, 4, N]`).
/// - **Mask coefficients**: 3D whose `shape[1]` matches the protos' `mask_dim`.
/// - **Scores**: the remaining 3D tensor (`[1, num_classes, N]`).
///
/// Because ARA-2 output ordering is not guaranteed, a retry pass is performed
/// when the mask-coefficient tensor appears before the protos tensor in the
/// output list (since the first pass cannot match `mask_dim` until protos is
/// found).
///
/// Returns `(boxes_index, scores_index, masks_index, protos_index)`.
fn identify_seg_outputs(shapes: &[Vec<usize>]) -> Result<(usize, usize, usize, usize), String> {
    if shapes.len() < 4 {
        return Err(format!("segmentation needs 4 outputs, got {}", shapes.len()));
    }
    let (mut scores, mut boxes, mut masks, mut protos) = (None, None, None, None);
    for (i, s) in shapes.iter().enumerate() {
        if s.len() == 4 {
            protos = Some(i);
        } else if s.len() == 3 && s[1] == 4 {
            boxes = Some(i);
        } else if s.len() == 3
            && masks.is_none()
            && protos.map_or(false, |p| shapes[p].get(1) == s.get(1))
        {
            masks = Some(i);
        } else if scores.is_none() {
            scores = Some(i);
        }
    }
    // Retry mask detection when proto tensor appears after mask-coeff tensor.
    if masks.is_none() {
        if let Some(pi) = protos {
            for (i, s) in shapes.iter().enumerate() {
                if s.len() == 3
                    && s[1] != 4
                    && Some(i) != scores
                    && Some(i) != boxes
                    && shapes[pi].get(1) == s.get(1)
                {
                    masks = Some(i);
                    break;
                }
            }
        }
    }
    Ok((
        boxes.ok_or("cannot identify boxes")?,
        scores.ok_or("cannot identify scores")?,
        masks.ok_or("cannot identify mask_coeff")?,
        protos.ok_or("cannot identify protos")?,
    ))
}

/// Compute a letterbox [`Crop`] that scales the source image into the
/// destination area while preserving aspect ratio.
///
/// The image is centered in the destination with a neutral grey
/// (`[114, 114, 114]`) fill, which is the standard YOLO letterbox padding.
///
/// - `src_w`, `src_h` — source (camera) frame dimensions.
/// - `dst_w`, `dst_h` — destination (model input) dimensions.
///
/// Returns a [`Crop`] with the `dst_rect` and `dst_color` configured for
/// the HAL `convert` operation.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn compute_letterbox(src_w: usize, src_h: usize, dst_w: usize, dst_h: usize) -> Crop {
    let scale = (dst_w as f32 / src_w as f32).min(dst_h as f32 / src_h as f32);
    let new_w = (src_w as f32 * scale) as usize;
    let new_h = (src_h as f32 * scale) as usize;
    Crop::new()
        .with_dst_rect(Some(Rect::new((dst_w - new_w) / 2, (dst_h - new_h) / 2, new_w, new_h)))
        .with_dst_color(Some([114, 114, 114, 255]))
}

// ── Pixel format mapping ────────────────────────────────────────────────────

/// DRM fourcc for NV12 — semi-planar YUV 4:2:0 (luma plane + interleaved UV plane).
///
/// This is the most common zero-copy camera format on embedded Linux platforms.
const LIBCAM_NV12: libcamera::pixel_format::PixelFormat =
    libcamera::pixel_format::PixelFormat::new(u32::from_le_bytes([b'N', b'V', b'1', b'2']), 0);

/// DRM fourcc for YUYV — packed YUV 4:2:2 (single plane, 2 bytes per pixel).
///
/// Used by USB cameras (UVC) that do not support NV12.
const LIBCAM_YUYV: libcamera::pixel_format::PixelFormat =
    libcamera::pixel_format::PixelFormat::new(u32::from_le_bytes([b'Y', b'U', b'Y', b'V']), 0);

/// Map a user-facing format name to both the libcamera and HAL pixel format enums.
///
/// # Panics
///
/// Exits the process if `name` is not a recognized format (`"nv12"`, `"yuyv"`, or `"yuy2"`).
fn parse_capture_format(name: &str) -> (libcamera::pixel_format::PixelFormat, PixelFormat) {
    match name {
        "nv12" => (LIBCAM_NV12, PixelFormat::Nv12),
        "yuyv" | "yuy2" => (LIBCAM_YUYV, PixelFormat::Yuyv),
        other => {
            eprintln!("Unknown format '{other}'. Supported: nv12, yuyv");
            std::process::exit(1);
        }
    }
}

// ── Main ────────────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();
    let cam_w = args.width;
    let cam_h = args.height;
    let (libcam_fmt, hal_fmt) = parse_capture_format(&args.format);

    // ── 1. Read model metadata ──────────────────────────────────────────
    // Load the DVM binary, extract embedded metadata (task type, compilation
    // target, PPA stats) and class labels. Falls back to COCO-80 labels if
    // none are embedded in the model.
    let dvm_data = std::fs::read(&args.model)?;
    let metadata = dvm_metadata::read_metadata(&dvm_data)?;
    let dvm_labels = dvm_metadata::read_labels(&dvm_data)?;
    let labels: Vec<&str> = if dvm_labels.is_empty() {
        COCO.to_vec()
    } else {
        dvm_labels.iter().map(|s| s.as_str()).collect()
    };

    let task = metadata
        .as_ref()
        .and_then(|m| m.task())
        .map(|t| if t == "segment" { Task::Segment } else { Task::Detect })
        .unwrap_or(Task::Detect);

    if let Some(ref m) = metadata {
        println!("Model: {}", args.model.display());
        println!("Task: {task:?}, Classes: {}", labels.len());
        if let Some(ref comp) = m.compilation {
            if let Some(ref ppa) = comp.ppa {
                println!(
                    "Target: {:?}, IPS: {:.0}, Power: {:.0} mW",
                    comp.target,
                    ppa.ips.unwrap_or(0.0),
                    ppa.power_mw.unwrap_or(0.0)
                );
            }
        }
    }

    // ── 2. Connect to ARA-2 and load model ──────────────────────────────
    // Establish a session with the ARA-2 proxy over a Unix socket, load the
    // DVM model onto the first available NPU endpoint, and allocate
    // DMA-BUF-backed input/output tensors.
    let session = Session::create_via_unix_socket(&args.socket)?;
    let endpoints = session.list_endpoints()?;
    if endpoints.is_empty() {
        eprintln!("No ARA-2 endpoints found.  Is ara2-proxy running?");
        std::process::exit(1);
    }
    let endpoint = &endpoints[0];
    let stats = endpoint.dram_statistics()?;
    println!(
        "Endpoint: {:?}, DRAM: {:.0} / {:.0} MB free",
        endpoint.check_status()?,
        stats.free_size as f64 / 1048576.0,
        stats.dram_size as f64 / 1048576.0,
    );

    let mut model = endpoint.load_model_from_file(&args.model)?;
    model.allocate_tensors(Some(TensorMemory::Dma))?;

    let input_shape = model.input_shape(0);
    let (in_c, in_h, in_w) = (input_shape[0], input_shape[1], input_shape[2]);
    let input_dim = in_w.max(in_h) as f32;
    println!("Input: {in_c}x{in_h}x{in_w} (CHW)");

    // ── 3. Build decoder ────────────────────────────────────────────────
    // Normalize output shapes, extract quantization parameters, identify
    // which output index corresponds to boxes/scores/masks/protos, and
    // build the appropriate HAL post-processing decoder (detection or
    // segmentation).
    let n_outputs = model.n_outputs();
    let mut shapes = Vec::with_capacity(n_outputs);
    let mut quants = Vec::with_capacity(n_outputs);

    for i in 0..n_outputs {
        let shape = normalize_shape(model.output_shape(i));
        let info = model.output_info(i)?;
        let is_box = shape.len() == 3 && shape[1] == 4;
        let scale = if is_box && input_dim > 1.0 {
            info.quant.qn / input_dim
        } else {
            info.quant.qn
        };
        quants.push((scale, info.quant.offset, info.bpp, info.quant.is_signed));
        shapes.push(shape);
    }

    let task = if task == Task::Detect && shapes.iter().any(|s| s.len() == 4) {
        Task::Segment
    } else {
        task
    };

    let decoder = match task {
        Task::Detect => {
            let (bi, si) = identify_det_outputs(&shapes)?;
            DecoderBuilder::new()
                .with_config_yolo_split_det(
                    configs::Boxes {
                        decoder: DecoderType::Ultralytics,
                        quantization: Some(QuantTuple(quants[bi].0, quants[bi].1)),
                        shape: shapes[bi].clone(),
                        normalized: Some(true),
                        ..Default::default()
                    },
                    configs::Scores {
                        decoder: DecoderType::Ultralytics,
                        quantization: Some(QuantTuple(quants[si].0, quants[si].1)),
                        shape: shapes[si].clone(),
                        ..Default::default()
                    },
                )
                .with_score_threshold(args.threshold)
                .with_iou_threshold(args.iou)
                .build()?
        }
        Task::Segment => {
            let (bi, si, mi, pi) = identify_seg_outputs(&shapes)?;
            DecoderBuilder::new()
                .with_config_yolo_split_segdet(
                    configs::Boxes {
                        decoder: DecoderType::Ultralytics,
                        quantization: Some(QuantTuple(quants[bi].0, quants[bi].1)),
                        shape: shapes[bi].clone(),
                        normalized: Some(true),
                        ..Default::default()
                    },
                    configs::Scores {
                        decoder: DecoderType::Ultralytics,
                        quantization: Some(QuantTuple(quants[si].0, quants[si].1)),
                        shape: shapes[si].clone(),
                        ..Default::default()
                    },
                    configs::MaskCoefficients {
                        decoder: DecoderType::Ultralytics,
                        quantization: Some(QuantTuple(quants[mi].0, quants[mi].1)),
                        shape: shapes[mi].clone(),
                        ..Default::default()
                    },
                    configs::Protos {
                        decoder: DecoderType::Ultralytics,
                        quantization: Some(QuantTuple(quants[pi].0, quants[pi].1)),
                        shape: shapes[pi].clone(),
                        ..Default::default()
                    },
                )
                .with_score_threshold(args.threshold)
                .with_iou_threshold(args.iou)
                .build()?
        }
    };

    // ── 4. Setup HAL processor and model I/O tensors ────────────────────
    // Create the HAL image processor and import the model's DMA-BUF
    // input tensor as a PlanarRGB image. Also compute the letterbox crop
    // and wrap each output tensor fd for later decoder use.
    let mut processor = ImageProcessor::new()?;

    let input_quant = model.input_quants(0);
    let input_dtype = if input_quant.is_signed { DType::I8 } else { DType::U8 };
    let input_fd = model.input_tensor(0).clone_fd()?;
    let plane = PlaneDescriptor::new(input_fd.as_fd())?;
    let mut model_input =
        processor.import_image(plane, None, in_w, in_h, PixelFormat::PlanarRgb, input_dtype)?;

    let letterbox = compute_letterbox(cam_w, cam_h, in_w, in_h);
    let _letterbox_norm = MaskOverlay::default()
        .with_letterbox_crop(&letterbox, in_w, in_h)
        .letterbox;

    let output_tensors: Vec<TensorDyn> = (0..n_outputs)
        .map(|i| {
            let dtype = match (quants[i].2, quants[i].3) {
                (1, false) => DType::U8,
                (1, true) => DType::I8,
                (2, false) => DType::U16,
                (2, true) => DType::I16,
                _ => panic!("unsupported bpp={} signed={}", quants[i].2, quants[i].3),
            };
            let fd = model.output_tensor(i).clone_fd().expect("clone output fd");
            TensorDyn::from_fd(fd, &shapes[i], dtype, None).expect("wrap output tensor")
        })
        .collect();
    let output_refs: Vec<&TensorDyn> = output_tensors.iter().collect();

    // ── 5. Output canvas ────────────────────────────────────────────────
    // Allocate a DMA-BUF-backed RGBA canvas at camera resolution. This is
    // the final render target: draw_masks composites the camera frame with
    // detection overlays into this buffer, which is then submitted to Wayland.
    let mut canvas =
        processor.create_image(cam_w, cam_h, PixelFormat::Rgba, DType::U8, None)?;
    let canvas_fd = canvas.clone_fd()?;
    let canvas_raw_fd = canvas_fd.as_raw_fd();

    // ── 6. Setup libcamera ──────────────────────────────────────────────
    // Open the camera, configure a single VideoRecording stream at the
    // requested resolution and pixel format, allocate DMA-BUF frame
    // buffers, and create one capture request per buffer.
    let mgr = CameraManager::new()?;
    let cameras = mgr.cameras();
    if cameras.is_empty() {
        eprintln!("No cameras found");
        std::process::exit(1);
    }

    let cam_index = if let Some(ref name) = args.camera_name {
        cameras
            .iter()
            .position(|c| c.id() == name)
            .unwrap_or_else(|| {
                eprintln!("Camera '{name}' not found. Available:");
                for c in cameras.iter() {
                    eprintln!("  {}", c.id());
                }
                std::process::exit(1);
            })
    } else {
        0
    };

    let cam = cameras.get(cam_index).unwrap();
    println!("Camera: {}", cam.id());
    let mut cam = cam.acquire().expect("Failed to acquire camera");

    let mut cfgs = cam
        .generate_configuration(&[StreamRole::VideoRecording])
        .expect("Failed to generate camera configuration");
    {
        let mut cfg = cfgs.get_mut(0).unwrap();
        cfg.set_pixel_format(libcam_fmt);
        cfg.set_size(libcamera::geometry::Size {
            width: cam_w as u32,
            height: cam_h as u32,
        });
    }
    match cfgs.validate() {
        CameraConfigurationStatus::Valid => {}
        CameraConfigurationStatus::Adjusted => {
            let cfg = cfgs.get(0).unwrap();
            let size = cfg.get_size();
            eprintln!(
                "Camera configuration adjusted: {}x{} {:?}",
                size.width,
                size.height,
                cfg.get_pixel_format()
            );
        }
        CameraConfigurationStatus::Invalid => {
            eprintln!("Invalid camera configuration");
            std::process::exit(1);
        }
    }
    cam.configure(&mut cfgs)
        .expect("Failed to configure camera");

    let stream = cfgs.get(0).unwrap().stream().unwrap();

    let mut alloc = FrameBufferAllocator::new(&cam);
    let buffers = alloc
        .alloc(&stream)
        .expect("Failed to allocate camera buffers");
    let n_buffers = buffers.len();
    println!("Allocated {n_buffers} camera buffers");

    let reqs: Vec<_> = buffers
        .into_iter()
        .enumerate()
        .map(|(i, buf)| {
            let mut req = cam.create_request(Some(i as u64)).unwrap();
            req.add_buffer(&stream, buf).unwrap();
            req
        })
        .collect();

    let (tx, rx) = mpsc::channel();
    cam.on_request_completed(move |req| {
        let _ = tx.send(req);
    });

    // ── 7. Setup display ────────────────────────────────────────────────
    // Create a Wayland window at camera resolution for DMA-BUF presentation.
    let mut display = WaylandDisplay::new(cam_w, cam_h, "ARA-2 YOLOv8 Live")?;

    // ── 8. Start camera and warmup ──────────────────────────────────────
    // Start the camera stream, queue all capture requests, then run one
    // full inference pass (import -> convert -> NPU -> draw) to warm up
    // caches, JIT paths, and ISP auto-exposure before entering the live loop.
    cam.start(None).expect("Failed to start camera");
    for req in reqs {
        cam.queue_request(req).map_err(|(_, e)| e)?;
    }

    let mut frame_cache = FrameCache::new(n_buffers);

    // Warmup: process first frame end-to-end
    let mut req = rx
        .recv_timeout(Duration::from_secs(5))
        .expect("No frames from camera");
    {
        let idx = req.cookie() as usize;
        let buf: &FrameBuffer = req.buffer(&stream).unwrap();
        let src = frame_cache.get_or_import(idx, &processor, buf, cam_w, cam_h, hal_fmt)?;
        req.reuse(ReuseFlag::REUSE_BUFFERS);
        cam.queue_request(req).map_err(|(_, e)| e)?;

        processor.convert(src, &mut model_input, Rotation::None, Flip::None, letterbox.clone())?;
        model.run()?;

        let overlay = MaskOverlay::default()
            .with_background(src)
            .with_letterbox_crop(&letterbox, in_w, in_h)
            .with_color_mode(ColorMode::Instance);
        processor.draw_masks(&decoder, &output_refs, &mut canvas, overlay)?;

        if !display.render_dmabuf(canvas_raw_fd) {
            return Err("DMA-BUF display failed. Compositor may not support zwp_linux_dmabuf_v1.".into());
        }
    }

    println!("\nWarmup complete");
    println!("Capturing {cam_w}x{cam_h} -- press Ctrl+C to stop\n");

    // ── 9. Live inference loop ──────────────────────────────────────────
    // Main loop: pull the newest camera frame (dropping stale ones), run
    // the full pipeline (import -> convert -> NPU -> draw -> display),
    // and print per-stage timing statistics every 30 frames.
    let mut frame_count: u64 = 0;
    let t_start = Instant::now();

    let mut t_pull = 0.0_f64;
    let mut t_import = 0.0_f64;
    let mut t_convert = 0.0_f64;
    let mut t_npu = 0.0_f64;
    let mut t_draw = 0.0_f64;
    let mut t_display = 0.0_f64;
    let mut total_dropped: u64 = 0;

    while display.is_open() {
        let t0 = Instant::now();

        // Pull the latest frame, dropping stale queued frames.
        let mut req = match rx.recv_timeout(Duration::from_secs(5)) {
            Ok(r) => r,
            Err(_) => {
                eprintln!("Camera timeout");
                break;
            }
        };
        let mut dropped = 0u64;
        while let Ok(newer) = rx.try_recv() {
            req.reuse(ReuseFlag::REUSE_BUFFERS);
            cam.queue_request(req).map_err(|(_, e)| e)?;
            req = newer;
            dropped += 1;
        }
        let t1 = Instant::now();

        let idx = req.cookie() as usize;
        let buf: &FrameBuffer = req.buffer(&stream).unwrap();
        let src = frame_cache.get_or_import(idx, &processor, buf, cam_w, cam_h, hal_fmt)?;

        // Requeue immediately — the cached tensor owns a dup'd fd,
        // so the libcamera buffer can be reused by the camera while
        // we process.  This prevents buffer starvation during ISP
        // adjustments (AWB/AE) that hold buffers internally.
        req.reuse(ReuseFlag::REUSE_BUFFERS);
        cam.queue_request(req).map_err(|(_, e)| e)?;
        let t2 = Instant::now();

        processor.convert(src, &mut model_input, Rotation::None, Flip::None, letterbox.clone())?;
        let t3 = Instant::now();

        model.run()?;
        let t4 = Instant::now();

        let overlay = MaskOverlay::default()
            .with_background(src)
            .with_letterbox_crop(&letterbox, in_w, in_h)
            .with_color_mode(ColorMode::Instance);
        let detections = processor.draw_masks(&decoder, &output_refs, &mut canvas, overlay)?;
        let t5 = Instant::now();

        display.render_dmabuf(canvas_raw_fd);
        let t6 = Instant::now();

        t_pull += (t1 - t0).as_secs_f64();
        t_import += (t2 - t1).as_secs_f64();
        t_convert += (t3 - t2).as_secs_f64();
        t_npu += (t4 - t3).as_secs_f64();
        t_draw += (t5 - t4).as_secs_f64();
        t_display += (t6 - t5).as_secs_f64();
        total_dropped += dropped;

        frame_count += 1;
        if frame_count % 30 == 0 {
            let elapsed = (Instant::now() - t_start).as_secs_f64();
            let fps = frame_count as f64 / elapsed;
            let n = 30.0;
            eprint!(
                "\r  FPS: {:5.1}  \
                 pull:{:5.1} imp:{:4.1} cvt:{:4.1} npu:{:5.1} \
                 draw:{:5.1} disp:{:4.1} \
                 tot:{:5.1}ms drop:{} det:{} f:{}",
                fps,
                t_pull / n * 1000.0,
                t_import / n * 1000.0,
                t_convert / n * 1000.0,
                t_npu / n * 1000.0,
                t_draw / n * 1000.0,
                t_display / n * 1000.0,
                (t_pull + t_import + t_convert + t_npu + t_draw + t_display) / n * 1000.0,
                total_dropped,
                detections.len(),
                frame_count,
            );
            t_pull = 0.0;
            t_import = 0.0;
            t_convert = 0.0;
            t_npu = 0.0;
            t_draw = 0.0;
            t_display = 0.0;
        }
    }

    // ── 10. Shutdown ────────────────────────────────────────────────────
    // Stop the camera stream and print the overall performance summary.
    eprintln!();
    cam.stop().ok();
    if frame_count > 0 {
        let elapsed = (Instant::now() - t_start).as_secs_f64();
        println!(
            "Processed {} frames in {:.1}s ({:.1} FPS average)",
            frame_count,
            elapsed,
            frame_count as f64 / elapsed,
        );
    }

    Ok(())
}
