#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
"""
YOLOv8 live camera inference on ARA-2 NPU with Wayland display.

Captures NV12 frames from a camera via the native libcamera Python
bindings, runs YOLOv8 detection + instance segmentation on the ARA-2
NPU, and displays results in a Wayland window using direct DMA-BUF
submission (no EGL, OpenGL, GStreamer, or compiled C libraries).

Pipeline::

    libcamera capture       HAL / ARA-2                 Wayland display
    ┌──────────────┐    ┌───────────────────┐    ┌──────────────────────┐
    │ libcamera     │    │ cached import     │    │ pywayland            │
    │ NV12 DMA-BUF  │ →  │ → convert         │ →  │  DMA-BUF → wl_buffer│
    │               │    │ → NPU inference   │    │  → wl_surface_attach │
    │               │    │ → draw_masks      │    │  → wl_surface_commit │
    └──────────────┘    │ → RGBA canvas     │    └──────────────────────┘
                         └───────────────────┘

Display:
  Uses ``pywayland`` with the ``zwp_linux_dmabuf_v1`` protocol to submit
  the RGBA canvas DMA-BUF directly to the Wayland compositor.

Usage::

    python yolov8_live.py model.dvm
    python yolov8_live.py model.dvm --camera-name '/base/soc/...' --width 1920 --height 1080

Requirements:
    edgefirst-ara2  edgefirst-hal  numpy  pywayland
    libcamera Python bindings (ship with libcamera)
    Wayland compositor running (e.g. Weston) with zwp_linux_dmabuf_v1
"""

from __future__ import annotations

import argparse
import os
import select
import signal
import sys
import time

import numpy as np

import edgefirst_ara2 as ara2
import edgefirst_hal as hal

# Camera capture via native libcamera Python bindings
try:
    import libcamera
except ImportError as exc:
    sys.exit(
        f"libcamera Python bindings not available: {exc}\n"
        "These ship with libcamera — install the libcamera-python package."
    )

# Wayland display via pywayland (no EGL/GL needed)
try:
    from pywayland.client import Display as WlDisplay
    from pywayland.protocol.wayland import WlCompositor, WlSurface, WlCallback, WlBuffer
    from pywayland.protocol.xdg_shell import XdgWmBase, XdgSurface, XdgToplevel
    from pywayland.protocol.linux_dmabuf_unstable_v1 import (
        ZwpLinuxDmabufV1,
        ZwpLinuxBufferParamsV1,
    )
except ImportError as exc:
    sys.exit(
        f"pywayland not available: {exc}\n"
        "Install: pip install pywayland"
    )


# ── Wayland display (direct DMA-BUF submission via pywayland) ────────────────

# DRM fourcc for ABGR8888 (little-endian 'AB24').  This maps to the memory
# layout R-G-B-A which matches HAL's PixelFormat.Rgba byte order.
DRM_FORMAT_ABGR8888 = 0x34324241


class WaylandDisplay:
    """Wayland window that displays DMA-BUF RGBA buffers directly.

    Uses ``zwp_linux_dmabuf_v1`` to submit DMA-BUF buffers to the
    compositor as ``wl_buffer`` objects — no EGL or OpenGL needed.
    Frame pacing is handled via ``wl_surface.frame`` callbacks.
    """

    def __init__(self, width: int, height: int, title: str = "ARA-2 YOLOv8"):
        """Connect to Wayland and create an xdg-shell toplevel window.

        Performs a registry roundtrip to bind three required compositor
        globals (``wl_compositor``, ``xdg_wm_base``, ``zwp_linux_dmabuf_v1``),
        then creates an xdg-shell toplevel window and blocks until the
        compositor sends the initial ``configure`` event.

        Args:
            width:  Window width in pixels (must match the DMA-BUF buffer
                    width passed to ``render_dmabuf``).
            height: Window height in pixels.
            title:  Window title shown in the compositor's task bar.

        Raises:
            RuntimeError: If any of the three required Wayland globals are
                missing (e.g. the compositor does not support
                ``zwp_linux_dmabuf_v1``).
        """
        self.width = width
        self.height = height
        self._closed = False
        self._configured = False
        self._frame_done = True
        self._buffer_cache: dict[int, WlBuffer] = {}

        # Globals
        self._compositor: WlCompositor | None = None
        self._wm_base: XdgWmBase | None = None
        self._dmabuf: ZwpLinuxDmabufV1 | None = None

        # Connect and bind globals
        self._display = WlDisplay()
        self._display.connect()
        registry = self._display.get_registry()
        registry.dispatcher["global"] = self._on_global
        self._display.roundtrip()

        if not self._compositor or not self._wm_base or not self._dmabuf:
            raise RuntimeError(
                "Compositor missing required globals "
                "(wl_compositor, xdg_wm_base, zwp_linux_dmabuf_v1). "
                "Is Weston running?"
            )

        # Create surface + xdg shell window
        self._surface = self._compositor.create_surface()
        self._xdg_surface = self._wm_base.get_xdg_surface(self._surface)
        self._xdg_surface.dispatcher["configure"] = self._on_xdg_configure
        self._toplevel = self._xdg_surface.get_toplevel()
        self._toplevel.set_title(title)
        self._toplevel.set_app_id("ara2-demo")
        self._toplevel.dispatcher["close"] = self._on_close
        self._surface.commit()

        # Wait for configure
        while not self._configured:
            self._display.dispatch(block=True)

        print(f"display: {width}x{height} wayland dmabuf")

    def _on_global(self, registry, id_num, iface_name, version):
        """Handle wl_registry.global events to bind required interfaces.

        Binds three interfaces when advertised by the compositor:

        * ``wl_compositor`` (v4) -- creates surfaces.
        * ``xdg_wm_base`` (v1) -- provides the xdg-shell window lifecycle;
          a ``ping`` handler is attached immediately to keep the connection
          alive.
        * ``zwp_linux_dmabuf_v1`` (v3) -- imports DMA-BUF fds as
          ``wl_buffer`` objects without GPU rendering.
        """
        if iface_name == "wl_compositor":
            self._compositor = registry.bind(id_num, WlCompositor, min(version, 4))
        elif iface_name == "xdg_wm_base":
            self._wm_base = registry.bind(id_num, XdgWmBase, min(version, 1))
            self._wm_base.dispatcher["ping"] = self._on_ping
        elif iface_name == "zwp_linux_dmabuf_v1":
            self._dmabuf = registry.bind(id_num, ZwpLinuxDmabufV1, min(version, 3))

    def _on_ping(self, wm_base, serial):
        """Respond to the compositor's keep-alive ping.

        The xdg_wm_base protocol requires the client to reply with
        ``pong`` promptly; failure to do so causes the compositor to
        consider the client unresponsive and may grey-out the window.
        """
        wm_base.pong(serial)

    def _on_xdg_configure(self, xdg_surface, serial):
        """Acknowledge the compositor's xdg_surface configure event.

        The xdg-shell protocol requires the client to ``ack_configure``
        before it may attach buffers.  The first configure completes the
        window setup handshake started in ``__init__``.
        """
        xdg_surface.ack_configure(serial)
        self._configured = True

    def _on_close(self, *_args):
        """Mark the window as closed when the user (or compositor) requests it.

        Sets ``_closed`` so that ``is_open`` and ``render_dmabuf`` will
        cause the main loop to exit cleanly.
        """
        self._closed = True

    def _on_frame_done(self, callback, _time):
        """Signal that the compositor is ready for the next frame.

        The ``wl_surface.frame`` callback fires after the compositor has
        consumed the previous buffer.  This implements frame pacing --
        ``render_dmabuf`` skips buffer submission while ``_frame_done``
        is ``False``, preventing the client from outrunning the display
        refresh rate.
        """
        self._frame_done = True
        callback._destroy()

    def _get_or_create_buffer(self, fd: int) -> WlBuffer:
        """Return a ``wl_buffer`` for *fd*, creating one on first use.

        On a cache miss the method uses ``zwp_linux_dmabuf_v1`` to wrap
        the DMA-BUF file descriptor in a ``wl_buffer``:

        1. ``create_params()`` -- allocate a ``zwp_linux_buffer_params_v1``.
        2. ``params.add(fd, plane=0, offset=0, stride, ...)`` -- describe
           a single RGBA plane with stride = ``width * 4`` bytes.
        3. ``params.create_immed(...)`` -- import synchronously as
           ``DRM_FORMAT_ABGR8888``.

        The resulting ``wl_buffer`` is cached by *fd* so that repeated
        renders of the same canvas buffer skip the import path entirely.

        Args:
            fd: DMA-BUF file descriptor for an RGBA buffer whose
                dimensions match ``self.width`` x ``self.height``.

        Returns:
            A ``wl_buffer`` backed by the DMA-BUF.
        """
        buf = self._buffer_cache.get(fd)
        if buf is not None:
            return buf

        params = self._dmabuf.create_params()
        params.add(fd, 0, 0, self.width * 4, 0, 0)
        buf = params.create_immed(
            self.width, self.height, DRM_FORMAT_ABGR8888, 0,
        )
        params.destroy()
        self._buffer_cache[fd] = buf
        return buf

    def render_dmabuf(self, fd: int) -> bool:
        """Submit a DMA-BUF RGBA buffer to the compositor.

        Implements frame-paced display:

        * If the previous frame callback has not yet fired the submission
          is skipped and the method returns immediately.  This avoids
          queuing frames faster than the display can present them.
        * Otherwise the buffer is attached, the full surface is marked
          as damaged, a new frame callback is registered, and the
          surface is committed.

        Args:
            fd: DMA-BUF file descriptor for the RGBA canvas to display.

        Returns:
            ``True`` if the window is still open (the caller should
            continue rendering), ``False`` if the window was closed.
        """
        self._display.dispatch(block=False)

        if self._closed:
            return False

        # Only submit when compositor is ready (frame callback fired)
        if not self._frame_done:
            self._display.flush()
            return True

        buffer = self._get_or_create_buffer(fd)
        self._surface.attach(buffer, 0, 0)
        self._surface.damage_buffer(0, 0, self.width, self.height)

        # Request frame callback for pacing
        callback = self._surface.frame()
        callback.dispatcher["done"] = self._on_frame_done
        self._frame_done = False

        self._surface.commit()
        self._display.flush()
        return True

    def is_open(self) -> bool:
        """Dispatch pending Wayland events and return window liveness.

        Performs a non-blocking event dispatch so that close events from
        the compositor are processed before checking ``_closed``.

        Returns:
            ``True`` if the window is still open.
        """
        self._display.dispatch(block=False)
        self._display.flush()
        return not self._closed

    def destroy(self):
        """Disconnect from the Wayland display server.

        Releases the connection; cached ``wl_buffer`` objects are
        invalidated by the disconnect.  Safe to call more than once.
        """
        if self._display:
            self._display.disconnect()
            self._display = None


# ── Helpers (shared with yolov8.py) ───────────────────────────────────────────


def normalize_shape(raw: tuple[int, int, int]) -> list[int]:
    """Normalize an ARA-2 output tensor shape for the HAL decoder.

    ARA-2 reports output shapes as 3-D ``(C, H, W)`` with trailing
    dimensions of 1 used as padding.  The HAL decoder expects a batch
    dimension and no trailing padding, so this function strips trailing
    1s and prepends ``batch=1``.

    Examples::

        (80, 8400, 1) -> [1, 80, 8400]
        (32, 160, 160) -> [1, 32, 160, 160]
        (1, 8400, 1)  -> [1, 1, 8400]

    Args:
        raw: 3-D shape tuple ``(C, H, W)`` as returned by
            ``model.output_shape()``.

    Returns:
        Shape list with ``batch=1`` prepended and trailing 1s removed.
    """
    shape = list(raw)
    while len(shape) > 1 and shape[-1] == 1:
        shape.pop()
    shape.insert(0, 1)
    return shape


def compute_letterbox(
    src_w: int, src_h: int, dst_w: int, dst_h: int
) -> tuple[hal.Rect, tuple[float, float, float, float]]:
    """Compute a letterbox transform that fits *src* into *dst*.

    Preserves the source aspect ratio and centres the image within the
    destination rectangle.  The remaining border is filled with YOLO's
    standard gray-114 padding by the caller.

    Args:
        src_w: Source (camera) width in pixels.
        src_h: Source (camera) height in pixels.
        dst_w: Destination (model input) width in pixels.
        dst_h: Destination (model input) height in pixels.

    Returns:
        A 2-tuple of:

        * ``hal.Rect(x, y, w, h)`` -- pixel-coordinate crop rectangle
          for ``processor.convert()``.
        * ``(x0, y0, x1, y1)`` -- the same rectangle normalised to
          ``[0, 1]`` for ``processor.draw_masks(letterbox=...)``.
    """
    scale = min(dst_w / src_w, dst_h / src_h)
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)
    x = (dst_w - new_w) // 2
    y = (dst_h - new_h) // 2
    rect = hal.Rect(x, y, new_w, new_h)
    norm = (x / dst_w, y / dst_h, (x + new_w) / dst_w, (y + new_h) / dst_h)
    return rect, norm


def output_dtype(bpp: int, signed: bool) -> str:
    """Map ARA-2 output tensor bit-width and sign to a HAL dtype string.

    Args:
        bpp: Bytes per element (1 for 8-bit tensors, 2 for 16-bit).
        signed: Whether the tensor uses signed integers.

    Returns:
        One of ``"int8"``, ``"uint8"``, ``"int16"``, or ``"uint16"``.
    """
    if bpp == 1:
        return "int8" if signed else "uint8"
    return "int16" if signed else "uint16"


def build_decoder(
    shapes: list[list[int]],
    quants: list,
    input_dim: float,
    threshold: float,
    iou: float,
) -> hal.Decoder:
    """Build a HAL YOLOv8 decoder from model output metadata.

    Classifies each output tensor into one of four roles based on its
    shape, so that the HAL decoder knows how to interpret the raw NPU
    output:

    * **Protos** (4-D, e.g. ``[1, 32, 160, 160]``) -- segmentation
      prototype masks.
    * **Boxes** (3-D with ``dim[1] == 4``) -- bounding-box coordinates.
      The quantization scale is normalised by *input_dim* so that
      coordinates are returned in ``[0, 1]``.
    * **Mask coefficients** (3-D with ``dim[1] == n_proto_channels``) --
      per-detection coefficients that combine with the protos.
    * **Scores** (everything else) -- class confidence scores.

    Args:
        shapes: Normalised output shapes (batch-prefixed) from
            ``normalize_shape``.
        quants: Per-output quantization parameters from
            ``model.output_quants()``.
        input_dim: Largest model input dimension (used to normalise box
            coordinates).
        threshold: Minimum score to keep a detection.
        iou: IoU threshold for non-maximum suppression.

    Returns:
        A configured ``hal.Decoder`` ready for
        ``processor.draw_masks()``.
    """
    proto_shape = next((s for s in shapes if len(s) == 4), None)
    n_proto_ch = proto_shape[1] if proto_shape else None

    outputs = []
    for i, shape in enumerate(shapes):
        qn, offset = quants[i].qn, quants[i].offset

        if len(shape) == 4:
            out = hal.Output.protos(shape=shape, decoder=hal.DecoderType.Ultralytics)
            out = out.with_quantization(qn, offset)
        elif len(shape) == 3 and shape[1] == 4:
            scale = qn / input_dim if input_dim > 1 else qn
            out = hal.Output.boxes(shape=shape, decoder=hal.DecoderType.Ultralytics)
            out = out.with_quantization(scale, offset).with_normalized(True)
        elif n_proto_ch and len(shape) == 3 and shape[1] == n_proto_ch:
            out = hal.Output.mask_coefficients(
                shape=shape, decoder=hal.DecoderType.Ultralytics
            )
            out = out.with_quantization(qn, offset)
        else:
            out = hal.Output.scores(shape=shape, decoder=hal.DecoderType.Ultralytics)
            out = out.with_quantization(qn, offset)

        outputs.append(out)

    return hal.Decoder.new_from_outputs(
        outputs,
        score_threshold=threshold,
        iou_threshold=iou,
        decoder_version=hal.DecoderVersion.Yolov8,
    )


# ── DMA-BUF tensor cache (by buffer index) ──────────────────────────────────


class FrameCache:
    """Cache HAL tensors by libcamera buffer index.

    libcamera's ``FrameBufferAllocator`` pre-allocates stable buffers
    (unlike GStreamer which recycles fds).  We cache by request cookie
    (buffer index) so ``import_image`` is called only once per slot.
    """

    def __init__(self, capacity: int) -> None:
        """Pre-allocate a fixed-size cache for imported camera buffers.

        Args:
            capacity: Number of slots, must equal the number of buffers
                allocated by ``libcamera.FrameBufferAllocator``.  Each
                slot maps 1:1 to a libcamera buffer index (cookie).
        """
        self._entries: list[hal.Tensor | None] = [None] * capacity

    def get_or_import(
        self,
        index: int,
        processor: hal.ImageProcessor,
        framebuffer: libcamera.FrameBuffer,
        width: int,
        height: int,
        fmt: hal.PixelFormat = hal.PixelFormat.Nv12,
    ) -> hal.Tensor:
        """Return a cached HAL tensor, importing the DMA-BUF on first use.

        libcamera's ``FrameBufferAllocator`` pre-allocates a fixed pool
        of buffers whose DMA-BUF file descriptors remain stable for the
        lifetime of the camera.  On the first call for a given *index*
        the buffer is imported into the HAL (a relatively expensive
        ``import_image`` call); subsequent calls return the cached
        tensor immediately.

        For semi-planar formats like NV12 the luma (Y) and chroma (UV)
        planes may live in separate DMA-BUFs or at different offsets
        within the same allocation.  When a second plane exists and the
        format is NV12 the chroma fd and offset are passed through so
        that ``import_image`` can map both planes correctly.

        Args:
            index: Buffer index (the libcamera request cookie), used as
                the cache key.
            processor: HAL image processor used to import the DMA-BUF.
            framebuffer: The ``libcamera.FrameBuffer`` whose planes
                provide the DMA-BUF fds.
            width:  Frame width in pixels.
            height: Frame height in pixels.
            fmt: Pixel format of the camera buffer (default NV12).

        Returns:
            A ``hal.Tensor`` wrapping the imported DMA-BUF, suitable for
            passing to ``processor.convert()`` or ``draw_masks()``.
        """
        tensor = self._entries[index]
        if tensor is None:
            planes = framebuffer.planes
            fd0 = planes[0].fd
            # Semi-planar formats (NV12) may have a separate chroma plane
            if len(planes) >= 2 and fmt == hal.PixelFormat.Nv12:
                chroma_fd = planes[1].fd
                chroma_offset = planes[1].offset or None
            else:
                chroma_fd = None
                chroma_offset = None
            tensor = processor.import_image(
                fd0, width, height, fmt,
                chroma_fd=chroma_fd,
                chroma_offset=chroma_offset,
            )
            self._entries[index] = tensor
        return tensor

    def __len__(self) -> int:
        """Return the number of slots that have been imported so far."""
        return sum(1 for e in self._entries if e is not None)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    """Run the YOLOv8 live camera inference pipeline.

    Orchestrates the full lifecycle in numbered stages:

    1. **Model metadata** -- read labels and compilation stats from the
       ``.dvm`` file for display.
    3. **ARA-2 session** -- connect to the NPU proxy, load the model,
       and allocate DMA-BUF tensors.
    4. **Decoder** -- build the HAL post-processor from output shapes
       and quantization parameters.
    5. **HAL processor** -- import the model's input tensor and set up
       the letterbox transform for aspect-ratio-preserving resize.
    6. **Output canvas** -- allocate a single RGBA DMA-BUF that
       ``draw_masks`` renders into and the compositor displays.
    7. **libcamera** -- configure the camera, allocate frame buffers,
       and populate the frame cache.
    8. **Wayland display** -- create the window for DMA-BUF presentation.
    9. **Start camera** -- begin streaming and register a SIGINT handler.
    10. **Warmup** -- run one full inference pass to JIT-compile any
        GPU shaders before the timed loop.
    11. **Live loop** -- capture, infer, draw, and display in a tight
        loop with per-stage timing.
    12. **Shutdown** -- stop the camera and release resources.
    """
    ap = argparse.ArgumentParser(
        description="YOLOv8 live camera inference on ARA-2 with Wayland display",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("model", help="Path to compiled .dvm model file")
    ap.add_argument("--threshold", type=float, default=0.50)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--width", type=int, default=1920, help="Camera width")
    ap.add_argument("--height", type=int, default=1080, help="Camera height")
    ap.add_argument(
        "--camera-name", default=None,
        help="libcamera camera ID",
    )
    ap.add_argument(
        "--format", default="nv12", choices=["nv12", "yuyv"],
        help="Camera pixel format (default: nv12)",
    )
    ap.add_argument("--socket", default=ara2.DEFAULT_SOCKET)
    args = ap.parse_args()

    cam_w, cam_h = args.width, args.height

    # Map --format to libcamera and HAL pixel formats
    format_map = {
        "nv12": (libcamera.formats.NV12, hal.PixelFormat.Nv12),
        "yuyv": (libcamera.formats.YUYV, hal.PixelFormat.Yuyv),
    }
    libcam_fmt, hal_fmt = format_map[args.format]

    # ── 1. Read model metadata ───────────────────────────────────────────
    metadata = ara2.read_metadata(args.model)
    labels = ara2.read_labels(args.model) or COCO_LABELS
    if metadata:
        print(f"Model: {args.model}")
        print(f"Task: {metadata.task}, Classes: {len(labels)}")
        if metadata.compilation and metadata.compilation.ppa:
            ppa = metadata.compilation.ppa
            print(
                f"Target: {metadata.compilation.target}, "
                f"IPS: {ppa.ips:.0f}, Power: {ppa.power_mw:.0f} mW"
            )

    # ── 3. Connect to ARA-2 and load model ───────────────────────────────
    session = ara2.Session.create_via_unix_socket(args.socket)
    endpoints = session.list_endpoints()
    if not endpoints:
        sys.exit("No ARA-2 endpoints found.  Is ara2-proxy running?")

    endpoint = endpoints[0]
    stats = endpoint.dram_statistics()
    print(
        f"Endpoint: {endpoint.check_status()}, "
        f"DRAM: {stats.free_size / 1048576:.0f} / "
        f"{stats.dram_size / 1048576:.0f} MB free"
    )

    with endpoint.load_model(args.model) as model:
        model.allocate_tensors("dma")
        c, h, w = model.input_shape(0)
        input_dim = float(max(w, h))
        iq = model.input_quants(0)
        print(f"Input: {c}x{h}x{w} (CHW)")

        # ── 4. Build decoder ─────────────────────────────────────────────
        shapes, quants = [], []
        for i in range(model.n_outputs):
            shapes.append(normalize_shape(model.output_shape(i)))
            quants.append(model.output_quants(i))

        decoder = build_decoder(shapes, quants, input_dim, args.threshold, args.iou)

        # ── 5. Setup HAL processor and model I/O tensors ─────────────────
        processor = hal.ImageProcessor()

        input_fd = model.input_tensor_fd(0)
        try:
            model_input = processor.import_image(
                input_fd, w, h, hal.PixelFormat.PlanarRgb,
                dtype="int8" if iq.is_signed else "uint8",
            )
        finally:
            os.close(input_fd)

        letterbox_rect, letterbox_norm = compute_letterbox(cam_w, cam_h, w, h)
        pad_color = (114, 114, 114, 255)

        output_tensors = []
        for i in range(model.n_outputs):
            fd = model.output_tensor_fd(i)
            output_tensors.append(
                hal.Tensor.from_fd(
                    fd, shapes[i],
                    output_dtype(model.output_info(i).bpp, model.output_quants(i).is_signed),
                )
            )

        # ── 6. Output canvas for draw_masks ─────────────────────────────
        # Single RGBA canvas.  HAL calls glFinish() before returning from
        # draw_masks(), so the DMA-BUF is fully written and safe to submit
        # to the Wayland compositor.
        canvas = processor.create_image(cam_w, cam_h, hal.PixelFormat.Rgba)
        canvas_fd = canvas.fd  # dup'd fd, we own it

        # ── 7. Setup libcamera ───────────────────────────────────────────
        cm = libcamera.CameraManager.singleton()
        cameras = cm.cameras
        if not cameras:
            sys.exit("No cameras found")

        if args.camera_name:
            cam_obj = next(
                (c for c in cameras if c.id == args.camera_name), None
            )
            if cam_obj is None:
                print(f"Camera '{args.camera_name}' not found. Available:")
                for c in cameras:
                    print(f"  {c.id}")
                sys.exit(1)
        else:
            cam_obj = cameras[0]

        print(f"Camera: {cam_obj.id}")
        cam_obj.acquire()

        cam_config = cam_obj.generate_configuration(
            [libcamera.StreamRole.VideoRecording]
        )
        stream_cfg = cam_config.at(0)
        stream_cfg.pixel_format = libcam_fmt
        stream_cfg.size = libcamera.Size(cam_w, cam_h)
        status = cam_config.validate()
        if status == libcamera.CameraConfiguration.Status.Invalid:
            sys.exit("Invalid camera configuration")
        cam_obj.configure(cam_config)

        stream = stream_cfg.stream
        alloc = libcamera.FrameBufferAllocator(cam_obj)
        alloc.allocate(stream)
        buffers = alloc.buffers(stream)
        n_buffers = len(buffers)
        print(f"Allocated {n_buffers} camera buffers")

        cam_reqs = []
        for i, fb in enumerate(buffers):
            req = cam_obj.create_request(i)
            req.add_buffer(stream, fb)
            cam_reqs.append(req)

        frame_cache = FrameCache(n_buffers)

        # ── 8. Create Wayland display window ─────────────────────────────
        display = WaylandDisplay(cam_w, cam_h, "ARA-2 YOLOv8 Live")

        # ── 9. Start camera ──────────────────────────────────────────────
        cam_obj.start()
        for req in cam_reqs:
            cam_obj.queue_request(req)
        event_fd = cm.event_fd

        # Graceful shutdown
        running = True

        def _on_sigint(_sig, _frame):
            nonlocal running
            running = False

        signal.signal(signal.SIGINT, _on_sigint)

        print(f"\nCapturing {cam_w}x{cam_h} — press Ctrl+C to stop\n")

        # ── 10. Warmup ──────────────────────────────────────────────────
        select.select([event_fd], [], [], 5.0)
        ready = cm.get_ready_requests()
        if not ready:
            sys.exit("No frames from camera")
        req = ready[0]
        idx = req.cookie
        fb = req.buffers[stream]
        src = frame_cache.get_or_import(idx, processor, fb, cam_w, cam_h, hal_fmt)
        req.reuse()
        cam_obj.queue_request(req)
        # Requeue any extra ready requests
        for r in ready[1:]:
            r.reuse()
            cam_obj.queue_request(r)

        processor.convert(
            src, model_input,
            dst_crop=letterbox_rect, dst_color=pad_color,
        )
        model.run()
        processor.draw_masks(
            decoder=decoder,
            model_output=output_tensors,
            dst=canvas,
            background=src,
            letterbox=letterbox_norm,
            color_mode=hal.ColorMode.Instance,
        )
        display.render_dmabuf(canvas_fd)

        print(f"Warmup complete, cache: {len(frame_cache)} bufs")

        # ── 11. Live inference loop ──────────────────────────────────────
        frame_count = 0
        t_start = time.monotonic()

        # Per-stage timing accumulators (in seconds)
        t_pull = 0.0; t_import = 0.0; t_convert = 0.0
        t_npu = 0.0; t_draw = 0.0; t_display = 0.0; t_sync = 0.0
        total_dropped = 0

        while running and display.is_open():
            t0 = time.monotonic()

            # Wait for a completed frame, then drain to get the latest
            select.select([event_fd], [], [], 5.0)
            ready = cm.get_ready_requests()
            if not ready:
                break
            dropped = len(ready) - 1
            # Requeue all but the latest
            for r in ready[:-1]:
                r.reuse()
                cam_obj.queue_request(r)
            req = ready[-1]
            t1 = time.monotonic()

            idx = req.cookie
            fb = req.buffers[stream]
            src = frame_cache.get_or_import(idx, processor, fb, cam_w, cam_h, hal_fmt)
            req.reuse()
            cam_obj.queue_request(req)
            t2 = time.monotonic()

            processor.convert(
                src, model_input,
                dst_crop=letterbox_rect, dst_color=pad_color,
            )
            t3 = time.monotonic()

            timing = model.run()
            t4 = time.monotonic()

            boxes, scores, classes = processor.draw_masks(
                decoder=decoder,
                model_output=output_tensors,
                dst=canvas,
                background=src,
                letterbox=letterbox_norm,
                color_mode=hal.ColorMode.Instance,
            )
            t5 = time.monotonic()

            display.render_dmabuf(canvas_fd)
            t6 = time.monotonic()

            t_pull += t1 - t0
            t_import += t2 - t1
            t_convert += t3 - t2
            t_npu += t4 - t3
            t_draw += t5 - t4
            t_display += t6 - t5
            t_sync += t6 - t0
            total_dropped += dropped

            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.monotonic() - t_start
                fps = frame_count / elapsed
                n = 30  # frames in this reporting window
                print(
                    f"\r  FPS: {fps:5.1f}  "
                    f"pull:{t_pull/n*1000:5.1f} "
                    f"imp:{t_import/n*1000:4.1f} "
                    f"cvt:{t_convert/n*1000:4.1f} "
                    f"npu:{t_npu/n*1000:5.1f} "
                    f"draw:{t_draw/n*1000:5.1f} "
                    f"disp:{t_display/n*1000:4.1f} "
                    f"tot:{t_sync/n*1000:5.1f}ms "
                    f"drop:{total_dropped} "
                    f"det:{len(scores)} "
                    f"f:{frame_count}",
                    end="", flush=True,
                )
                t_pull = t_import = t_convert = 0.0
                t_npu = t_draw = t_display = t_sync = 0.0

        # ── 12. Shutdown ─────────────────────────────────────────────────
        print("\n")
        if frame_count > 0:
            elapsed = time.monotonic() - t_start
            print(
                f"Processed {frame_count} frames in {elapsed:.1f}s "
                f"({frame_count / elapsed:.1f} FPS average)"
            )

        cam_obj.stop()
        cam_obj.release()
        os.close(canvas_fd)
        display.destroy()


# ── COCO labels (fallback) ────────────────────────────────────────────────────

COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


if __name__ == "__main__":
    main()
