#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
"""
YOLOv8 live camera inference on ARA-2 NPU with EGL/Wayland display.

Captures video from a camera (via GStreamer), runs YOLOv8 detection +
instance segmentation on the ARA-2 NPU, and displays results in a
native Wayland/EGL window.

Pipeline::

    GStreamer capture        HAL / ARA-2                 EGL display
    ┌──────────────┐    ┌───────────────────┐    ┌──────────────────────┐
    │ libcamerasrc  │    │ inode-cached      │    │ libwlegl_display.so  │
    │   or v4l2src  │ →  │ import → convert  │ →  │  dmabuf → EGLImage  │
    │ → appsink     │    │ → NPU inference   │    │  → GL texture       │
    │ (NV12 dmabuf) │    │ → draw_masks      │    │  → eglSwapBuffers   │
    └──────────────┘    │ → RGBA canvas     │    └──────────────────────┘
                         └───────────────────┘

Camera sources:
  - ``libcamerasrc`` — for platforms with libcamera (e.g. i.MX 95 NeoISP)
  - ``v4l2src`` — for platforms with V4L2 cameras (e.g. i.MX 8M Plus)

Display:
  The ``wlegl_display.c`` companion library creates a Wayland/EGL window
  and renders DMA-BUF textures via ``EGL_EXT_image_dma_buf_import``.
  A memcpy fallback (``display_render_pixels``) is used if DMA-BUF
  import fails.

Usage::

    # libcamera (i.MX 95):
    python yolov8_live.py model.dvm --source libcamera

    # V4L2 camera:
    python yolov8_live.py model.dvm --source v4l2 --device /dev/video0

    # Custom GStreamer pipeline:
    python yolov8_live.py model.dvm --source 'videotestsrc ! video/x-raw,format=NV12,width=640,height=480'

Requirements:
    edgefirst-ara2  edgefirst-hal  numpy  PyGObject
    GStreamer 1.20+ with libcamerasrc or v4l2src
    libwlegl_display.so (compiled from wlegl_display.c)
    Wayland compositor running (e.g. Weston)
"""

from __future__ import annotations

import argparse
import ctypes
import os
import signal
import sys
import time

import numpy as np

import edgefirst_ara2 as ara2
import edgefirst_hal as hal

# GStreamer is only used for camera capture
try:
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstAllocators", "1.0")
    from gi.repository import Gst, GstAllocators
except (ImportError, ValueError) as exc:
    sys.exit(
        f"GStreamer Python bindings not available: {exc}\n"
        "Install: python3-gi gstreamer1.0-plugins-base gstreamer1.0-plugins-good"
    )


# ── EGL Display Library (ctypes) ─────────────────────────────────────────────


class EglDisplay:
    """Python wrapper around libwlegl_display.so."""

    def __init__(self, width: int, height: int, title: str = "ARA-2 YOLOv8"):
        self._lib = ctypes.cdll.LoadLibrary("libwlegl_display.so")

        # void *display_init(int w, int h, const char *title)
        self._lib.display_init.restype = ctypes.c_void_p
        self._lib.display_init.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_char_p,
        ]
        # int display_render_dmabuf(void *d, int fd, int w, int h)
        self._lib.display_render_dmabuf.restype = ctypes.c_int
        self._lib.display_render_dmabuf.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        # int display_render_pixels(void *d, const void *rgba, int w, int h)
        self._lib.display_render_pixels.restype = ctypes.c_int
        self._lib.display_render_pixels.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ]
        # int display_is_open(void *d)
        self._lib.display_is_open.restype = ctypes.c_int
        self._lib.display_is_open.argtypes = [ctypes.c_void_p]
        # void display_destroy(void *d)
        self._lib.display_destroy.restype = None
        self._lib.display_destroy.argtypes = [ctypes.c_void_p]

        self._ptr = self._lib.display_init(width, height, title.encode())
        if not self._ptr:
            raise RuntimeError(
                "Failed to create EGL display window. "
                "Is a Wayland compositor (Weston) running?"
            )
        self.width = width
        self.height = height
        self._dmabuf_ok = True  # optimistic; falls back on first failure

    def render_dmabuf(self, fd: int, width: int, height: int) -> bool:
        """Render a DMA-BUF RGBA texture. Returns True on success."""
        if not self._dmabuf_ok:
            return False
        ret = self._lib.display_render_dmabuf(self._ptr, fd, width, height)
        if ret != 0:
            self._dmabuf_ok = False
            return False
        return True

    def render_pixels(self, data: np.ndarray, width: int, height: int) -> bool:
        """Render RGBA pixels from a numpy array (memcpy path)."""
        ptr = data.ctypes.data_as(ctypes.c_void_p)
        return self._lib.display_render_pixels(self._ptr, ptr, width, height) == 0

    def is_open(self) -> bool:
        return bool(self._lib.display_is_open(self._ptr))

    def destroy(self):
        if self._ptr:
            self._lib.display_destroy(self._ptr)
            self._ptr = None

    def __del__(self):
        self.destroy()


# ── Helpers (shared with yolov8.py) ───────────────────────────────────────────


def normalize_shape(raw: tuple[int, int, int]) -> list[int]:
    """Normalize an ARA-2 output shape for the HAL decoder.

    ARA-2 reports 3D shapes [C, H, W].  Strip trailing 1s, prepend batch=1.
    """
    shape = list(raw)
    while len(shape) > 1 and shape[-1] == 1:
        shape.pop()
    shape.insert(0, 1)
    return shape


def compute_letterbox(
    src_w: int, src_h: int, dst_w: int, dst_h: int
) -> tuple[hal.Rect, tuple[float, float, float, float]]:
    """Fit *src* into *dst* preserving aspect ratio (YOLO gray-114 padding)."""
    scale = min(dst_w / src_w, dst_h / src_h)
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)
    x = (dst_w - new_w) // 2
    y = (dst_h - new_h) // 2
    rect = hal.Rect(x, y, new_w, new_h)
    norm = (x / dst_w, y / dst_h, (x + new_w) / dst_w, (y + new_h) / dst_h)
    return rect, norm


def output_dtype(bpp: int, signed: bool) -> str:
    """Map ARA-2 output tensor bpp/signed flags to a HAL dtype string."""
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
    """Build a HAL Decoder from model output metadata."""
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


# ── Inode-based DMA-BUF tensor cache ─────────────────────────────────────────


class InodeTensorCache:
    """Cache HAL tensors keyed by DMA-BUF ``(inode, offset)``.

    File descriptors are *not* stable identifiers for DMA-BUF buffers:
    when GStreamer unrefs a buffer the fd may be recycled.  The Linux
    kernel assigns a unique inode to each ``dma_buf`` object which remains
    constant regardless of fd recycling.

    See HAL ``ARCHITECTURE.md`` § "DMA-BUF Inode as Stable Identity".
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[int, int], hal.Tensor] = {}

    def get_or_import(
        self,
        processor: hal.ImageProcessor,
        fd: int,
        width: int,
        height: int,
        fmt: hal.PixelFormat,
        *,
        offset: int = 0,
        chroma_fd: int | None = None,
        chroma_offset: int | None = None,
    ) -> hal.Tensor:
        key = (os.fstat(fd).st_ino, offset)
        tensor = self._cache.get(key)
        if tensor is None:
            tensor = processor.import_image(
                fd, width, height, fmt,
                offset=offset if offset else None,
                chroma_fd=chroma_fd,
                chroma_offset=chroma_offset,
            )
            self._cache[key] = tensor
        return tensor

    def __len__(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        self._cache.clear()


# ── Camera capture ────────────────────────────────────────────────────────────


def build_capture_pipeline(args: argparse.Namespace) -> Gst.Element:
    """Build a GStreamer pipeline that delivers NV12 DMA-BUF frames."""
    cam_w, cam_h = args.width, args.height

    if args.source == "libcamera":
        src = "libcamerasrc"
        if args.camera_name:
            src += f" camera-name={args.camera_name}"
        desc = (
            f"{src} ! "
            f"video/x-raw,format=NV12,width={cam_w},height={cam_h} ! "
            f"appsink name=capture emit-signals=false sync=false "
            f"max-buffers=1 drop=true"
        )
    elif args.source == "v4l2":
        desc = (
            f"v4l2src device={args.device} ! "
            f"video/x-raw,format=NV12,width={cam_w},height={cam_h} ! "
            f"appsink name=capture emit-signals=false sync=false "
            f"max-buffers=1 drop=true"
        )
    else:
        # Custom pipeline string — user provides everything before appsink
        desc = (
            f"{args.source} ! "
            f"appsink name=capture emit-signals=false sync=false "
            f"max-buffers=1 drop=true"
        )

    return Gst.parse_launch(desc)


def is_dmabuf_buffer(buffer: Gst.Buffer) -> bool:
    """Check if a GstBuffer contains DMA-BUF memory."""
    return GstAllocators.is_dmabuf_memory(buffer.peek_memory(0))


def get_dmabuf_fds(buffer: Gst.Buffer) -> tuple[int, int | None, int | None]:
    """Extract DMA-BUF fds and chroma offset from a GstBuffer.

    Returns ``(luma_fd, chroma_fd, chroma_offset)``.
    For single-plane formats, chroma values are None.
    """
    mem0 = buffer.peek_memory(0)
    luma_fd = GstAllocators.dmabuf_memory_get_fd(mem0)

    if buffer.n_memory() >= 2:
        mem1 = buffer.peek_memory(1)
        chroma_fd = GstAllocators.dmabuf_memory_get_fd(mem1)
        _, chroma_offset, _ = mem1.get_sizes()
        return luma_fd, chroma_fd, chroma_offset

    return luma_fd, None, None


def copy_sysmem_to_tensor(buffer: Gst.Buffer, tensor: hal.Tensor) -> None:
    """Copy system-memory GstBuffer data into a HAL tensor via TensorMap.

    Used as a fallback when the camera produces system memory instead
    of DMA-BUF (e.g. videotestsrc, USB cameras without DMA-BUF).
    """
    ok, info = buffer.map(Gst.MapFlags.READ)
    if not ok:
        raise RuntimeError("Failed to map GstBuffer")
    try:
        mapped = tensor.map()
        dst = np.frombuffer(mapped, dtype=np.uint8)
        src = np.frombuffer(info.data, dtype=np.uint8)
        n = min(len(src), len(dst))
        dst[:n] = src[:n]
        mapped.unmap()
    finally:
        buffer.unmap(info)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description="YOLOv8 live camera inference on ARA-2 with EGL display",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("model", help="Path to compiled .dvm model file")
    ap.add_argument("--threshold", type=float, default=0.50)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--width", type=int, default=1920, help="Camera width")
    ap.add_argument("--height", type=int, default=1080, help="Camera height")
    ap.add_argument(
        "--source", default="libcamera",
        help="Camera source: 'libcamera', 'v4l2', or a custom GStreamer pipeline",
    )
    ap.add_argument(
        "--device", default="/dev/video0",
        help="V4L2 device path (only used with --source v4l2)",
    )
    ap.add_argument(
        "--camera-name", default=None,
        help="libcamerasrc camera-name property",
    )
    ap.add_argument("--socket", default=ara2.DEFAULT_SOCKET)
    ap.add_argument(
        "--display-mode", choices=["dmabuf", "memcpy", "auto"], default="auto",
        help="Display mode: dmabuf (zero-copy), memcpy (fallback), auto (try dmabuf first)",
    )
    args = ap.parse_args()

    cam_w, cam_h = args.width, args.height

    # ── 1. Initialize GStreamer ──────────────────────────────────────────
    Gst.init(None)

    # ── 2. Read model metadata ───────────────────────────────────────────
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
        # draw_masks(), so the DMA-BUF is fully written and safe to pass
        # to the display's EGL context.
        canvas = processor.create_image(cam_w, cam_h, hal.PixelFormat.Rgba)
        canvas_fd = canvas.fd  # dup'd fd, we own it

        # Memcpy fallback buffer
        canvas_np = np.zeros((cam_h, cam_w, 4), dtype=np.uint8)

        # ── 7. Inode-based tensor cache ──────────────────────────────────
        input_cache = InodeTensorCache()

        # ── 8. Create EGL display window ─────────────────────────────────
        display = EglDisplay(cam_w, cam_h, "ARA-2 YOLOv8 Live")

        use_dmabuf = args.display_mode != "memcpy"

        # ── 9. Build and start capture pipeline ──────────────────────────
        capture_pipe = build_capture_pipeline(args)
        appsink = capture_pipe.get_by_name("capture")

        ret = capture_pipe.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            sys.exit("Failed to start capture pipeline")

        # Graceful shutdown
        running = True

        def _on_sigint(_sig, _frame):
            nonlocal running
            running = False

        signal.signal(signal.SIGINT, _on_sigint)

        print(f"\nCapturing {cam_w}x{cam_h} — press Ctrl+C to stop\n")

        # ── 10. Warmup — detect capture memory type ─────────────────────
        sample = appsink.emit("pull-sample")
        if sample is None:
            sys.exit("Failed to pull initial sample from camera")

        buf = sample.get_buffer()
        capture_is_dmabuf = is_dmabuf_buffer(buf)

        # For system-memory sources, create a HAL tensor to receive copies
        sysmem_input = None
        if not capture_is_dmabuf:
            sysmem_input = processor.create_image(cam_w, cam_h, hal.PixelFormat.Nv12)
            print(f"Capture: system memory (will memcpy into HAL tensor)")
        else:
            print(f"Capture: DMA-BUF (zero-copy)")

        def import_frame(buf: Gst.Buffer) -> hal.Tensor:
            """Import a GstBuffer as a HAL tensor (DMA-BUF or sysmem)."""
            if capture_is_dmabuf:
                luma_fd, chroma_fd, chroma_off = get_dmabuf_fds(buf)
                return input_cache.get_or_import(
                    processor, luma_fd, cam_w, cam_h, hal.PixelFormat.Nv12,
                    chroma_fd=chroma_fd, chroma_offset=chroma_off,
                )
            else:
                copy_sysmem_to_tensor(buf, sysmem_input)
                return sysmem_input

        src = import_frame(buf)
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

        # Test display path
        if use_dmabuf:
            if not display.render_dmabuf(canvas_fd, cam_w, cam_h):
                print("DMA-BUF display failed, falling back to memcpy")
                use_dmabuf = False

        if not use_dmabuf:
            canvas.normalize_to_numpy(canvas_np)
            display.render_pixels(canvas_np, cam_w, cam_h)

        print(f"Warmup complete, cache: {len(input_cache)} bufs, "
              f"display: {'dmabuf' if use_dmabuf else 'memcpy'}")

        # ── 11. Live inference loop ──────────────────────────────────────
        frame_count = 0
        t_start = time.monotonic()

        # Per-stage timing accumulators (in seconds)
        t_pull = 0.0; t_import = 0.0; t_convert = 0.0
        t_npu = 0.0; t_draw = 0.0; t_display = 0.0; t_sync = 0.0
        total_dropped = 0

        while running and display.is_open():
            t0 = time.monotonic()

            # Pull the latest frame, dropping any stale queued frames.
            # This prevents pipeline buffering from adding latency.
            sample = appsink.emit("pull-sample")
            if sample is None:
                break
            dropped = 0
            while True:
                newer = appsink.emit("try-pull-sample", 0)
                if newer is None:
                    break
                sample = newer
                dropped += 1
            t1 = time.monotonic()

            buf = sample.get_buffer()
            src = import_frame(buf)
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

            if use_dmabuf:
                display.render_dmabuf(canvas_fd, cam_w, cam_h)
            else:
                canvas.normalize_to_numpy(canvas_np)
                display.render_pixels(canvas_np, cam_w, cam_h)
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

        capture_pipe.set_state(Gst.State.NULL)
        os.close(canvas_fd)
        display.destroy()
        input_cache.clear()


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
