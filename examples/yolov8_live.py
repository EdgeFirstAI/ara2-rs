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

# DRM_FORMAT_ABGR8888 — matches HAL's RGBA pixel layout.
DRM_FORMAT_ABGR8888 = 0x34324241


class WaylandDisplay:
    """Wayland window that displays DMA-BUF RGBA buffers directly.

    Uses ``zwp_linux_dmabuf_v1`` to submit DMA-BUF buffers to the
    compositor as ``wl_buffer`` objects — no EGL or OpenGL needed.
    Frame pacing is handled via ``wl_surface.frame`` callbacks.
    """

    def __init__(self, width: int, height: int, title: str = "ARA-2 YOLOv8"):
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
        if iface_name == "wl_compositor":
            self._compositor = registry.bind(id_num, WlCompositor, min(version, 4))
        elif iface_name == "xdg_wm_base":
            self._wm_base = registry.bind(id_num, XdgWmBase, min(version, 1))
            self._wm_base.dispatcher["ping"] = self._on_ping
        elif iface_name == "zwp_linux_dmabuf_v1":
            self._dmabuf = registry.bind(id_num, ZwpLinuxDmabufV1, min(version, 3))

    def _on_ping(self, wm_base, serial):
        wm_base.pong(serial)

    def _on_xdg_configure(self, xdg_surface, serial):
        xdg_surface.ack_configure(serial)
        self._configured = True

    def _on_close(self, *_args):
        self._closed = True

    def _on_frame_done(self, callback, _time):
        self._frame_done = True
        callback._destroy()

    def _get_or_create_buffer(self, fd: int) -> WlBuffer:
        """Get a cached wl_buffer or create one from a DMA-BUF fd."""
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
        """Submit a DMA-BUF RGBA buffer to the compositor."""
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
        self._display.dispatch(block=False)
        self._display.flush()
        return not self._closed

    def destroy(self):
        if self._display:
            self._display.disconnect()
            self._display = None


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


# ── DMA-BUF tensor cache (by buffer index) ──────────────────────────────────


class FrameCache:
    """Cache HAL tensors by libcamera buffer index.

    libcamera's ``FrameBufferAllocator`` pre-allocates stable buffers
    (unlike GStreamer which recycles fds).  We cache by request cookie
    (buffer index) so ``import_image`` is called only once per slot.
    """

    def __init__(self, capacity: int) -> None:
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
        return sum(1 for e in self._entries if e is not None)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
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
