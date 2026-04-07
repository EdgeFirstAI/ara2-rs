# ARA-2 Examples

End-to-end inference examples for the ARA-2 NPU using the EdgeFirst HAL
for GPU preprocessing, decoding, and overlay rendering.  All examples use
zero-copy DMA-BUF buffers throughout the pipeline.

## Prerequisites

- ARA-2 proxy service running: `systemctl status ara2.service`
- ARA-2 PCIe device visible: `lspci | grep -i kinara`
- A compiled DVM model, e.g. `yolov8m-seg_640x640.dvm`

| File | Description |
|------|-------------|
| `yolov8.rs` | Static image inference (Rust) |
| `yolov8_live.rs` | Live camera inference with libcamera (Rust) |
| `yolov8_live.py` | Live camera inference with GStreamer (Python) |
| `wlegl_display.c` | Wayland/EGL display library (C, shared by live examples) |

---

## yolov8 -- Static Image Inference (Rust)

Runs YOLOv8 detection or instance segmentation on a single JPEG image.
Supports benchmarking with `--benchmark N` for per-stage timing statistics.

### Build

```bash
# Cross-compile (from development host):
cargo zigbuild --release --target aarch64-unknown-linux-gnu --example yolov8

# Or on target:
cargo build --release --example yolov8
```

### Run

```bash
yolov8 <model.dvm> <image.jpg> [--save] [--threshold 0.25] [--iou 0.45] [--benchmark N]
```

| Flag | Default | Description |
|------|---------|-------------|
| `model` | (required) | Path to `.dvm` model file |
| `image` | (required) | Path to input JPEG image |
| `--save` | off | Save overlay result as `<image>_overlay.jpg` |
| `--threshold` | 0.25 | Detection confidence threshold |
| `--iou` | 0.45 | NMS IoU threshold |
| `--benchmark` | 0 | Run N iterations with timing statistics |

---

## yolov8_live -- Live Camera Inference (Rust + libcamera)

Captures NV12 frames from a camera via libcamera, runs YOLOv8 inference on
the ARA-2 NPU, and displays results in a Wayland/EGL window.  This is a
minimal serial (single-threaded) pipeline.

### Architecture

```
libcamera (NV12 DMA-BUF)
  -> HAL import (cached by buffer index)
  -> HAL convert (NV12 -> PlanarRGB letterbox)
  -> ARA-2 NPU inference
  -> HAL draw_masks (decode + composite -> RGBA canvas)
  -> EGL display (DMA-BUF -> EGLImage -> GL texture)
```

### Build

The libcamera crate requires `libcamera-dev` headers, so this example must
be compiled on-target (cannot cross-compile with zigbuild).

```bash
# On target:
cargo build --release --example yolov8_live
```

### Run

```bash
export XDG_RUNTIME_DIR=/run/user/0
export WAYLAND_DISPLAY=wayland-0
export LIBCAMERA_PIPELINES_MATCH_LIST='nxp/neo,imx8-isi,simple'

yolov8_live /root/models/yolov8m-seg_640x640.dvm \
    --camera-name '/base/soc/bus@42000000/i2c@42540000/os08a20_mipi@36' \
    --width 1920 --height 1080
```

| Flag | Default | Description |
|------|---------|-------------|
| `model` | (required) | Path to `.dvm` model file |
| `--camera-name` | first available | libcamera camera ID |
| `--width` | 1920 | Camera capture width |
| `--height` | 1080 | Camera capture height |
| `--threshold` | 0.50 | Detection confidence threshold |
| `--iou` | 0.45 | NMS IoU threshold |
| `--socket` | `/var/run/ara2.sock` | ARA-2 proxy UNIX socket path |

---

## yolov8_live.py -- Live Camera Inference (Python + GStreamer)

Python version of the live pipeline using GStreamer for camera capture.
Requires PyGObject and GStreamer Python bindings.

### Python environment

```bash
python3 -m venv --system-site-packages /root/venv
source /root/venv/bin/activate
pip install edgefirst-ara2 edgefirst-hal numpy
```

The `--system-site-packages` flag is required to pick up PyGObject and
GStreamer bindings from system packages.

### Run

```bash
source /root/venv/bin/activate
python3 yolov8_live.py /root/models/yolov8m-seg_640x640.dvm \
    --source libcamera \
    --camera-name '/base/soc/bus@42000000/i2c@42540000/os08a20_mipi@36' \
    --width 1920 --height 1080
```

| Flag | Default | Description |
|------|---------|-------------|
| `model` | (required) | Path to `.dvm` model file |
| `--source` | `libcamera` | `libcamera`, `v4l2`, or a custom GStreamer pipeline |
| `--camera-name` | auto | libcamerasrc `camera-name` property |
| `--device` | `/dev/video0` | V4L2 device path (with `--source v4l2`) |
| `--width` | 1920 | Camera capture width |
| `--height` | 1080 | Camera capture height |
| `--threshold` | 0.50 | Detection confidence threshold |
| `--iou` | 0.45 | NMS IoU threshold |
| `--display-mode` | `auto` | `dmabuf`, `memcpy`, or `auto` |
| `--socket` | `/run/ara2.sock` | ARA-2 proxy UNIX socket path |

### GStreamer appsink API note

The Python GI bindings for `GstApp.AppSink` do not expose `.pull_sample()`
as a method.  Use the signal-emission API instead:

```python
sample = appsink.emit("pull-sample")        # blocking
sample = appsink.emit("try-pull-sample", 0)  # non-blocking
```

### Inode-based DMA-BUF tensor cache

GStreamer recycles DMA-BUF file descriptors -- when a buffer is unreffed
the fd number may be reused for a different buffer.  The Python example
caches by inode (`os.fstat(fd).st_ino`) instead of fd to avoid stale hits.
The Rust example uses libcamera's stable buffer indices and does not need
this workaround.

---

## Display Library (wlegl_display.c)

Minimal Wayland/EGL shared library for rendering DMA-BUF RGBA textures.
Used by both the Rust and Python live examples via FFI/ctypes.

### Build (on target)

```bash
gcc -shared -fPIC -o libwlegl_display.so wlegl_display.c \
    $(pkg-config --cflags --libs wayland-client wayland-egl egl glesv2) \
    -DLINUX -DWL_EGL_PLATFORM

cp libwlegl_display.so /usr/lib/
ldconfig
```

### Dependencies

- `libwayland-dev`, `libwayland-egl1`
- `libegl-dev`, `libgles2-dev`
- `pkg-config`, `gcc`

The xdg-shell Wayland protocol is implemented inline -- no
`wayland-scanner` or protocol XML files are needed.

### DMA-BUF synchronization

HAL's `ImageProcessor` uses its own headless EGL context for GPU
operations (`convert`, `draw_masks`) and calls `glFinish()` before
returning, so the output DMA-BUF is fully written when the caller
regains control.  The display library calls `DMA_BUF_IOCTL_SYNC` with
`SYNC_START|READ` before binding the texture and `SYNC_END|READ` after
rendering, ensuring the GPU texture cache sees the latest data written
by HAL's separate EGL context.

### System memory fallback

When DMA-BUF import fails (e.g. the GPU driver lacks
`EGL_EXT_image_dma_buf_import`), the examples fall back to reading
canvas pixels to CPU and uploading via `glTexImage2D`.  This path is
functional but slower.

---

## Performance (measured on imx95-frdm)

With `yolov8m-seg_640x640.dvm` at 1920x1080 NV12 input:

| Stage | Time |
|-------|------|
| Pull (capture + drain) | 0.2 ms |
| Import (cache lookup) | 0.6 ms |
| Convert (NV12 -> RGB letterbox) | 6.0 ms |
| NPU inference | 34.7 ms |
| Draw masks (composite) | 4.5 ms |
| Display (EGL texture + swap) | 1.1 ms |
| **Total** | **47 ms (~21 FPS)** |

NPU inference wall-clock time (34.7 ms) includes PCIe DMA overhead
beyond raw compute (~26 ms).

## Known issues

### Frame rate

These are serial single-threaded pipelines.  At ~21 FPS the per-frame
timing is consistent (~47 ms), dominated by NPU inference (~35 ms).
Higher frame rates require overlapping capture with inference (e.g. a
capture thread), which is outside the scope of these minimal examples.

### Compositor latency

`eglSwapInterval(0)` is set, so the display library does not block on
vsync.  However, Weston triple-buffers internally which can add 1-2
compositor frames (~16-33 ms) of perceived lag.  This is inherent to
the composited Wayland path.

### NV12 DMA-BUF planes

The NeoISP on i.MX 95 produces NV12 frames as two separate DMA-BUF
memory objects (luma and chroma) with different inodes.  Both examples
handle this via separate luma/chroma plane descriptors.

### Debug logging

```bash
# ARA-2 debug logging
RUST_LOG=debug yolov8_live ...

# libcamera debug logging
export LIBCAMERA_LOG_LEVELS='NxpNeo:DEBUG,ISI:DEBUG'
```
