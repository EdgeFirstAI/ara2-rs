# ARA-2 Examples

End-to-end inference examples for the ARA-2 NPU using the EdgeFirst HAL
for GPU preprocessing, decoding, and overlay rendering.  All examples use
zero-copy DMA-BUF buffers throughout the pipeline.

## Prerequisites

- ARA-2 proxy service running: `systemctl status ara2.service`
- ARA-2 PCIe device visible: `lspci | grep -i kinara`
- Wayland compositor running (Weston) with `zwp_linux_dmabuf_v1` support
- A compiled DVM model, e.g. `yolov8n-seg_640x640.dvm`

| File | Description |
|------|-------------|
| `yolov8.rs` | Static image inference (Rust) |
| `yolov8_live.rs` | Live camera inference (Rust + libcamera + wayland-client) |
| `yolov8_live.py` | Live camera inference (Python + libcamera + pywayland) |
| `yolov8.py` | Static image inference (Python) |

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
| `--color-mode` | `class` | Mask coloring: `class`, `instance`, or `track` |

---

## yolov8_live -- Live Camera Inference (Rust)

Captures NV12 or YUYV frames from a camera via libcamera, runs YOLOv8
inference on the ARA-2 NPU, and displays results in a Wayland window.
This is a minimal serial (single-threaded) pipeline.

Display uses the `zwp_linux_dmabuf_v1` Wayland protocol to submit the
RGBA canvas DMA-BUF directly to the compositor -- no EGL or OpenGL.

### Architecture

```
libcamera (NV12 or YUYV DMA-BUF)
  -> HAL import (cached by buffer index)
  -> HAL convert (NV12|YUYV -> PlanarRGB letterbox)
  -> ARA-2 NPU inference
  -> HAL draw_masks (decode + composite -> RGBA canvas)
  -> Wayland display (DMA-BUF -> wl_buffer -> compositor)
```

### Build

Cross-compile using the Yocto SDK (zigbuild cannot handle `libcamera-sys`
C++ dependencies):

```bash
SDK=/opt/yocto-sdk-imx95-frdm
SYSROOT=$SDK/sysroots/armv8a-poky-linux
export PATH=$SDK/sysroots/x86_64-pokysdk-linux/usr/bin/aarch64-poky-linux:$PATH
export CC_aarch64_unknown_linux_gnu="aarch64-poky-linux-gcc --sysroot=$SYSROOT"
export CXX_aarch64_unknown_linux_gnu="aarch64-poky-linux-g++ --sysroot=$SYSROOT"
export AR_aarch64_unknown_linux_gnu="aarch64-poky-linux-ar"
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER="aarch64-poky-linux-gcc"
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS="-C link-arg=--sysroot=$SYSROOT"
export PKG_CONFIG_SYSROOT_DIR=$SYSROOT
export PKG_CONFIG_PATH=$SYSROOT/usr/lib/pkgconfig
export PKG_CONFIG_ALLOW_CROSS=1
GCC_INC=$SYSROOT/usr/lib/gcc/aarch64-poky-linux/14.3.0/include
CXX_INC=$SYSROOT/usr/include/c++/14.3.0
export BINDGEN_EXTRA_CLANG_ARGS="--target=aarch64-poky-linux --sysroot=$SYSROOT \
    -I$GCC_INC -I$CXX_INC -I$CXX_INC/aarch64-poky-linux -I$SYSROOT/usr/include"

cargo build --release --target aarch64-unknown-linux-gnu --example yolov8_live
```

### Run (imx95-frdm with on-board `os08a20` MIPI-CSI sensor)

```bash
export XDG_RUNTIME_DIR=/run/user/0
export WAYLAND_DISPLAY=wayland-0
export LIBCAMERA_PIPELINES_MATCH_LIST='nxp/neo'

# Default NV12 capture:
yolov8_live /root/models/yolov8m-seg_640x640.dvm

# YUYV capture:
yolov8_live /root/models/yolov8m-seg_640x640.dvm --format yuyv
```

**`LIBCAMERA_PIPELINES_MATCH_LIST='nxp/neo'` is required on i.MX95.** The
i.MX95 sensor appears in two media graphs: `mxc-isi` (raw bypass, handled
by the `imx8-isi` pipeline) and `neoisp` (the NXP Neo ISP, handled by
`nxp/neo`).  Without this hint libcamera picks `imx8-isi` first, which
only exposes raw Bayer (`SBGGR10`/`SBGGR12`) and fails with
`Cannot find a supported YUV/RGB format` for any YUV/RGB request.
Only the `nxp/neo` pipeline exposes NV12, YUYV, NV16, UYVY, RGB, etc.

| Flag | Default | Description |
|------|---------|-------------|
| `model` | (required) | Path to `.dvm` model file |
| `--format` | `nv12` | Capture pixel format: `nv12` or `yuyv` |
| `--camera-name` | first available | libcamera camera ID |
| `--width` | 1920 | Camera capture width |
| `--height` | 1080 | Camera capture height |
| `--threshold` | 0.50 | Detection confidence threshold |
| `--iou` | 0.45 | NMS IoU threshold |
| `--color-mode` | `class` | Mask coloring: `class`, `instance`, or `track` |
| `--socket` | `/var/run/ara2.sock` | ARA-2 proxy UNIX socket path |

On i.MX95 with the Neo ISP, only `1920x1080` and `3840x2160` are supported
capture sizes; other values will be silently adjusted.

---

## yolov8_live.py -- Live Camera Inference (Python)

Python version of the live pipeline using the native libcamera Python
bindings for camera capture and pywayland for display.  No GStreamer,
EGL, OpenGL, or compiled C libraries needed.

### Python environment

```bash
python3 -m venv --system-site-packages /root/venv
source /root/venv/bin/activate
pip install edgefirst-ara2 edgefirst-hal numpy pywayland
```

The `--system-site-packages` flag is required to pick up the libcamera
Python bindings from system packages.

### Run (imx95-frdm with on-board `os08a20` MIPI-CSI sensor)

```bash
export XDG_RUNTIME_DIR=/run/user/0
export WAYLAND_DISPLAY=wayland-0
export LIBCAMERA_PIPELINES_MATCH_LIST='nxp/neo'

source /root/venv/bin/activate

# Default NV12 capture:
python3 yolov8_live.py /root/models/yolov8m-seg_640x640.dvm

# YUYV capture:
python3 yolov8_live.py /root/models/yolov8m-seg_640x640.dvm --format yuyv
```

See the Rust `yolov8_live` section above for why
`LIBCAMERA_PIPELINES_MATCH_LIST='nxp/neo'` is required on i.MX95.

| Flag | Default | Description |
|------|---------|-------------|
| `model` | (required) | Path to `.dvm` model file |
| `--format` | `nv12` | Capture pixel format: `nv12` or `yuyv` |
| `--camera-name` | first available | libcamera camera ID |
| `--width` | 1920 | Camera capture width |
| `--height` | 1080 | Camera capture height |
| `--threshold` | 0.50 | Detection confidence threshold |
| `--iou` | 0.45 | NMS IoU threshold |
| `--color-mode` | `class` | Mask coloring: `class`, `instance`, or `track` |
| `--socket` | `/var/run/ara2.sock` | ARA-2 proxy UNIX socket path |

---

## Performance (measured on imx95-frdm)

### yolov8n-seg (nano) at 1920x1080

| Stage | Time |
|-------|------|
| Pull (capture + drain) | 0.2 ms |
| Import (cache lookup) | 0.1 ms |
| Convert (NV12 -> RGB letterbox) | 4.5 ms |
| NPU inference | 12.6 ms |
| Draw masks (no detections) | 4.2 ms |
| Display (wl_buffer attach+commit) | 0.1 ms |
| **Total** | **~22 ms (~45 FPS)** |

### yolov8m-seg (medium) at 1920x1080

| Stage | Time |
|-------|------|
| Pull (capture + drain) | 0.2 ms |
| Import (cache lookup) | 0.1 ms |
| Convert (NV12 -> RGB letterbox) | 5.0 ms |
| NPU inference | 34.7 ms |
| Draw masks (no detections) | 4.0 ms |
| Display (wl_buffer attach+commit) | 0.1 ms |
| **Total** | **~44 ms (~23 FPS)** |

NPU inference wall-clock time includes PCIe DMA overhead beyond raw
compute.  Draw masks time increases with the number of detections
(mask compositing is proportional to detection count).

## Known issues

### Frame rate

These are serial single-threaded pipelines.  Frame rate is dominated
by NPU inference time.  Higher frame rates require overlapping capture
with inference (e.g. a capture thread), which is outside the scope of
these minimal examples.

### `Cannot find a supported YUV/RGB format` on i.MX95

If you see this libcamera error followed by `Invalid camera configuration`:

```
WARN  ISI imx8-isi.cpp:301  Cannot find a supported YUV/RGB format
ERROR ISI imx8-isi.cpp:456  Cannot adjust pixelformat YUYV
Invalid camera configuration
```

libcamera picked the `imx8-isi` pipeline handler instead of `nxp/neo`.
On i.MX95 the `imx8-isi` path only exposes raw Bayer for the `os08a20`
sensor and has no YUV/RGB output.  Fix by exporting
`LIBCAMERA_PIPELINES_MATCH_LIST='nxp/neo'` before running the example.

### ISP adjustment stutter

Visible stutter occurs when the camera ISP adjusts auto-exposure or
auto-white-balance (e.g. when lighting changes or an object moves close
to the lens).  A raw `libcamerasrc ! waylandsink` GStreamer pipeline
does not exhibit this stutter under the same conditions.

### Debug logging

```bash
# ARA-2 debug logging
RUST_LOG=debug yolov8_live ...

# libcamera debug logging
export LIBCAMERA_LOG_LEVELS='NxpNeo:DEBUG,ISI:DEBUG'
```
