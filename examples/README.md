# ARA-2 Examples

End-to-end inference examples for the ARA-2 NPU using the EdgeFirst HAL
for GPU preprocessing, decoding, and overlay rendering.  All examples use
zero-copy DMA-BUF buffers throughout the pipeline.

The HAL ships as one library per concern rather than a single
`edgefirst-hal` package: `edgefirst-tensor` (memory), `edgefirst-codec`
(JPEG/PNG decode), `edgefirst-image` (convert and render), and
`edgefirst-decoder` (YOLO post-processing).  In Rust they are four crates;
in Python they are four wheels sharing the `edgefirst.` namespace.

## Prerequisites

- ARA-2 proxy service running: `systemctl status rt-sdk-ara2`
- ARA-2 PCIe device visible: `lspci | grep -i kinara`
- Wayland compositor running (Weston) with `zwp_linux_dmabuf_v1` support
- A compiled DVM model — see [Models](#models) below

| File | Description |
|------|-------------|
| `yolov8.rs` | Static image inference (Rust) |
| `yolov8_live/` | Live camera inference (Rust + libcamera + wayland-client), a threaded pipeline |
| `yolov8_live.py` | Live camera inference (Python + libcamera + pywayland) |
| `yolov8.py` | Static image inference (Python) |
| `yolov8_outputs.rs`, `yolov8_outputs.py` | Output identification shared by the YOLOv8 examples; copy it alongside them |
| `telemetry.rs` | Telemetry report and sampling-cost measurement (Rust) |
| `telemetry.py` | Telemetry report (Python) |

## Models

The official pre-trained models are published in the EdgeFirst model zoo on
Hugging Face:

| Task | Repository |
| --- | --- |
| Detection | <https://huggingface.co/EdgeFirst/yolov8-det> |
| Segmentation | <https://huggingface.co/EdgeFirst/yolov8-seg> |

Each repository ships one directory per target — `tflite/`, `imx95/`,
`onnx/`, `hailo/`, `jetson/`, `qnn/`.  The ARA-2 builds are the int16 `.dvm`
exports under `ara240/`, in `n`/`s`/`m` sizes:

```bash
# On the target
curl -LO https://huggingface.co/EdgeFirst/yolov8-det/resolve/main/ara240/yolov8n-det-int16.dvm
curl -LO https://huggingface.co/EdgeFirst/yolov8-seg/resolve/main/ara240/yolov8n-seg-int16.dvm
```

Each export embeds an `edgefirst.json` schema and the class `labels.txt` in a
ZIP trailer appended to the `.dvm`.  That is where the per-output
`normalized` flag and named `dshape` come from — the NPU runtime reports
neither, and a model whose boxes are in pixel space is misread without them.
An export with no trailer still runs, falling back to the built-in COCO
labels and a canonical Ultralytics axis naming.

Both examples pin `DecoderVersion::Yolov8`.  The zoo's `yolo11` and `yolo26`
repositories ship `ara240/` builds too, but need the matching decoder
version.

---

## yolov8 -- Static Image Inference (Rust)

Runs YOLOv8 detection or instance segmentation on a single JPEG image.
Supports benchmarking with `--benchmark N` for per-stage timing statistics.

### Build

The `decoder` feature is off by default -- it adds
`OutputSpec::dshape_typed()`, which resolves the metadata's axis names onto
the HAL decoder's `DimName` and which this example hands straight to a
`DecoderBuilder` -- so it has to be named on the build:

```bash
# Cross-compile (from development host):
cargo zigbuild --release --target aarch64-unknown-linux-gnu \
    --features decoder --example yolov8

# Or on target:
cargo build --release --features decoder --example yolov8
```

### Run

```bash
yolov8 <model.dvm> <image.jpg> [--save] [--threshold 0.25] [--iou 0.45] [--benchmark N]

# e.g., with the model-zoo segmentation export
yolov8 yolov8n-seg-int16.dvm zidane.jpg --save --benchmark 30
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

## telemetry -- Telemetry Report (Rust and Python)

Prints everything the runtime reports: the DVAPI version and component versions, `Endpoint::info()`, `Endpoint::statistics()`, `is_busy()`, the model's version, name and type, and the full `InferenceStats` of one inference.
Fields the loaded DVAPI version does not provide print as `None` or `n/a`, and calls it lacks (`is_busy()`) print a message saying so.

The Rust example also measures the cost of sampling: it times `iterations` inferences alone, then again while a second thread calls `Endpoint::statistics()` every `sample_ms` milliseconds, and prints both frame rates and the call latency.
It needs no `decoder` feature and no image.

```bash
# Cross-compile (from development host):
cargo zigbuild --release --target aarch64-unknown-linux-gnu --example telemetry

# On target:
telemetry <model.dvm> [iterations] [sample_ms]    # defaults: 200 iterations, 100 ms
python3 telemetry.py <model.dvm>
```

On DVAPI 3.0 the `input_start`, `inference_start` and `output_start` timestamps are on `CLOCK_MONOTONIC`, so they line up with a Perfetto trace taken on the same host.
On DVAPI 1.x they are `None`.
See [Availability by DVAPI version](../README.md#availability-by-dvapi-version) for which other fields are absent on which version.

---

## yolov8_live -- Live Camera Inference (Rust)

Captures NV12 or YUYV frames from a camera via libcamera, runs YOLOv8
inference on the ARA-2 NPU, and displays results in a Wayland window.

Display uses the `zwp_linux_dmabuf_v1` Wayland protocol to submit the
RGBA canvas DMA-BUF directly to the compositor -- no EGL or OpenGL.

### Architecture

```
libcamera (NV12 or YUYV DMA-BUF)
  -> edgefirst-image import (cached by buffer index)
  -> edgefirst-image convert (NV12|YUYV -> PlanarRGB letterbox)
  -> ARA-2 NPU inference (into a pooled output set)
  -> edgefirst-decoder decode (in place on the output DMA-BUFs)
  -> edgefirst-image draw (composite -> RGBA canvas)
  -> Wayland display (DMA-BUF -> wl_buffer -> compositor)
```

The letterbox, inference and decode stages run on their own threads, so consecutive frames overlap: the GPU letterboxes the next frame into one input buffer (a *slot*) while the ARA-2 reads the other, and the decoder works on an earlier frame's outputs. The main thread captures, draws and presents. When the pipeline is busy the newest camera frame replaces the waiting one, and the status line counts the frames dropped. Each frame is drawn into one of two canvases, the one the compositor is not showing.

Over a TCP proxy, which cannot pass file descriptors, the model's input and outputs are heap buffers, copied to and from the HAL images around each inference.

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

cargo build --release --target aarch64-unknown-linux-gnu --features camera --example yolov8_live
```

### Run (FRDM i.MX 95 with on-board `os08a20` MIPI-CSI sensor)

```bash
export XDG_RUNTIME_DIR=/run/user/0
export WAYLAND_DISPLAY=wayland-0
export LIBCAMERA_PIPELINES_MATCH_LIST='nxp/neo'

# Default NV12 capture:
yolov8_live /root/models/yolov8m-seg-int16.dvm

# YUYV capture:
yolov8_live /root/models/yolov8m-seg-int16.dvm --format yuyv
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
| `--socket` | none: the proxy is found as `Session::connect()` does | ARA-2 proxy UNIX socket path |
| `--nms` | `auto` | NMS policy: `auto` follows the model's `edgefirst.json`; `agnostic` also suppresses overlapping boxes of different classes, removing the double-labelled objects a model unsure between two classes produces; `aware` suppresses only boxes of the same class |
| `--fullscreen` | off | Open a fullscreen window at the output's resolution; the camera frame is scaled to fill it, so capture at the display's aspect ratio to keep the picture undistorted |
| `--class-colors` | HAL palette | Box colour of each class, in label order, as comma-separated hex `RRGGBB` values (up to 20) |
| `--line-width` | 3 | Box outline width in display pixels |
| `--slots` | 2 | Input buffers the letterbox and inference stages alternate between; 1 runs them back to back |
| `--exposure US --gain N` | camera AGC | Turn the camera's automatic exposure off and hold this exposure time in microseconds and analogue gain; give both together |
| `--auto-exposure` | | Leave exposure to the camera; cancels an earlier `--exposure` and `--gain` |

On i.MX95 with the Neo ISP, only `1920x1080` and `3840x2160` are supported capture sizes; the example stops with an error naming the size the camera offered instead.

### Exposure

The camera's automatic exposure keeps adjusting the gain from frame to frame, which shows as a slight flicker. `--exposure` and `--gain` hold the sensor at fixed values instead. Mains lighting flickers at twice the mains frequency, so choose an exposure of whole flicker periods: 8333 or 16666 µs under 60 Hz lighting, 10000 µs under 50 Hz. Brightness follows exposure × gain, so set the gain for the room.

At start-up the example prints the range of each exposure control the camera offers, and a `WARNING:` for any control it lacks or a gain outside its range. Each status line ends with the exposure and gain the frames report, and the second status line is checked against the requested values: a `WARNING:` says when the camera ran at other values, which means it ignored or clamped them. Some pipelines report the requested values rather than the sensor's, so a match is not proof.

Some sensors also run an automatic exposure of their own, beneath libcamera, which libcamera leaves on. The Phytec AR0234 does: it fights `--exposure` and `--gain`, so the picture flickers while the frames still report steady values. Turn it off before starting the example:

```bash
v4l2-ctl -d /dev/cam-csi1-port0 -c auto_exposure=1,autogain_analogue=0
```

The setting stays in place across runs of the example; the driver restores its defaults at boot. Reading it back with `v4l2-ctl -d /dev/cam-csi1-port0 -C auto_exposure,autogain_analogue` should print `auto_exposure: 1 (Manual Mode)` and `autogain_analogue: 0`.

---

## yolov8_live.py -- Live Camera Inference (Python)

Python version of the live pipeline using the native libcamera Python
bindings for camera capture and pywayland for display.  No GStreamer,
EGL, OpenGL, or compiled C libraries needed.

### Python environment

```bash
python3 -m venv --system-site-packages /root/venv
source /root/venv/bin/activate
pip install edgefirst-ara2 'edgefirst-decoder>=0.31' 'edgefirst-image>=0.31' \
    numpy pywayland
```

The `--system-site-packages` flag is required to pick up the libcamera
Python bindings from system packages.

### Run (FRDM i.MX 95 with on-board `os08a20` MIPI-CSI sensor)

```bash
export XDG_RUNTIME_DIR=/run/user/0
export WAYLAND_DISPLAY=wayland-0
export LIBCAMERA_PIPELINES_MATCH_LIST='nxp/neo'

source /root/venv/bin/activate

# Default NV12 capture:
python3 yolov8_live.py /root/models/yolov8m-seg-int16.dvm

# YUYV capture:
python3 yolov8_live.py /root/models/yolov8m-seg-int16.dvm --format yuyv
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
| `--socket` | none: the proxy is found as `Session::connect()` does | ARA-2 proxy UNIX socket path |

---

## Performance (measured on FRDM i.MX 95)

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
