#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
"""
YOLOv8 detection on ARA-2 NPU — complete Python inference example.

Demonstrates the full zero-copy DMA-BUF pipeline using edgefirst-ara2
and edgefirst-hal:

  1. Read model metadata and class labels from the DVM file
  2. Connect to the ARA-2 proxy and load the model onto the NPU
  3. Allocate DMA-BUF tensors and set up GPU preprocessing
  4. Run GPU convert (RGBA → PlanarRgb CHW) + NPU inference
  5. Decode quantized outputs with the HAL decoder (dequant + NMS)
  6. Print detections with bounding boxes scaled to image coordinates

Steady-state performance on i.MX8MP + ARA-2 (yolov8n, 640x640):

  GPU preprocess (convert):     6.37 ms
  Inference (wall clock):       9.13 ms
    NPU execution:              3.33 ms
    DMA input upload:           2.20 ms
    DMA output download:        1.96 ms
  Postprocess (decode+NMS):     2.53 ms
  Total pipeline:              18.03 ms  (55.5 FPS)

Usage:
    python yolov8.py <model.dvm> <image.jpg>
    python yolov8.py <model.dvm> <image.jpg> --benchmark 50

Requirements:
    pip install edgefirst-ara2 edgefirst-hal numpy
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

import edgefirst_ara2 as ara2
import edgefirst_hal as hal


# ── Helpers ──────────────────────────────────────────────────────────────────


def normalize_shape(raw: tuple[int, int, int]) -> list[int]:
    """Normalize an ARA-2 output shape for the HAL decoder.

    ARA-2 always reports 3D shapes [C, H, W]. For logically 2D outputs
    (e.g., scores [80, 8400]), the shape is padded as [80, 8400, 1].
    We strip trailing 1s and prepend batch=1 for the decoder.

      [80, 8400, 1]  → [1, 80, 8400]   (scores)
      [4, 8400, 1]   → [1, 4, 8400]    (boxes)
      [32, 160, 160] → [1, 32, 160, 160] (protos)
    """
    shape = list(raw)
    while len(shape) > 1 and shape[-1] == 1:
        shape.pop()
    shape.insert(0, 1)
    return shape


def typed_output(raw: np.ndarray, shape: list[int], bpp: int, signed: bool) -> np.ndarray:
    """Reinterpret raw uint8 bytes as the correct integer type and shape."""
    if bpp == 2 and signed:
        return raw.view(np.int16).reshape(shape)
    elif bpp == 2:
        return raw.view(np.uint16).reshape(shape)
    elif bpp == 1 and signed:
        return raw.view(np.int8).reshape(shape)
    else:
        return raw.reshape(shape)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="YOLOv8 detection on ARA-2 NPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: python yolov8.py yolov8n_640x640.dvm zidane.jpg --benchmark 20",
    )
    parser.add_argument("model", help="Path to compiled .dvm model file")
    parser.add_argument("image", help="Path to input image (JPEG/PNG)")
    parser.add_argument("--threshold", type=float, default=0.25, help="Score threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--benchmark", type=int, default=0,
                        help="Run N iterations and print timing statistics")
    parser.add_argument("--socket", default=ara2.DEFAULT_SOCKET, help="Proxy socket path")
    args = parser.parse_args()

    # ── 1. Metadata ──────────────────────────────────────────────────────
    print(f"edgefirst-ara2 v{ara2.__version__}, edgefirst-hal v{hal.version()}")

    metadata = ara2.read_metadata(args.model)
    labels = ara2.read_labels(args.model)
    if metadata:
        print(f"Task: {metadata.task}, Classes: {len(labels)}")
        if metadata.compilation and metadata.compilation.ppa:
            ppa = metadata.compilation.ppa
            print(f"Target: {metadata.compilation.target}, "
                  f"IPS: {ppa.ips:.0f}, Power: {ppa.power_mw:.0f} mW")

    # ── 2. Connect and load model ────────────────────────────────────────
    session = ara2.Session.create_via_unix_socket(args.socket)
    endpoints = session.list_endpoints()
    if not endpoints:
        print("No ARA-2 endpoints found. Is the proxy running?")
        sys.exit(1)

    endpoint = endpoints[0]
    state = endpoint.check_status()
    stats = endpoint.dram_statistics()
    print(f"Endpoint: {state}, "
          f"DRAM: {stats.free_size / 1048576:.0f} / {stats.dram_size / 1048576:.0f} MB free")

    with endpoint.load_model(args.model) as model:
        model.allocate_tensors("dma")
        c, h, w = model.input_shape(0)
        input_dim = float(max(w, h))
        iq = model.input_quants(0)
        print(f"Input: {c}x{h}x{w} (CHW), signed={iq.is_signed}")

        # ── 3. Build HAL decoder from output tensor metadata ─────────
        decoder_outputs = []
        for i in range(model.n_outputs):
            shape = normalize_shape(model.output_shape(i))
            oq = model.output_quants(i)
            info = model.output_info(i)

            # Box outputs have shape [1, 4, N] — specifically dim[1]==4 for
            # Ultralytics split format. Plain `4 in shape` would misidentify
            # score tensors from models with exactly 4 classes.
            is_box = len(shape) == 3 and shape[1] == 4

            if is_box:
                # Box output: normalize qn by input dimension
                scale = oq.qn / input_dim if input_dim > 1 else oq.qn
                out = hal.Output.boxes(shape=shape, decoder=hal.DecoderType.Ultralytics)
                out = out.with_quantization(scale, oq.offset).with_normalized(True)
                kind = "boxes"
            else:
                out = hal.Output.scores(shape=shape, decoder=hal.DecoderType.Ultralytics)
                out = out.with_quantization(oq.qn, oq.offset)
                kind = "scores"

            print(f"  output[{i}]: {kind} {shape} bpp={info.bpp} "
                  f"qn={oq.qn:.4f} signed={oq.is_signed}")
            decoder_outputs.append(out)

        decoder = hal.Decoder.new_from_outputs(
            decoder_outputs,
            score_threshold=args.threshold,
            iou_threshold=args.iou,
            decoder_version=hal.DecoderVersion.Yolov8,
        )

        # ── 4. Set up persistent DMA-BUF tensors ────────────────────
        processor = hal.ImageProcessor()

        # Load source image into a DMA-BUF tensor (one-time decode)
        with open(args.image, "rb") as f:
            jpeg_bytes = f.read()
        src = hal.Tensor.load_from_bytes(
            jpeg_bytes, format=hal.PixelFormat.Rgba, mem=hal.TensorMemory.DMA,
        )
        img_w, img_h = src.width, src.height
        print(f"Image: {img_w}x{img_h}")

        # Import model input DMA-BUF as PlanarRgb (CHW layout)
        # Use int8 dtype for signed input quantization
        input_fd = model.input_tensor_fd(0)
        try:
            dtype = "int8" if iq.is_signed else "uint8"
            dst = processor.import_image(
                input_fd, w, h, hal.PixelFormat.PlanarRgb, dtype=dtype,
            )
        finally:
            os.close(input_fd)

        # ── 5. Warmup pass ───────────────────────────────────────────
        processor.convert(src, dst)
        model.run()

        # ── 6. Inference + decode ────────────────────────────────────
        t_pre = time.monotonic()
        processor.convert(src, dst)
        t_inf = time.monotonic()
        timing = model.run()
        t_post = time.monotonic()

        # Read and reshape outputs for the decoder
        raw_outputs = []
        for i in range(model.n_outputs):
            raw = model.get_output_tensor(i)
            shape = normalize_shape(model.output_shape(i))
            oq = model.output_quants(i)
            info = model.output_info(i)
            raw_outputs.append(typed_output(raw, shape, info.bpp, oq.is_signed))

        boxes, scores, class_ids, _masks = decoder.decode(raw_outputs)
        t_done = time.monotonic()

        n_det = len(scores)
        pre_ms = (t_inf - t_pre) * 1000
        inf_ms = (t_post - t_inf) * 1000
        post_ms = (t_done - t_post) * 1000

        # ── 7. Print detections ──────────────────────────────────────
        print(f"\n--- Detections ({n_det}) ---")
        for i in range(n_det):
            cls = int(class_ids[i])
            name = labels[cls] if cls < len(labels) else f"class_{cls}"
            x1, y1 = boxes[i, 0] * img_w, boxes[i, 1] * img_h
            x2, y2 = boxes[i, 2] * img_w, boxes[i, 3] * img_h
            print(f"  {name:>12} {scores[i]*100:5.1f}%  "
                  f"[{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

        # ── 8. Timing ────────────────────────────────────────────────
        print(f"\n--- Timing ---")
        print(f"  GPU preprocess (convert):  {pre_ms:6.2f} ms")
        print(f"  Inference (wall clock):    {inf_ms:6.2f} ms")
        print(f"    NPU execution:           {timing.run_time_us/1000:6.2f} ms")
        print(f"    DMA input upload:        {timing.input_time_us/1000:6.2f} ms")
        print(f"    DMA output download:     {timing.output_time_us/1000:6.2f} ms")
        print(f"  Postprocess (decode+NMS):  {post_ms:6.2f} ms")
        total = pre_ms + inf_ms + post_ms
        print(f"  Total pipeline:            {total:6.2f} ms  ({1000/total:.1f} FPS)")

        # ── 9. Benchmark (optional) ──────────────────────────────────
        if args.benchmark > 0:
            _benchmark(
                model, processor, src, dst, decoder, labels,
                img_w, img_h, args.benchmark,
            )


def _benchmark(
    model: ara2.Model,
    processor: hal.ImageProcessor,
    src: hal.Tensor,
    dst: hal.Tensor,
    decoder: hal.Decoder,
    labels: list[str],
    img_w: int,
    img_h: int,
    n_iter: int,
) -> None:
    """Run N iterations and print timing statistics."""
    pre_times = np.empty(n_iter)
    inf_times = np.empty(n_iter)
    npu_times = np.empty(n_iter)
    dma_in_times = np.empty(n_iter)
    dma_out_times = np.empty(n_iter)
    post_times = np.empty(n_iter)

    for j in range(n_iter):
        t0 = time.monotonic()
        processor.convert(src, dst)
        t1 = time.monotonic()
        timing = model.run()
        t2 = time.monotonic()

        raw_outputs = []
        for i in range(model.n_outputs):
            raw = model.get_output_tensor(i)
            shape = normalize_shape(model.output_shape(i))
            oq = model.output_quants(i)
            info = model.output_info(i)
            raw_outputs.append(typed_output(raw, shape, info.bpp, oq.is_signed))
        decoder.decode(raw_outputs)
        t3 = time.monotonic()

        pre_times[j] = (t1 - t0) * 1000
        inf_times[j] = (t2 - t1) * 1000
        npu_times[j] = timing.run_time_us / 1000
        dma_in_times[j] = timing.input_time_us / 1000
        dma_out_times[j] = timing.output_time_us / 1000
        post_times[j] = (t3 - t2) * 1000

    total = pre_times + inf_times + post_times

    print(f"\n--- Benchmark ({n_iter} iterations) ---")
    print(f"                              {'mean':>7}  {'min':>7}  {'max':>7}  {'std':>7}")
    _row("GPU preprocess (convert):", pre_times)
    _row("Inference (wall clock):", inf_times)
    _row("  NPU execution:", npu_times)
    _row("  DMA input upload:", dma_in_times)
    _row("  DMA output download:", dma_out_times)
    _row("Postprocess (decode+NMS):", post_times)
    print(f"  {'─' * 54}")
    _row("Total pipeline:", total)
    print(f"  Throughput: {1000 / total.mean():.1f} FPS")


def _row(label: str, data: np.ndarray) -> None:
    print(f"  {label:<30} {data.mean():6.2f}  {data.min():6.2f}  "
          f"{data.max():6.2f}  {data.std():6.2f}")


if __name__ == "__main__":
    main()
