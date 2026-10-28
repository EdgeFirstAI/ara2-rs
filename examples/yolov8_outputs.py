# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
"""YOLOv8 output identification shared by ``yolov8.py`` and ``yolov8_live.py``.

Which ARA-2 output is boxes, scores, mask coefficients or protos, and which
``edgefirst.json`` entry describes it. Pure Python, so it has no HAL imports.
"""

from __future__ import annotations


def normalize_shape(raw: tuple[int, int, int]) -> list[int]:
    """Normalize an ARA-2 output shape for the EdgeFirst decoder.

    ARA-2 always reports 3D shapes [C, H, W]. For logically 2D outputs
    (e.g., scores [80, 8400]), the shape is padded as [80, 8400, 1].
    We strip trailing 1s and prepend batch=1 for the decoder.

      [80, 8400, 1]  → [1, 80, 8400]     (scores)
      [4, 8400, 1]   → [1, 4, 8400]      (boxes)
      [32, 160, 160] → [1, 32, 160, 160] (protos)
    """
    shape = list(raw)
    while len(shape) > 1 and shape[-1] == 1:
        shape.pop()
    shape.insert(0, 1)
    return shape


def strip_trailing_ones(raw) -> list[int]:
    """Strip trailing 1s from a metadata shape.

    Metadata declares shapes with the leading batch and any trailing
    ``padding`` axes; the NPU runtime strips trailing 1s before returning the
    per-output shape. Normalising both sides the same way lets a runtime shape
    be matched to its ``OutputSpec``.
    """
    shape = [int(d) for d in raw]
    while len(shape) > 1 and shape[-1] == 1:
        shape.pop()
    return shape


# The ``type`` an ``edgefirst.json`` output entry gives each role.
METADATA_TYPES = {
    "boxes": "boxes",
    "scores": "scores",
    "mask_coefficients": "mask_coefs",
    "protos": "protos",
}


def index_metadata(metadata) -> tuple[dict[str, object], dict[tuple[int, ...], object]]:
    """Index a model's metadata outputs by name and by normalised shape.

    The runtime reports shape and quantization; ``edgefirst.json`` inside the
    ``.dvm`` adds the semantics the runtime cannot know — the named ``dshape``
    and, for box outputs, whether coordinates are already normalised. Older
    ``.dvm`` files carry neither, and the caller falls back accordingly.

    A shape declared by more than one output (32 classes and 32 protos) is
    ambiguous and maps to ``None``.
    """
    by_name, by_shape = {}, {}
    if metadata is None:
        return by_name, by_shape
    for o in metadata.outputs:
        key = tuple(strip_trailing_ones(o.shape))
        by_shape[key] = None if key in by_shape else o
        if o.name:
            by_name[o.name] = o
    return by_name, by_shape


def metadata_spec(index, name: str, shape: list[int], role: str):
    """The ``OutputSpec`` for the output with layer *name* and *shape*.

    Matched by name, else by shape. An entry whose ``type`` disagrees with
    *role* is not used, leaving the output to the canonical layout for its role.
    """
    by_name, by_shape = index
    spec = by_name.get(name)
    if spec is None:
        spec = by_shape.get(tuple(shape))
    if spec is None or spec.output_type != METADATA_TYPES[role]:
        return None
    return spec


def is_scores_name(name: str) -> bool:
    """Whether a layer name marks the class scores.

    Ultralytics exports end the class branch in a ``Sigmoid``.
    """
    name = name.lower()
    return "sigmoid" in name or "score" in name


def pick_non_scores(candidates: list[int], names: list[str], what: str) -> int:
    """The one output of *candidates* that is not the class scores.

    Two outputs share a shape when the class count equals a box or mask
    coefficient count (4 classes, or 32 classes in a segmentation model), and
    then only the layer names tell them apart.
    """
    if not candidates:
        raise ValueError(f"cannot identify {what} output")
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 2:
        raise ValueError(
            f"cannot identify {what} output: {len(candidates)} outputs have its shape"
        )
    a, b = candidates
    named = (is_scores_name(names[a]), is_scores_name(names[b]))
    if named == (False, True):
        return a
    if named == (True, False):
        return b
    raise ValueError(
        f"cannot identify {what} output: {names[a]!r} and {names[b]!r} have the "
        "same shape and neither layer name marks the scores"
    )


def output_layout(shapes: list[list[int]], names: list[str]) -> list[tuple[int, str]]:
    """The runtime index and role of each output, in the order the decoder
    takes them: boxes, scores, then mask coefficients and protos.

    Roles are identified by shape:
      - Protos:             rank-4  [1, 32, H, W]
      - Boxes:              rank-3  [1,  4, N]  (shape[1] == 4)
      - Mask coefficients:  rank-3  [1, 32, N]  (shape[1] == n_proto_channels)
      - Scores:             rank-3  [1,  C, N]  (the remaining one)

    with layer names breaking a tie between outputs of the same shape. The
    decoder gives each role the first tensor whose shape matches it, so
    outputs sharing a shape must reach it in role order, whatever order the
    runtime uses.
    """
    rank3 = [i for i, s in enumerate(shapes) if len(s) == 3]
    boxes = pick_non_scores([i for i in rank3 if shapes[i][1] == 4], names, "boxes")
    protos = next((i for i, s in enumerate(shapes) if len(s) == 4), None)
    masks = None
    if protos is not None:
        n_proto_ch = shapes[protos][1]
        masks = pick_non_scores(
            [i for i in rank3 if i != boxes and shapes[i][1] == n_proto_ch],
            names,
            "mask coefficients",
        )
    scores = next((i for i in rank3 if i not in (boxes, masks)), None)
    if scores is None:
        raise ValueError("cannot identify scores output")
    layout = [(boxes, "boxes"), (scores, "scores")]
    if protos is not None:
        layout += [(masks, "mask_coefficients"), (protos, "protos")]
    return layout
