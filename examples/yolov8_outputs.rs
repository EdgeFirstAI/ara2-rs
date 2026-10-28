// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! YOLOv8 output identification shared by the `yolov8` and `yolov8_live`
//! examples: which ARA-2 output is boxes, scores, mask coefficients or
//! protos, and which `edgefirst.json` entry describes it.

use ara2::dvm_metadata;
use edgefirst_decoder::configs::DimName;
use std::collections::HashMap;

/// Whether this model performs detection only or detection + segmentation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Task {
    Detect,
    Segment,
}

/// Normalize an ARA-2 output shape for the HAL decoder.
///
/// ARA-2's FFI struct always reports 3 dimensions `[nch, height, width]`.
/// For logically 2D outputs (e.g. scores `[80, 8400]`), the struct pads
/// width=1, producing `[80, 8400, 1]`. We strip that trailing 1 to recover
/// the true rank.
///
/// The HAL decoder expects an ONNX-style leading batch dimension (it slices
/// `[0, .., ..]` internally to remove it), so we prepend batch=1.
///
/// Examples:
///   `[80, 8400, 1]` → `[1, 80, 8400]`  (scores)
///   `[4, 8400, 1]`  → `[1, 4, 8400]`   (boxes)
///   `[32, 160, 160]` → `[1, 32, 160, 160]` (protos — no trailing 1 to strip)
pub fn normalize_shape(raw: [usize; 3]) -> Vec<usize> {
    let mut shape: Vec<usize> = raw.to_vec();
    while shape.len() > 1 && shape.last() == Some(&1) {
        shape.pop();
    }
    shape.insert(0, 1);
    shape
}

/// Strip trailing 1s from a metadata shape (Vec<i64>) and widen to usize.
///
/// Metadata declares shapes with the leading batch and any trailing
/// `padding` axes; the NPU runtime strips trailing 1s before returning
/// the per-output shape. Normalising both sides the same way lets us
/// match a runtime shape to its [`dvm_metadata::OutputSpec`].
pub fn strip_trailing_ones_i64(raw: &[i64]) -> Vec<usize> {
    let mut shape: Vec<usize> = raw.iter().map(|&d| d as usize).collect();
    while shape.len() > 1 && shape.last() == Some(&1) {
        shape.pop();
    }
    shape
}

/// Whether an output's layer name marks it as class scores: Ultralytics
/// exports end the class branch in a `Sigmoid`.
pub fn is_scores_name(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    name.contains("sigmoid") || name.contains("score")
}

/// The one output of `candidates` that is not the class scores.
///
/// Two outputs share a shape when the class count equals a box or mask
/// coefficient count (4 classes, or 32 classes in a segmentation model), and
/// then only the layer names tell them apart.
pub fn pick_non_scores(
    candidates: &[usize],
    names: &[String],
    what: &str,
) -> Result<usize, String> {
    match *candidates {
        [] => Err(format!("cannot identify {what} output")),
        [only] => Ok(only),
        [a, b] => match (is_scores_name(&names[a]), is_scores_name(&names[b])) {
            (false, true) => Ok(a),
            (true, false) => Ok(b),
            _ => Err(format!(
                "cannot identify {what} output: {:?} and {:?} have the same shape \
                 and neither layer name marks the scores",
                names[a], names[b]
            )),
        },
        _ => Err(format!(
            "cannot identify {what} output: {} outputs have its shape",
            candidates.len()
        )),
    }
}

/// The boxes output: the `[1, 4, N]` output not named as scores.
pub fn identify_boxes(shapes: &[Vec<usize>], names: &[String]) -> Result<usize, String> {
    let candidates: Vec<usize> = (0..shapes.len())
        .filter(|&i| shapes[i].len() == 3 && shapes[i][1] == 4)
        .collect();
    pick_non_scores(&candidates, names, "boxes")
}

/// Identify detection output indices: boxes `[1, 4, N]` and scores `[1, C, N]`.
pub fn identify_det_outputs(
    shapes: &[Vec<usize>],
    names: &[String],
) -> Result<(usize, usize), String> {
    let boxes = identify_boxes(shapes, names)?;
    let scores = (0..shapes.len())
        .find(|&i| i != boxes && shapes[i].len() == 3)
        .ok_or("cannot identify scores output")?;
    Ok((boxes, scores))
}

/// Identify segmentation output indices: boxes, scores, mask_coeff, protos.
///
/// Protos are the rank-4 output `[1, 32, H, W]` and the mask coefficients
/// the rank-3 output with as many channels. ARA-2 does not keep the export's
/// output order.
pub fn identify_seg_outputs(
    shapes: &[Vec<usize>],
    names: &[String],
) -> Result<(usize, usize, usize, usize), String> {
    let protos = shapes
        .iter()
        .position(|s| s.len() == 4)
        .ok_or("cannot identify protos output")?;
    let boxes = identify_boxes(shapes, names)?;
    let candidates: Vec<usize> = (0..shapes.len())
        .filter(|&i| i != boxes && shapes[i].len() == 3 && shapes[i][1] == shapes[protos][1])
        .collect();
    let masks = pick_non_scores(&candidates, names, "mask_coeff")?;
    let scores = (0..shapes.len())
        .find(|&i| i != boxes && i != masks && shapes[i].len() == 3)
        .ok_or("cannot identify scores output")?;
    Ok((boxes, scores, masks, protos))
}

/// The runtime index and role of each output the decoder reads, in the
/// order it takes them: boxes, scores, then mask coefficients and protos.
///
/// The decoder gives each role the first tensor whose shape matches its
/// configuration, so outputs sharing a shape (a 4-class model's scores and
/// boxes) must reach it in role order, whatever order the runtime uses.
pub fn output_layout(
    task: Task,
    shapes: &[Vec<usize>],
    names: &[String],
) -> Result<Vec<(usize, OutputRole)>, String> {
    Ok(match (task, shapes.len()) {
        (Task::Detect, 1) => vec![(0, OutputRole::Detection)],
        (Task::Detect, _) => {
            let (bi, si) = identify_det_outputs(shapes, names)?;
            vec![(bi, OutputRole::Boxes), (si, OutputRole::Scores)]
        }
        (Task::Segment, _) => {
            let (bi, si, mi, pi) = identify_seg_outputs(shapes, names)?;
            vec![
                (bi, OutputRole::Boxes),
                (si, OutputRole::Scores),
                (mi, OutputRole::MaskCoefficients),
                (pi, OutputRole::Protos),
            ]
        }
    })
}

/// Build a canonical Ultralytics `dshape` for a role, used when the DVM
/// did not ship an `edgefirst.json` (e.g. Kinara-1.2.1 official models) or
/// when the metadata entry for that output was missing.
///
/// The HAL decoder treats an empty `dshape` as "shape is already in
/// canonical order"; this is true for boxes / scores / mask coefficients,
/// but for protos the canonical order is NHWC while NPU outputs are NCHW,
/// so the materializer would otherwise mis-identify `num_protos`.
pub fn canonical_dshape(role: OutputRole, shape: &[usize]) -> Vec<(DimName, usize)> {
    let names: &[DimName] = match role {
        OutputRole::Detection => &[DimName::Batch, DimName::NumFeatures, DimName::NumBoxes],
        OutputRole::Boxes => &[DimName::Batch, DimName::BoxCoords, DimName::NumBoxes],
        OutputRole::Scores => &[DimName::Batch, DimName::NumClasses, DimName::NumBoxes],
        OutputRole::MaskCoefficients => &[DimName::Batch, DimName::NumProtos, DimName::NumBoxes],
        OutputRole::Protos => &[
            DimName::Batch,
            DimName::NumProtos,
            DimName::Height,
            DimName::Width,
        ],
    };
    // Pair element-wise; if the runtime rank is shorter than the canonical
    // names list (e.g. a `padding=1` axis was stripped) the zip stops at the
    // shorter side, which keeps `dshape.len() == shape.len()` as the HAL
    // decoder's validator requires.
    shape
        .iter()
        .copied()
        .zip(names.iter().copied())
        .map(|(n, name)| (name, n))
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OutputRole {
    Detection,
    Boxes,
    Scores,
    MaskCoefficients,
    Protos,
}

impl OutputRole {
    /// The `type` an `edgefirst.json` output entry gives this role.
    pub fn metadata_type(self) -> &'static str {
        match self {
            OutputRole::Detection => "detection",
            OutputRole::Boxes => "boxes",
            OutputRole::Scores => "scores",
            OutputRole::MaskCoefficients => "mask_coefs",
            OutputRole::Protos => "protos",
        }
    }
}

/// Return `dshapes[i]` if it was populated from metadata, otherwise the
/// canonical Ultralytics dshape for the role at `shape`.
pub fn dshape_or_canonical(
    dshapes: &[Vec<(DimName, usize)>],
    i: usize,
    role: OutputRole,
    shape: &[usize],
) -> Vec<(DimName, usize)> {
    if dshapes[i].is_empty() {
        canonical_dshape(role, shape)
    } else {
        dshapes[i].clone()
    }
}

/// A model's `edgefirst.json` output entries, which add each output's
/// dshape and box `normalized` flag to what the runtime reports. Older .dvm
/// files without these fields still work, from runtime shape and
/// quantization alone.
pub struct OutputMetadata<'a> {
    by_name: HashMap<&'a str, &'a dvm_metadata::OutputSpec>,
    /// `None` for a shape declared by more than one output (e.g. 32 classes
    /// and 32 protos), which is ambiguous.
    by_shape: HashMap<Vec<usize>, Option<&'a dvm_metadata::OutputSpec>>,
}

impl<'a> OutputMetadata<'a> {
    pub fn new(outputs: impl IntoIterator<Item = &'a dvm_metadata::OutputSpec>) -> Self {
        let mut by_name = HashMap::new();
        let mut by_shape = HashMap::new();
        for o in outputs {
            by_shape
                .entry(strip_trailing_ones_i64(&o.shape))
                .and_modify(|e| *e = None)
                .or_insert(Some(o));
            if let Some(name) = o.name.as_deref() {
                by_name.insert(name, o);
            }
        }
        Self { by_name, by_shape }
    }

    /// The entry for the output with layer `name` and normalized `shape`,
    /// matched by name, else by shape. An entry whose `type` disagrees with
    /// the output's `role` is not used, leaving the output to the canonical
    /// layout for its role.
    pub fn spec(
        &self,
        name: &str,
        shape: &[usize],
        role: OutputRole,
    ) -> Option<&'a dvm_metadata::OutputSpec> {
        self.by_name
            .get(name)
            .copied()
            .or_else(|| self.by_shape.get(shape).copied().flatten())
            .filter(|s| s.output_type.as_deref() == Some(role.metadata_type()))
    }
}
