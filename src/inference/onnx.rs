// CensorChip – ONNX Runtime inference backend
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Loads any ONNX model placed in the `models/` directory and runs inference
// using the `ort` crate (ONNX Runtime Rust bindings v2).
//
// SUPPORTED MODEL ARCHITECTURES:
//   - YOLOv8/v9/v10/v11 (detection head)
//   - YOLOv8-seg / YOLO11-seg (detection + segmentation)
//   - NudeNet
//   - RT-DETR
//   - Any ONNX model with a YOLO-compatible output tensor
//
// Output tensor conventions assumed:
//   [1, 4+C(+32), N]  (transposed YOLO)  or  [1, N, 4+C(+32)]  (standard)
//   Second output (seg models): [1, 32, mask_h, mask_w] prototype masks.
//   Filtered by confidence threshold and greedy NMS before returning.

use super::{Detection, DetectionLabel, InferenceEngine, SegMask};
use crate::capture::frame::CapturedFrame;
use crate::config::DetectionConfig;
use crate::inference::label_config::ModelLabelConfig;
use anyhow::{Context, Result};
use ndarray::{Array, ArrayD, Axis, IxDyn};
use ort::{
    execution_providers::DirectMLExecutionProvider,
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
};
use std::path::Path;

#[cfg(target_os = "linux")]
use ort::execution_providers::CUDAExecutionProvider;

// ── Internal intermediate type ────────────────────────────────────────────

/// Intermediate detection before segmentation masks are applied.
struct RawDet {
    detection: Detection,
    /// Mask prototype coefficients (empty for detection-only models).
    mask_coeffs: Vec<f32>,
}

// ── Engine ────────────────────────────────────────────────────────────────

/// ONNX Runtime based inference engine.
pub struct OnnxEngine {
    session: Session,
    name: String,
    input_width: u32,
    input_height: u32,
    /// Mask prototype channels in the 2nd output tensor (0 = detection-only).
    /// YOLOv8/YOLO11 -seg models have 32 mask protos; NudeNet has 0.
    num_mask_protos: usize,
    /// Per-class label configuration loaded at model-load time.
    label_config: ModelLabelConfig,
}

impl OnnxEngine {
    /// Load a model from disk.
    /// When `gpu` is true, DirectML (Windows) or CUDA (Linux) is requested;
    /// the runtime silently falls back to CPU if the EP is unavailable.
    pub fn load(model_path: &Path, gpu: bool) -> Result<Self> {
        let name = model_path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        // Register execution providers before building the session.
        if gpu {
            #[cfg(target_os = "windows")]
            {
                let _ = ort::init()
                    .with_execution_providers([DirectMLExecutionProvider::default().build()])
                    .commit();
            }
            #[cfg(target_os = "linux")]
            {
                let _ = ort::init()
                    .with_execution_providers([CUDAExecutionProvider::default().build()])
                    .commit();
            }
        } else {
            let _ = ort::init().commit();
        }

        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("creating ONNX session builder: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("setting optimization level: {e}"))?
            .with_intra_threads(num_cpus::get().min(4))
            .map_err(|e| anyhow::anyhow!("setting intra-op threads: {e}"))?
            .commit_from_file(model_path)
            .map_err(|e| anyhow::anyhow!("loading ONNX model from {}: {e}", model_path.display()))?;

        let (input_height, input_width) = parse_input_dims(&session);
        let num_mask_protos = detect_mask_protos(&session);

        // Load label config: sidecar file first, then filename heuristic.
        let mut label_config = ModelLabelConfig::load_for_model(model_path);

        // Override has_segmentation based on actual ONNX output shape.
        if num_mask_protos > 0 {
            label_config.has_segmentation = true;
        }

        // Write a sidecar .labels.toml so users can customise labels.
        label_config.write_sidecar(model_path);

        log::info!(
            "Model '{}' loaded — input {}×{}, {} input(s), {} output(s){}, {} classes",
            name,
            input_width,
            input_height,
            session.inputs().len(),
            session.outputs().len(),
            if num_mask_protos > 0 {
                format!(" (seg, {} mask protos)", num_mask_protos)
            } else {
                String::new()
            },
            label_config.classes.len(),
        );

        Ok(Self {
            session,
            name,
            input_width,
            input_height,
            num_mask_protos,
            label_config,
        })
    }

    // ── Preprocessing ────────────────────────────────────────────────────

    /// Resize `frame` to the model's expected input size and convert to
    /// a float32 NCHW tensor normalised to [0, 1].
    fn preprocess(&self, frame: &CapturedFrame) -> Result<ArrayD<f32>> {
        use image::imageops::FilterType;

        let img = image::RgbaImage::from_raw(frame.width, frame.height, frame.data.clone())
            .context("frame data dimensions mismatch")?;
        let rgb = image::DynamicImage::ImageRgba8(img).to_rgb8();
        let resized = image::imageops::resize(
            &rgb,
            self.input_width,
            self.input_height,
            FilterType::Nearest, // fastest for real-time use
        );

        let (h, w) = (self.input_height as usize, self.input_width as usize);
        let mut tensor = Array::<f32, _>::zeros(IxDyn(&[1, 3, h, w]));
        for y in 0..h {
            for x in 0..w {
                let px = resized.get_pixel(x as u32, y as u32);
                tensor[[0, 0, y, x]] = px[0] as f32 / 255.0;
                tensor[[0, 1, y, x]] = px[1] as f32 / 255.0;
                tensor[[0, 2, y, x]] = px[2] as f32 / 255.0;
            }
        }
        Ok(tensor)
    }

    // ── Output parsing ───────────────────────────────────────────────────

    /// Decode a YOLO-style output tensor into `RawDet` structs.
    ///
    /// Supported shapes (after batch squeeze):
    ///   [4+C(+P), N] — YOLO transposed output (YOLOv8 default export)
    ///   [N, 4+C(+P)] — standard row-per-detection
    ///
    /// For segmentation models the last `num_mask_protos` columns/rows are
    /// mask coefficients, returned in `RawDet.mask_coeffs`.
    fn parse_raw_dets(
        &self,
        raw: &ArrayD<f32>,
        conf_threshold: f32,
        filter: &DetectionConfig,
    ) -> Vec<RawDet> {
        // Squeeze batch dim if present.
        let data = if raw.ndim() == 3 && raw.shape()[0] == 1 {
            raw.index_axis(Axis(0), 0).into_dyn()
        } else {
            raw.view().into_dyn()
        };

        let shape = data.shape();
        if shape.len() != 2 {
            return Vec::new();
        }

        // Transposed [4+C, N]: the attrs axis is the small one.
        let transposed = shape[0] < shape[1];
        let (num_attrs, num_dets) = if transposed {
            (shape[0], shape[1])
        } else {
            (shape[1], shape[0])
        };

        // Subtract bbox (4) + mask-proto coefficients so we only score true
        // class channels.  For segmentation models (YOLOv8-seg, YOLO11-seg)
        // the last `num_mask_protos` attrs are mask coefficients, not classes.
        let num_classes = match num_attrs.checked_sub(4 + self.num_mask_protos) {
            Some(c) if c > 0 => c,
            _ => return Vec::new(),
        };

        let mut raw_dets = Vec::new();

        for i in 0..num_dets {
            let (cx, cy, bw, bh) = if transposed {
                (data[[0, i]], data[[1, i]], data[[2, i]], data[[3, i]])
            } else {
                (data[[i, 0]], data[[i, 1]], data[[i, 2]], data[[i, 3]])
            };

            let mut best_class = 0usize;
            let mut best_score = f32::NEG_INFINITY;
            for c in 0..num_classes {
                let score = if transposed {
                    data[[4 + c, i]]
                } else {
                    data[[i, 4 + c]]
                };
                if score > best_score {
                    best_score = score;
                    best_class = c;
                }
            }

            if best_score < conf_threshold {
                continue;
            }

            let label = match DetectionLabel::from_class_index(best_class, &self.label_config) {
                Some(l) if l.is_enabled(filter) => l,
                _ => continue,
            };

            // Extract mask coefficients for segmentation models.
            let mask_coeffs: Vec<f32> = if self.num_mask_protos > 0 {
                (0..self.num_mask_protos)
                    .map(|p| {
                        if transposed {
                            data[[4 + num_classes + p, i]]
                        } else {
                            data[[i, 4 + num_classes + p]]
                        }
                    })
                    .collect()
            } else {
                Vec::new()
            };

            let x = (cx - bw / 2.0).max(0.0);
            let y = (cy - bh / 2.0).max(0.0);

            raw_dets.push(RawDet {
                detection: Detection {
                    x,
                    y,
                    w: bw,
                    h: bh,
                    confidence: best_score,
                    label,
                    mask: None,
                },
                mask_coeffs,
            });
        }

        nms_raw(&mut raw_dets, 0.4);
        raw_dets
    }
}

impl InferenceEngine for OnnxEngine {
    fn infer(
        &mut self,
        frame: &CapturedFrame,
        filter: &DetectionConfig,
    ) -> Result<Vec<Detection>> {
        let input_array = self.preprocess(frame)?;

        // Name of the first input (required by ort::inputs! macro).
        let input_name = self.session.inputs()[0].name().to_owned();
        let input_tensor = Tensor::from_array(input_array)
            .map_err(|e| anyhow::anyhow!("creating input tensor: {e}"))?;

        let outputs = self
            .session
            .run(ort::inputs![input_name => input_tensor])
            .map_err(|e| anyhow::anyhow!("running ONNX inference: {e}"))?;

        // ── Extract first output (detections) ────────────────────────────
        let (_, out0) = outputs
            .iter()
            .next()
            .context("model produced no outputs")?;

        let (shape0, data0) = out0
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("output is not an f32 tensor: {e}"))?;

        let dims0: Vec<usize> = shape0.iter().map(|&d| d as usize).collect();
        let raw0 = ArrayD::from_shape_vec(dims0, data0.to_vec())
            .context("reshaping first output tensor")?;

        // ── Extract second output (seg prototype masks) ───────────────────
        let protos_opt: Option<ArrayD<f32>> = if self.num_mask_protos > 0 {
            outputs.iter().nth(1).and_then(|(_, out1)| {
                out1.try_extract_tensor::<f32>()
                    .ok()
                    .and_then(|(shape1, data1)| {
                        let dims: Vec<usize> = shape1.iter().map(|&d| d as usize).collect();
                        ArrayD::from_shape_vec(dims, data1.to_vec()).ok()
                    })
            })
        } else {
            None
        };

        // Drop `outputs` (and the mutable borrow of session) before further work.
        drop(outputs);

        // ── Parse detections ──────────────────────────────────────────────
        let mut raw_dets = self.parse_raw_dets(&raw0, filter.confidence_threshold, filter);

        // ── Compute segmentation masks ────────────────────────────────────
        if let Some(ref protos) = protos_opt {
            for rd in &mut raw_dets {
                if !rd.mask_coeffs.is_empty() {
                    rd.detection.mask = compute_seg_mask(
                        &rd.mask_coeffs,
                        protos,
                        rd.detection.x,
                        rd.detection.y,
                        rd.detection.w,
                        rd.detection.h,
                        self.input_width,
                        self.input_height,
                    );
                }
            }
        }

        let mut detections: Vec<Detection> =
            raw_dets.into_iter().map(|rd| rd.detection).collect();

        // ── Scale coords from model space to frame space ──────────────────
        // `parse_raw_dets` yields coordinates in model-input space
        // (e.g. 0..640).  Scale them back to the coordinate space of `frame`
        // so that the pipeline's scale_x/scale_y (small→original) is correct.
        let sx = frame.width as f32 / self.input_width as f32;
        let sy = frame.height as f32 / self.input_height as f32;
        if sx != 1.0 || sy != 1.0 {
            for det in &mut detections {
                det.x *= sx;
                det.y *= sy;
                det.w *= sx;
                det.h *= sy;
            }
        }

        Ok(detections)
    }

    fn model_name(&self) -> &str {
        &self.name
    }
}

// ── Segmentation mask computation ─────────────────────────────────────────

/// Compute a binary segmentation mask for a single detection.
///
/// `coeffs`  – mask coefficients from the detection output (typically 32).
/// `protos`  – prototype mask tensor, shape [1, protos, mask_h, mask_w].
/// `x,y,w,h` – bounding box in YOLO model-input space (e.g. 0..640).
/// `model_w/h` – YOLO model input size (used to compute mask/model scale ratio).
fn compute_seg_mask(
    coeffs: &[f32],
    protos: &ArrayD<f32>,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    model_w: u32,
    model_h: u32,
) -> Option<SegMask> {
    // Squeeze batch dim: [1, protos, mask_h, mask_w] → [protos, mask_h, mask_w]
    let protos_3d = if protos.ndim() == 4 && protos.shape()[0] == 1 {
        protos.index_axis(Axis(0), 0).into_dyn()
    } else {
        protos.view().into_dyn()
    };

    if protos_3d.ndim() != 3 {
        return None;
    }
    let (num_protos, mask_h, mask_w) = (
        protos_3d.shape()[0],
        protos_3d.shape()[1],
        protos_3d.shape()[2],
    );
    if num_protos != coeffs.len() || mask_h == 0 || mask_w == 0 {
        return None;
    }

    // Compute full mask as a weighted sum of prototype layers, then sigmoid.
    // mask[h,w] = sigmoid( sum_p  coeffs[p] * protos[p, h, w] )
    let npix = mask_h * mask_w;
    let mut mask_vals = vec![0.0f32; npix];

    for p in 0..num_protos {
        let c = coeffs[p];
        if c.abs() < 1e-7 {
            continue;
        }
        let layer = protos_3d.index_axis(Axis(0), p);
        for (i, &v) in layer.iter().enumerate() {
            mask_vals[i] += c * v;
        }
    }
    for v in &mut mask_vals {
        *v = 1.0 / (1.0 + (-*v).exp()); // sigmoid
    }

    // Crop to the bounding box region in mask space.
    let scale_x = mask_w as f32 / model_w as f32;
    let scale_y = mask_h as f32 / model_h as f32;

    let mx0 = ((x * scale_x).floor() as usize).min(mask_w);
    let my0 = ((y * scale_y).floor() as usize).min(mask_h);
    let mx1 = (((x + w) * scale_x).ceil() as usize).min(mask_w);
    let my1 = (((y + h) * scale_y).ceil() as usize).min(mask_h);

    if mx1 <= mx0 || my1 <= my0 {
        return None;
    }

    let cw = mx1 - mx0;
    let ch = my1 - my0;
    let mut pixels = vec![0u8; cw * ch];

    for cy in 0..ch {
        for cx in 0..cw {
            if mask_vals[(my0 + cy) * mask_w + (mx0 + cx)] > 0.5 {
                pixels[cy * cw + cx] = 255;
            }
        }
    }

    Some(SegMask {
        width: cw as u32,
        height: ch as u32,
        pixels,
    })
}

// ── Helpers: input dimension extraction ──────────────────────────────────

/// Heuristically extract the expected spatial (H, W) from the first model input.
/// Falls back to 640×640 if the shape cannot be determined.
/// For YOLOv8-seg models the second output is `[1, num_protos, mask_h, mask_w]`.
/// Returns 0 if the model has no second output (detection-only model).
fn detect_mask_protos(session: &Session) -> usize {
    if let Some(shape) = session.outputs().get(1).and_then(|o| o.dtype().tensor_shape()) {
        let dims: Vec<i64> = shape.iter().copied().collect();
        if dims.len() == 4 && dims[1] > 0 {
            return dims[1] as usize;
        }
    }
    0
}

fn parse_input_dims(session: &Session) -> (u32, u32) {
    if let Some(shape) = session.inputs().first().and_then(|o| o.dtype().tensor_shape()) {
        let dims: Vec<i64> = shape.iter().copied().collect();
        if dims.len() == 4 {
            let h = if dims[2] > 0 { dims[2] as u32 } else { 640 };
            let w = if dims[3] > 0 { dims[3] as u32 } else { 640 };
            return (h, w);
        }
    }
    (640, 640)
}

// ── Non-maximum suppression (greedy) ─────────────────────────────────────

fn iou(a: &Detection, b: &Detection) -> f32 {
    let x1 = a.x.max(b.x);
    let y1 = a.y.max(b.y);
    let x2 = (a.x + a.w).min(b.x + b.w);
    let y2 = (a.y + a.h).min(b.y + b.h);
    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area_a = a.w * a.h;
    let area_b = b.w * b.h;
    let union = area_a + area_b - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

fn nms_raw(dets: &mut Vec<RawDet>, iou_threshold: f32) {
    dets.sort_by(|a, b| {
        b.detection
            .confidence
            .partial_cmp(&a.detection.confidence)
            .unwrap()
    });
    let mut keep = vec![true; dets.len()];
    for i in 0..dets.len() {
        if !keep[i] {
            continue;
        }
        for j in (i + 1)..dets.len() {
            if !keep[j] {
                continue;
            }
            if dets[i].detection.label == dets[j].detection.label
                && iou(&dets[i].detection, &dets[j].detection) > iou_threshold
            {
                keep[j] = false;
            }
        }
    }
    let mut idx = 0;
    dets.retain(|_| {
        let k = keep[idx];
        idx += 1;
        k
    });
}
