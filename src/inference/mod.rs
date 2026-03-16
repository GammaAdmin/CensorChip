// CensorChip – AI inference module
// SPDX-License-Identifier: GPL-3.0-or-later

pub mod label_config;
pub mod model_registry;
pub mod onnx;

use crate::capture::frame::CapturedFrame;
use crate::config::{AppConfig, DetectionConfig};
use anyhow::Result;
use std::path::Path;

// ── Detection output ─────────────────────────────────────────────────────

/// A detected class label.  The exact meaning depends on the model.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DetectionLabel {
    pub class_index: usize,
    /// Machine key, matches `ClassDef.key` and `DetectionConfig.enabled_labels`.
    pub key: String,
    /// Human-readable display name.
    pub name: String,
}

impl DetectionLabel {
    pub fn from_class_index(
        idx: usize,
        cfg: &label_config::ModelLabelConfig,
    ) -> Option<Self> {
        cfg.class_by_index(idx).map(|cd| Self {
            class_index: idx,
            key: cd.key.clone(),
            name: cd.name.clone(),
        })
    }

    /// Returns true when this label is enabled in the config.
    /// Unknown keys default to `true` (censor everything not explicitly off).
    pub fn is_enabled(&self, det_cfg: &DetectionConfig) -> bool {
        det_cfg.is_label_enabled(&self.key)
    }
}

/// A binary segmentation mask cropped to the detection bounding-box region.
/// `pixels` are row-major, 0 = transparent, 255 = part of segment.
#[derive(Debug, Clone)]
pub struct SegMask {
    pub width:  u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

/// A single detection result.
#[derive(Debug, Clone)]
pub struct Detection {
    pub x:          f32,
    pub y:          f32,
    pub w:          f32,
    pub h:          f32,
    pub confidence: f32,
    pub label:      DetectionLabel,
    /// Present for segmentation models.
    pub mask:       Option<SegMask>,
}

// ── Engine trait ─────────────────────────────────────────────────────────

pub trait InferenceEngine: Send {
    fn infer(
        &mut self,
        frame: &CapturedFrame,
        filter: &DetectionConfig,
    ) -> Result<Vec<Detection>>;
    fn model_name(&self) -> &str;
}

pub fn load_engine(model_path: &Path, config: &AppConfig) -> Result<Box<dyn InferenceEngine>> {
    onnx::OnnxEngine::load(model_path, config.performance.gpu_acceleration)
        .map(|e| Box::new(e) as Box<dyn InferenceEngine>)
}

