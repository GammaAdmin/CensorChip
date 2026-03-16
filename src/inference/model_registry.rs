// CensorChip – Model registry / scanner
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Scans the `models/` directory for ONNX files and presents them to the UI
// as a selectable list.  The registry is lazily refreshed so newly dropped
// models appear without restarting the app.

use crate::inference::label_config::ModelLabelConfig;
use std::path::{Path, PathBuf};

/// Info about a discovered model file.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Filename (e.g. "yolov8n.onnx").
    pub filename: String,
    /// Full path on disk.
    pub path: PathBuf,
    /// Approximate file size in MiB.
    pub size_mib: f64,
    /// User-friendly category guess based on the filename.
    pub tier: ModelTier,
    /// Per-class label configuration for this model.
    pub label_config: ModelLabelConfig,
}

impl ModelInfo {
    /// Returns true when the model supports segmentation masks.
    pub fn has_segmentation(&self) -> bool {
        self.label_config.has_segmentation
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelTier {
    Light,
    Balanced,
    High,
    Unknown,
}

impl std::fmt::Display for ModelTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Light    => write!(f, "Light"),
            Self::Balanced => write!(f, "Balanced"),
            Self::High     => write!(f, "High"),
            Self::Unknown  => write!(f, "?"),
        }
    }
}

/// Scan `dir` for *.onnx files.
pub fn scan_models(dir: &Path) -> Vec<ModelInfo> {
    let mut models = Vec::new();
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return models,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("onnx") {
            let filename = path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            let size_mib = entry
                .metadata()
                .map(|m| m.len() as f64 / (1024.0 * 1024.0))
                .unwrap_or(0.0);
            let tier = guess_tier(&filename);
            let label_config = ModelLabelConfig::load_for_model(&path);
            models.push(ModelInfo {
                filename,
                path,
                size_mib,
                tier,
                label_config,
            });
        }
    }
    models.sort_by(|a, b| a.filename.cmp(&b.filename));
    models
}

fn guess_tier(name: &str) -> ModelTier {
    let lower = name.to_ascii_lowercase();
    if lower.contains("v8n")
        || lower.contains("lite")
        || lower.contains("mobilenet")
        || lower.contains("nano")
        || lower.contains("tiny")
    {
        ModelTier::Light
    } else if lower.contains("v8s")
        || lower.contains("v9s")
        || lower.contains("nudenet")
        || lower.contains("small")
    {
        ModelTier::Balanced
    } else if lower.contains("v10")
        || lower.contains("v11")
        || lower.contains("rt-detr")
        || lower.contains("rtdetr")
        || lower.contains("large")
    {
        ModelTier::High
    } else {
        ModelTier::Unknown
    }
}
