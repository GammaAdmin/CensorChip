// CensorChip – Per-model label configuration
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Each ONNX model has its own class taxonomy.  This module provides:
//   • A `ModelLabelConfig` that lists classes with their display names and
//     default enable/disable state.
//   • A loader that first checks for a sidecar `<model>.labels.toml` file,
//     then falls back to built-in configs for known model families.
//
// Users can drop a `<model>.labels.toml` next to the ONNX file to override
// any heuristic.

use serde::{Deserialize, Serialize};
use std::path::Path;

fn default_true() -> bool { true }

/// One class/label entry in the model's output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassDef {
    /// ONNX class index (0-based).
    pub index: usize,
    /// Machine key used in the config HashMap (must be stable).
    pub key: String,
    /// Human-readable display name shown in the UI.
    pub name: String,
    /// Whether to enable censoring of this class by default.
    #[serde(default = "default_true")]
    pub default_enabled: bool,
}

/// Full label configuration for one model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelLabelConfig {
    /// True when the model outputs segmentation masks (second output tensor).
    #[serde(default)]
    pub has_segmentation: bool,
    /// Ordered list of class definitions.
    #[serde(default)]
    pub classes: Vec<ClassDef>,
}

impl Default for ModelLabelConfig {
    fn default() -> Self {
        generic_config()
    }
}

impl ModelLabelConfig {
    /// Load for `model_path`:
    ///  1. Try `<stem>.labels.toml` sidecar in the same directory.
    ///  2. Fall back to built-in heuristic based on filename.
    pub fn load_for_model(model_path: &Path) -> Self {
        // Sidecar:  foo.onnx  →  foo.labels.toml
        // `with_extension` replaces the last extension, so use stem manually.
        let stem = model_path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy();
        let sidecar = model_path
            .parent()
            .unwrap_or(Path::new("."))
            .join(format!("{stem}.labels.toml"));

        if sidecar.is_file() {
            if let Ok(text) = std::fs::read_to_string(&sidecar) {
                match toml::from_str::<ModelLabelConfig>(&text) {
                    Ok(cfg) => {
                        log::info!("[labels] loaded '{}' from sidecar", stem);
                        return cfg;
                    }
                    Err(e) => {
                        log::warn!("[labels] failed to parse sidecar {sidecar:?}: {e}");
                    }
                }
            }
        }

        // Heuristic from filename.
        let lower = stem.to_ascii_lowercase();
        if lower.contains("nudenet") || lower.contains("320n") || lower.contains("640m") {
            log::debug!("[labels] using NudeNet-v3 built-in config for '{stem}'");
            nudenet_v3_config()
        } else if lower.contains("erax") {
            log::debug!("[labels] using EraX built-in config for '{stem}'");
            erax_config()
        } else {
            log::debug!("[labels] using generic fallback config for '{stem}'");
            generic_config()
        }
    }

    /// Write a `.labels.toml` sidecar so users can customise it.
    /// Called automatically when we first encounter an unknown model.
    pub fn write_sidecar(&self, model_path: &Path) {
        let stem = model_path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy();
        let sidecar = model_path
            .parent()
            .unwrap_or(Path::new("."))
            .join(format!("{stem}.labels.toml"));
        if sidecar.exists() {
            return; // never overwrite
        }
        if let Ok(text) = toml::to_string_pretty(self) {
            let _ = std::fs::write(&sidecar, text);
        }
    }

    /// Look up a class definition by its output index.
    pub fn class_by_index(&self, idx: usize) -> Option<&ClassDef> {
        self.classes.iter().find(|c| c.index == idx)
    }
}

// ── Built-in configurations ──────────────────────────────────────────────

fn c(index: usize, key: &str, name: &str, default_enabled: bool) -> ClassDef {
    ClassDef { index, key: key.into(), name: name.into(), default_enabled }
}

/// NudeNet v3 – 18 classes.
pub fn nudenet_v3_config() -> ModelLabelConfig {
    ModelLabelConfig {
        has_segmentation: false,
        classes: vec![
            c(0,  "female_genitalia_covered",  "Female genitalia (covered)",  true ),
            c(1,  "face_female",               "Female face",                 false),
            c(2,  "buttocks_exposed",          "Buttocks (exposed)",          true ),
            c(3,  "female_breast_exposed",     "Female breast (exposed)",     true ),
            c(4,  "female_genitalia_exposed",  "Female genitalia (exposed)",  true ),
            c(5,  "male_breast_exposed",       "Male breast (exposed)",       false),
            c(6,  "anus_exposed",              "Anus (exposed)",              true ),
            c(7,  "feet_exposed",              "Feet (exposed)",              true ),
            c(8,  "belly_covered",             "Belly (covered)",             false),
            c(9,  "feet_covered",              "Feet (covered)",              true ),
            c(10, "armpits_covered",           "Armpits (covered)",           false),
            c(11, "armpits_exposed",           "Armpits (exposed)",           false),
            c(12, "face_male",                 "Male face",                   false),
            c(13, "belly_exposed",             "Belly (exposed)",             false),
            c(14, "male_genitalia_exposed",    "Male genitalia (exposed)",    true ),
            c(15, "anus_covered",              "Anus (covered)",              true ),
            c(16, "female_breast_covered",     "Female breast (covered)",     true ),
            c(17, "buttocks_covered",          "Buttocks (covered)",          false),
        ],
    }
}

/// EraX Anti-NSFW YOLO11 – 5 classes.
pub fn erax_config() -> ModelLabelConfig {
    ModelLabelConfig {
        has_segmentation: false,
        classes: vec![
            c(0, "nipple",    "Nipple",   true),
            c(1, "make_love", "Sex act",  true),
            c(2, "vagina",    "Vagina",   true),
            c(3, "anus",      "Anus",     true),
            c(4, "penis",     "Penis",    true),
        ],
    }
}

/// Generic single-class fallback (unknown models).
pub fn generic_config() -> ModelLabelConfig {
    ModelLabelConfig {
        has_segmentation: false,
        classes: vec![
            c(0, "detected", "Detected", true),
        ],
    }
}
