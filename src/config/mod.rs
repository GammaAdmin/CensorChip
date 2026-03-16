// CensorChip – Configuration module
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Handles loading, saving, and live-updating application configuration via
// config.toml.  Uses serde for deserialization and parking_lot for fast
// concurrent reads from the pipeline threads.

use anyhow::{Context, Result};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Root configuration – maps 1-to-1 with config.toml.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    #[serde(default)]
    pub general: GeneralConfig,
    #[serde(default)]
    pub detection: DetectionConfig,
    #[serde(default)]
    pub censorship: CensorshipConfig,
    #[serde(default)]
    pub performance: PerformanceConfig,
    #[serde(default)]
    pub hotkeys: HotkeyConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralConfig {
    /// Path to the models directory.
    #[serde(default = "default_models_dir")]
    pub models_dir: String,
    /// Currently selected model filename (e.g. "yolov8n.onnx").
    #[serde(default = "default_model")]
    pub selected_model: String,
    /// Capture the whole screen or a specific window title.
    #[serde(default)]
    pub window_title: Option<String>,
}

/// Per-label detection toggles.  Keys match `ClassDef.key` in the label
/// config.  Unknown keys default to `true` (censor if unsure).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionConfig {
    /// Map of label key → enabled.  Missing keys default to `true`.
    #[serde(default)]
    pub enabled_labels: HashMap<String, bool>,

    /// Confidence threshold (0.0 – 1.0).
    #[serde(default = "default_confidence")]
    pub confidence_threshold: f32,
}

impl DetectionConfig {
    pub fn is_label_enabled(&self, key: &str) -> bool {
        *self.enabled_labels.get(key).unwrap_or(&true)
    }

    /// Seed any missing keys from the given label config using their defaults.
    /// Existing user settings are not overwritten.
    pub fn seed_defaults(&mut self, label_cfg: &crate::inference::label_config::ModelLabelConfig) {
        for cd in &label_cfg.classes {
            self.enabled_labels
                .entry(cd.key.clone())
                .or_insert(cd.default_enabled);
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CensorshipConfig {
    /// Active censorship method.
    #[serde(default)]
    pub method: CensorMethod,
    /// Blur kernel radius (for Blur method).
    #[serde(default = "default_blur_radius")]
    pub blur_radius: u32,
    /// Pixel block size (for Pixelation method).
    #[serde(default = "default_pixel_size")]
    pub pixel_size: u32,
    /// Text to display (for TextOverlay method).
    #[serde(default = "default_censor_text")]
    pub overlay_text: String,
    /// Path to a custom texture (for TextureOverlay method).
    #[serde(default)]
    pub texture_path: Option<String>,
    /// Masking mode.
    #[serde(default)]
    pub mask_mode: MaskMode,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum CensorMethod {
    Blur,
    Pixelation,
    BlackBar,
    TextureOverlay,
    TextOverlay,
}

impl Default for CensorMethod {
    fn default() -> Self {
        Self::Blur
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum MaskMode {
    BoundingBox,
    Segmentation,
}

impl Default for MaskMode {
    fn default() -> Self {
        Self::BoundingBox
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum PerformancePreset {
    Low,
    Balanced,
    High,
}

impl Default for PerformancePreset {
    fn default() -> Self {
        Self::Balanced
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    #[serde(default)]
    pub preset: PerformancePreset,
    /// Run inference every N frames (skip frames to save CPU).
    #[serde(default = "default_skip_frames")]
    pub inference_skip_frames: u32,
    /// Downscale factor applied before inference (1 = no downscale, 2 = half, etc.).
    #[serde(default = "default_downscale")]
    pub downscale_factor: u32,
    /// Use GPU acceleration if available.
    #[serde(default)]
    pub gpu_acceleration: bool,
    /// Maximum capture FPS (0 = unlimited).
    #[serde(default)]
    pub max_fps: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotkeyConfig {
    /// Toggle censorship on/off.
    #[serde(default = "default_toggle_key")]
    pub toggle_censorship: String,
    /// Pause/resume capture.
    #[serde(default = "default_pause_key")]
    pub pause_capture: String,
}

// ── Default helpers ──────────────────────────────────────────────────────

fn default_models_dir() -> String {
    "models".into()
}
fn default_model() -> String {
    "yolov8n.onnx".into()
}
fn default_confidence() -> f32 {
    0.4
}
fn default_blur_radius() -> u32 {
    25
}
fn default_pixel_size() -> u32 {
    12
}
fn default_censor_text() -> String {
    "CENSORED".into()
}
fn default_skip_frames() -> u32 {
    2
}
fn default_downscale() -> u32 {
    2
}
fn default_toggle_key() -> String {
    "F9".into()
}
fn default_pause_key() -> String {
    "F10".into()
}

// ── Defaults for top-level structs ───────────────────────────────────────

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            general: GeneralConfig::default(),
            detection: DetectionConfig::default(),
            censorship: CensorshipConfig::default(),
            performance: PerformanceConfig::default(),
            hotkeys: HotkeyConfig::default(),
        }
    }
}

impl Default for GeneralConfig {
    fn default() -> Self {
        Self {
            models_dir: default_models_dir(),
            selected_model: default_model(),
            window_title: None,
        }
    }
}

impl Default for DetectionConfig {
    fn default() -> Self {
        // Start empty; keys are seeded from the model's label config via
        // `seed_defaults` when a model is selected.
        Self {
            enabled_labels: HashMap::new(),
            confidence_threshold: default_confidence(),
        }
    }
}

impl Default for CensorshipConfig {
    fn default() -> Self {
        Self {
            method: CensorMethod::default(),
            blur_radius: default_blur_radius(),
            pixel_size: default_pixel_size(),
            overlay_text: default_censor_text(),
            texture_path: None,
            mask_mode: MaskMode::default(),
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            preset: PerformancePreset::default(),
            inference_skip_frames: default_skip_frames(),
            downscale_factor: default_downscale(),
            gpu_acceleration: false,
            max_fps: 0,
        }
    }
}

impl Default for HotkeyConfig {
    fn default() -> Self {
        Self {
            toggle_censorship: default_toggle_key(),
            pause_capture: default_pause_key(),
        }
    }
}

// ── Shared config handle ─────────────────────────────────────────────────

/// Thread-safe handle to the live configuration.
pub type SharedConfig = Arc<RwLock<AppConfig>>;

/// Create a new shared config, loading from `path` if it exists.
pub fn load_or_default(path: &Path) -> Result<SharedConfig> {
    let cfg = if path.exists() {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading config from {}", path.display()))?;
        toml::from_str::<AppConfig>(&text)
            .with_context(|| format!("parsing config from {}", path.display()))?
    } else {
        AppConfig::default()
    };
    Ok(Arc::new(RwLock::new(cfg)))
}

/// Persist the current configuration to disk.
pub fn save(config: &AppConfig, path: &Path) -> Result<()> {
    let text = toml::to_string_pretty(config).context("serializing config")?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, text).with_context(|| format!("writing config to {}", path.display()))?;
    Ok(())
}

/// Return the default config path next to the executable.
pub fn default_config_path() -> PathBuf {
    std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join("config.toml")
}
