// CensorChip – Model downloader
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Provides a catalogue of pre-vetted open ONNX models and a background
// download helper that reports progress to the UI via shared state.

use anyhow::{Context, Result};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use parking_lot::Mutex;

// ── Model catalogue ──────────────────────────────────────────────────────

/// A known, downloadable model definition.
pub struct ModelDefinition {
    pub name: &'static str,
    pub filename: &'static str,
    /// Direct HTTPS URL pointing to the ONNX file.
    pub url: &'static str,
    pub size_human: &'static str,
    pub description: &'static str,
}

/// Official NudeNet v3 / open NSFW ONNX models.
/// Primary source: HuggingFace (no login required).
pub const AVAILABLE_MODELS: &[ModelDefinition] = &[
    ModelDefinition {
        name: "NudeNet 320n  ·  Light",
        filename: "320n.onnx",
        url: "https://huggingface.co/zhangsongbo365/nudenet_onnx/resolve/main/320n.onnx",
        size_human: "~12 MiB",
        description: "Fast, lower accuracy. Best for real-time on CPU.",
    },
    ModelDefinition {
        name: "NudeNet 640m  ·  Balanced",
        filename: "640m.onnx",
        url: "https://huggingface.co/zhangsongbo365/nudenet_onnx/resolve/main/640m.onnx",
        size_human: "~104 MiB",
        description: "Balanced speed/accuracy. Recommended for GPU.",
    },
    ModelDefinition {
        name: "YOLO11n-seg  ·  General Segmentation",
        filename: "yolo11n-seg.onnx",
        url: "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n-seg.onnx",
        size_human: "~14 MiB",
        description: "General-purpose YOLOv8 segmentation (COCO classes). Good for person detection.",
    },
];

// ── Download state ───────────────────────────────────────────────────────

/// Current state of a (possibly in-progress) download.
#[derive(Debug, Clone, Default)]
pub struct DownloadState {
    pub phase: DownloadPhase,
    pub bytes_done: usize,
    pub bytes_total: usize,
}

impl DownloadState {
    /// 0.0 – 1.0 progress fraction, or `None` if total is unknown.
    pub fn fraction(&self) -> Option<f32> {
        if self.bytes_total > 0 {
            Some(self.bytes_done as f32 / self.bytes_total as f32)
        } else {
            None
        }
    }

    pub fn human_progress(&self) -> String {
        let done_mib = self.bytes_done as f32 / 1_048_576.0;
        if self.bytes_total > 0 {
            let total_mib = self.bytes_total as f32 / 1_048_576.0;
            format!("{done_mib:.1} / {total_mib:.1} MiB")
        } else {
            format!("{done_mib:.1} MiB")
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub enum DownloadPhase {
    #[default]
    Idle,
    Connecting,
    Downloading,
    Done,
    Error(String),
}

pub type SharedDownloadState = Arc<Mutex<DownloadState>>;

// ── Download logic ───────────────────────────────────────────────────────

/// Spawn a background thread to download `url` into `dest_path`.
/// Progress is written to `state`; will not block the caller.
pub fn start_download(url: &'static str, dest_path: PathBuf, state: SharedDownloadState) {
    std::thread::spawn(move || {
        {
            let mut s = state.lock();
            s.phase = DownloadPhase::Connecting;
            s.bytes_done = 0;
            s.bytes_total = 0;
        }

        // Catch panics so a ureq/TLS error never poisons shared state.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            do_download(url, &dest_path, &state)
        }));

        match result {
            Ok(Ok(())) => state.lock().phase = DownloadPhase::Done,
            Ok(Err(e)) => state.lock().phase = DownloadPhase::Error(format!("{e:#}")),
            Err(_) => state.lock().phase = DownloadPhase::Error("download panicked (internal error)".into()),
        }
    });
}

fn do_download(url: &str, dest_path: &Path, state: &SharedDownloadState) -> Result<()> {
    let response = ureq::get(url)
        .header("User-Agent", "CensorChip/0.1")
        .call()
        .context("HTTP request failed — check your internet connection")?;

    let status = response.status();
    if status.as_u16() < 200 || status.as_u16() >= 300 {
        anyhow::bail!("server returned HTTP {}", status.as_u16());
    }

    // Read content-length if server provides it.
    let total_bytes: usize = response
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0);

    {
        let mut s = state.lock();
        s.phase = DownloadPhase::Downloading;
        s.bytes_total = total_bytes;
    }

    // Stream into memory with chunked progress updates.
    let mut reader = response.into_body().into_reader();
    let mut buf = [0u8; 65_536];
    let mut data: Vec<u8> = if total_bytes > 0 {
        Vec::with_capacity(total_bytes)
    } else {
        Vec::new()
    };

    loop {
        let n = reader.read(&mut buf).context("reading download body")?;
        if n == 0 {
            break;
        }
        data.extend_from_slice(&buf[..n]);
        state.lock().bytes_done = data.len();
    }

    // Ensure the parent directory exists.
    if let Some(parent) = dest_path.parent() {
        std::fs::create_dir_all(parent).context("creating models directory")?;
    }

    std::fs::write(dest_path, &data).context("writing model file to disk")?;
    Ok(())
}
