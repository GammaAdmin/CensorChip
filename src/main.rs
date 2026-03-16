// CensorChip – Entry point
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Initialises logging, loads configuration, and launches the egui UI.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console on release

pub mod capture;
pub mod censor;
pub mod config;
pub mod downloader;
pub mod inference;
pub mod overlay;
pub mod pipeline;
pub mod ui;

use anyhow::Result;
use log::info;

fn main() -> Result<()> {
    // Initialise logger – reads RUST_LOG env var, defaults to info.
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .filter_module("wgpu_core", log::LevelFilter::Warn)
        .filter_module("wgpu_hal",  log::LevelFilter::Warn)
        .filter_module("naga",      log::LevelFilter::Warn)
        // Suppress Vulkan validation layer chatter from third-party hooks
        // (OBS, Overwolf, …).  These are cosmetic version-mismatch warnings
        // that don't affect functionality.
        .filter_module("wgpu_hal::vulkan", log::LevelFilter::Error)
        .format_timestamp_millis()
        .init();

    info!("CensorChip v{}", env!("CARGO_PKG_VERSION"));

    // Load (or create) configuration.
    let config_path = config::default_config_path();
    info!("Config path: {}", config_path.display());
    let shared_config = config::load_or_default(&config_path)?;

    // Ensure the models directory exists.
    {
        let models_dir = &shared_config.read().general.models_dir;
        std::fs::create_dir_all(models_dir)?;
    }

    // Launch the UI.
    let icon = eframe::icon_data::from_png_bytes(include_bytes!("../icon.png"))
        .unwrap_or_default();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("CensorChip")
            .with_icon(icon)
            .with_inner_size([1024.0, 720.0])
            .with_min_inner_size([640.0, 480.0]),
        // Use wgpu/Direct3D backend – unlike glow/OpenGL it supports
        // per-pixel alpha compositing for the transparent overlay viewport.
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };

    eframe::run_native(
        "CensorChip",
        options,
        Box::new(move |_cc| {
            Ok(Box::new(ui::CensorChipApp::new(
                shared_config,
                config_path,
            )))
        }),
    )
    .map_err(|e| anyhow::anyhow!("eframe error: {e}"))?;

    Ok(())
}
