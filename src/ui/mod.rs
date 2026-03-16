// CensorChip – egui-based user interface
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Lightweight control panel + live preview built with eframe / egui.
// The UI runs on the main thread; the heavy lifting happens on dedicated
// pipeline threads which push finished frames to an Arc<Mutex<Option<Frame>>>.

use crate::config::{
    self, CensorMethod, MaskMode, PerformancePreset, SharedConfig,
};
use crate::downloader::{self as downloader, DownloadPhase, SharedDownloadState};
use crate::inference::model_registry;
use crate::pipeline::PipelineHandle;
use eframe::egui;
use log::info;
use std::path::PathBuf;
use std::sync::atomic::Ordering;

/// Which content is shown in the central preview panel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum PreviewMode {
    /// Post-censor frame – what an observer would see on your screen.
    #[default]
    Censored,
    /// Raw captured frame before any censor effect is applied.
    Clean,
    /// No preview – hides the central panel entirely (saves GPU bandwidth).
    Off,
}

/// Application state for eframe.
pub struct CensorChipApp {
    config: SharedConfig,
    config_path: PathBuf,
    pipeline: Option<PipelineHandle>,
    models: Vec<model_registry::ModelInfo>,
    models_dir: PathBuf,
    /// egui texture handle for the live preview.
    preview_texture: Option<egui::TextureHandle>,
    /// Cached frame dimensions for detecting changes.
    preview_size: (u32, u32),
    /// Status line shown in the UI.
    status: String,
    /// Whether the full-screen transparent overlay window is active.
    overlay_enabled: bool,
    /// Which content to show in the central preview panel.
    preview_mode: PreviewMode,
    /// Win32 layered-window overlay (Some while pipeline is running).
    win32_overlay: Option<crate::overlay::Win32Overlay>,
    /// Per-model download progress state.
    download_states: Vec<Option<SharedDownloadState>>,
    /// Whether the "Download Models" popup is open.
    show_download_popup: bool,
    /// Cached list of open window titles for the Capture Source dropdown.
    /// Populated by clicking "↻ Refresh" in that section.
    available_windows: Vec<String>,
    /// Last preset applied; used to detect changes so we only write
    /// the preset's defaults once rather than overwriting sliders every frame.
    last_preset: Option<PerformancePreset>,
}

impl CensorChipApp {
    pub fn new(config: SharedConfig, config_path: PathBuf) -> Self {
        let models_dir = {
            let cfg = config.read();
            PathBuf::from(&cfg.general.models_dir)
        };
        let models = model_registry::scan_models(&models_dir);

        Self {
            config,
            config_path,
            pipeline: None,
            models,
            models_dir,
            preview_texture: None,
            preview_size: (0, 0),
            status: "Idle".into(),
            overlay_enabled: true,
            preview_mode: PreviewMode::Off,
            win32_overlay: None,
            download_states: vec![None; downloader::AVAILABLE_MODELS.len()],
            show_download_popup: false,
            available_windows: Vec::new(),
            last_preset: None,
        }
    }

    fn start_pipeline(&mut self) {
        if self.pipeline.is_some() {
            return;
        }
        // Launch the overlay first so we can pass its affinity_ok Arc
        // to the pipeline.  The preprocess thread uses it to decide whether
        // the overlay is visible in the captured frames; when affinity is
        // confirmed active it skips blanking so the model can see content.
        let overlay = crate::overlay::Win32Overlay::launch();
        let affinity_ok = overlay.affinity_ok.clone();
        match crate::pipeline::start(self.config.clone(), affinity_ok) {
            Ok(handle) => {
                self.win32_overlay = Some(overlay);
                self.pipeline = Some(handle);
                self.overlay_enabled = true;
                self.status = "Running".into();
                info!("Pipeline started");
            }
            Err(e) => {
                self.status = format!("Failed to start: {e:#}");
            }
        }
    }

    fn stop_pipeline(&mut self) {
        if let Some(handle) = self.pipeline.take() {
            handle.stop();
            self.status = "Stopped".into();
            info!("Pipeline stopped");
        }
        self.win32_overlay = None; // hides and destroys the overlay window
        self.overlay_enabled = false;
    }

    fn refresh_models(&mut self) {
        self.models = model_registry::scan_models(&self.models_dir);
    }

    fn save_config(&self) {
        let cfg = self.config.read().clone();
        if let Err(e) = config::save(&cfg, &self.config_path) {
            log::error!("Failed to save config: {e:#}");
        }
    }

    /// Spawn a background download thread for the model at `AVAILABLE_MODELS[idx]`.
    fn kick_download(&mut self, idx: usize, dest_path: std::path::PathBuf) {
        let shared = std::sync::Arc::new(parking_lot::Mutex::new(
            downloader::DownloadState::default(),
        ));
        self.download_states[idx] = Some(shared.clone());
        downloader::start_download(
            downloader::AVAILABLE_MODELS[idx].url,
            dest_path,
            shared,
        );
    }
}

impl eframe::App for CensorChipApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Handle window close: stop the pipeline immediately so threads don't
        // keep the process alive.
        if ctx.input(|i| i.viewport().close_requested()) {
            self.stop_pipeline();
        }

        // Request repaints at ~60 fps while the pipeline is running.
        // Using request_repaint_after suppresses the wgpu Device::maintain
        // spam that occurs when repaints are triggered on every GPU flush.
        if self.pipeline.is_some() {
            ctx.request_repaint_after(std::time::Duration::from_millis(16));
        }

        // ── Push overlay frame to Win32 layered window ──────────────────
        if let Some(ref ov) = self.win32_overlay {
            if self.overlay_enabled {
                if let Some(ref handle) = self.pipeline {
                    let frame_opt = handle.latest_overlay.lock().clone();
                    if let Some(ref frame) = frame_opt {
                        ov.update_frame(&frame.data, frame.width, frame.height,
                                        frame.screen_x, frame.screen_y);
                    } else {
                        ov.hide();
                    }
                }
            } else {
                ov.hide();
            }
        }

        // ── Side panel (controls) ────────────────────────────────────────
        egui::SidePanel::left("controls")
            .min_width(300.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().auto_shrink([false; 2]).show(ui, |ui| {
                ui.heading("CensorChip");
                ui.separator();

                // ── Start / Stop ─────────────────────────────────────────
                ui.horizontal(|ui| {
                    let running = self.pipeline.is_some();
                    if running {
                        if ui.button("Stop").clicked() {
                            self.stop_pipeline();
                        }
                    } else if ui.button("Start").clicked() {
                        self.start_pipeline();
                    }

                    // Censorship toggle.
                    if let Some(ref handle) = self.pipeline {
                        let mut enabled =
                            handle.censorship_enabled.load(Ordering::Relaxed);
                        if ui.checkbox(&mut enabled, "Censorship").changed() {
                            handle
                                .censorship_enabled
                                .store(enabled, Ordering::Relaxed);
                        }
                    }
                });

                // FPS display.
                if let Some(ref handle) = self.pipeline {
                    let fps          = handle.fps.load(Ordering::Relaxed);
                    let inf_fps      = handle.inference_fps.load(Ordering::Relaxed);
                    let frames       = handle.frame_count.load(Ordering::Relaxed);
                    let censored     = handle.censor_count.load(Ordering::Relaxed);
                    ui.label(format!(
                        "Cap: {} fps | Inf: {}/s | Out: {} | Cen: {}",
                        fps, inf_fps, frames, censored
                    ))
                    .on_hover_text(
                        "Cap = capture FPS\nInf = inference calls/s\nOut = total frames rendered\nCen = total censored frames"
                    );

                    if handle.model_loading.load(Ordering::Relaxed) {
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label(egui::RichText::new("Loading model…")
                                .color(egui::Color32::YELLOW));
                        });
                    }
                }
                ui.label(&self.status);

                // Overlay toggle — only meaningful while pipeline is running.
                ui.horizontal(|ui| {
                    let was_enabled = self.overlay_enabled;
                    let label = if self.overlay_enabled {
                        "Overlay: ON"
                    } else {
                        "Overlay: OFF"
                    };
                    if ui.toggle_value(&mut self.overlay_enabled, label).clicked() {
                        if self.overlay_enabled && self.pipeline.is_none() {
                            self.overlay_enabled = false;
                            self.status = "Start the pipeline first".into();
                        }
                    }
                    // Resize window when overlay mode changes.
                    if self.overlay_enabled != was_enabled {
                        if self.overlay_enabled {
                            ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(
                                egui::vec2(340.0, 480.0),
                            ));
                        } else {
                            ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(
                                egui::vec2(1024.0, 720.0),
                            ));
                        }
                    }
                    ui.label(egui::RichText::new("ℹ")
                        .small()
                        .color(egui::Color32::GRAY))
                        .on_hover_text("Opens a transparent fullscreen overlay that shows\ncensored regions directly on your desktop.");
                });

                // ── Overlay affinity warning ──────────────────────────────
                // Show a prominent warning when:
                //  - The overlay is enabled, AND
                //  - SetWindowDisplayAffinity completely failed (VM / RDP), AND
                //  - The user is still on full-screen DXGI capture.
                // In that situation the overlay is visible to screen recorders,
                // which re-creates the feedback-loop blink.
                let affinity_failed = self.win32_overlay
                    .as_ref()
                    .map(|o| !o.affinity_ok.load(Ordering::Relaxed))
                    .unwrap_or(false);
                let using_full_screen = self.config.read().general.window_title.is_none();

                if self.overlay_enabled && affinity_failed && using_full_screen {
                    egui::Frame::none()
                        .fill(egui::Color32::from_rgb(120, 60, 0))
                        .inner_margin(egui::Margin::same(6.0))
                        .rounding(4.0)
                        .show(ui, |ui| {
                            ui.label(egui::RichText::new(
                                "[!] Overlay visible to screen recorders \
                                 (SetWindowDisplayAffinity unavailable on this GPU/OS).\n\
                                 Fix: open \u{22}Capture Source\u{22} below and select \
                                 your target application. Window capture bypasses the \
                                 overlay completely."
                            ).color(egui::Color32::WHITE).small());
                        });
                }

                // ── Preview mode ───────────────────────────────────────────
                ui.horizontal(|ui| {
                    ui.label("Preview:");
                    ui.selectable_value(
                        &mut self.preview_mode, PreviewMode::Censored, "Censored");
                    ui.selectable_value(
                        &mut self.preview_mode, PreviewMode::Clean, "Clean");
                    ui.selectable_value(
                        &mut self.preview_mode, PreviewMode::Off, "Off");
                });

                ui.separator();

                // ── Capture source ────────────────────────────────────────
                ui.collapsing("Capture Source", |ui| {
                    // Refresh enumerates all visible windows right now.
                    if ui.button("Refresh window list").clicked() {
                        self.available_windows = crate::capture::list_windows();
                    }

                    let current = self.config.read().general.window_title.clone();
                    let display = current.clone().unwrap_or_else(|| "Full screen (DXGI)".into());

                    egui::ComboBox::from_id_salt("window_picker")
                        .selected_text(&display)
                        .width(ui.available_width())
                        .show_ui(ui, |ui| {
                            // Full-screen option first.
                            let sel = current.is_none();
                            if ui.selectable_label(sel, "Full screen (DXGI)").clicked() {
                                self.config.write().general.window_title = None;
                            }
                            if self.available_windows.is_empty() {
                                ui.label(egui::RichText::new(
                                    "(click \"Refresh window list\" to see open windows)"
                                ).italics().small());
                            } else {
                                for title in self.available_windows.clone() {
                                    let sel = current.as_deref() == Some(&title);
                                    if ui.selectable_label(sel, &title).clicked() {
                                        self.config.write().general.window_title =
                                            Some(title.clone());
                                    }
                                }
                            }
                        });

                    ui.label(
                        egui::RichText::new(
                            "(i) Window capture reads directly from the app's GPU surface \
                             \u{2014} the censor overlay is never included regardless of OS support.\n\
                             Restart the pipeline after changing the source."
                        ).small().color(egui::Color32::GRAY)
                    );
                });

                ui.separator();

                // ── Model selection ──────────────────────────────────────
                ui.collapsing("Model", |ui| {
                    if ui.button("Refresh models").clicked() {
                        self.refresh_models();
                    }
                    let mut cfg = self.config.write();
                    for model in &self.models {
                        let selected = cfg.general.selected_model == model.filename;
                        let num_cls = model.label_config.classes.len();
                        let seg_tag = if model.has_segmentation() { ", seg" } else { "" };
                        let label = format!(
                            "{} ({:.1} MiB) [{}] [{} cls{}]",
                            model.filename, model.size_mib, model.tier, num_cls, seg_tag
                        );
                        if ui.selectable_label(selected, &label).clicked() {
                            cfg.general.selected_model.clone_from(&model.filename);
                            // Auto-apply EraX defaults when an EraX model is selected.
                            if model.filename.to_lowercase().contains("erax") {
                                cfg.detection.confidence_threshold = 0.35;
                                cfg.censorship.mask_mode = MaskMode::BoundingBox;
                                // Seed EraX label defaults (enable all by default).
                                cfg.detection.seed_defaults(&model.label_config);
                            } else {
                                // Automatically switch mask mode to match the new model.
                                if model.has_segmentation() {
                                    cfg.censorship.mask_mode = MaskMode::Segmentation;
                                } else {
                                    cfg.censorship.mask_mode = MaskMode::BoundingBox;
                                }
                            }
                        }
                    }
                    if self.models.is_empty() {
                        ui.label("No .onnx models found in models/ directory.");
                    }
                    // Warn when the selected model uses the generic (unknown) config.
                    let selected_generic = self.models.iter()
                        .find(|m| m.filename == cfg.general.selected_model)
                        .map(|m| m.label_config.classes.len() == 1
                            && m.label_config.classes[0].key == "detected")
                        .unwrap_or(false);
                    if selected_generic {
                        ui.colored_label(egui::Color32::YELLOW,
                            "[!] Unknown model type - may not detect nudity.\nDownload a NudeNet or EraX model.");
                    }
                });

                // ── Download Models (popup button) ────────────────────────
                if ui.button("Get Models...").clicked() {
                    self.show_download_popup = true;
                }

                // ── Performance ──────────────────────────────────────────
                ui.collapsing("Performance", |ui| {
                    let mut cfg = self.config.write();
                    ui.horizontal(|ui| {
                        ui.label("Preset:");
                        egui::ComboBox::from_id_salt("perf_preset")
                            .selected_text(format!("{:?}", cfg.performance.preset))
                            .show_ui(ui, |ui| {
                                ui.selectable_value(
                                    &mut cfg.performance.preset,
                                    PerformancePreset::Low,
                                    "Low",
                                );
                                ui.selectable_value(
                                    &mut cfg.performance.preset,
                                    PerformancePreset::Balanced,
                                    "Balanced",
                                );
                                ui.selectable_value(
                                    &mut cfg.performance.preset,
                                    PerformancePreset::High,
                                    "High",
                                );
                            });
                    });

                    // Apply preset side-effects only when the preset actually changes.
                    // Running this every frame would overwrite the skip-frames and
                    // downscale sliders each repaint, making them non-functional.
                    if Some(cfg.performance.preset) != self.last_preset {
                        self.last_preset = Some(cfg.performance.preset);
                        match cfg.performance.preset {
                            PerformancePreset::Low => {
                                cfg.performance.inference_skip_frames = 4;
                                cfg.performance.downscale_factor = 4;
                            }
                            PerformancePreset::Balanced => {
                                cfg.performance.inference_skip_frames = 2;
                                cfg.performance.downscale_factor = 2;
                            }
                            PerformancePreset::High => {
                                cfg.performance.inference_skip_frames = 0;
                                cfg.performance.downscale_factor = 1;
                            }
                        }
                    }

                    ui.checkbox(&mut cfg.performance.gpu_acceleration, "GPU acceleration");

                    ui.add(
                        egui::Slider::new(&mut cfg.performance.max_fps, 0..=240)
                            .text("Max FPS (0 = unlimited)"),
                    );
                    ui.add(
                        egui::Slider::new(&mut cfg.performance.inference_skip_frames, 0..=10)
                            .text("Skip frames"),
                    );
                    ui.add(
                        egui::Slider::new(&mut cfg.performance.downscale_factor, 1..=8)
                            .text("Downscale factor"),
                    );
                });

                // ── Detection filters ────────────────────────────────────
                ui.collapsing("Detection Filters", |ui| {
                    let selected_model_file = self.config.read().general.selected_model.clone();
                    let label_cfg_opt: Option<crate::inference::label_config::ModelLabelConfig> =
                        self.models
                            .iter()
                            .find(|m| m.filename == selected_model_file)
                            .map(|m| m.label_config.clone());

                    let mut cfg = self.config.write();

                    ui.add(
                        egui::Slider::new(&mut cfg.detection.confidence_threshold, 0.1..=1.0)
                            .text("Confidence"),
                    );

                    if let Some(ref label_cfg) = label_cfg_opt {
                        cfg.detection.seed_defaults(label_cfg);
                        ui.separator();
                        for class in &label_cfg.classes {
                            let mut enabled = cfg.detection.is_label_enabled(&class.key);
                            if ui.checkbox(&mut enabled, &class.name).changed() {
                                cfg.detection.enabled_labels.insert(class.key.clone(), enabled);
                            }
                        }
                    } else {
                        ui.label(egui::RichText::new("No model selected").color(egui::Color32::GRAY));
                    }
                });

                // ── Censorship style ─────────────────────────────────────
                ui.collapsing("Censorship Style", |ui| {
                    let mut cfg = self.config.write();
                    ui.horizontal(|ui| {
                        ui.label("Method:");
                        egui::ComboBox::from_id_salt("censor_method")
                            .selected_text(format!("{:?}", cfg.censorship.method))
                            .show_ui(ui, |ui| {
                                ui.selectable_value(
                                    &mut cfg.censorship.method,
                                    CensorMethod::Blur,
                                    "Blur",
                                );
                                ui.selectable_value(
                                    &mut cfg.censorship.method,
                                    CensorMethod::Pixelation,
                                    "Pixelation",
                                );
                                ui.selectable_value(
                                    &mut cfg.censorship.method,
                                    CensorMethod::BlackBar,
                                    "Black Bar",
                                );
                                ui.selectable_value(
                                    &mut cfg.censorship.method,
                                    CensorMethod::TextOverlay,
                                    "Text Overlay",
                                );
                                ui.selectable_value(
                                    &mut cfg.censorship.method,
                                    CensorMethod::TextureOverlay,
                                    "Texture Overlay",
                                );
                            });
                    });

                    match cfg.censorship.method {
                        CensorMethod::Blur => {
                            ui.add(
                                egui::Slider::new(&mut cfg.censorship.blur_radius, 1..=100)
                                    .text("Blur radius"),
                            );
                        }
                        CensorMethod::Pixelation => {
                            ui.add(
                                egui::Slider::new(&mut cfg.censorship.pixel_size, 2..=64)
                                    .text("Pixel size"),
                            );
                        }
                        CensorMethod::TextOverlay => {
                            ui.label("Overlay text:");
                            ui.text_edit_singleline(&mut cfg.censorship.overlay_text);
                        }
                        CensorMethod::TextureOverlay => {
                            ui.label("Texture path:");
                            let mut path_str = cfg
                                .censorship
                                .texture_path
                                .clone()
                                .unwrap_or_default();
                            if ui.text_edit_singleline(&mut path_str).changed() {
                                cfg.censorship.texture_path = if path_str.is_empty() {
                                    None
                                } else {
                                    Some(path_str)
                                };
                            }
                        }
                        CensorMethod::BlackBar => {}
                    }

                    // Only offer Segmentation mask mode when the selected model
                    // actually outputs segmentation masks.
                    let selected_model_file = cfg.general.selected_model.clone();
                    drop(cfg); // release write lock before the read below
                    let has_seg = self.models.iter()
                        .find(|m| m.filename == selected_model_file)
                        .map(|m| m.has_segmentation())
                        .unwrap_or(false);

                    if has_seg {
                        let mut cfg2 = self.config.write();
                        ui.separator();
                        ui.horizontal(|ui| {
                            ui.label("Mask mode:");
                            egui::ComboBox::from_id_salt("mask_mode")
                                .selected_text(format!("{:?}", cfg2.censorship.mask_mode))
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(
                                        &mut cfg2.censorship.mask_mode,
                                        MaskMode::BoundingBox,
                                        "Bounding Box",
                                    );
                                    ui.selectable_value(
                                        &mut cfg2.censorship.mask_mode,
                                        MaskMode::Segmentation,
                                        "Segmentation",
                                    );
                                });
                        });
                    } else {
                        // Reset to BoundingBox silently when model doesn't support seg.
                        let mut cfg2 = self.config.write();
                        cfg2.censorship.mask_mode = MaskMode::BoundingBox;
                    }
                });

                ui.separator();
                if ui.button("Save config").clicked() {
                    self.save_config();
                    self.status = "Config saved".into();
                }
                }); // end ScrollArea
            });

        // ── Download Models popup ────────────────────────────────────────
        if self.show_download_popup {
            // Snapshot state needed inside the closure (avoids &mut self conflict).
            let models_dir = self.models_dir.clone();
            let phase_snapshot: Vec<DownloadPhase> = self
                .download_states
                .iter()
                .map(|s| {
                    s.as_ref()
                        .map(|sh| sh.lock().phase.clone())
                        .unwrap_or(DownloadPhase::Idle)
                })
                .collect();
            let progress_snapshot: Vec<Option<(f32, String)>> = self
                .download_states
                .iter()
                .map(|s| {
                    s.as_ref().map(|sh| {
                        let g = sh.lock();
                        (g.fraction().unwrap_or(0.0), g.human_progress())
                    })
                })
                .collect();

            // Deferred actions.
            let mut kick: Option<usize> = None;
            let mut reset: Option<usize> = None;
            let mut refresh = false;

            egui::Window::new("Download Models")
                .open(&mut self.show_download_popup)
                .resizable(true)
                .default_width(520.0)
                .show(ctx, |ui| {
                    ui.label("Download ONNX models directly into your models folder.");
                    ui.add_space(4.0);

                    // EraX conversion note
                    ui.collapsing("EraX model - manual conversion required", |ui| {
                        ui.label(
                            "EraX-Anti-NSFW-V1.1 is a high-accuracy NSFW detector \
                             that must be converted to ONNX before use. \
                             It is NOT downloadable automatically."
                        );
                        ui.add_space(4.0);
                        ui.label("Steps:");
                        ui.label("  1. Download the PyTorch weights from HuggingFace:\n     \
                                    https://huggingface.co/erax-ai/EraX-Anti-NSFW-V1.1");
                        ui.label("  2. Install Ultralytics:\n     pip install ultralytics");
                        ui.label("  3. Export to ONNX (replace path as needed):");
                        ui.add_space(2.0);
                        ui.code("yolo export model=erax_anti_nsfw_v1.1.pt format=onnx imgsz=640");
                        ui.add_space(2.0);
                        ui.label("  4. Move the .onnx file into the models/ folder and \
                                    click \"Refresh models\".");
                    });
                    ui.add_space(4.0);

                    for (idx, model) in downloader::AVAILABLE_MODELS.iter().enumerate() {
                        let dest = models_dir.join(model.filename);
                        let already_exists = dest.exists();

                        ui.group(|ui| {
                            ui.horizontal(|ui| {
                                ui.strong(model.name);
                                ui.label(
                                    egui::RichText::new(model.size_human)
                                        .small()
                                        .color(egui::Color32::GRAY),
                                );
                            });
                            ui.label(
                                egui::RichText::new(model.description)
                                    .small()
                                    .italics(),
                            );

                            let phase = phase_snapshot
                                .get(idx)
                                .cloned()
                                .unwrap_or(DownloadPhase::Idle);

                            match phase {
                                DownloadPhase::Idle => {
                                    if already_exists {
                                        ui.label(
                                            egui::RichText::new("Already installed")
                                                .color(egui::Color32::GREEN),
                                        );
                                        if ui.small_button("Re-download").clicked() {
                                            kick = Some(idx);
                                        }
                                    } else if ui.button("Download").clicked() {
                                        kick = Some(idx);
                                    }
                                }
                                DownloadPhase::Connecting => {
                                    ui.horizontal(|ui| {
                                        ui.spinner();
                                        ui.label("Connecting…");
                                    });
                                }
                                DownloadPhase::Downloading => {
                                    if let Some((progress, label)) =
                                        progress_snapshot.get(idx).and_then(|p| p.clone())
                                    {
                                        ui.add(
                                            egui::ProgressBar::new(progress)
                                                .text(label)
                                                .animate(true),
                                        );
                                    }
                                }
                                DownloadPhase::Done => {
                                    ui.label(
                                        egui::RichText::new("Download complete!")
                                            .color(egui::Color32::GREEN),
                                    );
                                    reset = Some(idx);
                                    refresh = true;
                                }
                                DownloadPhase::Error(ref msg) => {
                                    ui.colored_label(
                                        egui::Color32::RED,
                                        format!("Error: {msg}"),
                                    );
                                    if ui.small_button("Retry").clicked() {
                                        kick = Some(idx);
                                    }
                                }
                            }
                        });
                        ui.add_space(2.0);
                    }
                });

            // Apply deferred mutations now that the closure has ended.
            if let Some(idx) = kick {
                let dest = self.models_dir.join(downloader::AVAILABLE_MODELS[idx].filename);
                self.kick_download(idx, dest);
            }
            if let Some(idx) = reset {
                self.download_states[idx] = None;
            }
            if refresh {
                self.refresh_models();
            }
        }

        // ── Central panel (live preview) ────────────────────────────────────────
        egui::CentralPanel::default().show(ctx, |ui| {
            if self.preview_mode == PreviewMode::Off {
                ui.centered_and_justified(|ui| {
                    ui.label(egui::RichText::new("Preview off").color(egui::Color32::GRAY));
                });
            } else if let Some(ref handle) = self.pipeline {                    let loading = handle.model_loading.load(Ordering::Relaxed);
                    let frame_opt = match self.preview_mode {
                        PreviewMode::Censored => handle.latest_frame.lock().clone(),
                        PreviewMode::Clean    => handle.latest_raw_frame.lock().clone(),
                        PreviewMode::Off      => unreachable!(),
                    };
                    if let Some(frame) = frame_opt {
                        let size = [frame.width as usize, frame.height as usize];
                        let color_image = egui::ColorImage::from_rgba_unmultiplied(size, &frame.data);

                        match &mut self.preview_texture {
                            Some(tex)
                                if self.preview_size == (frame.width, frame.height) =>
                            {
                                tex.set(color_image, egui::TextureOptions::NEAREST);
                            }
                            _ => {
                                self.preview_texture = Some(ctx.load_texture(
                                    "preview",
                                    color_image,
                                    egui::TextureOptions::NEAREST,
                                ));
                                self.preview_size = (frame.width, frame.height);
                            }
                        }

                        if let Some(ref tex) = self.preview_texture {
                            let available = ui.available_size();
                            let aspect = frame.width as f32 / frame.height as f32;
                            let (disp_w, disp_h) = fit_aspect(
                                available.x,
                                available.y,
                                aspect,
                            );
                            ui.image(egui::load::SizedTexture::new(
                                tex.id(),
                                egui::vec2(disp_w, disp_h),
                            ));
                        }
                    } else {
                        ui.centered_and_justified(|ui| {
                            if loading {
                                ui.vertical_centered(|ui| {
                                    ui.spinner();
                                    ui.label("Loading model, capture will start shortly…");
                                });
                            } else {
                                ui.label("Waiting for first frame...");
                            }
                        });
                    }
                } else {
                    ui.centered_and_justified(|ui| {
                        ui.label("Press ▶ Start to begin capture");
                    });
                }
            });
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        self.stop_pipeline();
        // Force-exit to prevent stuck capture/inference threads from hanging
        // the process (e.g. DXGI Desktop Duplication blocking).
        std::process::exit(0);
    }
}

/// Fit a rectangle with the given aspect ratio inside max_w × max_h.
fn fit_aspect(max_w: f32, max_h: f32, aspect: f32) -> (f32, f32) {
    let w = max_w;
    let h = w / aspect;
    if h <= max_h {
        (w, h)
    } else {
        let h = max_h;
        let w = h * aspect;
        (w, h)
    }
}
