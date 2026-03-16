// CensorChip – Processing pipeline
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Implements the hyper-fast multi-threaded pipeline:
//
//   capture → resize → inference → mask generation → censor → render
//
// Each stage runs on a dedicated OS thread.  Stages are connected by bounded
// crossbeam channels (lock-free MPMC queues) so producers never block
// consumers and vice-versa.
//
// PERFORMANCE NOTES:
//  • Bounded channels with capacity 2 keep latency low while allowing one
//    frame of buffering.
//  • Inference is the bottleneck – the `inference_skip_frames` setting lets
//    us reuse the last detection result for N frames, dramatically reducing
//    GPU/CPU load on low-end hardware.
//  • All image resizing happens in the preprocessing thread so the inference
//    thread only ever sees small images.

pub mod tracker;

use crate::capture;
use crate::capture::frame::CapturedFrame;
use crate::censor::CensorEngine;
use crate::config::SharedConfig;
use crate::inference::{Detection, InferenceEngine};
use anyhow::Result;
use crossbeam_channel::{bounded, Receiver, Sender};
use log::{debug, info, warn};
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use tracker::SimpleTracker;

/// Messages flowing through the pipeline.
struct PreprocessedFrame {
    /// Original full-resolution frame (for overlay rendering).
    original: CapturedFrame,
    /// Downscaled frame ready for inference.
    small: CapturedFrame,
}

struct InferredFrame {
    original: CapturedFrame,
    detections: Vec<Detection>,
    /// Scale factors to map detection boxes back to original resolution.
    scale_x: f32,
    scale_y: f32,
}

/// Handle to the running pipeline.  Drop it to request shutdown.
pub struct PipelineHandle {
    /// Set to `true` to request all threads to stop.
    pub running: Arc<AtomicBool>,
    /// Censorship enabled flag (toggled by hotkey / UI).
    pub censorship_enabled: Arc<AtomicBool>,
    /// Latest censored frame that the UI should display.
    pub latest_frame: Arc<Mutex<Option<CapturedFrame>>>,
    /// Latest overlay frame (fully transparent except at censored regions).
    pub latest_overlay: Arc<Mutex<Option<CapturedFrame>>>,
    /// Raw (pre-censor) frame – used by the in-app "Clean" preview mode.
    pub latest_raw_frame: Arc<Mutex<Option<CapturedFrame>>>,
    /// FPS as measured by the censor+render thread.
    pub fps: Arc<AtomicU64>,
    /// How many times per second the AI model actually runs inference.
    /// Lower than `fps` when `inference_skip_frames` > 0.  Use this to
    /// compare model speed across checkpoints.
    pub inference_fps: Arc<AtomicU64>,
    /// Raw capture frame counter (set by capture thread).
    pub frame_count: Arc<AtomicU64>,
    /// Frames successfully processed all the way through to latest_frame.
    pub censor_count: Arc<AtomicU64>,
    /// `true` while a model is being loaded in the background.
    pub model_loading: Arc<AtomicBool>,
    threads: Vec<thread::JoinHandle<()>>,
}

impl PipelineHandle {
    /// Signal all pipeline threads to stop.
    ///
    /// Threads are *not* joined on the calling thread — they exit
    /// asynchronously so the UI never freezes.  The `Drop` impl also
    /// signals shutdown as a safety net.
    pub fn stop(mut self) {
        self.running.store(false, Ordering::SeqCst);
        // Detach thread handles instead of joining on the UI thread.
        // Each thread checks `running` with a 100 ms timeout and will
        // exit on its own.
        drop(std::mem::take(&mut self.threads));
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }
}

impl Drop for PipelineHandle {
    fn drop(&mut self) {
        self.running.store(false, Ordering::SeqCst);
    }
}

/// Spin up the full multi-threaded pipeline and return a handle.
pub fn start(config: SharedConfig, affinity_ok: Arc<AtomicBool>) -> Result<PipelineHandle> {
    // ── Quick sanity check: can we capture the screen at all? ─────────
    {
        let mut test_capturer = capture::create_capturer(0, None);
        match test_capturer.capture_frame() {
            Ok(f) => {
                info!(
                    "Screen capture test: OK ({}×{}, {} bytes)",
                    f.width, f.height, f.data.len()
                );
            }
            Err(e) => {
                return Err(anyhow::anyhow!(
                    "Screen capture failed: {e:#}. Check display permissions."
                ));
            }
        }
    }

    let running = Arc::new(AtomicBool::new(true));
    let censorship_enabled = Arc::new(AtomicBool::new(true));
    let latest_frame: Arc<Mutex<Option<CapturedFrame>>> = Arc::new(Mutex::new(None));
    let latest_overlay: Arc<Mutex<Option<CapturedFrame>>> = Arc::new(Mutex::new(None));
    let latest_raw_frame: Arc<Mutex<Option<CapturedFrame>>> = Arc::new(Mutex::new(None));
    let fps = Arc::new(AtomicU64::new(0));
    let inference_fps = Arc::new(AtomicU64::new(0));
    let frame_count = Arc::new(AtomicU64::new(0));
    let censor_count = Arc::new(AtomicU64::new(0));
    let model_loading = Arc::new(AtomicBool::new(false));

    // Bounded channels – capacity 2 keeps latency low.
    let (cap_tx, cap_rx): (Sender<CapturedFrame>, Receiver<CapturedFrame>) = bounded(2);
    let (pre_tx, pre_rx): (Sender<PreprocessedFrame>, Receiver<PreprocessedFrame>) = bounded(2);
    let (inf_tx, inf_rx): (Sender<InferredFrame>, Receiver<InferredFrame>) = bounded(2);

    let mut threads = Vec::new();

    // ── Thread 1: Screen capture ─────────────────────────────────────────
    {
        let running = running.clone();
        let config = config.clone();
        let frame_count = frame_count.clone();
        threads.push(thread::Builder::new().name("capture".into()).spawn(move || {
            info!("[capture] thread started");
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut current_title = config.read().general.window_title.clone();
                let mut capturer = capture::create_capturer(0, current_title.as_deref());
                info!("[capture] backend: {}", capturer.backend_name());

                let mut counter: u64 = 0;
                let mut consecutive_errors: u32 = 0;
                while running.load(Ordering::Relaxed) {
                    let cfg = config.read();
                    let max_fps = cfg.performance.max_fps;
                    let new_title = cfg.general.window_title.clone();
                    drop(cfg);

                    // Recreate the capturer whenever the user switches source.
                    if new_title != current_title {
                        current_title = new_title;
                        capturer = capture::create_capturer(0, current_title.as_deref());
                        info!("[capture] switched backend: {}", capturer.backend_name());
                    }

                    let t0 = Instant::now();
                    match capturer.capture_frame() {
                        Ok(mut frame) => {
                            consecutive_errors = 0;
                            counter += 1;
                            frame.frame_id = counter;
                            frame_count.store(counter, Ordering::Relaxed);
                            if counter == 1 {
                                info!("[capture] first frame captured ({}x{})", frame.width, frame.height);
                            }
                            if cap_tx.try_send(frame).is_err() {
                                debug!("[capture] channel full, dropping frame");
                            }
                        }
                        Err(e) => {
                            consecutive_errors += 1;
                            if consecutive_errors <= 3 {
                                warn!("[capture] error (#{consecutive_errors}): {e:#}");
                            }
                            thread::sleep(Duration::from_millis(100));
                            if consecutive_errors > 100 {
                                log::error!("[capture] too many consecutive errors, stopping");
                                break;
                            }
                        }
                    }

                    if max_fps > 0 {
                        let target = Duration::from_secs_f64(1.0 / max_fps as f64);
                        let elapsed = t0.elapsed();
                        if elapsed < target {
                            thread::sleep(target - elapsed);
                        }
                    }
                }
            }));
            if let Err(e) = result {
                log::error!("[capture] thread panicked: {:?}", e);
            }
            info!("[capture] thread exiting");
        })?);
    }

    // ── Thread 2: Preprocessing (downscale) ──────────────────────────────
    {
        let running = running.clone();
        let config = config.clone();
        // Read-only reference to the current overlay so the preprocess thread
        // can blank out already-censored pixels before they reach inference.
        // This stops the model from seeing its own censor bars when
        // SetWindowDisplayAffinity is unavailable (VM, RDP, old Windows).
        //
        // IMPORTANT: only blank when the captured frame is known to contain
        // the overlay.  When WDA_EXCLUDEFROMCAPTURE / WDA_MONITOR is active
        // (affinity_ok == true) the DXGI frame is already clean – blanking
        // would erase real content and cause periodic blinking.  Similarly,
        // window capture (PrintWindow) never includes the overlay at all.
        let latest_overlay_preprocess = latest_overlay.clone();
        let affinity_ok_pre = affinity_ok.clone();
        let latest_raw_frame_pre = latest_raw_frame.clone();
        threads.push(thread::Builder::new().name("preprocess".into()).spawn(move || {
            info!("[preprocess] thread started");
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                while running.load(Ordering::Relaxed) {
                    match cap_rx.recv_timeout(Duration::from_millis(100)) {
                        Ok(frame) => {
                            // Update the raw preview at capture rate so the in-app
                            // preview runs smoothly rather than at inference rate.
                            *latest_raw_frame_pre.lock() = Some(frame.clone());

                            let factor = config.read().performance.downscale_factor;
                            let mut small = frame.downscale(factor);

                            // Blank overlay pixels in the inference frame ONLY
                            // when the overlay is known to be visible to the
                            // capturer: full-screen DXGI mode (window_title is
                            // None) AND WDA affinity has not yet been confirmed.
                            let is_window_mode = config.read().general.window_title.is_some();
                            let affinity_active = affinity_ok_pre.load(Ordering::Relaxed);
                            if !is_window_mode && !affinity_active {
                                if let Some(ov_guard) = latest_overlay_preprocess.try_lock() {
                                    if let Some(ref ov) = *ov_guard {
                                        let sw = small.width  as usize;
                                        let sh = small.height as usize;
                                        let ow = ov.width  as usize;
                                        let oh = ov.height as usize;
                                        for y in 0..sh {
                                            for x in 0..sw {
                                                let ox = (x * ow) / sw;
                                                let oy = (y * oh) / sh;
                                                let oi = (oy * ow + ox) * 4 + 3;
                                                if ov.data.get(oi).copied().unwrap_or(0) > 0 {
                                                    let si = (y * sw + x) * 4;
                                                    small.data[si]     = 127;
                                                    small.data[si + 1] = 127;
                                                    small.data[si + 2] = 127;
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            if frame.frame_id == 1 {
                                info!("[preprocess] first frame forwarded ({}x{} → {}x{})",
                                    frame.width, frame.height, small.width, small.height);
                            }
                            let msg = PreprocessedFrame {
                                original: frame,
                                small,
                            };
                            if pre_tx.try_send(msg).is_err() {
                                debug!("[preprocess] channel full, dropping frame");
                            }
                        }
                        Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
                        Err(_) => break,
                    }
                }
            }));
            if let Err(e) = result {
                log::error!("[preprocess] thread panicked: {:?}", e);
            }
            info!("[preprocess] thread exiting");
        })?);
    }

    // ── Thread 3: AI inference ───────────────────────────────────────────
    // Model loading is kicked off in a *separate* OS thread so the frame
    // pipeline (capture → preprocess → censor+render) keeps flowing while
    // the ONNX Runtime initialises.  Frames processed before the engine is
    // ready are forwarded with empty detections (no censorship yet).
    {
        let running = running.clone();
        let config = config.clone();
        let model_loading = model_loading.clone();
        let inference_fps_arc = inference_fps.clone();
        threads.push(thread::Builder::new().name("inference".into()).spawn(move || {
            info!("[inference] thread started");
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut engine: Option<Box<dyn InferenceEngine>> = None;
                let mut last_model = String::new();
                let mut last_detections: Vec<Detection> = Vec::new();
                let mut frame_since_inference: u32 = 0;
                let mut tracker = SimpleTracker::new();
                // Inference FPS counter.
                let mut inf_count: u32 = 0;
                let mut inf_fps_timer = Instant::now();
                // Pixel-based scroll / motion compensation.
                // After each positive inference we save per-row luminance sums of
                // the small frame as a reference.  On subsequent frames we compute
                // the vertical shift between the current frame and that reference
                // and translate all detection boxes accordingly, giving the overlay
                // a cheap sub-inference-frame "scroll follow" effect.
                let mut reference_luma: Option<Vec<f32>> = None;
                let mut baseline_detections: Vec<Detection> = Vec::new();

                // Background model-loader channel (capacity 1 – one pending load at a time).
                type LoadResult = (String, anyhow::Result<Box<dyn InferenceEngine>>);
                let (load_tx, load_rx) = crossbeam_channel::bounded::<LoadResult>(1);
                let mut is_loading = false;

                let mut last_load_attempt: Option<Instant> = None;
                const LOAD_RETRY_COOLDOWN: Duration = Duration::from_secs(5);

                // Detection hold: keep the last positive detections alive for
                // this duration after the final positive inference result.
                // Primary anti-blink mechanism: while the overlay covers the
                // detected region the model returns empty, but we hold the
                // previous detections so the overlay stays visible instead of
                // toggling off and on.
                let mut last_detection_time: Option<Instant> = None;
                const DETECTION_HOLD: Duration = Duration::from_millis(2500);

                while running.load(Ordering::Relaxed) {
                    // ── Poll for completed background model load ──────────
                    if let Ok((loaded_name, load_result)) = load_rx.try_recv() {
                        is_loading = false;
                        model_loading.store(false, Ordering::Relaxed);
                        match load_result {
                            Ok(eng) => {
                                info!("[inference] model '{}' loaded OK", loaded_name);
                                engine = Some(eng);
                                last_model = loaded_name;
                            }
                            Err(e) => {
                                warn!("[inference] model '{}' failed to load: {e:#}", loaded_name);
                                last_load_attempt = Some(Instant::now());
                            }
                        }
                    }

                    match pre_rx.recv_timeout(Duration::from_millis(100)) {
                        Ok(msg) => {
                            let cfg = config.read().clone();

                            // Kick off a background load if the model changed and we have
                            // no load in flight.
                            if !is_loading {
                                let want_reload = engine.is_none()
                                    || last_model != cfg.general.selected_model;
                                let cooldown_ok = last_load_attempt
                                    .map(|t| t.elapsed() >= LOAD_RETRY_COOLDOWN)
                                    .unwrap_or(true);

                                if want_reload && cooldown_ok {
                                    let model_path = std::path::Path::new(&cfg.general.models_dir)
                                        .join(&cfg.general.selected_model);

                                    if model_path.is_file() {
                                        info!(
                                            "[inference] loading '{}' in background thread…",
                                            cfg.general.selected_model
                                        );
                                        is_loading = true;
                                        model_loading.store(true, Ordering::Relaxed);
                                        last_load_attempt = Some(Instant::now());

                                        let tx   = load_tx.clone();
                                        let name = cfg.general.selected_model.clone();
                                        let gpu  = cfg.performance.gpu_acceleration;

                                        thread::Builder::new()
                                            .name("model-load".into())
                                            .spawn(move || {
                                                let res = std::panic::catch_unwind(
                                                    std::panic::AssertUnwindSafe(|| {
                                                        crate::inference::onnx::OnnxEngine::load(
                                                            &model_path,
                                                            gpu,
                                                        )
                                                        .map(|e| {
                                                            Box::new(e) as Box<dyn InferenceEngine>
                                                        })
                                                    }),
                                                );
                                                let load_result = match res {
                                                    Ok(r) => r,
                                                    Err(_) => Err(anyhow::anyhow!(
                                                        "model load panicked"
                                                    )),
                                                };
                                                let _ = tx.send((name, load_result));
                                            })
                                            .ok(); // ignore spawn error
                                    }
                                    // else: file doesn't exist yet, skip silently
                                }
                            }

                            // Always forward frames — empty detections if no engine yet.
                            frame_since_inference += 1;
                            let should_infer =
                                frame_since_inference > cfg.performance.inference_skip_frames;

                            let scale_x =
                                msg.original.width as f32 / msg.small.width as f32;
                            let scale_y =
                                msg.original.height as f32 / msg.small.height as f32;

                            if should_infer {
                                frame_since_inference = 0;
                                if let Some(ref mut eng) = engine {
                                    match eng.infer(&msg.small, &cfg.detection) {
                                        Ok(dets) => {
                                            if !dets.is_empty() {
                                                // Fresh positive detections – update tracker
                                                // and reset the hold timer.
                                                tracker.update(&dets);
                                                last_detections = dets;
                                                last_detection_time = Some(Instant::now());
                                                // Snap a new luminance reference so the
                                                // scroll-compensation baseline is fresh.
                                                reference_luma = Some(row_luma_sums(&msg.small));
                                                baseline_detections = last_detections.clone();
                                                // Count this inference for fps tracking.
                                                inf_count += 1;
                                                if inf_fps_timer.elapsed() >= Duration::from_secs(1) {
                                                    inference_fps_arc.store(
                                                        inf_count as u64, Ordering::Relaxed);
                                                    inf_count = 0;
                                                    inf_fps_timer = Instant::now();
                                                }
                                            } else {
                                                // Inference returned empty.  Hold previous
                                                // detections for DETECTION_HOLD after the
                                                // last positive result so the overlay doesn't
                                                // blink off while it's obscuring the content.
                                                let hold_active = last_detection_time
                                                    .map(|t| t.elapsed() < DETECTION_HOLD)
                                                    .unwrap_or(false);
                                                if hold_active {
                                                    // If the frame content shifted by >20 % of the
                                                    // frame height the user almost certainly navigated
                                                    // to a new page – clear immediately rather than
                                                    // holding stale detections over new content.
                                                    let large_scene_change = reference_luma
                                                        .as_ref()
                                                        .map(|ref_luma| {
                                                            let curr = row_luma_sums(&msg.small);
                                                            let dy = estimate_v_shift(ref_luma, &curr, 64);
                                                            dy.abs() > msg.small.height as f32 * 0.20
                                                        })
                                                        .unwrap_or(false);
                                                    if large_scene_change {
                                                        tracker.update(&dets);
                                                        last_detections = dets;
                                                        last_detection_time = None;
                                                        reference_luma = None;
                                                        baseline_detections.clear();
                                                    } else {
                                                        apply_motion_shift(
                                                            &reference_luma,
                                                            &baseline_detections,
                                                            &msg.small,
                                                            &mut last_detections,
                                                            &tracker,
                                                        );
                                                    }
                                                } else {
                                                    tracker.update(&dets);
                                                    last_detections = dets; // genuinely clear
                                                    reference_luma = None;
                                                    baseline_detections.clear();
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            warn!("[inference] infer error: {e:#}");
                                        }
                                    }
                                }
                            } else {
                                // Skipping inference this frame: only apply motion
                                // compensation during the active hold window.
                                // Once the hold expires we keep last_detections as-is
                                // (set by the most recent inference) rather than
                                // restoring from the baseline, which would undo a
                                // "genuinely empty" result and cause a stale overlay.
                                let hold_active = last_detection_time
                                    .map(|t| t.elapsed() < DETECTION_HOLD)
                                    .unwrap_or(false);
                                if hold_active {
                                    apply_motion_shift(
                                        &reference_luma,
                                        &baseline_detections,
                                        &msg.small,
                                        &mut last_detections,
                                        &tracker,
                                    );
                                }
                            }

                            let out = InferredFrame {
                                original: msg.original,
                                detections: last_detections.clone(),
                                scale_x,
                                scale_y,
                            };
                            if inf_tx.try_send(out).is_err() {
                                debug!("[inference] channel full, dropping frame");
                            }
                        }
                        Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
                        Err(_) => break,
                    }
                }
            })); // end catch_unwind
            if let Err(e) = result {
                log::error!("[inference] thread panicked: {:?}", e);
            }
            info!("[inference] thread exiting");
        })?);
    }

    // ── Thread 4+5: Post-processing / censorship + render ────────────────
    {
        let running = running.clone();
        let config = config.clone();
        let censorship_enabled = censorship_enabled.clone();
        let latest_frame = latest_frame.clone();
        let latest_overlay = latest_overlay.clone();
        let latest_raw_frame = latest_raw_frame.clone();
        let fps = fps.clone();
        let censor_count = censor_count.clone();
        threads.push(thread::Builder::new().name("censor+render".into()).spawn(move || {
            info!("[censor+render] thread started");
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut censor_engine = CensorEngine::new();
                let mut fps_counter: u32 = 0;
                let mut fps_timer = Instant::now();

                while running.load(Ordering::Relaxed) {
                    match inf_rx.recv_timeout(Duration::from_millis(100)) {
                        Ok(msg) => {
                            let cfg = config.read().clone();
                            // Clone before censoring for the clean-preview mode.
                            let raw = msg.original.clone();
                            let mut frame = msg.original;

                            if censorship_enabled.load(Ordering::Relaxed) && !msg.detections.is_empty()
                            {
                                let overlay = censor_engine.apply_and_get_overlay(
                                    &mut frame,
                                    &msg.detections,
                                    msg.scale_x,
                                    msg.scale_y,
                                    &cfg.censorship,
                                );
                                *latest_overlay.lock() = Some(overlay);
                            } else {
                                *latest_overlay.lock() = None;
                            }

                            *latest_frame.lock() = Some(frame);
                            *latest_raw_frame.lock() = Some(raw);

                            let n = censor_count.fetch_add(1, Ordering::Relaxed) + 1;
                            if n == 1 {
                                info!("[censor+render] first frame written to latest_frame");
                            }

                            fps_counter += 1;
                            let elapsed = fps_timer.elapsed();
                            if elapsed >= Duration::from_secs(1) {
                                fps.store(fps_counter as u64, Ordering::Relaxed);
                                fps_counter = 0;
                                fps_timer = Instant::now();
                            }
                        }
                        Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
                        Err(_) => break,
                    }
                }
            }));
            if let Err(e) = result {
                log::error!("[censor+render] thread panicked: {:?}", e);
            }
            info!("[censor+render] thread exiting");
        })?);
    }

    Ok(PipelineHandle {
        running,
        censorship_enabled,
        latest_frame,
        latest_overlay,
        latest_raw_frame,
        fps,
        inference_fps,
        frame_count,
        censor_count,
        model_loading,
        threads,
    })
}

// ── Scroll / motion compensation helpers ─────────────────────────────────
//
// Between inference runs the tracker applies a constant-velocity prediction
// which handles slow, steady movement well.  For abrupt scroll events it
// often lags by several frames.
//
// The pixel-based approach below computes the *actual* vertical displacement
// of the scene between the current frame and the frame at which we last saw
// detections, then translates all stored detection boxes by that amount.
// The result is that the censor overlay "follows" the content when the user
// scrolls a page, even before the next model inference completes.
//
// Algorithm: reduce each frame to a 1-D signal of per-row average luminance
// (only the centre horizontal band is used to avoid noisy edges), then
// find the integer-pixel shift that minimises the sum-of-absolute-differences
// between the reference signal and the current signal.  This is O(H × Δmax)
// and runs in < 0.5 ms on a typical 640×360 small frame.

/// Compute per-row average luminance from the centre quarter-width band.
fn row_luma_sums(frame: &CapturedFrame) -> Vec<f32> {
    let w = frame.width as usize;
    let h = frame.height as usize;
    let x0 = w / 4;
    let x1 = 3 * w / 4;
    // Sample ~80 columns evenly across that band.
    let step = ((x1 - x0) / 80).max(1);
    let n_samples = (x1 - x0 + step - 1) / step;
    (0..h)
        .map(|y| {
            let mut sum = 0.0f32;
            let mut x = x0;
            while x < x1 {
                let i = (y * w + x) * 4;
                if i + 2 < frame.data.len() {
                    sum += 0.299 * frame.data[i] as f32
                        + 0.587 * frame.data[i + 1] as f32
                        + 0.114 * frame.data[i + 2] as f32;
                }
                x += step;
            }
            sum / n_samples as f32
        })
        .collect()
}

/// Find the vertical shift (in pixels, small-frame space) by which `curr`
/// appears to have moved relative to `prev`.
/// Positive return value means content moved *down* (the user scrolled up).
fn estimate_v_shift(prev: &[f32], curr: &[f32], max_shift: i32) -> f32 {
    let n = prev.len().min(curr.len()) as i32;
    let ms = max_shift.min(n / 4);
    let (mut best_dy, mut best_sad) = (0i32, f32::MAX);
    for dy in -ms..=ms {
        let mut sad = 0.0f32;
        let mut cnt = 0i32;
        for y in 0..n {
            let py = y - dy;
            if py >= 0 && py < n {
                sad += (prev[py as usize] - curr[y as usize]).abs();
                cnt += 1;
            }
        }
        if cnt > 0 {
            let norm = sad / cnt as f32;
            if norm < best_sad {
                best_sad = norm;
                best_dy = dy;
            }
        }
    }
    best_dy as f32
}

/// Translate `last_detections` to follow scene motion.
/// If a luminance reference is available the pixel-based shift is used;
/// otherwise falls back to the constant-velocity tracker prediction.
fn apply_motion_shift(
    reference_luma: &Option<Vec<f32>>,
    baseline: &[Detection],
    small: &CapturedFrame,
    last_detections: &mut Vec<Detection>,
    tracker: &SimpleTracker,
) {
    if let Some(ref ref_luma) = reference_luma {
        if !baseline.is_empty() {
            let curr_luma = row_luma_sums(small);
            let dy = estimate_v_shift(ref_luma, &curr_luma, 64);
            *last_detections = baseline
                .iter()
                .map(|d| {
                    let mut s = d.clone();
                    s.y += dy;
                    s
                })
                .collect();
            return;
        }
    }
    tracker.predict(last_detections);
}

