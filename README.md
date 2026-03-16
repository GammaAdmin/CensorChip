# CensorChip

> Real-time AI-powered screen censorship - written in Rust.

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Build](https://github.com/GammaAdmin/CensorChip/actions/workflows/ci.yml/badge.svg)](https://github.com/GammaAdmin/CensorChip/actions)
[![Platform](https://img.shields.io/badge/platform-Windows-blue)](https://github.com/GammaAdmin/CensorChip)
[![Rust](https://img.shields.io/badge/built%20with-Rust-orange)](https://www.rust-lang.org)

CensorChip captures your screen in real time, runs AI-based detection to identify NSFW content, and draws a transparent censorship overlay - all without touching the underlying application.

---

## Features

- **Real-time screen capture** - DXGI Desktop Duplication (GPU-accelerated) or per-window capture via `PrintWindow`
- **AI object detection** - ONNX Runtime with DirectML / CUDA / CPU fallback
- **Censorship styles** - Blur, Pixelation, Black Bar, Text Overlay, Texture Overlay
- **Detection categories** - genitals, chest, buttocks, anus, face, eyes, full skin regions
- **Gender filtering** - apply rules selectively to female / male detections
- **Hot-swappable models** - drop any `.onnx` file into `models/` and select it in the UI
- **Inter-frame tracker** - smooth overlay without running inference every single frame
- **Sticky window capture** - browser tab / title changes no longer break capture
- **Built-in model downloader** - NudeNet models download in one click
- **Live preview** - embedded censored or clean preview panel
- **Transparent overlay** - rendered on a click-through topmost window; invisible to screen recorders
- **Hotkeys** - F9 toggle censorship, F10 pause (configurable)
- **config.toml** - all settings persisted automatically

---

## Performance

| Hardware | Typical FPS |
|---|---|
| CPU only (low-end) | 20–30 |
| CPU (mid-range) | 40–60 |
| GPU (DirectML / CUDA) | 100–200+ |

---

## Getting Started

### Download a release

Grab the latest `.zip` from the [Releases](https://github.com/GammaAdmin/CensorChip/releases) page and extract it.

On first launch, CensorChip will automatically:
1. Detect your system architecture (Windows x64 / ARM64)
2. Download the appropriate `onnxruntime.dll` from the official GitHub releases
3. Extract it to the same folder as the executable

Just run `censorchip.exe` and wait for the first-launch setup to complete. All downloads use HTTPS and verify via GitHub's official release infrastructure.

### Build from source

**Prerequisites**

- Rust ≥ 1.78 - install via [rustup.rs](https://rustup.rs)
- Visual Studio Build Tools 2022 (MSVC linker)

**Build steps**

```powershell
git clone https://github.com/GammaAdmin/CensorChip.git
cd censorchip
cargo build --release
```

The binary will be at `target\release\censorchip.exe`.

On first run, the app will automatically download `onnxruntime.dll` for your platform. No manual DLL setup needed!

The app is then ready to distribute - just package `censorchip.exe` and the user downloads models in-app.

---

## Models

Place `.onnx` files in the `models/` directory (default location shown in the UI). The app lists them automatically - no restart required.

### One-click downloads (built-in)

Open the app, click **Get Models...** and download any of:

| Model | Size | Best for |
|---|---|---|
| NudeNet 320n · Light | ~12 MiB | Real-time on CPU |
| NudeNet 640m · Balanced | ~104 MiB | GPU, higher accuracy |
| YOLO11n-seg · Segmentation | ~14 MiB | General person detection |

### EraX Anti-NSFW (manual conversion required)

[EraX-Anti-NSFW-V1.1](https://huggingface.co/erax-ai/EraX-Anti-NSFW-V1.1) is a high-accuracy NSFW detector distributed as PyTorch weights. Convert it to ONNX before use:

```bash
# 1. Download weights from HuggingFace
#    https://huggingface.co/erax-ai/EraX-Anti-NSFW-V1.1

# 2. Install Ultralytics
pip install ultralytics

# 3. Export to ONNX
yolo export model=erax_anti_nsfw_v1.1.pt format=onnx imgsz=640

# 4. Move the .onnx file into the models/ folder
```

Then click **Refresh models** in the UI and select the exported file.

### Any YOLOv8-v11 model

```bash
pip install ultralytics
yolo export model=yolov8n.pt format=onnx imgsz=640
```

All models must output bounding boxes in YOLO format. See the [label config](src/inference/label_config.rs) to map custom class IDs to detection categories.

---

## Usage

1. **Start** - click **Start** to launch the pipeline. The transparent overlay window opens automatically.
2. **Capture source** - choose *Full screen (DXGI)* for the primary monitor or pick a specific window from the **Refresh window list** dropdown. Window capture works even for minimised or occluded windows.
3. **Model** - select a `.onnx` model from the dropdown. Click **Refresh models** after adding new files.
4. **Censorship style** - Blur, Pixelate, Black Bar, Text Overlay, or Texture Overlay.
5. **Detection toggles** - enable/disable individual body-region labels and gender filters.
6. **Performance preset** - Light / Balanced / Quality adjusts inference skip frames and downscale factor together. Individual sliders can be tuned further.
7. **Overlay toggle** - show/hide the censorship overlay without stopping the pipeline.
8. **Preview** - switch between *Censored*, *Clean*, and *Off* in the central panel.
9. **Save config** - writes current settings to `config.toml`. Settings are also auto-saved on exit.

### Hotkeys (default)

| Key | Action |
|---|---|
| F9 | Toggle censorship on/off |
| F10 | Pause / resume capture |

Rebind in `config.toml` under `[hotkeys]`.

---

## Configuration

`config.toml` is auto-generated on first run next to the executable. A fully annotated template (including EraX settings) is available at [config.example.toml](config.example.toml). Key options:

```toml
[general]
models_dir     = "models"
selected_model = "640m.onnx"

[detection]
genitals  = true
chest     = true
buttocks  = true
face      = false
eyes      = false

[censorship]
method       = "Blur"     # Blur | Pixelation | BlackBar | TextOverlay | TextureOverlay
blur_radius  = 25
block_size   = 20         # Pixelation block size

[performance]
preset                = "Balanced"
gpu_acceleration      = false
inference_skip_frames = 2
downscale_factor      = 2

[hotkeys]
toggle_censorship = "F9"
pause_capture     = "F10"
```

---

## Architecture

```
Capture ──► Preprocess ──► Inference ──► Censor+Render ──► egui UI
(Thread 1)  (Thread 2)     (Thread 3)    (Thread 4)
```

Stages are connected with bounded **crossbeam** channels (lock-free, capacity 2).  
Inference is skipped every N frames and stale detections are propagated with a constant-velocity bounding-box tracker.

### Source layout

```
src/
├── main.rs              Entry point, eframe bootstrap
├── capture/
│   ├── mod.rs           ScreenCapture trait, DXGI + PrintWindow + generic backends
│   └── frame.rs         CapturedFrame (raw RGBA8 buffer + screen-space origin)
├── pipeline/
│   ├── mod.rs           5-stage threaded pipeline, PipelineHandle
│   └── tracker.rs       Constant-velocity bounding-box tracker
├── inference/
│   ├── mod.rs           InferenceEngine trait, DetectionLabel, Detection
│   ├── onnx.rs          ONNX Runtime backend (NMS, letterbox pre/post-processing)
│   ├── model_registry.rs  Scans models/ directory
│   └── label_config.rs  Maps model class IDs to body-region labels
├── censor/
│   └── mod.rs           Blur, Pixelate, BlackBar, TextOverlay, Texture censor effects
├── overlay/
│   └── mod.rs           Win32 layered topmost window, WDA affinity protection
├── ui/
│   └── mod.rs           CensorChipApp (eframe::App), live preview panel
├── config/
│   └── mod.rs           AppConfig, SharedConfig (Arc<RwLock>), load/save helpers
└── downloader.rs        Background HTTPS model downloader with progress state
```

---

## Contributing

Pull requests are welcome. Please open an issue first for large changes.  
All contributions must be compatible with **GPL-3.0-or-later**.

---

## License

CensorChip is free software released under the **GNU General Public License v3** (or later).  
See [LICENSE](LICENSE) for the full text.


[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Build](https://github.com/GammaAdmin/CensorChip/actions/workflows/ci.yml/badge.svg)](https://github.com/GammaAdmin/CensorChip/actions)