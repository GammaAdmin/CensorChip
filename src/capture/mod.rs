// CensorChip – Screen capture module
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Abstracts platform-specific screen capture behind a uniform `ScreenCapture`
// trait.  The `create_capturer()` factory picks the best backend for the host
// OS at compile time.
//
// PERFORMANCE NOTE: The capture thread runs on its own OS thread and pushes
// raw RGBA frames into a lock-free crossbeam channel so the rest of the
// pipeline never blocks on I/O.

pub mod frame;

use anyhow::Result;
use frame::CapturedFrame;

/// Trait that every platform capturer must implement.
pub trait ScreenCapture: Send {
    /// Grab a single frame. Should be as fast as possible.
    fn capture_frame(&mut self) -> Result<CapturedFrame>;

    /// Human-readable backend name (for logging / UI).
    fn backend_name(&self) -> &'static str;
}

// ── Windows: DXGI Desktop Duplication ────────────────────────────────────
//
// The `screenshots` crate uses GDI (StretchBlt) on Windows, which does NOT
// honour SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE).  That means the
// overlay window is captured verbatim, the model sees its own censor patches,
// stops detecting the original content, hides the overlay – and the cycle
// repeats (the visible flicker the user reports).
//
// DXGI Desktop Duplication operates at the DWM compositor level.  DWM
// maintains a separate "capture" image that excludes WDA_EXCLUDEFROMCAPTURE
// windows: the excluded window's area shows the desktop content *behind* it,
// as if the overlay never existed.  This breaks the feedback loop entirely.

#[cfg(target_os = "windows")]
mod dxgi_dup {
    use anyhow::{anyhow, Result};
    use super::{CapturedFrame, ScreenCapture};
    use windows::{
        core::Interface,
        Win32::{
            Foundation::HMODULE,
            Graphics::{
                Direct3D::D3D_DRIVER_TYPE_HARDWARE,
                Direct3D11::{
                    D3D11CreateDevice, ID3D11Device, ID3D11DeviceContext,
                    ID3D11Resource, ID3D11Texture2D,
                    D3D11_CPU_ACCESS_READ, D3D11_CREATE_DEVICE_FLAG,
                    D3D11_MAP_READ, D3D11_MAPPED_SUBRESOURCE,
                    D3D11_SDK_VERSION, D3D11_TEXTURE2D_DESC,
                    D3D11_USAGE_STAGING,
                },
                Dxgi::{
                    Common::{DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_SAMPLE_DESC},
                    IDXGIDevice, IDXGIOutput1, IDXGIOutputDuplication,
                    IDXGIResource, DXGI_OUTDUPL_FRAME_INFO,
                },
            },
        },
    };

    struct Inner {
        ctx:     ID3D11DeviceContext,
        dup:     IDXGIOutputDuplication,
        staging: ID3D11Texture2D,
        width:   u32,
        height:  u32,
    }

    // SAFETY: D3D11/DXGI objects are only accessed from the single capture
    // thread; no concurrent use ever occurs.
    unsafe impl Send for Inner {}

    pub struct DxgiCapturer {
        screen_idx: usize,
        inner: Option<Inner>,
        /// Last successfully captured frame – returned on DXGI timeout so the
        /// pipeline keeps flowing when the desktop hasn't changed.
        last_frame: Option<CapturedFrame>,
    }

    impl DxgiCapturer {
        pub fn new(screen_idx: usize) -> Self {
            Self { screen_idx, inner: None, last_frame: None }
        }
    }

    impl ScreenCapture for DxgiCapturer {
        fn capture_frame(&mut self) -> Result<CapturedFrame> {
            if self.inner.is_none() {
                self.inner = Some(unsafe { init(self.screen_idx)? });
            }
            match unsafe { capture(self.inner.as_ref().unwrap()) } {
                Ok(Some(f)) => {
                    self.last_frame = Some(f.clone());
                    Ok(f)
                }
                Ok(None) => {
                    // DXGI timeout: desktop hasn't changed – reuse last frame.
                    self.last_frame
                        .clone()
                        .ok_or_else(|| anyhow!("DXGI: no frame captured yet"))
                }
                Err(e) => {
                    // Fatal DXGI error (access lost, mode change, etc.)
                    // – discard the duplication handle; next call re-inits.
                    self.inner = None;
                    self.last_frame = None;
                    Err(e)
                }
            }
        }

        fn backend_name(&self) -> &'static str {
            "DXGI Desktop Duplication"
        }
    }

    unsafe fn init(screen_idx: usize) -> Result<Inner> {
        let mut device: Option<ID3D11Device> = None;
        let mut ctx: Option<ID3D11DeviceContext> = None;
        D3D11CreateDevice(
            None,
            D3D_DRIVER_TYPE_HARDWARE,
            HMODULE::default(),
            D3D11_CREATE_DEVICE_FLAG(0),
            None,
            D3D11_SDK_VERSION,
            Some(&mut device),
            None,
            Some(&mut ctx),
        )?;
        let device = device.ok_or_else(|| anyhow!("D3D11CreateDevice: null device"))?;
        let ctx    = ctx.ok_or_else(|| anyhow!("D3D11CreateDevice: null context"))?;

        let dxgi_dev: IDXGIDevice = device.cast()?;
        let adapter = dxgi_dev.GetAdapter()?;
        let output  = adapter.EnumOutputs(screen_idx as u32)?;
        let out1: IDXGIOutput1 = output.cast()?;

        let dup = out1.DuplicateOutput(&device)?;

        // Use the actual frame dimensions from the duplication descriptor, not
        // the logical desktop coordinates – they may differ under DPI scaling.
        let dup_desc = dup.GetDesc();
        let width  = dup_desc.ModeDesc.Width;
        let height = dup_desc.ModeDesc.Height;

        // CPU-readable staging texture – same format as the desktop surface.
        let tex_desc = D3D11_TEXTURE2D_DESC {
            Width:          width,
            Height:         height,
            MipLevels:      1,
            ArraySize:      1,
            Format:         DXGI_FORMAT_B8G8R8A8_UNORM,
            SampleDesc:     DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Usage:          D3D11_USAGE_STAGING,
            BindFlags:      0,
            CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
            MiscFlags:      0,
        };
        let mut staging_opt: Option<ID3D11Texture2D> = None;
        device.CreateTexture2D(&tex_desc, None, Some(&mut staging_opt))?;
        let staging = staging_opt.ok_or_else(|| anyhow!("CreateTexture2D: null staging"))?;

        Ok(Inner { ctx, dup, staging, width, height })
    }

    unsafe fn capture(inner: &Inner) -> Result<Option<CapturedFrame>> {
        // DXGI_ERROR_WAIT_TIMEOUT (0x887A0027) – no new desktop frame in the
        // given window.  This is NOT a fatal error; it just means the desktop
        // hasn't changed.  Do NOT treat it as a reason to re-create the whole
        // duplication handle – that would cause an expensive re-init storm on
        // any quiet (non-animating) desktop.
        const WAIT_TIMEOUT: i32 = 0x887A0027u32 as i32;

        let mut info = DXGI_OUTDUPL_FRAME_INFO::default();
        let mut res: Option<IDXGIResource> = None;

        match inner.dup.AcquireNextFrame(16, &mut info, &mut res) {
            Ok(_) => {}
            Err(e) if e.code().0 == WAIT_TIMEOUT => return Ok(None),
            Err(e) => return Err(anyhow::anyhow!("AcquireNextFrame: {e}")),
        }

        // ReleaseFrame MUST be called after every successful AcquireNextFrame.
        let result = copy_frame(inner, res);
        let _ = inner.dup.ReleaseFrame();
        result.map(Some)
    }

    unsafe fn copy_frame(
        inner: &Inner,
        res: Option<IDXGIResource>,
    ) -> Result<CapturedFrame> {
        let texture: ID3D11Texture2D = res
            .ok_or_else(|| anyhow!("AcquireNextFrame: no desktop resource"))?
            .cast()?;

        // GPU copy: desktop texture → CPU-accessible staging texture.
        let staging_res: ID3D11Resource = inner.staging.cast()?;
        let texture_res: ID3D11Resource = texture.cast()?;
        inner.ctx.CopyResource(&staging_res, &texture_res);

        // Map the staging texture for CPU read-back.
        let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
        inner.ctx.Map(&staging_res, 0, D3D11_MAP_READ, 0, Some(&mut mapped))?;

        let w     = inner.width  as usize;
        let h     = inner.height as usize;
        let pitch = mapped.RowPitch as usize;
        let src   = std::slice::from_raw_parts(mapped.pData as *const u8, pitch * h);

        // BGRA (DXGI native) → RGBA (pipeline convention).
        let mut rgba = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let si = y * pitch + x * 4;
                let di = (y * w + x) * 4;
                rgba[di]     = src[si + 2]; // R ← B
                rgba[di + 1] = src[si + 1]; // G
                rgba[di + 2] = src[si];     // B ← R
                rgba[di + 3] = 255;         // A – desktop surface is opaque
            }
        }

        inner.ctx.Unmap(&staging_res, 0);
        Ok(CapturedFrame::new(rgba, inner.width, inner.height))
    }
}

// ── Windows: per-window capture via PrintWindow ───────────────────────────
//
// PrintWindow asks the target window to re-render itself into a GDI
// compatible bitmap.  Crucially it reads straight from the window's own
// rendering pipeline BEFORE DWM composition, so:
//   • Overlays drawn on top in screen coordinates are invisible.
//   • WDA_EXCLUDEFROMCAPTURE / VM / RDP restrictions are irrelevant.
//   • The window doesn't even have to be visible or on top.
//
// PW_RENDERFULLCONTENT (0x2) forces GPU-accelerated content (D3D, WebGL …)
// to be included in addition to classic GDI content.

#[cfg(target_os = "windows")]
mod win_capture {
    use anyhow::{anyhow, Result};
    use super::{CapturedFrame, ScreenCapture};
    use std::mem;

    type HWND  = isize;
    type HDC   = isize;
    type HBITMAP = isize;

    #[repr(C)]
    #[allow(non_snake_case)]
    struct BITMAPINFOHEADER {
        biSize:          u32,
        biWidth:         i32,
        biHeight:        i32,
        biPlanes:        u16,
        biBitCount:      u16,
        biCompression:   u32, // BI_RGB = 0
        biSizeImage:     u32,
        biXPelsPerMeter: i32,
        biYPelsPerMeter: i32,
        biClrUsed:       u32,
        biClrImportant:  u32,
    }

    #[repr(C)]
    #[allow(non_snake_case)]
    struct BITMAPINFO {
        bmiHeader: BITMAPINFOHEADER,
        bmiColors: [u32; 1],
    }

    #[repr(C)]
    struct RECT { left: i32, top: i32, right: i32, bottom: i32 }

    #[repr(C)]
    struct POINT { x: i32, y: i32 }

    #[link(name = "user32")]
    extern "system" {
        fn GetClientRect(hWnd: HWND, lpRect: *mut RECT) -> i32;
        fn ClientToScreen(hWnd: HWND, lpPoint: *mut POINT) -> i32;
        fn GetDC(hWnd: HWND) -> HDC;
        fn ReleaseDC(hWnd: HWND, hDC: HDC) -> i32;
        fn PrintWindow(hWnd: HWND, hdcBlt: HDC, nFlags: u32) -> i32;
        fn IsWindowVisible(hWnd: HWND) -> i32;
        fn IsWindow(hWnd: HWND) -> i32;
        fn EnumWindows(
            lpEnumFunc: unsafe extern "system" fn(isize, isize) -> i32,
            lParam: isize,
        ) -> i32;
        fn InternalGetWindowText(hWnd: isize, pString: *mut u16, cchMaxCount: i32) -> i32;
    }

    #[link(name = "gdi32")]
    extern "system" {
        fn CreateCompatibleDC(hdc: HDC) -> HDC;
        fn CreateDIBSection(
            hdc: HDC, pbmi: *const BITMAPINFO, usage: u32,
            ppvBits: *mut *mut u8, hSection: isize, offset: u32,
        ) -> HBITMAP;
        fn SelectObject(hdc: HDC, h: isize) -> isize;
        fn DeleteObject(ho: isize) -> i32;
        fn DeleteDC(hdc: HDC) -> i32;
    }

    const DIB_RGB_COLORS:       u32 = 0;
    const PW_RENDERFULLCONTENT: u32 = 0x2;

    pub struct WindowCapturer {
        title: String,
        /// Cached HWND so title changes (browser tab navigation) don't
        /// break capture — the HWND stays valid as long as the window lives.
        cached_hwnd: Option<HWND>,
    }

    impl WindowCapturer {
        pub fn new(title: String) -> Self {
            Self { title, cached_hwnd: None }
        }

        /// Find the first visible window whose title *contains* `self.title`
        /// as a case-insensitive substring.  This handles browsers / players
        /// that prepend/append the document name to the window title.
        fn find_hwnd(&self) -> Result<HWND> {
            // Pack the search needle + output slot into an lParam struct.
            struct Ctx { needle: String, result: HWND }

            unsafe extern "system" fn find_proc(hwnd: isize, lparam: isize) -> i32 {
                let ctx = unsafe { &mut *(lparam as *mut Ctx) };
                if unsafe { IsWindowVisible(hwnd) } == 0 { return 1; }
                let mut buf = [0u16; 512];
                let len = unsafe {
                    InternalGetWindowText(hwnd, buf.as_mut_ptr(), buf.len() as i32)
                };
                if len > 0 {
                    let title = String::from_utf16_lossy(&buf[..len as usize]);
                    if title.to_lowercase().contains(&ctx.needle.to_lowercase()) {
                        ctx.result = hwnd;
                        return 0; // stop enumeration
                    }
                }
                1
            }

            let mut ctx = Ctx { needle: self.title.clone(), result: 0 };
            unsafe { EnumWindows(find_proc, &mut ctx as *mut Ctx as isize) };

            if ctx.result == 0 {
                Err(anyhow!(
                    "No visible window whose title contains \"{}\". \
                     Use the dropdown in Capture Source to pick from currently open windows.",
                    self.title
                ))
            } else {
                Ok(ctx.result)
            }
        }
    }

    impl ScreenCapture for WindowCapturer {
        fn capture_frame(&mut self) -> Result<CapturedFrame> {
            // Reuse the cached HWND while the window is still alive.
            // Tab navigation and page-title changes don't create a new HWND,
            // so this keeps capture running even when the title no longer
            // matches the user's search string.
            if let Some(hwnd) = self.cached_hwnd {
                if unsafe { IsWindow(hwnd) } != 0 {
                    return unsafe { capture_window(hwnd) };
                }
                // Window was closed — discard and fall through to re-search.
                self.cached_hwnd = None;
            }
            let hwnd = self.find_hwnd()?;
            self.cached_hwnd = Some(hwnd);
            unsafe { capture_window(hwnd) }
        }

        fn backend_name(&self) -> &'static str {
            "PrintWindow (window capture)"
        }
    }

    unsafe fn capture_window(hwnd: HWND) -> Result<CapturedFrame> {
        let mut rc: RECT = mem::zeroed();
        if GetClientRect(hwnd, &mut rc) == 0 {
            return Err(anyhow!("GetClientRect failed"));
        }
        let w = (rc.right  - rc.left) as i32;
        let h = (rc.bottom - rc.top)  as i32;
        if w <= 0 || h <= 0 {
            return Err(anyhow!("Window has zero client area (minimised?)"));
        }

        let hdc_screen = GetDC(0);
        let hdc_mem    = CreateCompatibleDC(hdc_screen);

        let bi = BITMAPINFO {
            bmiHeader: BITMAPINFOHEADER {
                biSize:          mem::size_of::<BITMAPINFOHEADER>() as u32,
                biWidth:         w,
                biHeight:        -h, // negative = top-down
                biPlanes:        1,
                biBitCount:      32,
                biCompression:   0, // BI_RGB
                biSizeImage:     0,
                biXPelsPerMeter: 0,
                biYPelsPerMeter: 0,
                biClrUsed:       0,
                biClrImportant:  0,
            },
            bmiColors: [0],
        };

        let mut bits: *mut u8 = std::ptr::null_mut();
        let hbm = CreateDIBSection(hdc_screen, &bi, DIB_RGB_COLORS, &mut bits, 0, 0);
        if hbm == 0 || bits.is_null() {
            ReleaseDC(0, hdc_screen);
            DeleteDC(hdc_mem);
            return Err(anyhow!("CreateDIBSection failed"));
        }
        SelectObject(hdc_mem, hbm);

        // Ask the window to render itself into our DC.
        // PW_RENDERFULLCONTENT includes D3D / layered child content.
        let ok = PrintWindow(hwnd, hdc_mem, PW_RENDERFULLCONTENT);

        let result = if ok != 0 {
            let byte_count = (w as usize) * (h as usize) * 4;
            let src = std::slice::from_raw_parts(bits as *const u8, byte_count);

            // DIB is BGRA – convert to RGBA and force A=255.
            let mut rgba = vec![0u8; byte_count];
            for i in (0..byte_count).step_by(4) {
                rgba[i]     = src[i + 2]; // R
                rgba[i + 1] = src[i + 1]; // G
                rgba[i + 2] = src[i];     // B
                rgba[i + 3] = 255;
            }
            let mut origin = POINT { x: 0, y: 0 };
            ClientToScreen(hwnd, &mut origin);
            let mut frame = CapturedFrame::new(rgba, w as u32, h as u32);
            frame.screen_x = origin.x;
            frame.screen_y = origin.y;
            Ok(frame)
        } else {
            Err(anyhow!("PrintWindow returned 0 (rendering failed)"))
        };

        DeleteObject(hbm);
        DeleteDC(hdc_mem);
        ReleaseDC(0, hdc_screen);
        result
    }
}

// ── Platform backend (screenshots crate – cross-platform) ────────────────

/// Cross-platform capturer powered by the `screenshots` crate.
/// Works on Windows (DXGI), Linux (X11/PipeWire) and macOS (CGDisplay).
pub struct GenericCapturer {
    screen_idx: usize,
    /// Cached screen object – avoids re-enumerating monitors (and recreating
    /// DXGI Desktop Duplication) on every single frame.
    cached_screen: Option<screenshots::Screen>,
}

impl GenericCapturer {
    pub fn new(screen_idx: usize) -> Self {
        Self { screen_idx, cached_screen: None }
    }

    /// Lazily initialise (or re-initialise on error) the cached screen.
    fn get_or_init_screen(&mut self) -> Result<&screenshots::Screen> {
        if self.cached_screen.is_none() {
            use screenshots::Screen;
            let screens = Screen::all()
                .map_err(|e| anyhow::anyhow!("enumerate screens: {e}"))?;
            let screen = screens
                .into_iter()
                .nth(self.screen_idx)
                .ok_or_else(|| anyhow::anyhow!("screen index {} out of range", self.screen_idx))?;
            self.cached_screen = Some(screen);
        }
        Ok(self.cached_screen.as_ref().unwrap())
    }
}

impl ScreenCapture for GenericCapturer {
    fn capture_frame(&mut self) -> Result<CapturedFrame> {
        let img = match self.get_or_init_screen() {
            Ok(screen) => match screen.capture() {
                Ok(img) => img,
                Err(e) => {
                    // On DXGI errors the cached screen may be stale — drop it so
                    // we re-create on the next call.
                    self.cached_screen = None;
                    return Err(anyhow::anyhow!("screen capture failed: {e}"));
                }
            },
            Err(e) => return Err(e),
        };

        let width = img.width();
        let height = img.height();
        let rgba = img.into_raw();

        let expected = (width as usize) * (height as usize) * 4;
        if rgba.len() != expected {
            self.cached_screen = None; // re-init next frame
            anyhow::bail!(
                "frame data length mismatch: got {} bytes, expected {} ({}×{}×4)",
                rgba.len(), expected, width, height
            );
        }

        Ok(CapturedFrame::new(rgba, width, height))
    }

    fn backend_name(&self) -> &'static str {
        #[cfg(target_os = "windows")]
        {
            "GDI (screenshots)"
        }
        #[cfg(target_os = "linux")]
        {
            "X11/PipeWire (screenshots)"
        }
        #[cfg(target_os = "macos")]
        {
            "CGDisplay (screenshots)"
        }
        #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
        {
            "screenshots"
        }
    }
}

// ── Window enumeration (for the UI dropdown) ────────────────────────────
//
// list_windows() produces a sorted list of every Alt-Tab–style window for
// the Capture Source dropdown.
//
// WHY InternalGetWindowText instead of GetWindowTextW:
//   GetWindowTextW works by posting WM_GETTEXT to the target window – this
//   fails silently (returns 0 chars) when the target runs at a higher
//   integrity level (UAC elevation).  InternalGetWindowText reads the title
//   directly from the kernel's window structure and is unaffected by UAC.
//
// WHY GetWindowLongW / WS_EX_TOOLWINDOW:
//   EnumWindows returns ALL top-level windows, including system tray pop-ups,
//   tooltip holders, and DWM ghost windows.  We mimic the Alt-Tab list:
//   exclude WS_EX_TOOLWINDOW windows unless they also bear WS_EX_APPWINDOW.
//
// WHY lParam instead of thread_local:
//   The lParam pointer pattern is the canonical Win32 way to collect results
//   in an EnumWindows callback.  No thread juggling, no static state.

#[cfg(target_os = "windows")]
mod win_enum {
    pub const GWL_EXSTYLE:       i32 = -20;
    pub const WS_EX_TOOLWINDOW:  i32 = 0x0000_0080;
    pub const WS_EX_APPWINDOW:   i32 = 0x0004_0000;

    #[link(name = "user32")]
    extern "system" {
        pub fn EnumWindows(
            lpEnumFunc: unsafe extern "system" fn(isize, isize) -> i32,
            lParam: isize,
        ) -> i32;
        pub fn IsWindowVisible(hWnd: isize) -> i32;
        pub fn GetWindowLongW(hWnd: isize, nIndex: i32) -> i32;
        /// Reads window title directly from kernel – works for elevated processes.
        pub fn InternalGetWindowText(hWnd: isize, pString: *mut u16, cchMaxCount: i32) -> i32;
    }
}

#[cfg(target_os = "windows")]
unsafe extern "system" fn _list_enum_proc(hwnd: isize, lparam: isize) -> i32 {
    use win_enum::*;
    if unsafe { IsWindowVisible(hwnd) } == 0 {
        return 1;
    }
    let ex = unsafe { GetWindowLongW(hwnd, GWL_EXSTYLE) };
    let is_tool = (ex & WS_EX_TOOLWINDOW) != 0;
    let is_app  = (ex & WS_EX_APPWINDOW)  != 0;
    if is_tool && !is_app {
        return 1;
    }
    let mut buf = [0u16; 512];
    let len = unsafe { InternalGetWindowText(hwnd, buf.as_mut_ptr(), buf.len() as i32) };
    if len > 0 {
        let title = String::from_utf16_lossy(&buf[..len as usize])
            .trim()
            .to_string();
        if !title.is_empty() {
            let out = unsafe { &mut *(lparam as *mut Vec<String>) };
            out.push(title);
        }
    }
    1
}

/// Return a sorted, deduplicated list of Alt-Tab–style visible window titles.
/// Uses `InternalGetWindowText` so elevated-process windows are included.
pub fn list_windows() -> Vec<String> {
    #[cfg(target_os = "windows")]
    {
        let mut titles: Vec<String> = Vec::new();
        unsafe {
            win_enum::EnumWindows(
                _list_enum_proc,
                &mut titles as *mut Vec<String> as isize,
            )
        };
        titles.sort_unstable();
        titles.dedup();
        titles
    }
    #[cfg(not(target_os = "windows"))]
    {
        Vec::new()
    }
}


/// Factory – picks the best capturer for the host platform.
/// Pass a window title to capture only that window (works even when a
/// transparent overlay sits on top of it).  Pass `None` to capture the
/// full desktop via DXGI Desktop Duplication.
pub fn create_capturer(screen_idx: usize, window_title: Option<&str>) -> Box<dyn ScreenCapture> {
    // On Windows, prefer the window-specific PrintWindow backend when a title
    // is provided – it reads directly from the GPU surface of that window and
    // is completely unaffected by anything drawn on top (overlays, other
    // windows, WDA_EXCLUDEFROMCAPTURE failures, VM / RDP limitations, etc.).
    #[cfg(target_os = "windows")]
    {
        if let Some(title) = window_title {
            return Box::new(win_capture::WindowCapturer::new(title.to_string()));
        }
        Box::new(dxgi_dup::DxgiCapturer::new(screen_idx))
    }
    #[cfg(not(target_os = "windows"))]
    {
        let _ = window_title;
        Box::new(GenericCapturer::new(screen_idx))
    }
}
