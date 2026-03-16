// CensorChip – Win32 layered-window overlay
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Provides a borderless, always-on-top, fully-transparent Win32 window that
// draws only the censored patches over the live desktop.
//
// APPROACH: `UpdateLayeredWindow` with `ULW_ALPHA` + `AC_SRC_ALPHA`.
// This is the only fully reliable way to achieve per-pixel alpha on Windows
// 10/11 without requiring DirectComposition or a special DXGI swap chain.
// The overlay window is WS_EX_TRANSPARENT so all mouse/keyboard events pass
// through to whatever is behind it.

use crossbeam_channel::{bounded, Receiver, Sender};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

/// An RGBA frame to paint on the overlay.
struct FrameMsg {
    pixels: Vec<u8>,
    width:  u32,
    height: u32,
    /// Top-left screen corner at which to blit the frame.
    offset_x: i32,
    offset_y: i32,
}

/// Manages the persistent Win32 layered overlay window.  Drop to destroy.
pub struct Win32Overlay {
    tx: Sender<Option<FrameMsg>>,
    /// `true` once `SetWindowDisplayAffinity` succeeds (either
    /// WDA_EXCLUDEFROMCAPTURE or the WDA_MONITOR fallback).
    /// The UI reads this to decide whether to show a warning banner.
    pub affinity_ok: Arc<AtomicBool>,
}

impl Win32Overlay {
    /// Spawn the background window thread and return a handle.
    pub fn launch() -> Self {
        let (tx, rx) = bounded::<Option<FrameMsg>>(1);
        let affinity_ok = Arc::new(AtomicBool::new(false));
        let aff = affinity_ok.clone();
        thread::Builder::new()
            .name("win32-overlay".into())
            .spawn(move || {
                #[cfg(windows)]
                unsafe { run_thread(rx, aff) };
                #[cfg(not(windows))]
                drop(rx);
            })
            .expect("failed to spawn win32-overlay thread");
        Self { tx, affinity_ok }
    }

    /// Send a new RGBA overlay frame (non-blocking; stale frames are dropped).
    pub fn update_frame(&self, pixels: &[u8], width: u32, height: u32, offset_x: i32, offset_y: i32) {
        let _ = self.tx.try_send(Some(FrameMsg {
            pixels: pixels.to_vec(),
            width,
            height,
            offset_x,
            offset_y,
        }));
    }

    /// Make the overlay invisible (clears to fully transparent).
    pub fn hide(&self) {
        let _ = self.tx.try_send(None);
    }
}

impl Drop for Win32Overlay {
    fn drop(&mut self) {
        // Tell the thread to exit by closing the channel.
        // try_send(None) ensures the DIB is cleared before the thread exits.
        let _ = self.tx.try_send(None);
    }
}

// ── Windows implementation ───────────────────────────────────────────────

#[cfg(windows)]
#[allow(non_snake_case, non_camel_case_types, clippy::upper_case_acronyms)]
unsafe fn run_thread(rx: Receiver<Option<FrameMsg>>, affinity_ok: Arc<AtomicBool>) {
    use std::mem;

    // ── Win32 types ──────────────────────────────────────────────────────
    type WNDPROC = unsafe extern "system" fn(isize, u32, usize, isize) -> isize;

    #[repr(C)]
    struct WNDCLASSEXW {
        cbSize: u32,
        style: u32,
        lpfnWndProc: Option<WNDPROC>,
        cbClsExtra: i32,
        cbWndExtra: i32,
        hInstance: isize,
        hIcon: isize,
        hCursor: isize,
        hbrBackground: isize,
        lpszMenuName: *const u16,
        lpszClassName: *const u16,
        hIconSm: isize,
    }

    #[repr(C)] struct POINT { x: i32, y: i32 }
    #[repr(C)] struct SIZE  { cx: i32, cy: i32 }

    #[repr(C)]
    struct BLENDFUNCTION {
        BlendOp: u8,
        BlendFlags: u8,
        SourceConstantAlpha: u8,
        AlphaFormat: u8,
    }

    #[repr(C)]
    struct BITMAPINFOHEADER {
        biSize: u32,
        biWidth: i32,
        biHeight: i32,
        biPlanes: u16,
        biBitCount: u16,
        biCompression: u32,
        biSizeImage: u32,
        biXPelsPerMeter: i32,
        biYPelsPerMeter: i32,
        biClrUsed: u32,
        biClrImportant: u32,
    }

    #[repr(C)]
    struct BITMAPINFO {
        bmiHeader: BITMAPINFOHEADER,
        bmiColors: [u32; 1],
    }

    #[repr(C)]
    struct MSG {
        hwnd: isize,
        message: u32,
        wParam: usize,
        lParam: isize,
        time: u32,
        pt: POINT,
        lPrivate: u32,
    }

    // ── External declarations ────────────────────────────────────────────
    #[link(name = "user32")]
    extern "system" {
        fn GetModuleHandleW(lpModuleName: *const u16) -> isize;
        fn RegisterClassExW(lpwcx: *const WNDCLASSEXW) -> u16;
        fn CreateWindowExW(
            dwExStyle: u32,
            lpClassName: *const u16,
            lpWindowName: *const u16,
            dwStyle: u32,
            X: i32, Y: i32, nWidth: i32, nHeight: i32,
            hWndParent: isize,
            hMenu: isize,
            hInstance: isize,
            lpParam: *mut std::ffi::c_void,
        ) -> isize;
        fn ShowWindow(hWnd: isize, nCmdShow: i32) -> i32;
        fn DestroyWindow(hWnd: isize) -> i32;
        fn DefWindowProcW(hWnd: isize, Msg: u32, wParam: usize, lParam: isize) -> isize;
        fn GetDC(hWnd: isize) -> isize;
        fn ReleaseDC(hWnd: isize, hDC: isize) -> i32;
        fn UpdateLayeredWindow(
            hWnd: isize,
            hdcDst: isize,
            pptDst: *const POINT,
            psize: *const SIZE,
            hdcSrc: isize,
            pptSrc: *const POINT,
            crKey: u32,
            pblend: *const BLENDFUNCTION,
            dwFlags: u32,
        ) -> i32;
        fn PeekMessageW(
            lpMsg: *mut MSG,
            hWnd: isize,
            wMsgFilterMin: u32,
            wMsgFilterMax: u32,
            wRemoveMsg: u32,
        ) -> i32;
        fn TranslateMessage(lpMsg: *const MSG) -> i32;
        fn DispatchMessageW(lpMsg: *const MSG) -> isize;
        fn GetSystemMetrics(nIndex: i32) -> i32;
        fn SetWindowDisplayAffinity(hWnd: isize, nAffinity: u32) -> i32;
        fn GetLastError() -> u32;
    }

    #[link(name = "gdi32")]
    extern "system" {
        fn CreateCompatibleDC(hdc: isize) -> isize;
        fn CreateDIBSection(
            hdc: isize,
            pbmi: *const BITMAPINFO,
            usage: u32,
            ppvBits: *mut *mut u8,
            hSection: isize,
            offset: u32,
        ) -> isize;
        fn SelectObject(hdc: isize, h: isize) -> isize;
        fn DeleteObject(ho: isize) -> i32;
        fn DeleteDC(hdc: isize) -> i32;
    }

    #[link(name = "ntdll")]
    extern "system" {
        /// Retrieves the real Windows version bypassing the app-compat shimming
        /// that makes GetVersionEx lie on Windows 8.1+.
        fn RtlGetVersion(lpVersionInformation: *mut OsVersionInfoW) -> u32;
    }

    #[repr(C)]
    struct OsVersionInfoW {
        dwOSVersionInfoSize: u32,
        dwMajorVersion:      u32,
        dwMinorVersion:      u32,
        dwBuildNumber:       u32,
        dwPlatformId:        u32,
        szCSDVersion:        [u16; 128],
    }

    // ── Constants ────────────────────────────────────────────────────────
    const WS_POPUP:          u32 = 0x80000000;
    const WS_EX_LAYERED:     u32 = 0x00080000;
    const WS_EX_TRANSPARENT: u32 = 0x00000020;
    const WS_EX_TOPMOST:     u32 = 0x00000008;
    const WS_EX_NOACTIVATE:  u32 = 0x08000000;
    const WS_EX_TOOLWINDOW:  u32 = 0x00000080;
    const SW_SHOWNOACTIVATE: i32 = 4;
    const ULW_ALPHA:         u32 = 2;
    const AC_SRC_OVER:       u8  = 0;
    const AC_SRC_ALPHA:      u8  = 1;
    const BI_RGB:            u32 = 0;
    const DIB_RGB_COLORS:    u32 = 0;
    const SM_CXSCREEN:       i32 = 0;
    const SM_CYSCREEN:       i32 = 1;
    const PM_REMOVE:         u32 = 1;
    const WM_QUIT:           u32 = 0x0012;
    const SM_REMOTESESSION:  i32 = 0x1000; // non-zero when running over RDP/Citrix
    const WDA_MONITOR:            u32 = 0x00000001;
    const WDA_EXCLUDEFROMCAPTURE: u32 = 0x00000011;

    // ── Create the overlay window ─────────────────────────────────────────
    let hinstance = GetModuleHandleW(std::ptr::null());

    let class_name:  Vec<u16> = "CensorChipOvClass\0".encode_utf16().collect();
    let window_name: Vec<u16> = "CensorChip Overlay\0".encode_utf16().collect();

    let wc = WNDCLASSEXW {
        cbSize:        mem::size_of::<WNDCLASSEXW>() as u32,
        style:         0,
        lpfnWndProc:   Some(DefWindowProcW),
        cbClsExtra:    0,
        cbWndExtra:    0,
        hInstance:     hinstance,
        hIcon:         0,
        hCursor:       0,
        hbrBackground: 0,
        lpszMenuName:  std::ptr::null(),
        lpszClassName: class_name.as_ptr(),
        hIconSm:       0,
    };
    RegisterClassExW(&wc);

    let sw = GetSystemMetrics(SM_CXSCREEN);
    let sh = GetSystemMetrics(SM_CYSCREEN);

    let hwnd = CreateWindowExW(
        WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST
            | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW,
        class_name.as_ptr(),
        window_name.as_ptr(),
        WS_POPUP,
        0, 0, sw, sh,
        0, 0, hinstance,
        std::ptr::null_mut(),
    );

    if hwnd == 0 {
        log::error!("[win32-overlay] CreateWindowExW failed");
        return;
    }

    // Exclude the overlay from screen capture so the pipeline sees the real desktop.
    // Gather Windows version info first – it drives which API is available and
    // produces actionable diagnostic messages when the call fails.
    let mut ver: OsVersionInfoW = mem::zeroed();
    ver.dwOSVersionInfoSize = mem::size_of::<OsVersionInfoW>() as u32;
    RtlGetVersion(&mut ver); // always succeeds (ntdll, no compat shim)
    let build = ver.dwBuildNumber;
    let is_rdp = GetSystemMetrics(SM_REMOTESESSION) != 0;

    let aff_ok = SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE);
    if aff_ok == 0 {
        let err = GetLastError();
        // Error 8 = ERROR_NOT_ENOUGH_MEMORY – confusingly returned by DWM when the
        // window affinity sub-system is unavailable (VM graphics driver, RDP, or an
        // OS build that doesn't support the flag).
        if is_rdp {
            log::warn!(
                "[win32-overlay] SetWindowDisplayAffinity failed \
                 (err {err}, Windows build {build}): Remote Desktop / Citrix session – \
                 WDA_EXCLUDEFROMCAPTURE is not supported over RDP. \
                 External recorders on the physical host will NOT see the overlay. \
                 In-session recorders (OBS inside the RDP window) will see it. \
                 Pipeline feedback-loop protection (detection-hold + frame masking) is active."
            );
        } else if build < 19041 {
            // WDA_EXCLUDEFROMCAPTURE requires Windows 10 version 2004 (build 19041).
            // Try WDA_MONITOR instead – available since Vista – which shows a solid
            // black rectangle to capturers rather than the overlay content.
            log::warn!(
                "[win32-overlay] WDA_EXCLUDEFROMCAPTURE unsupported on build {build} \
                 (requires 19041 / Windows 10 2004+, err {err}). \
                 Trying WDA_MONITOR (black-out to recorders)…"
            );
            let mon_ok = SetWindowDisplayAffinity(hwnd, WDA_MONITOR);
            if mon_ok == 0 {
                log::warn!(
                    "[win32-overlay] WDA_MONITOR also failed (err {}). \
                     Overlay will be visible to screen recorders. \
                     Pipeline feedback-loop is still blocked by detection-hold + frame masking. \
                     ▶ Recorder tip: in OBS use 'Window Capture' on the target app \
                     instead of 'Display Capture'.",
                    GetLastError()
                );
            } else {                affinity_ok.store(true, Ordering::Relaxed);                log::info!("[win32-overlay] WDA_MONITOR active – overlay shows as black bars to recorders");
            }
        } else {
            // build >= 19041 but EXCLUDEFROMCAPTURE still failed.
            // Most common cause: running inside a VM (Hyper-V, VMware, VirtualBox)
            // whose virtual GPU driver does not fully implement DWM composition
            // (error 8 = DWM affinity subsystem unavailable).
            log::warn!(
                "[win32-overlay] WDA_EXCLUDEFROMCAPTURE failed (err {err}, build {build}). \
                 Likely cause: VM with a virtual/para-virtual GPU (Hyper-V basic session, \
                 VMware SVGA, VirtualBox VBGA) where DWM affinity is not implemented. \
                 Trying WDA_MONITOR fallback…"
            );
            let mon_ok = SetWindowDisplayAffinity(hwnd, WDA_MONITOR);
            if mon_ok == 0 {
                log::warn!(
                    "[win32-overlay] WDA_MONITOR also failed (err {}). \
                     Overlay is visible to screen recorders. \
                     ▶ Recorder tip: in OBS use 'Window Capture' on the target app \
                     instead of 'Display Capture'. \
                     Pipeline feedback-loop is still blocked by detection-hold + frame masking.",
                    GetLastError()
                );
            } else {
                affinity_ok.store(true, Ordering::Relaxed);
                log::info!("[win32-overlay] WDA_MONITOR active - overlay shows as black to recorders");
            }
        }
    } else {
        affinity_ok.store(true, Ordering::Relaxed);
        log::info!(
            "[win32-overlay] WDA_EXCLUDEFROMCAPTURE set (build {build}) – \
             overlay fully hidden from DXGI/WGC screen recorders"
        );
    }

    ShowWindow(hwnd, SW_SHOWNOACTIVATE);

    // ── Allocate a persistent DIB ─────────────────────────────────────────
    // The DIB is created once at screen size and reused every frame.
    let hdc_screen = GetDC(0);
    let hdc_mem    = CreateCompatibleDC(hdc_screen);

    let mut dib_ptr: *mut u8 = std::ptr::null_mut();
    let bi = BITMAPINFO {
        bmiHeader: BITMAPINFOHEADER {
            biSize:          mem::size_of::<BITMAPINFOHEADER>() as u32,
            biWidth:         sw,
            biHeight:        -sh,  // negative = top-down DIB
            biPlanes:        1,
            biBitCount:      32,
            biCompression:   BI_RGB,
            biSizeImage:     0,
            biXPelsPerMeter: 0,
            biYPelsPerMeter: 0,
            biClrUsed:       0,
            biClrImportant:  0,
        },
        bmiColors: [0],
    };

    let hbm = CreateDIBSection(
        hdc_screen, &bi, DIB_RGB_COLORS, &mut dib_ptr, 0, 0,
    );
    if hbm == 0 || dib_ptr.is_null() {
        log::error!("[win32-overlay] CreateDIBSection failed");
        ReleaseDC(0, hdc_screen);
        DeleteDC(hdc_mem);
        DestroyWindow(hwnd);
        return;
    }
    SelectObject(hdc_mem, hbm);

    let dib_bytes      = (sw * sh * 4) as usize;
    let dib_row_stride = sw as usize * 4;

    // Start fully transparent.
    std::ptr::write_bytes(dib_ptr, 0, dib_bytes);

    let blend  = BLENDFUNCTION { BlendOp: AC_SRC_OVER, BlendFlags: 0,
                                  SourceConstantAlpha: 255, AlphaFormat: AC_SRC_ALPHA };
    let pt_src = POINT { x: 0, y: 0 };
    let pt_dst = POINT { x: 0, y: 0 };
    let win_sz = SIZE  { cx: sw, cy: sh };

    UpdateLayeredWindow(
        hwnd, hdc_screen, &pt_dst, &win_sz,
        hdc_mem, &pt_src, 0, &blend, ULW_ALPHA,
    );

    // ── Event / render loop ───────────────────────────────────────────────
    // We sleep 8 ms per iteration (~120 fps cap) and drain the channel on
    // each wake to pick up the latest frame.  Win32 messages are pumped to
    // keep the window alive.
    let mut last_frame: Option<FrameMsg> = None;
    let mut dirty = true; // initial clear already done above

    'outer: loop {
        // Pump pending Win32 messages.
        let mut msg: MSG = mem::zeroed();
        while PeekMessageW(&mut msg, 0, 0, 0, PM_REMOVE) != 0 {
            if msg.message == WM_QUIT { break 'outer; }
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }

        // Drain the channel – keep only the most recent message.
        loop {
            match rx.try_recv() {
                Ok(frame_opt) => {
                    last_frame = frame_opt;
                    dirty = true;
                }
                Err(crossbeam_channel::TryRecvError::Empty) => break,
                Err(crossbeam_channel::TryRecvError::Disconnected) => {
                    break 'outer;
                }
            }
        }

        if dirty {
            // Clear the DIB to fully transparent.
            std::ptr::write_bytes(dib_ptr, 0, dib_bytes);

            if let Some(ref frame) = last_frame {
                // RGBA (pipeline) → BGRA (GDI DIB) with pre-multiplied alpha.
                // Our overlay pixels are either alpha=0 (transparent) or
                // alpha=255 (censored patch), so no actual pre-multiply needed.
                let src        = &frame.pixels;
                let copy_w     = frame.width  as usize;
                let copy_h     = frame.height as usize;
                let src_stride = frame.width as usize * 4;
                let ox = frame.offset_x as isize;
                let oy = frame.offset_y as isize;

                for y in 0..copy_h {
                    let dst_y = y as isize + oy;
                    if dst_y < 0 || dst_y >= sh as isize { continue; }
                    let dst_row = dib_ptr.add(dst_y as usize * dib_row_stride);
                    let src_off = y * src_stride;
                    for x in 0..copy_w {
                        let dst_x = x as isize + ox;
                        if dst_x < 0 || dst_x >= sw as isize { continue; }
                        let si = src_off + x * 4;
                        let di = dst_x as usize * 4;
                        if si + 3 < src.len() {
                            let a = src[si + 3];
                            if a > 0 {
                                *dst_row.add(di)     = src[si + 2]; // B
                                *dst_row.add(di + 1) = src[si + 1]; // G
                                *dst_row.add(di + 2) = src[si];     // R
                                *dst_row.add(di + 3) = a;           // A
                            }
                            // a == 0 pixels are already zero from write_bytes
                        }
                    }
                }
            }

            UpdateLayeredWindow(
                hwnd, hdc_screen, &pt_dst, &win_sz,
                hdc_mem, &pt_src, 0, &blend, ULW_ALPHA,
            );
            dirty = false;
        }

        std::thread::sleep(std::time::Duration::from_millis(8));
    }

    // ── Cleanup ───────────────────────────────────────────────────────────
    DeleteObject(hbm);
    DeleteDC(hdc_mem);
    ReleaseDC(0, hdc_screen);
    DestroyWindow(hwnd);
    log::info!("[win32-overlay] thread exiting");
}
