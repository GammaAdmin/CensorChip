// CensorChip – Captured frame representation
// SPDX-License-Identifier: GPL-3.0-or-later
//
// A lightweight struct that carries raw pixel data through the pipeline.
// Keeps allocations to a minimum by reusing Vec buffers where possible.

/// A single captured frame in RGBA8 format.
#[derive(Clone)]
pub struct CapturedFrame {
    /// Raw pixel bytes in RGBA order, row-major, top-left origin.
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    /// Monotonic frame counter (set by the pipeline, not the capturer).
    pub frame_id: u64,
    /// Top-left corner of the captured window in screen coordinates.
    /// (0, 0) for full-screen DXGI capture; set by WindowCapturer.
    pub screen_x: i32,
    pub screen_y: i32,
}

impl CapturedFrame {
    pub fn new(data: Vec<u8>, width: u32, height: u32) -> Self {
        Self {
            data,
            width,
            height,
            frame_id: 0,
            screen_x: 0,
            screen_y: 0,
        }
    }

    /// Number of pixels.
    #[inline]
    pub fn pixel_count(&self) -> usize {
        self.width as usize * self.height as usize
    }

    /// Total byte length (should equal data.len()).
    #[inline]
    pub fn byte_len(&self) -> usize {
        self.pixel_count() * 4
    }

    /// Downscale the frame by an integer factor using nearest-neighbour.
    /// Very fast – used before AI inference to shrink frames cheaply.
    pub fn downscale(&self, factor: u32) -> CapturedFrame {
        if factor <= 1 {
            return self.clone();
        }
        let factor = factor as usize;
        let new_w = self.width as usize / factor;
        let new_h = self.height as usize / factor;
        let stride = self.width as usize * 4;
        let mut out = Vec::with_capacity(new_w * new_h * 4);

        for y in 0..new_h {
            let src_y = y * factor;
            let row_start = src_y * stride;
            for x in 0..new_w {
                let src_x = x * factor;
                let px = row_start + src_x * 4;
                out.extend_from_slice(&self.data[px..px + 4]);
            }
        }

        CapturedFrame {
            data: out,
            width: new_w as u32,
            height: new_h as u32,
            frame_id: self.frame_id,
            screen_x: 0,
            screen_y: 0,
        }
    }
}
