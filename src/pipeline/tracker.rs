// CensorChip – Simple bounding-box tracker (SORT-inspired)
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Between inference frames the pipeline reuses the last set of detections.
// This tracker applies a constant-velocity prediction to keep boxes roughly
// aligned with moving content, reducing jitter without running inference
// every frame.
//
// This is *not* a full SORT/DeepSORT implementation but is extremely cheap
// and good enough for a censorship overlay where pixel-perfect tracking is
// not required.

use crate::inference::Detection;

/// Lightweight inter-frame tracker.
pub struct SimpleTracker {
    /// Per-detection velocity estimates (dx, dy) from previous update.
    velocities: Vec<(f32, f32)>,
    /// Previous positions for velocity computation.
    prev_positions: Vec<(f32, f32)>,
}

impl SimpleTracker {
    pub fn new() -> Self {
        Self {
            velocities: Vec::new(),
            prev_positions: Vec::new(),
        }
    }

    /// Called when we *did* run inference and have fresh detections.
    pub fn update(&mut self, detections: &[Detection]) {
        let new_positions: Vec<(f32, f32)> = detections
            .iter()
            .map(|d| (d.x + d.w / 2.0, d.y + d.h / 2.0))
            .collect();

        // Naively match by index (works when detection count is stable frame
        // to frame).  A production system would use Hungarian matching.
        self.velocities.clear();
        for (i, pos) in new_positions.iter().enumerate() {
            if let Some(prev) = self.prev_positions.get(i) {
                self.velocities.push((pos.0 - prev.0, pos.1 - prev.1));
            } else {
                self.velocities.push((0.0, 0.0));
            }
        }
        self.prev_positions = new_positions;
    }

    /// Called when we *skipped* inference.  Shifts each detection by its
    /// estimated velocity so the overlay follows moving content.
    pub fn predict(&self, detections: &mut [Detection]) {
        for (i, det) in detections.iter_mut().enumerate() {
            if let Some(&(vx, vy)) = self.velocities.get(i) {
                det.x += vx;
                det.y += vy;
            }
        }
    }
}
