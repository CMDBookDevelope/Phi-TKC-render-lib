//! Deterministic hand planning used by chart rendering.
//!
//! The legacy experimental neural APIs remain available to callers, but chart
//! rendering never starts their GPU, trainer, framebuffer readback or workers.

pub mod ai;
pub(crate) mod network;
pub(crate) mod planner;
pub(crate) mod trainer;
pub mod vit;

pub use ai::{Finger, PhiTKAdvancedAI};

pub const AI_IMAGE_W: usize = 80;
pub const AI_IMAGE_H: usize = 45;
pub const AI_IMAGE_CHANNELS: usize = 3;
pub const AI_IMAGE_SIZE: usize = AI_IMAGE_W * AI_IMAGE_H * AI_IMAGE_CHANNELS;
pub(crate) const MODEL_PATH: &str = "phitk_ai_model.bin";

/// The immutable layout/time model used by the renderer. Applications can
/// verify their packaged asset is identical to the model embedded in prpr.
pub fn bundled_hand_model() -> &'static [u8] {
    planner::bundled_model_bytes()
}

use crate::config::Config;
use crate::core::{Note, Vector};
use serde::{Deserialize, Serialize};

/// One rollout sample for callers of the legacy experimental PPO trainer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experience {
    pub state: Vec<f32>,
    pub actions: Vec<u8>,
    pub old_out: Vec<f32>,
    pub reward: f32,
    pub value: f32,
    pub advantage: f32,
    pub return_: f32,
}

/// Compatibility entry point for callers that already have world-space note
/// positions. It assigns immediately and never invokes `make_image`. The
/// renderer uses the chart-wide planner, so chords on different lines share
/// the same hands and this work is performed only once per chart geometry.
pub fn assign_hands_unified_perspective(
    notes: &mut [Note],
    config: &Config,
    _line_id: usize,
    _rotation: f32,
    _time: f32,
    world_positions: &[Vector],
    _make_image: impl FnOnce() -> Vec<f32>,
) {
    if !config.hand_split {
        return;
    }
    planner::assign_notes(notes, world_positions, config.flip_x());
}
