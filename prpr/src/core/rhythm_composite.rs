//! Append a rhythm strip after the unmodified game render has resolved.
use super::{MSRenderTarget, RhythmLayout, RhythmStrip};
use crate::ui::{FontArc, TextPainter, Ui};
use anyhow::Result;
use macroquad::prelude::*;
use std::cell::{Cell, RefCell};

const SAIRA_FONT: &[u8] = include_bytes!("../../assets/fonts/Saira-SemiBold.ttf");

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct RhythmFrameSample {
    time: Option<f32>,
    previous_time: Option<f32>,
}

impl RhythmFrameSample {
    fn advance(mut self, chart_time: Option<f32>) -> Self {
        if let Some(time) = chart_time.filter(|time| time.is_finite()) {
            if self.time != Some(time) {
                self.previous_time = self.time;
                self.time = Some(time);
            }
        }
        // Same-time retries, other slots and Ending reuse the same interval.
        self
    }
}

pub struct RhythmComposite {
    pub layout: RhythmLayout,
    strip: Option<RhythmStrip>,
    strip_painter: Option<RefCell<TextPainter>>,
    targets: Vec<(RefCell<MSRenderTarget>, MSRenderTarget)>,
    frame_sample: Cell<RhythmFrameSample>,
    preserve_alpha: bool,
    hdr: bool,
}

impl RhythmComposite {
    pub fn new(width: u32, height: u32, strip: Option<RhythmStrip>, slots: usize) -> Result<Self> {
        Self::new_with_alpha(width, height, strip, slots, false)
    }

    pub fn new_with_alpha(
        width: u32,
        height: u32,
        strip: Option<RhythmStrip>,
        slots: usize,
        preserve_alpha: bool,
    ) -> Result<Self> {
        Self::with_format(width, height, strip, slots, preserve_alpha, false)
    }

    pub fn new_hdr(width: u32, height: u32, strip: Option<RhythmStrip>, slots: usize) -> Result<Self> {
        Self::new_hdr_with_alpha(width, height, strip, slots, false)
    }

    pub fn new_hdr_with_alpha(
        width: u32,
        height: u32,
        strip: Option<RhythmStrip>,
        slots: usize,
        preserve_alpha: bool,
    ) -> Result<Self> {
        Self::with_format(width, height, strip, slots, preserve_alpha, true)
    }

    fn with_format(
        width: u32,
        height: u32,
        strip: Option<RhythmStrip>,
        slots: usize,
        preserve_alpha: bool,
        hdr: bool,
    ) -> Result<Self> {
        let layout = RhythmLayout::new(width, height, strip.is_some())?;
        // Keep the glyph cache local to this export's GPU context. Embedding the
        // licensed font also avoids platform-specific asset paths or fallbacks.
        let strip_painter = if strip.is_some() {
            Some(RefCell::new(TextPainter::new(FontArc::try_from_slice(SAIRA_FONT)?)))
        } else {
            None
        };
        let targets = if strip.is_some() {
            (0..slots)
                .map(|_| {
                    let strip = if hdr {
                        MSRenderTarget::new_hdr((width, layout.strip_height), 1)
                    } else if preserve_alpha {
                        MSRenderTarget::new_rgba((width, layout.strip_height), 1)
                    } else {
                        MSRenderTarget::new((width, layout.strip_height), 1)
                    };
                    let output = if hdr {
                        MSRenderTarget::new_hdr((width, layout.output_height), 1)
                    } else if preserve_alpha {
                        MSRenderTarget::new_rgba((width, layout.output_height), 1)
                    } else {
                        MSRenderTarget::new((width, layout.output_height), 1)
                    };
                    (RefCell::new(strip), output)
                })
                .collect()
        } else {
            Vec::new()
        };
        Ok(Self {
            layout,
            strip,
            strip_painter,
            targets,
            frame_sample: Cell::new(RhythmFrameSample::default()),
            preserve_alpha,
            hdr,
        })
    }

    /// The disabled path returns the original target without any extra draws.
    /// Each in-flight export frame owns its own strip and composite targets.
    pub fn compose(&self, slot: usize, game: &MSRenderTarget, chart_time: Option<f32>, _game_painter: &mut TextPainter) -> RenderTarget {
        let Some(strip) = &self.strip else { return game.output() };
        let sample = self.frame_sample.get().advance(chart_time);
        self.frame_sample.set(sample);
        let (strip_target, output_target) = &self.targets[slot];
        let mut strip_target = strip_target.borrow_mut();
        unsafe { get_internal_gl() }.flush();
        push_camera_state();
        let mut painter = self
            .strip_painter
            .as_ref()
            .expect("enabled rhythm strip has its own painter")
            .borrow_mut();
        let mut ui = Ui::new(&mut painter, Some((0, 0, self.layout.width as i32, self.layout.strip_height as i32)));
        let mut camera = ui.camera();
        camera.render_target = Some(strip_target.input());
        set_camera(&camera);
        clear_background(BLACK);
        strip.draw_sampled(sample.time.unwrap_or(0.), sample.previous_time, &mut ui);
        unsafe { get_internal_gl() }.flush();
        if self.hdr {
            strip_target.map_hdr(&crate::config::HdrLayerSettings::default());
        }

        let width = self.layout.width as f32;
        let game_height = self.layout.game_height as f32;
        let mut camera = Camera2D::from_display_rect(Rect::new(0., 0., width, self.layout.output_height as f32));
        camera.render_target = Some(output_target.input());
        set_camera(&camera);
        clear_background(if self.preserve_alpha { Color::new(0., 0., 0., 0.) } else { BLACK });
        draw_texture_ex(
            game.output().texture,
            0.,
            0.,
            WHITE,
            DrawTextureParams {
                dest_size: Some(vec2(width, game_height)),
                flip_y: true,
                ..Default::default()
            },
        );
        draw_texture_ex(
            strip_target.output().texture,
            0.,
            game_height,
            WHITE,
            DrawTextureParams {
                dest_size: Some(vec2(width, self.layout.strip_height as f32)),
                flip_y: true,
                ..Default::default()
            },
        );
        unsafe { get_internal_gl() }.flush();
        pop_camera_state();
        output_target.output()
    }
}

#[cfg(test)]
mod tests {
    use super::RhythmFrameSample;
    use super::super::rhythm::metronome_pulse_sampled;

    #[test]
    fn rhythm_sampled_composite_retries_slots_and_ending_preserve_peak_interval() {
        let first = RhythmFrameSample::default().advance(Some(0.499));
        assert_eq!(first.previous_time, None);
        let peak = first.advance(Some(0.51));
        let pulse = |sample: RhythmFrameSample| {
            let time = sample.time.unwrap_or(0.);
            metronome_pulse_sampled(time * 2., sample.previous_time.map(|time| time * 2.), sample.previous_time.map_or(0., |previous| time - previous))
        };
        assert_eq!(pulse(peak), 1.);
        for retry in [Some(0.51), None, Some(f32::NAN), Some(f32::INFINITY)] {
            assert_eq!(peak.advance(retry), peak);
            assert_eq!(pulse(peak.advance(retry)), 1.);
        }
        let next = peak.advance(Some(0.52));
        assert_eq!(next.previous_time, Some(0.51));
        assert!(pulse(next) < 1.);
        let seek = next.advance(Some(2.02));
        assert!(pulse(seek) < 1.);
        assert_eq!(seek.advance(None), seek);
        let backwards = seek.advance(Some(0.2));
        assert!(pulse(backwards) < 1.);
    }
}
