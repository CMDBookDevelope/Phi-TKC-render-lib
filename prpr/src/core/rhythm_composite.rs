//! Append a rhythm strip after the unmodified game render has resolved.
use super::{MSRenderTarget, RhythmLayout, RhythmStrip};
use crate::ui::{FontArc, TextPainter, Ui};
use anyhow::Result;
use macroquad::prelude::*;
use std::cell::{Cell, RefCell};

const SAIRA_FONT: &[u8] = include_bytes!("../../assets/fonts/Saira-SemiBold.ttf");

/// Source-over blit for associated SDR render-target RGB. Do not use for HDR
/// mapped layers: map_hdr deliberately outputs straight RGB for SrcAlpha draws.
/// Own the material per GPU context, rather than caching a handle across exports.
pub struct PremultipliedBlit(Material);

impl PremultipliedBlit {
    pub fn new() -> Result<Self> {
        use macroquad::miniquad::{BlendFactor, BlendState, BlendValue, Equation, PipelineParams};
        let source_over = BlendState::new(
            Equation::Add,
            BlendFactor::One,
            BlendFactor::OneMinusValue(BlendValue::SourceAlpha),
        );
        Ok(Self(load_material(
            PREMULTIPLIED_VERTEX,
            PREMULTIPLIED_FRAGMENT,
            MaterialParams {
                pipeline_params: PipelineParams {
                    color_blend: Some(source_over),
                    alpha_blend: Some(source_over),
                    ..Default::default()
                },
                ..Default::default()
            },
        )?))
    }

    pub fn draw(&self, texture: Texture2D, x: f32, y: f32, tint: Color, params: DrawTextureParams) {
        gl_use_material(self.0);
        draw_texture_ex(texture, x, y, tint, params);
        // QuadGl queues pipeline handles, not owned materials. Submit this blit
        // before it can be dropped on a subsequent render error; never flush
        // from Drop during error unwinding or after GPU teardown.
        unsafe { get_internal_gl() }.flush();
        gl_use_default_material();
    }
}

impl Drop for PremultipliedBlit {
    fn drop(&mut self) {
        self.0.delete();
    }
}

const PREMULTIPLIED_VERTEX: &str = r#"#version 450 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texcoord;
layout(location = 2) in uvec4 color0;
layout(set = 0, binding = 0) uniform Uniforms { mat4 Projection; mat4 Model; vec4 _Time; };
layout(location = 0) out vec2 uv;
layout(location = 1) out vec4 tint;
void main() {
    gl_Position = Projection * Model * vec4(position, 1.0);
    uv = texcoord;
    tint = vec4(color0) / 255.0;
}
"#;

const PREMULTIPLIED_FRAGMENT: &str = r#"#version 450 core
layout(location = 0) in vec2 uv;
layout(location = 1) in vec4 tint;
layout(set = 0, binding = 1) uniform sampler2D Texture;
layout(location = 0) out vec4 fragColor;
void main() {
    vec4 src = texture(Texture, uv);
    fragColor = vec4(src.rgb * tint.rgb * tint.a, src.a * tint.a);
}
"#;

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
    premultiplied_blit: Option<PremultipliedBlit>,
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
        let premultiplied_blit = if strip.is_some() && preserve_alpha && !hdr {
            Some(PremultipliedBlit::new()?)
        } else {
            None
        };
        Ok(Self {
            layout,
            strip,
            strip_painter,
            targets,
            frame_sample: Cell::new(RhythmFrameSample::default()),
            preserve_alpha,
            hdr,
            premultiplied_blit,
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
        // SDR alpha game targets are associated; copying them with SrcAlpha
        // would multiply coverage twice. The opaque strip is compatible too.
        // HDR mapped textures remain straight and keep the default material.
        for (texture, y, height) in [
            (game.output().texture, 0., game_height),
            (strip_target.output().texture, game_height, self.layout.strip_height as f32),
        ] {
            let params = DrawTextureParams {
                dest_size: Some(vec2(width, height)),
                flip_y: true,
                ..Default::default()
            };
            if let Some(blit) = &self.premultiplied_blit {
                blit.draw(texture, 0., y, WHITE, params);
            } else {
                draw_texture_ex(texture, 0., y, WHITE, params);
            }
        }
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
