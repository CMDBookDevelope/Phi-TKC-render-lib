//! Append a rhythm strip after the unmodified game render has resolved.
use super::{MSRenderTarget, RhythmLayout, RhythmStrip};
use crate::ui::{FontArc, TextPainter, Ui};
use anyhow::Result;
use macroquad::prelude::*;
use std::cell::{Cell, RefCell};

const SAIRA_FONT: &[u8] = include_bytes!("../../assets/fonts/Saira-SemiBold.ttf");

pub struct RhythmComposite {
    pub layout: RhythmLayout,
    strip: Option<RhythmStrip>,
    strip_painter: Option<RefCell<TextPainter>>,
    targets: Vec<(MSRenderTarget, MSRenderTarget)>,
    last_time: Cell<f32>,
}

impl RhythmComposite {
    pub fn new(width: u32, height: u32, strip: Option<RhythmStrip>, slots: usize) -> Result<Self> {
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
                .map(|_| (MSRenderTarget::new((width, layout.strip_height), 1), MSRenderTarget::new((width, layout.output_height), 1)))
                .collect()
        } else {
            Vec::new()
        };
        Ok(Self {
            layout,
            strip,
            strip_painter,
            targets,
            last_time: Cell::new(0.),
        })
    }

    /// The disabled path returns the original target without any extra draws.
    /// Each in-flight export frame owns its own strip and composite targets.
    pub fn compose(&self, slot: usize, game: &MSRenderTarget, chart_time: Option<f32>, _game_painter: &mut TextPainter) -> RenderTarget {
        let Some(strip) = &self.strip else { return game.output() };
        if let Some(time) = chart_time.filter(|time| time.is_finite()) {
            self.last_time.set(time);
        }
        let (strip_target, output_target) = &self.targets[slot];
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
        strip.draw(self.last_time.get(), &mut ui);
        unsafe { get_internal_gl() }.flush();

        let width = self.layout.width as f32;
        let game_height = self.layout.game_height as f32;
        let mut camera = Camera2D::from_display_rect(Rect::new(0., 0., width, self.layout.output_height as f32));
        camera.render_target = Some(output_target.input());
        set_camera(&camera);
        clear_background(BLACK);
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
