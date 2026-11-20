//! Musical analysis independent of chart geometry and GPU readback. The caller
//! gives this strip the exact `Scene::chart_time()` used for note judgement.
use super::{BpmList, Chart, NoteKind};
use crate::{config::Config, ui::Ui};
use anyhow::{ensure, Context, Result};
use image::RgbaImage;
use lyon::{math::point, path::Path};
use macroquad::prelude::{Color, Rect, WHITE};
use std::{borrow::Cow, f32::consts::{FRAC_PI_2, FRAC_PI_4, TAU}};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RhythmNoteSelection {
    pub tap: bool,
    pub drag: bool,
    pub hold: bool,
    pub flick: bool,
}

impl Default for RhythmNoteSelection {
    fn default() -> Self {
        Self {
            tap: true,
            drag: false,
            hold: true,
            flick: false,
        }
    }
}

impl From<&Config> for RhythmNoteSelection {
    fn from(config: &Config) -> Self {
        Self {
            tap: config.rhythm_tap,
            drag: config.rhythm_drag,
            hold: config.rhythm_hold,
            flick: config.rhythm_flick,
        }
    }
}

impl RhythmNoteSelection {
    fn includes(self, kind: &NoteKind) -> bool {
        match kind {
            NoteKind::Click => self.tap,
            NoteKind::Drag => self.drag,
            NoteKind::Hold { .. } => self.hold,
            NoteKind::Flick => self.flick,
        }
    }
}

/// Video codecs require even dimensions. Round W/10 to the nearest even pixel,
/// leaving the original game rectangle exactly W × H.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RhythmLayout {
    pub width: u32,
    pub game_height: u32,
    pub strip_height: u32,
    pub output_height: u32,
}

impl RhythmLayout {
    pub fn new(width: u32, height: u32, enabled: bool) -> Result<Self> {
        ensure!(width >= 16 && height >= 16 && width % 2 == 0 && height % 2 == 0, "video width and height must be even and at least 16");
        let strip_height = if enabled { (((width as u64 + 10) / 20) * 2).max(2) } else { 0 };
        let output_height = height as u64 + strip_height;
        ensure!(width <= i32::MAX as u32 && output_height <= i32::MAX as u64, "rhythm output dimensions exceed the graphics viewport limit");
        Ok(Self {
            width,
            game_height: height,
            strip_height: strip_height as u32,
            output_height: output_height as u32,
        })
    }

    pub fn judge_x_pixels(self) -> f32 {
        self.strip_height as f32 * 1.2
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RhythmShape {
    Circle,
    Diamond,
    Square,
    Triangle,
    DoubleTriangle,
    Hexagon,
    Octagon,
    Cross,
    Star,
}

/// Durations come from successive distinct selected onsets and stop at one
/// quarter-note beat. Holds contribute only their onset, like a tap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RhythmValue {
    Whole,
    DottedHalf,
    Half,
    DottedQuarter,
    Quarter,
    DottedEighth,
    Eighth,
    DottedSixteenth,
    Sixteenth,
    ThirtySecond,
    SixtyFourth,
    HalfTriplet,
    QuarterTriplet,
    EighthTriplet,
    SixteenthTriplet,
    ThirtySecondTriplet,
    Tuplet(u16),
    Ratio(u16, u16),
}

impl RhythmValue {
    pub const ALL: [Self; 17] = [
        Self::Whole,
        Self::DottedHalf,
        Self::Half,
        Self::DottedQuarter,
        Self::Quarter,
        Self::DottedEighth,
        Self::Eighth,
        Self::DottedSixteenth,
        Self::Sixteenth,
        Self::ThirtySecond,
        Self::SixtyFourth,
        Self::HalfTriplet,
        Self::QuarterTriplet,
        Self::EighthTriplet,
        Self::SixteenthTriplet,
        Self::ThirtySecondTriplet,
        Self::Tuplet(5),
    ];

    pub fn classify(beats: f32) -> Self {
        Self::classify_with_precision(beats, 0.)
    }

    fn classify_span(begin: f32, end: f32) -> Self {
        // Subtracting large chart positions loses more precision than a short
        // duration near zero. Account for the quantization of both endpoints.
        let precision = f32::EPSILON * begin.abs().max(end.abs()).max(1.);
        Self::classify_with_precision(end - begin, precision)
    }

    fn classify_with_precision(beats: f32, precision: f32) -> Self {
        if !beats.is_finite() || beats <= 0. {
            return Self::Quarter;
        }
        // A long interval contains one quarter-note onset followed by empty
        // beat guides. Do not invent notes or rest symbols in the silent gap.
        let beats = beats.min(1.);
        let tolerance = (beats * 0.0002).max(precision).max(0.000002);
        let candidates = [
            (4., Self::Whole),
            (3., Self::DottedHalf),
            (2., Self::Half),
            (1.5, Self::DottedQuarter),
            (1., Self::Quarter),
            (0.75, Self::DottedEighth),
            (0.5, Self::Eighth),
            (0.375, Self::DottedSixteenth),
            (0.25, Self::Sixteenth),
            (0.125, Self::ThirtySecond),
            (0.0625, Self::SixtyFourth),
            (4. / 3., Self::HalfTriplet),
            (2. / 3., Self::QuarterTriplet),
            (1. / 3., Self::EighthTriplet),
            (1. / 6., Self::SixteenthTriplet),
            (1. / 12., Self::ThirtySecondTriplet),
        ];
        if let Some((_, value)) = candidates
            .into_iter()
            .find(|(value, _)| (beats - value).abs() <= tolerance)
        {
            return value;
        }
        // Infer the simplest beat fraction within the source's f32 precision.
        // Removing binary subdivisions preserves the actual tuplet count:
        // 1/5, 2/5, and 1/10 all belong to five-note subdivision families.
        let mut ratio = None;
        for denominator in 1..=4096_u16 {
            let numerator = (beats as f64 * denominator as f64).round();
            if numerator < 1. || numerator > u16::MAX as f64 {
                continue;
            }
            if (beats as f64 - numerator / denominator as f64).abs() <= tolerance as f64 {
                ratio = Some((numerator as u16, denominator));
                break;
            }
        }
        let (mut numerator, mut denominator) = ratio.unwrap_or_else(|| {
            let denominator = (u16::MAX as f32 / beats).floor().clamp(1., 4096.) as u16;
            let numerator = (beats * denominator as f32).round().clamp(1., u16::MAX as f32) as u16;
            (numerator, denominator)
        });
        let mut a = numerator;
        let mut b = denominator;
        while b != 0 {
            (a, b) = (b, a % b);
        }
        numerator /= a;
        denominator /= a;
        let tuplet = denominator >> denominator.trailing_zeros();
        if tuplet > 1 {
            Self::Tuplet(tuplet)
        } else {
            Self::Ratio(numerator, denominator)
        }
    }

    /// Conventional values have distinct visual codes; arbitrary tuplets share
    /// a star and are distinguished by their actual numeric subdivision count.
    pub fn style(self) -> (RhythmShape, usize) {
        use RhythmShape::*;
        match self {
            Self::Whole => (Octagon, 0),
            Self::DottedHalf => (Octagon, 1),
            Self::Half => (Hexagon, 0),
            Self::DottedQuarter => (Circle, 1),
            Self::Quarter => (Circle, 0),
            Self::DottedEighth => (Diamond, 0),
            Self::Eighth => (Diamond, 1),
            Self::DottedSixteenth => (Square, 0),
            Self::Sixteenth => (Square, 2),
            Self::ThirtySecond => (Hexagon, 1),
            Self::SixtyFourth => (Cross, 2),
            Self::HalfTriplet => (DoubleTriangle, 0),
            Self::QuarterTriplet => (Triangle, 0),
            Self::EighthTriplet => (Triangle, 1),
            Self::SixteenthTriplet => (Triangle, 2),
            Self::ThirtySecondTriplet => (DoubleTriangle, 2),
            Self::Tuplet(_) | Self::Ratio(_, _) => (Star, 2),
        }
    }

    pub fn label(self) -> Cow<'static, str> {
        Cow::Borrowed(match self {
            Self::Whole | Self::DottedHalf | Self::Half | Self::DottedQuarter | Self::HalfTriplet => "4",
            Self::Quarter => "4",
            Self::DottedEighth => "16/3",
            Self::Eighth => "8",
            Self::DottedSixteenth => "32/3",
            Self::Sixteenth => "16",
            Self::ThirtySecond => "32",
            Self::SixtyFourth => "64",
            Self::QuarterTriplet => "3",
            Self::EighthTriplet => "6",
            Self::SixteenthTriplet => "12",
            Self::ThirtySecondTriplet => "24",
            Self::Tuplet(count) => return Cow::Owned(count.to_string()),
            Self::Ratio(numerator, denominator) => return Cow::Owned(format!("{numerator}/{denominator}")),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RhythmRenderLayer {
    All,
    Metronome,
    Note,
    Value,
}

impl RhythmRenderLayer {
    fn draws_metronome(self) -> bool { matches!(self, Self::All | Self::Metronome) }
    fn draws_notes(self) -> bool { matches!(self, Self::All | Self::Note) }
    fn draws_values(self) -> bool { matches!(self, Self::All | Self::Value) }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RhythmKind {
    Tap,
    Drag,
    Hold,
    Flick,
}

#[derive(Clone, Debug)]
pub struct RhythmNote {
    pub time: f32,
    pub end_time: f32,
    pub beat: f32,
    pub end_beat: f32,
    pub kind: RhythmKind,
    pub value: RhythmValue,
}

pub struct RhythmStrip {
    tempo: BpmList,
    notes: Vec<RhythmNote>,
    palette: [Color; 3],
    auto_palette: [Color; 3],
    speed: f32,
}

impl RhythmStrip {
    pub fn from_chart_bytes(chart: &Chart, selection: RhythmNoteSelection, cover: &[u8]) -> Result<Self> {
        let cover = image::load_from_memory(cover)
            .context("failed to decode rhythm analysis cover")?
            .into_rgba8();
        Ok(Self::from_chart(chart, selection, Some(&cover)))
    }

    /// Build the strip on the chart's seconds axis. `audio_offset` is the same
    /// chart + configuration + metadata offset subtracted by GameScene.
    pub fn from_chart_bytes_with_audio(
        chart: &Chart,
        selection: RhythmNoteSelection,
        cover: &[u8],
        music: &sasa::AudioClip,
        auto_bpm: bool,
        audio_offset: f32,
    ) -> Result<Self> {
        let cover = image::load_from_memory(cover)
            .context("failed to decode rhythm analysis cover")?
            .into_rgba8();
        Ok(Self::from_chart_with_audio(chart, selection, Some(&cover), music, auto_bpm, audio_offset))
    }

    pub fn from_chart_with_audio(
        chart: &Chart,
        selection: RhythmNoteSelection,
        cover: Option<&RgbaImage>,
        music: &sasa::AudioClip,
        auto_bpm: bool,
        audio_offset: f32,
    ) -> Self {
        // Authored maps and disabled inference must not even inspect audio.
        if !auto_bpm || chart.rhythm_bpm_authored {
            let tempo = chart.rhythm_bpm_list.clone().unwrap_or_else(|| chart.bpm_list.borrow().clone());
            return Self::from_chart_with_tempo(chart, selection, cover, tempo);
        }
        let original = chart.rhythm_bpm_list.clone().unwrap_or_else(|| chart.bpm_list.borrow().clone());
        let frames = music.frames();
        // Preserve shared stereo attacks while avoiding complete cancellation
        // when the recording's channels have opposite polarity.
        let finite = |value: f32| if value.is_finite() { value } else { 0. };
        let covariance = frames.iter().map(|frame| finite(frame.0) as f64 * finite(frame.1) as f64).sum::<f64>();
        let right_sign = if covariance < 0. { -1. } else { 1. };
        let estimate = super::rhythm_tempo::infer_if_missing(
            true, false, frames.len(), music.sample_rate(),
            |index| (finite(frames[index].0) + right_sign * finite(frames[index].1)) * 0.5,
            original.now_bpm(0.),
        );
        let tempo = estimate.map_or(original, |estimate| {
            tracing::info!(scale = estimate.scale, nodes = estimate.points.len(), "inferred rhythm BPM from audio");
            let offset = if audio_offset.is_finite() { audio_offset } else { 0. };
            let mut elements = Vec::with_capacity(estimate.points.len());
            let mut beat = -estimate.first_beat_time * estimate.points[0].bpm / 60.;
            let mut previous_time = 0.;
            let mut previous_bpm = estimate.points[0].bpm;
            for point in estimate.points {
                beat += (point.time - previous_time) * previous_bpm / 60.;
                elements.push((beat, point.time - offset, point.bpm));
                previous_time = point.time;
                previous_bpm = point.bpm;
            }
            BpmList { elements, cursor: 0 }
        });
        Self::from_chart_with_tempo(chart, selection, cover, tempo)
    }

    pub fn from_chart(chart: &Chart, selection: RhythmNoteSelection, cover: Option<&RgbaImage>) -> Self {
        let tempo = chart.rhythm_bpm_list.clone().unwrap_or_else(|| chart.bpm_list.borrow().clone());
        Self::from_chart_with_tempo(chart, selection, cover, tempo)
    }

    fn from_chart_with_tempo(chart: &Chart, selection: RhythmNoteSelection, cover: Option<&RgbaImage>, tempo: BpmList) -> Self {
        let events = chart
            .lines
            .iter()
            .flat_map(|line| &line.notes)
            .filter(|note| !note.fake && selection.includes(&note.kind) && note.time.is_finite())
            .map(|note| match note.kind {
                NoteKind::Click => (note.time, note.time, RhythmKind::Tap),
                NoteKind::Drag => (note.time, note.time, RhythmKind::Drag),
                NoteKind::Flick => (note.time, note.time, RhythmKind::Flick),
                NoteKind::Hold { .. } => (note.time, note.time, RhythmKind::Tap),
            })
            .collect();
        Self::from_events(tempo, events, cover_palette(cover))
    }

    fn from_events(tempo: BpmList, mut events: Vec<(f32, f32, RhythmKind)>, palette: [Color; 3]) -> Self {
        events.retain(|(time, _, _)| time.is_finite());
        events.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut notes = Vec::with_capacity(events.len());
        let mut begin = 0;
        while begin < events.len() {
            let time = events[begin].0;
            // Flattened events from every selected chart line share one onset.
            // RPE/PGR conversion may place that onset a few ulps apart; merge
            // only within 0.1ms and retain the earliest time as its canonical x.
            let mut end = begin + 1;
            while end < events.len() && (events[end].0 - time).abs() < 0.0001 {
                end += 1;
            }
            let next_beat = events.get(end).map(|event| tempo.beat(event.0));
            let mut kind = RhythmKind::Tap;
            for &(_, _, event_kind) in &events[begin..end] {
                match event_kind {
                    RhythmKind::Flick => kind = RhythmKind::Flick,
                    RhythmKind::Drag if kind != RhythmKind::Flick => kind = RhythmKind::Drag,
                    _ => {}
                }
            }
            let beat = tempo.beat(time);
            let end_time = time;
            let end_beat = beat;
            // All source kinds, including holds, use the next global onset.
            // No following onset can establish the terminal value.
            let value = next_beat.map_or(RhythmValue::Quarter, |end| RhythmValue::classify_span(beat, end));
            notes.push(RhythmNote { time, end_time, beat, end_beat, kind, value });
            begin = end;
        }
        Self {
            tempo,
            notes,
            palette,
            auto_palette: palette,
            speed: 1.,
        }
    }

    pub fn notes(&self) -> &[RhythmNote] {
        &self.notes
    }
    pub fn palette(&self) -> [Color; 3] {
        self.palette
    }

    /// Preserve valid user colors exactly; omitted or invalid overrides restore
    /// the same automatic palette used by the one-click extraction API.
    pub fn with_colors(mut self, colors: Option<&[String; 3]>) -> Self {
        self.palette = colors.and_then(|colors| {
            Some([parse_hex_color(&colors[0])?, parse_hex_color(&colors[1])?, parse_hex_color(&colors[2])?])
        }).unwrap_or(self.auto_palette);
        self
    }
    pub fn beat_at(&self, time: f32) -> f32 {
        self.tempo.beat(time)
    }
    pub fn bpm_at(&self, time: f32) -> f32 {
        self.tempo.now_bpm(time)
    }

    pub fn with_speed(mut self, speed: f32) -> Self {
        self.set_speed(speed);
        self
    }

    pub fn set_speed(&mut self, speed: f32) {
        self.speed = if speed.is_finite() { speed.clamp(0.1, 10.) } else { 1. };
    }

    pub fn speed(&self) -> f32 {
        self.speed
    }

    fn time_pitch(&self, top: f32) -> f32 {
        // A constant seconds-to-pixels scale keeps authored note motion
        // continuous through tempo changes. The first BPM preserves the old
        // default layout for charts whose tempo stays constant.
        spacing(top) * (self.bpm_at(0.) / 60.).max(0.001) * self.speed
    }

    fn time_position_x(&self, authored_time: f32, time: f32, top: f32) -> f32 {
        -1. + 2.4 * top + (authored_time - time) * self.time_pitch(top)
    }

    // Later judgements paint first, so an earlier note remains above any
    // overlapping future note. Stable source order breaks exact-time ties.
    fn visible_notes(&self, first: f32, last: f32) -> impl Iterator<Item = &RhythmNote> {
        let begin = self.notes.partition_point(|note| note.beat < first);
        let end = self.notes.partition_point(|note| note.beat <= last);
        self.notes[begin.min(end)..end].iter().rev()
    }

    /// x is normalized to the strip's -1..1 camera. At `note.time`, x is
    /// precisely the same judgement line used for the beat reference marks.
    pub fn position_x(&self, note: &RhythmNote, time: f32, top: f32) -> f32 {
        self.time_position_x(note.time, time, top)
    }

    /// Draw inside a dedicated Ui/camera viewport whose size is W × W/10.
    /// No music offset, playback speed, or preroll is applied here: the caller
    /// has already resolved those in the scene's judgement clock.
    pub fn draw(&self, chart_time: f32, ui: &mut Ui) {
        self.draw_sampled_layer(chart_time, None, ui, RhythmRenderLayer::All);
    }

    pub(super) fn draw_sampled(&self, chart_time: f32, previous_time: Option<f32>, ui: &mut Ui) {
        self.draw_sampled_layer(chart_time, previous_time, ui, RhythmRenderLayer::All);
    }

    pub(crate) fn draw_sampled_layer(
        &self,
        chart_time: f32,
        previous_time: Option<f32>,
        ui: &mut Ui,
        layer: RhythmRenderLayer,
    ) {
        let draw_metronome = layer.draws_metronome();
        let draw_notes = layer.draws_notes();
        let draw_values = layer.draws_values();
        let time = if chart_time.is_finite() { chart_time } else { 0. };
        let top = ui.top;
        let h = 2. * top;
        let left = -1. + h;
        let judge = -1. + 2.4 * top;
        let pitch = self.time_pitch(top);
        let now = self.beat_at(time);
        let pulse = metronome_pulse_sampled(now, previous_time.map(|time| self.beat_at(time)), previous_time.map_or(0., |previous| time - previous));
        let bg = mix(self.palette[0], Color::new(0.035, 0.043, 0.06, 1.), 0.88);
        if draw_metronome {
        ui.fill_rect(ui.screen_rect(), (mix(bg, self.palette[1], 0.10), (-1., -top), bg, (1., top)));
        ui.fill_rect(Rect::new(-1., -top, 2., h * 0.014), alpha(WHITE, 0.19));
        ui.fill_rect(Rect::new(left, -top, 2. - h, h), (alpha(WHITE, 0.065), (left, -top), alpha(WHITE, 0.), (left, top)));

        // Every real beat has the same luminous peak, independent of meter.
        // Only brightness pulses; all ring and halo dimensions stay fixed.
        let cx = -1. + top;
        let radius = h * 0.345;
        ui.fill_circle(cx, 0., radius * 1.16, alpha(self.palette[0], 0.035 + pulse * 0.09));
        ui.fill_circle(cx, 0., radius, (alpha(WHITE, 0.055), (cx, -radius), alpha(WHITE, 0.012), (cx, radius)));
        ui.stroke_circle(cx, 0., radius, h * 0.008, alpha(WHITE, 0.15));
        for layer in 0..4 {
            let extra = layer as f32 * h * 0.009;
            ui.stroke_circle(
                cx,
                0.,
                radius + extra,
                h * 0.01 + extra * 0.45,
                alpha(self.palette[0], pulse * (0.28 - layer as f32 * 0.055)),
            );
        }
        ui.stroke_circle(cx, 0., radius, h * 0.013, alpha(mix(self.palette[0], WHITE, 0.65), 0.2 + pulse * 0.75));
        text(ui, "BPM", cx, -h * 0.14, h * 1.1, alpha(WHITE, 0.58));
        let bpm = self.bpm_at(time);
        let bpm_text = if (bpm - bpm.round()).abs() < 0.05 {
            format!("{:.0}", bpm)
        } else {
            format!("{:.1}", bpm)
        };
        text(ui, &bpm_text, cx, h * 0.014, h * 2.95, alpha(WHITE, 0.96));
        text(ui, "BEAT", cx, h * 0.18, h * 0.83, alpha(self.palette[0], 0.8));
        ui.fill_rect(Rect::new(left, -top + h * 0.10, h * 0.004, h * 0.80), alpha(WHITE, 0.16));

        }

        let bands = RhythmBands::new(left, h);
        let note_margin = bands.notes.h * NOTE_HEIGHT_RATIO / 1.5 * 1.27;
        let first = self.beat_at(time + (left - judge - note_margin) / pitch);
        let last = self.beat_at(time + (1. - judge + note_margin) / pitch);
        if draw_metronome {
            ui.fill_rect(bands.labels, alpha(Color::new(0.02, 0.025, 0.035, 1.), 0.58));
            ui.fill_rect(Rect::new(left, bands.labels.y, 1. - left, h * 0.003), alpha(WHITE, 0.08));
        }
        if draw_notes {
        // Onset heads and their glows are physically clipped above the label
        // band, including when future notes overlap each other.
        ui.scissor(Some(bands.notes));
        // Guide beats sit behind the onset heads.
        let first_beat = self.beat_at(time + (left - judge) / pitch).floor() as i64;
        let last_beat = self.beat_at(time + (1. - judge) / pitch).ceil() as i64;
        let dash_width = (h * 0.012).max(2. / ui.viewport.2.max(1) as f32);
        for beat in first_beat..=last_beat {
            let x = self.time_position_x(self.tempo.time_beats(beat as f32), time, top);
            for dash in beat_dashes(bands.notes, x, dash_width) {
                ui.fill_rect(dash, alpha(mix(self.palette[0], WHITE, 0.82), 0.62));
            }
        }
        ui.fill_rect(Rect::new(left, -h * 0.003, 1. - left, h * 0.006), alpha(WHITE, 0.05));

        // Soft line glow never obscures the note glyph, which is drawn last.
        for layer in 0..3 {
            let w = h * (0.023 - layer as f32 * 0.007);
            ui.fill_rect(Rect::new(judge - w * 0.5, bands.notes.y, w, bands.notes.h), alpha(self.palette[0], 0.09));
        }
        ui.fill_rect(Rect::new(judge - h * 0.0025, bands.notes.y, h * 0.005, bands.notes.h), alpha(WHITE, 0.72));
        text(ui, "RHYTHM", left + h * 0.12, -h * 0.40, h * 0.84, alpha(WHITE, 0.36));

        for note in self.visible_notes(first, last) {
            let x = self.position_x(note, time, top);
            let (shape, color) = note.value.style();
            let (y, radius) = note_geometry(shape, bands.notes);
            let c = self.palette[color];
            if x + radius < left || x - radius > 1. {
                continue;
            }
            let hit = (1. - (time - note.time).abs() / 0.07).max(0.);
            glyph(ui, shape, x, y + radius * 0.07, radius * 1.07, alpha(Color::new(0., 0., 0., 1.), 0.35));
            glyph(ui, shape, x, y, radius * (1.14 + hit * 0.13), alpha(c, 0.12 + hit * 0.18));
            glyph(ui, shape, x, y, radius, mix(c, WHITE, 0.12));
            glyph(ui, shape, x, y, radius * NOTE_INNER_RATIO, c);
            ui.fill_circle(x - radius * 0.22, y - radius * 0.28, radius * 0.15, alpha(WHITE, 0.20));
            match note.kind {
                RhythmKind::Drag => ui.fill_circle(x, y, radius * 0.18, alpha(WHITE, 0.82)),
                RhythmKind::Flick => {
                    let mut b = ui.builder(alpha(WHITE, 0.82));
                    b.add(x - radius * 0.25, y + radius * 0.10);
                    b.add(x + radius * 0.20, y - radius * 0.33);
                    b.add(x + radius * 0.20, y + radius * 0.40);
                    b.triangle(0, 1, 2);
                    b.commit();
                }
                RhythmKind::Hold | RhythmKind::Tap => {}
            }
        }
        ui.scissor(None);
        }
        if draw_values {
        ui.scissor(Some(bands.labels));
        // Labels are painted after every note, in a dedicated larger band.
        // Global onset aggregation supplies exactly one label per analysis note.
        let begin = self.notes.partition_point(|note| note.beat < first);
        let end = self.notes.partition_point(|note| note.beat <= last);
        for index in (begin.min(end)..end).rev() {
            let note = &self.notes[index];
            let x = self.position_x(note, time, top);
            text(ui, &note.value.label(), x, bands.label_y, bands.label_size, alpha(WHITE, 0.86));
        }
        ui.scissor(None);
        }
    }
}

fn spacing(top: f32) -> f32 {
    ((2. - 2.4 * top - 0.08) / 6.).max(0.01)
}

fn metronome_pulse(beat: f32) -> f32 {
    if !beat.is_finite() {
        return 0.;
    }
    // Converting an authored integer beat to seconds and back can land one
    // ulp below the integer. Snap only that quantization interval to the peak.
    let tolerance = 2. * f32::EPSILON * beat.abs().max(1.);
    let phase = if (beat - beat.round()).abs() <= tolerance { 0. } else { beat.rem_euclid(1.) };
    (-phase * 20.).exp()
}

pub(super) fn metronome_pulse_sampled(beat: f32, previous_beat: Option<f32>, elapsed: f32) -> f32 {
    if elapsed.is_finite() && elapsed > 0. && elapsed <= 0.1 && beat.is_finite() {
        if let Some(previous) = previous_beat.filter(|previous| previous.is_finite()) {
            // Every real beat in the continuous frame interval gets its full
            // luminous peak, even when no frame lands exactly on that beat.
            if beat.floor() > previous.floor() {
                return 1.;
            }
        }
    }
    metronome_pulse(beat)
}

const NOTE_HEIGHT_RATIO: f32 = 0.70;
const NOTE_INNER_RATIO: f32 = 0.92;

fn note_geometry(shape: RhythmShape, notes: Rect) -> (f32, f32) {
    let (min_y, max_y) = glyph_unit_y_bounds(shape);
    (notes.y + notes.h * 0.5, notes.h * NOTE_HEIGHT_RATIO / (max_y - min_y))
}

fn beat_dashes(notes: Rect, x: f32, width: f32) -> impl Iterator<Item = Rect> {
    let step = notes.h / 8.;
    (0..8).map(move |index| Rect::new(x - width * 0.5, notes.y + index as f32 * step, width, step * 0.58))
}

struct RhythmBands {
    notes: Rect,
    labels: Rect,
    label_y: f32,
    label_size: f32,
}

impl RhythmBands {
    fn new(left: f32, h: f32) -> Self {
        Self {
            notes: Rect::new(left, -h * 0.5, 1. - left, h * 0.68),
            labels: Rect::new(left, h * 0.20, 1. - left, h * 0.30),
            label_y: h * 0.35,
            label_size: h * 3.6,
        }
    }
}

fn text(ui: &mut Ui, value: &str, x: f32, y: f32, size: f32, color: Color) {
    ui.text(value).pos(x, y).anchor(0.5, 0.5).no_baseline().size(size).color(color).draw();
}

fn alpha(mut color: Color, a: f32) -> Color {
    color.a = a;
    color
}
fn mix(a: Color, b: Color, t: f32) -> Color {
    Color::new(a.r + (b.r - a.r) * t, a.g + (b.g - a.g) * t, a.b + (b.b - a.b) * t, a.a + (b.a - a.a) * t)
}

fn glyph(ui: &mut Ui, shape: RhythmShape, x: f32, y: f32, radius: f32, color: Color) {
    if shape == RhythmShape::Circle {
        ui.fill_circle(x, y, radius, color);
        return;
    }
    ui.fill_path(&glyph_path(shape, x, y, radius), color);
    if shape == RhythmShape::DoubleTriangle {
        ui.fill_circle(x, y + radius * 0.16, radius * 0.16, alpha(WHITE, 0.8));
    }
}

fn glyph_path(shape: RhythmShape, x: f32, y: f32, radius: f32) -> Path {
    let (min_y, max_y) = glyph_unit_y_bounds(shape);
    let center_y = (min_y + max_y) * 0.5;
    let mut path = Path::builder();
    for (index, pt) in glyph_unit_vertices(shape).enumerate() {
        let pt = point(x + pt.x * radius, y + (pt.y - center_y) * radius);
        if index == 0 {
            path.begin(pt);
        } else {
            path.line_to(pt);
        }
    }
    path.end(true);
    path.build()
}

fn glyph_unit_y_bounds(shape: RhythmShape) -> (f32, f32) {
    if shape == RhythmShape::Circle {
        return (-1., 1.);
    }
    glyph_unit_vertices(shape).fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), pt| (min.min(pt.y), max.max(pt.y)))
}

fn glyph_unit_vertices(shape: RhythmShape) -> impl Iterator<Item = lyon::math::Point> {
    let (vertices, angle) = match shape {
        RhythmShape::Triangle | RhythmShape::DoubleTriangle => (3, -FRAC_PI_2),
        RhythmShape::Diamond => (4, -FRAC_PI_2),
        RhythmShape::Square => (4, -FRAC_PI_4),
        RhythmShape::Hexagon => (6, -FRAC_PI_2),
        RhythmShape::Octagon => (8, -FRAC_PI_4 * 0.5),
        RhythmShape::Star => (10, -FRAC_PI_2),
        RhythmShape::Cross => (12, -FRAC_PI_4),
        RhythmShape::Circle => unreachable!(),
    };
    (0..vertices).map(move |i| {
        let a = angle + TAU * i as f32 / vertices as f32;
        let scale = if shape == RhythmShape::Star && i % 2 == 1 {
            0.48
        } else if shape == RhythmShape::Cross && i % 3 == 1 {
            0.45
        } else {
            1.
        };
        point(a.cos() * scale, a.sin() * scale)
    })
}

// Three visible accents: source hues first, perceptual separation before reuse.
// Oklab matrices: https://bottosson.github.io/posts/oklab/
const MIN_ACCENT_DISTANCE: f32 = 0.16;

/// Extract exactly the hexadecimal colors used by automatic rhythm rendering.
pub fn extract_cover_palette(cover: &[u8]) -> Result<[String; 3]> {
    let cover = image::load_from_memory(cover)
        .context("failed to decode chart illustration for rhythm colors")?
        .to_rgba8();
    Ok(cover_palette(Some(&cover)).map(color_hex))
}

/// Return the exact RGB pixel at an artwork coordinate for the UI eyedropper.
/// Coordinates use the decoded image's top-left origin, matching the browser
/// preview's `object-fit: contain` mapping.
pub fn sample_cover_color(cover: &[u8], x: u32, y: u32) -> Result<String> {
    let cover = image::load_from_memory(cover)
        .context("failed to decode chart illustration for rhythm color sampling")?
        .to_rgba8();
    ensure!(x < cover.width() && y < cover.height(), "artwork sample is outside the image");
    let pixel = cover.get_pixel(x, y);
    Ok(format!("#{:02X}{:02X}{:02X}", pixel[0], pixel[1], pixel[2]))
}

fn color_hex(color: Color) -> String {
    let [r, g, b] = [color.r, color.g, color.b].map(|channel| (channel.clamp(0., 1.) * 255.).round() as u8);
    format!("#{r:02X}{g:02X}{b:02X}")
}

fn parse_hex_color(value: &str) -> Option<Color> {
    let bytes = value.as_bytes();
    if bytes.len() != 7 || bytes[0] != b'#' || !bytes[1..].iter().all(u8::is_ascii_hexdigit) {
        return None;
    }
    let channel = |start| u8::from_str_radix(&value[start..start + 2], 16).ok().map(|value| value as f32 / 255.);
    Some(Color::new(channel(1)?, channel(3)?, channel(5)?, 1.))
}

fn quantized_color(color: Color) -> Color {
    let [r, g, b] = [color.r, color.g, color.b].map(|channel| (channel.clamp(0., 1.) * 255.).round() as u8 as f32 / 255.);
    Color::new(r, g, b, 1.)
}

pub fn cover_palette(cover: Option<&RgbaImage>) -> [Color; 3] {
    let fallback = Color::new(0.56, 0.75, 0.96, 1.);
    let Some(cover) = cover.filter(|image| image.width() > 0 && image.height() > 0) else {
        return separated_accents(&[fallback]);
    };
    let small = palette_thumbnail(cover);
    let mut average = [0.; 3];
    let mut weight = 0.;
    for pixel in small.pixels() {
        let alpha = pixel[3] as f32 / 255.;
        weight += alpha;
        for channel in 0..3 {
            average[channel] += pixel[channel] as f32 / 255. * alpha;
        }
    }
    if weight == 0. {
        return separated_accents(&[fallback]);
    }
    let samples: Vec<_> = small.pixels().filter(|pixel| pixel[3] > 0).map(|pixel| {
        let color = Color::new(pixel[0] as f32 / 255., pixel[1] as f32 / 255., pixel[2] as f32 / 255., 1.);
        (color, oklab(color), pixel[3] as f32 / 255.)
    }).collect();
    let mut sources: Vec<Color> = color_thief::get_palette(small.as_raw(), color_thief::ColorFormat::Rgba, 1, 10)
        .unwrap_or_default()
        .into_iter()
        .filter_map(|color| {
            let center = oklab(Color::new(color.r as f32 / 255., color.g as f32 / 255., color.b as f32 / 255., 1.));
            // Median-cut can return colors from empty histogram boxes. Snap
            // genuine centers to a real source sample and reject empty boxes.
            let closest = samples.iter().min_by(|a, b| lab_distance(a.1, center).total_cmp(&lab_distance(b.1, center)))?;
            (lab_distance(closest.1, center) <= 0.06).then_some(closest.0)
        })
        .collect();
    if sources.is_empty() {
        let center = oklab(Color::new(average[0] / weight, average[1] / weight, average[2] / weight, 1.));
        if let Some(sample) = samples.iter().min_by(|a, b| lab_distance(a.1, center).total_cmp(&lab_distance(b.1, center))) {
            sources.push(sample.0);
        }
    }
    // Color Thief excludes pure white and low-opacity pixels. Retain a real
    // light/dark source cluster when it covers a meaningful part of the image,
    // rather than inserting an average color absent from a two-color cover.
    for lightest in [false, true] {
        let sample = if lightest {
            samples.iter().max_by(|a, b| a.1[0].total_cmp(&b.1[0]))
        } else {
            samples.iter().min_by(|a, b| a.1[0].total_cmp(&b.1[0]))
        };
        if let Some(sample) = sample {
            let population: f32 = samples.iter().filter(|other| lab_distance(sample.1, other.1) <= 0.06).map(|other| other.2).sum();
            if population >= weight * 0.05 {
                sources.push(sample.0);
            }
        }
    }
    separated_accents(&sources)
}

fn palette_thumbnail(cover: &RgbaImage) -> RgbaImage {
    let largest = cover.width().max(cover.height()) as u64;
    let width = (cover.width() as u64 * 128 / largest).max(1).min(cover.width() as u64) as u32;
    let height = (cover.height() as u64 * 128 / largest).max(1).min(cover.height() as u64) as u32;
    // Accumulate premultiplied colors while reducing, with a bounded output
    // allocation. Hidden RGB in transparent pixels never bleeds into covers.
    RgbaImage::from_fn(width, height, |x, y| {
        let x0 = (x as u64 * cover.width() as u64 / width as u64) as u32;
        let x1 = ((x as u64 + 1) * cover.width() as u64 / width as u64) as u32;
        let y0 = (y as u64 * cover.height() as u64 / height as u64) as u32;
        let y1 = ((y as u64 + 1) * cover.height() as u64 / height as u64) as u32;
        let mut sum = [0.; 3];
        let mut alpha = 0.;
        for y in y0..y1 { for x in x0..x1 {
            let pixel = cover.get_pixel(x, y);
            let weight = pixel[3] as f32 / 255.;
            alpha += weight;
            for channel in 0..3 { sum[channel] += pixel[channel] as f32 * weight; }
        } }
        if alpha == 0. { return image::Rgba([0, 0, 0, 0]); }
        image::Rgba([
            (sum[0] / alpha).round() as u8, (sum[1] / alpha).round() as u8, (sum[2] / alpha).round() as u8,
            (alpha * 255. / ((x1 - x0) as f32 * (y1 - y0) as f32)).round() as u8,
        ])
    })
}

fn separated_accents(sources: &[Color]) -> [Color; 3] {
    // Keep the illustration's natural lightness, chroma, and neutral colors.
    // A source already separated from earlier accents wins over any variant.
    let normalized: Vec<_> = sources.iter().copied().map(visible_accent).collect();
    let first = normalized.first().copied().unwrap_or_else(|| quantized_color(Color::new(0.56, 0.75, 0.96, 1.)));
    let lab = oklab(first);
    let hue = lab[2].atan2(lab[1]);
    let chroma = if lab[1].hypot(lab[2]) < 0.002 { 0. } else { lab[1].hypot(lab[2]) };
    // If the cover has fewer than three distinguishable colors, vary only the
    // same source hue's lightness. Do not invent complementary green/orange.
    let derived: Vec<_> = [0.30, 0.36, 0.42, 0.48, 0.54, 0.60, 0.66, 0.72, 0.78, 0.84, 0.90, 0.96]
        .into_iter().map(|light| quantized_color(gamut_mapped(light, chroma, hue))).collect();
    let mut accents = [first; 3];
    for index in 1..3 {
        let prior = &accents[..index];
        let score = |color: Color| prior.iter().map(|&other| distance(color, other)).fold(f32::INFINITY, f32::min);
        let source = normalized.iter().copied().max_by(|a, b| score(*a).total_cmp(&score(*b))).unwrap_or(first);
        accents[index] = if score(source) >= MIN_ACCENT_DISTANCE {
            source
        } else {
            derived.iter().copied().max_by(|a, b| score(*a).total_cmp(&score(*b))).unwrap()
        };
    }
    // Check after u8 quantization, which the extraction API and renderer share.
    // The bounded fallback also stays in the same source hue/neutral family.
    if (1..3).any(|i| (0..i).any(|j| distance(accents[i], accents[j]) < MIN_ACCENT_DISTANCE)) {
        return [
            quantized_color(gamut_mapped(0.30, chroma, hue)),
            quantized_color(gamut_mapped(0.63, chroma, hue)),
            quantized_color(gamut_mapped(0.96, chroma, hue)),
        ];
    }
    accents
}

fn visible_accent(color: Color) -> Color {
    let lab = oklab(color);
    if lab[0] >= 0.30 {
        return quantized_color(color);
    }
    // Only near-black source colors need a minimal visibility adjustment;
    // ordinary shadows and all other natural source colors remain untouched.
    let chroma = if lab[1].hypot(lab[2]) < 0.002 { 0. } else { lab[1].hypot(lab[2]) };
    quantized_color(gamut_mapped(0.30, chroma, lab[2].atan2(lab[1])))
}

fn gamut_mapped(light: f32, chroma: f32, hue: f32) -> Color {
    let rgb = |chroma: f32| linear_rgb([light, chroma * hue.cos(), chroma * hue.sin()]);
    let valid = |rgb: [f32; 3]| rgb.iter().all(|&channel| (0. ..=1.).contains(&channel));
    let mut low = 0.;
    let mut high = chroma;
    if !valid(rgb(high)) {
        for _ in 0..14 {
            let middle = (low + high) * 0.5;
            if valid(rgb(middle)) { low = middle; } else { high = middle; }
        }
        high = low;
    }
    let color = rgb(high).map(|channel| {
        let channel = channel.clamp(0., 1.);
        if channel <= 0.0031308 { channel * 12.92 } else { 1.055 * channel.powf(1. / 2.4) - 0.055 }
    });
    Color::new(color[0], color[1], color[2], 1.)
}

fn oklab(color: Color) -> [f32; 3] {
    let linear = [color.r, color.g, color.b].map(|channel| {
        if channel <= 0.04045 { channel / 12.92 } else { ((channel + 0.055) / 1.055).powf(2.4) }
    });
    let [r, g, b] = linear;
    let l = (0.41222146 * r + 0.53633255 * g + 0.051445995 * b).cbrt();
    let m = (0.21190350 * r + 0.68069955 * g + 0.10739696 * b).cbrt();
    let s = (0.08830246 * r + 0.28171885 * g + 0.62997870 * b).cbrt();
    [
        0.21045426 * l + 0.79361779 * m - 0.004072047 * s,
        1.9779985 * l - 2.4285922 * m + 0.45059371 * s,
        0.025904037 * l + 0.78277177 * m - 0.80867577 * s,
    ]
}

fn linear_rgb([light, a, b]: [f32; 3]) -> [f32; 3] {
    let l = (light + 0.39633778 * a + 0.21580376 * b).powi(3);
    let m = (light - 0.10556135 * a - 0.06385417 * b).powi(3);
    let s = (light - 0.08948418 * a - 1.2914855 * b).powi(3);
    [
        4.0767417 * l - 3.3077116 * m + 0.23096994 * s,
        -1.2684380 * l + 2.6097574 * m - 0.34131940 * s,
        -0.0041960863 * l - 0.70341861 * m + 1.7076147 * s,
    ]
}

fn distance(a: Color, b: Color) -> f32 {
    lab_distance(oklab(a), oklab(b))
}

fn lab_distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rhythm_render_layers_are_independently_selectable() {
        let all = RhythmRenderLayer::All;
        assert!(all.draws_metronome() && all.draws_notes() && all.draws_values());
        for layer in [RhythmRenderLayer::Metronome, RhythmRenderLayer::Note, RhythmRenderLayer::Value] {
            let flags = [layer.draws_metronome(), layer.draws_notes(), layer.draws_values()];
            assert_eq!(flags.into_iter().filter(|enabled| *enabled).count(), 1);
        }
        assert!(RhythmRenderLayer::Metronome.draws_metronome());
        assert!(RhythmRenderLayer::Note.draws_notes());
        assert!(RhythmRenderLayer::Value.draws_values());
    }

    fn strip(events: Vec<(f32, f32, RhythmKind)>, tempo: BpmList) -> RhythmStrip {
        RhythmStrip::from_events(tempo, events, cover_palette(None))
    }

    fn auto_bpm_fixture(authored: bool) -> (Chart, Vec<u8>, sasa::AudioClip) {
        use serde_json::json;
        let mut source = json!({"offset":0.,"judgeLineList":[{
            "bpm":120.,"judgeLineDisappearEvents":[],"judgeLineRotateEvents":[],
            "judgeLineMoveEvents":[],"speedEvents":[{"startTime":0.,"endTime":10000.,"value":1.}],
            "notesAbove":[
                {"type":1,"time":1152.,"positionX":0.,"holdTime":0.,"speed":1.,"floorPosition":0.},
                {"type":3,"time":1280.,"positionX":0.,"holdTime":128.,"speed":1.,"floorPosition":0.}
            ],"notesBelow":[]
        }]});
        if authored {
            source["BPMList"] = json!([
                {"startTime":[0,0,1],"bpm":120.},
                {"startTime":[4,0,1],"bpm":180.},
                {"startTime":[10,0,1],"bpm":90.}
            ]);
        }
        let chart = crate::parse::parse_phigros(&source.to_string(), super::super::ChartExtra::default()).unwrap();
        let mut cover = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(RgbaImage::from_pixel(2, 2, image::Rgba([40, 80, 120, 255])))
            .write_to(&mut cover, image::ImageOutputFormat::Png).unwrap();
        (chart, cover.into_inner(), auto_bpm_audio(&[60., 100.], &[16., 16.]))
    }

    fn auto_bpm_audio(bpms: &[f64], lengths: &[f64]) -> sasa::AudioClip {
        let mut frames = vec![sasa::Frame(0., 0.); ((lengths.iter().sum::<f64>() + 0.125) * 8000.) as usize];
        let mut segment_start = 0.125;
        for (&bpm, &length) in bpms.iter().zip(lengths) {
            let mut beat = segment_start;
            let end = segment_start + length;
            while beat < end - 1e-6 {
                let start = (beat * 8000.).round() as usize;
                for i in 0..400.min(frames.len() - start) {
                    let time = i as f64 / 8000.;
                    let sample = ((-time / 0.012).exp() * (std::f64::consts::TAU * 1800. * time).sin()) as f32;
                    // Complete antiphase cancellation under a naïve mono sum.
                    frames[start + i] = sasa::Frame(sample, -sample);
                }
                beat += 60. / bpm;
            }
            segment_start = end;
        }
        sasa::AudioClip::from_raw(frames, 8000)
    }

    #[test]
    fn rhythm_auto_bpm_defaults_and_authored_or_disabled_maps_remain_exact() {
        let legacy: Config = serde_json::from_str("{}").unwrap();
        assert!(legacy.rhythm_auto_bpm);
        let disabled: Config = serde_json::from_str(r#"{"rhythmAutoBpm":false}"#).unwrap();
        assert!(!disabled.rhythm_auto_bpm);
        let restored: Config = serde_json::from_str(&serde_json::to_string(&disabled).unwrap()).unwrap();
        assert!(!restored.rhythm_auto_bpm);
        for authored in [false, true] {
            let (chart, cover, music) = auto_bpm_fixture(authored);
            assert_eq!(chart.rhythm_bpm_authored, authored);
            let baseline = RhythmStrip::from_chart_bytes(&chart, RhythmNoteSelection::default(), &cover).unwrap();
            let inferred = RhythmStrip::from_chart_bytes_with_audio(
                &chart, RhythmNoteSelection::default(), &cover, &music, authored, 9.75,
            ).unwrap();
            assert_eq!(inferred.tempo.points(), baseline.tempo.points());
            let notes = |strip: &RhythmStrip| strip.notes.iter().map(|note|
                (note.time, note.end_time, note.beat, note.end_beat, note.value)).collect::<Vec<_>>();
            assert_eq!(notes(&inferred), notes(&baseline));
        }
    }

    #[test]
    fn rhythm_auto_bpm_antiphase_audio_fixed_scale_offsets_and_notes_share_seconds() {
        let (chart, cover, music) = auto_bpm_fixture(false);
        let build = |offset| RhythmStrip::from_chart_bytes_with_audio(
            &chart, RhythmNoteSelection::default(), &cover, &music, true, offset,
        ).unwrap();
        let raw_axis = build(0.);
        let chart_axis = build(0.375);
        assert_eq!(raw_axis.tempo.points().len(), 2);
        assert_eq!(raw_axis.bpm_at(0.), 120.);
        assert_eq!(raw_axis.bpm_at(18.), 200.);
        assert!((chart_axis.beat_at(0.128 - 0.375)).abs() < 0.02);
        for (raw, shifted) in raw_axis.tempo.points().iter().zip(chart_axis.tempo.points()) {
            assert_eq!(raw.0, shifted.0);
            assert_eq!(raw.2, shifted.2);
            assert!((raw.1 - shifted.1 - 0.375).abs() < 1e-5);
        }
        let baseline = RhythmStrip::from_chart(&chart, RhythmNoteSelection::default(), None);
        for (original, inferred) in baseline.notes.iter().zip(&chart_axis.notes) {
            assert_eq!(original.time, inferred.time);
            assert_eq!(original.end_time, inferred.end_time);
            assert_eq!(inferred.beat, chart_axis.beat_at(inferred.time));
            assert_eq!(inferred.end_beat, chart_axis.beat_at(inferred.end_time));
            assert_ne!(original.beat, inferred.beat);
        }
    }

    #[test]
    fn rhythm_auto_bpm_short_middle_run_integrates_a_strict_continuous_beat_axis() {
        let (chart, cover, _) = auto_bpm_fixture(false);
        let music = auto_bpm_audio(&[120., 180., 90.], &[16., 8., 16.]);
        let inferred = RhythmStrip::from_chart_bytes_with_audio(
            &chart, RhythmNoteSelection::default(), &cover, &music, true, 0.375,
        ).unwrap();
        assert_eq!(inferred.tempo.points().iter().map(|point| point.2.round() as i32).collect::<Vec<_>>(), vec![120, 180, 90]);
        for nodes in inferred.tempo.points().windows(2) {
            let (beat, time, bpm) = nodes[0];
            let (next_beat, next_time, _) = nodes[1];
            assert!(next_time > time && next_beat > beat);
            assert!((next_beat - beat - (next_time - time) * bpm / 60.).abs() < 1e-4);
            assert!((inferred.beat_at(next_time) - next_beat).abs() < 1e-5);
            let around = inferred.beat_at(next_time + 0.0001) - inferred.beat_at(next_time - 0.0001);
            assert!(around > 0. && around < 0.002, "node {next_time}: jump {around}");
        }
    }

    #[test]
    fn rhythm_adjust_layout_preserves_game_with_one_tenth_extension() {
        let layout = RhythmLayout::new(1920, 1080, true).unwrap();
        assert_eq!((layout.game_height, layout.strip_height, layout.output_height), (1080, 192, 1272));
        assert!((layout.judge_x_pixels() - 230.4).abs() < 0.0001);
        assert_eq!(RhythmLayout::new(320, 180, true).unwrap().output_height, 212);
        assert_eq!(RhythmLayout::new(1280, 720, true).unwrap().output_height, 848);
        assert_eq!(RhythmLayout::new(16, 16, true).unwrap().strip_height, 2);
        assert_eq!(RhythmLayout::new(1920, 1080, false).unwrap().output_height, 1080);
        assert!(RhythmLayout::new(1919, 1080, true).is_err());
        assert!(RhythmLayout::new(2_147_483_646, 2_147_483_646, true).is_err());
    }

    #[test]
    fn tempo_change_keeps_hit_and_reference_beats_exactly_on_the_line() {
        let tempo = BpmList::new(vec![(0., 120.), (4., 180.), (10., 90.)]);
        let times: Vec<_> = [0., 0.5, 1., 3.5, 4., 4.25, 6., 10., 10.5]
            .into_iter()
            .map(|beat| tempo.time_beats(beat))
            .collect();
        let strip = strip(times.iter().map(|&time| (time, time, RhythmKind::Tap)).collect(), tempo);
        for note in strip.notes() {
            assert!((strip.position_x(note, note.time, 1. / 7.) - (-1. + 2.4 / 7.)).abs() < 0.00001);
            assert!(strip.position_x(note, note.time - 0.01, 1. / 7.) > strip.position_x(note, note.time, 1. / 7.));
            assert!(strip.position_x(note, note.time + 0.01, 1. / 7.) < strip.position_x(note, note.time, 1. / 7.));
        }
        assert_eq!(strip.bpm_at(tempo_change_time()), 180.);
        fn tempo_change_time() -> f32 {
            2.
        }
    }

    #[test]
    fn rhythm_numeric_simultaneous_hold_starts_merge_with_taps_and_keep_kind_priority() {
        let tempo = BpmList::new(vec![(0., 120.), (4., 240.)]);
        let strip = strip(
            vec![
                (1.75, 2.25, RhythmKind::Hold),
                (1.75, 1.75, RhythmKind::Tap),
                (1.75, 1.75, RhythmKind::Flick),
                (1.75003, 2., RhythmKind::Hold),
                (2., 2., RhythmKind::Tap),
            ],
            tempo,
        );
        assert_eq!(strip.notes().len(), 2);
        let onset = &strip.notes()[0];
        assert_eq!(onset.time, 1.75);
        assert_eq!(onset.kind, RhythmKind::Flick);
        assert_eq!(onset.value, RhythmValue::Eighth);
        assert_eq!(onset.end_time, onset.time);
        assert_eq!(onset.end_beat, onset.beat);
        assert_eq!(strip.notes()[1].value, RhythmValue::Quarter);
    }

    #[test]
    fn rhythm_adjust_speed_scales_positions_without_changing_tempo_or_judgement() {
        let tempo = BpmList::new(vec![(0., 120.), (4., 180.)]);
        let mut strip = strip(vec![(1., 1., RhythmKind::Tap), (2., 2.5, RhythmKind::Hold)], tempo);
        let notes = strip.notes().to_vec();
        let top = 0.1;
        let now = 1.5;
        let judge = -1. + 2.4 * top;
        let gap = strip.position_x(&notes[1], now, top) - strip.position_x(&notes[0], now, top);
        let movement = strip.position_x(&notes[1], now, top) - strip.position_x(&notes[1], now + 0.05, top);
        for speed in [0.1, 0.5, 1., 2., 10.] {
            strip.set_speed(speed);
            for note in &notes {
                assert!((strip.position_x(note, note.time, top) - judge).abs() < 0.00001);
                assert!(strip.position_x(note, note.time - 0.01, top) > judge);
                assert!(strip.position_x(note, note.time + 0.01, top) < judge);
            }
            let scaled_gap = strip.position_x(&notes[1], now, top) - strip.position_x(&notes[0], now, top);
            let scaled_movement = strip.position_x(&notes[1], now, top) - strip.position_x(&notes[1], now + 0.05, top);
            assert!((scaled_gap - gap * speed).abs() < 0.00001);
            assert!((scaled_movement - movement * speed).abs() < 0.00001);
            assert_eq!(strip.bpm_at(now), 120.);
            assert_eq!(strip.bpm_at(2.1), 180.);
            assert_eq!(strip.beat_at(now), 3.);
            assert_eq!(strip.notes()[1].end_beat, notes[1].end_beat);
        }
    }

    #[test]
    fn rhythm_adjust_speed_defaults_and_invalid_boundaries_remain_finite() {
        let config: Config = serde_json::from_str("{}").unwrap();
        assert_eq!(config.rhythm_speed, 1.);
        let config: Config = serde_json::from_str(r#"{"rhythmSpeed":2.5}"#).unwrap();
        assert_eq!(config.rhythm_speed, 2.5);
        let mut strip = strip(Vec::new(), BpmList::default()).with_speed(config.rhythm_speed);
        assert_eq!(strip.speed(), 2.5);
        for (input, expected) in [
            (0., 0.1),
            (-2., 0.1),
            (0.1, 0.1),
            (10., 10.),
            (100., 10.),
            (f32::NAN, 1.),
            (f32::INFINITY, 1.),
            (f32::NEG_INFINITY, 1.),
        ] {
            strip.set_speed(input);
            assert_eq!(strip.speed(), expected);
            assert!(strip.time_pitch(0.1).is_finite() && strip.time_pitch(0.1) > 0.);
        }
    }

    #[test]
    fn rhythm_132_global_merged_heads_keep_earlier_judgements_on_top() {
        let strip = strip(
            vec![
                (0.5, 0.5, RhythmKind::Tap),
                (0.5, 0.5, RhythmKind::Drag),
                (1., 1., RhythmKind::Flick),
                (2., 3., RhythmKind::Hold),
            ],
            BpmList::new(vec![(0., 120.)]),
        );
        let order: Vec<_> = strip.visible_notes(0., 5.).map(|note| (note.time, note.kind)).collect();
        assert_eq!(
            order,
            vec![
                (2., RhythmKind::Tap),
                (1., RhythmKind::Flick),
                (0.5, RhythmKind::Drag)
            ]
        );
        let clipped: Vec<_> = strip.visible_notes(2., 4.).map(|note| note.time).collect();
        assert_eq!(clipped, vec![2., 1.]);
        assert!(strip.visible_notes(8., 9.).next().is_none());
    }

    #[test]
    fn rhythm_numeric_hold_starts_leave_no_visible_tail_after_the_head() {
        let strip = strip(
            vec![
                (0., 5., RhythmKind::Hold),
                (0.5, 0.75, RhythmKind::Hold),
                (1., 1., RhythmKind::Tap),
                (2., 2., RhythmKind::Flick),
                (3., 4., RhythmKind::Hold),
            ],
            BpmList::new(vec![(0., 120.)]),
        );
        // Only starts inside the visible range count. Long authored tails
        // must not keep an expired onset visible or fill otherwise empty beats.
        let order: Vec<_> = strip.visible_notes(3., 8.).map(|note| note.time).collect();
        assert_eq!(order, vec![3., 2.]);
        let order: Vec<_> = strip.visible_notes(0., 8.).map(|note| note.time).collect();
        assert_eq!(order, vec![3., 2., 1., 0.5, 0.]);
    }

    #[test]
    fn rhythm_132_solid_shapes_fill_notes_band_and_stay_visually_centered() {
        for h in [0.125, 0.2, 0.4] {
            let bands = RhythmBands::new(-1. + h, h);
            assert!(bands.notes.y + bands.notes.h < bands.labels.y);
            for value in RhythmValue::ALL {
                let shape = value.style().0;
                let (y, radius) = note_geometry(shape, bands.notes);
                let bounds = |scale: f32| {
                    if shape == RhythmShape::Circle {
                        return (y - radius * scale, y + radius * scale);
                    }
                    // Inspect the same centered solid paths sent to the GPU.
                    glyph_path(shape, 0., y, radius * scale).iter().fold(
                        (f32::INFINITY, f32::NEG_INFINITY),
                        |(min, max), event| {
                            let at = match event {
                                lyon::path::PathEvent::Begin { at } => at,
                                lyon::path::PathEvent::Line { to, .. } => to,
                                _ => return (min, max),
                            };
                            (min.min(at.y), max.max(at.y))
                        },
                    )
                };
                for scale in [1., NOTE_INNER_RATIO] {
                    let (min, max) = bounds(scale);
                    assert!((max - min - bands.notes.h * NOTE_HEIGHT_RATIO * scale).abs() < 0.000001);
                    assert!(((min + max) * 0.5 - (bands.notes.y + bands.notes.h * 0.5)).abs() < 0.000001);
                    assert!(min >= bands.notes.y && max <= bands.notes.y + bands.notes.h);
                    assert!(max < bands.labels.y);
                }
                // The largest hit glow fits the reduced 70% head height; the
                // production scissor still protects the separate label band.
                let (min, max) = bounds(1.27);
                assert!(min >= bands.notes.y && max <= bands.notes.y + bands.notes.h);
                assert!(y - radius * 0.32 > bands.notes.y);
                assert!(y + radius * 0.32 < bands.notes.y + bands.notes.h);
            }
            assert!(bands.label_y > bands.labels.y);
            assert!(bands.label_y < bands.labels.y + bands.labels.h);
            assert_eq!(bands.label_size, h * 3.6); // readable Saira caps in the exported video
        }
    }

    #[test]
    fn conventional_subdivisions_and_triplets_keep_distinct_visual_codes() {
        assert_eq!(RhythmValue::classify(1.), RhythmValue::Quarter);
        assert_eq!(RhythmValue::classify(0.5), RhythmValue::Eighth);
        assert_eq!(RhythmValue::classify(0.25), RhythmValue::Sixteenth);
        assert_eq!(RhythmValue::classify(1. / 3.), RhythmValue::EighthTriplet);
        assert_eq!(RhythmValue::classify(0.75), RhythmValue::DottedEighth);
        assert_eq!(RhythmValue::classify(0.42), RhythmValue::Tuplet(25));
        let styles: std::collections::HashSet<_> = RhythmValue::ALL.into_iter().map(RhythmValue::style).collect();
        assert_eq!(styles.len(), RhythmValue::ALL.len());
        assert_eq!(RhythmValue::Quarter.style().0, RhythmShape::Circle);
        assert_eq!(RhythmValue::EighthTriplet.style().0, RhythmShape::Triangle);
    }

    #[test]
    fn rhythm_numeric_labels_replace_triplet_and_dotted_notation() {
        for (beats, label) in [
            (1., "4"), (0.5, "8"), (0.25, "16"),
            (2. / 3., "3"), (1. / 3., "6"), (1. / 6., "12"), (1. / 12., "24"),
            (0.75, "16/3"), (0.375, "32/3"),
        ] {
            assert_eq!(RhythmValue::classify(beats).label(), label);
        }
        for value in RhythmValue::ALL {
            assert!(value.label().chars().all(|character| character.is_ascii_digit() || character == '/'));
        }
        for count in [5, 7, 13] {
            assert_eq!(RhythmValue::classify(1. / count as f32).label(), count.to_string());
        }
    }

    #[test]
    fn rhythm_numeric_long_gaps_have_one_quarter_onset_and_empty_following_beats() {
        for beats in [1., 1.5, 2., 3., 4., 12.] {
            assert_eq!(RhythmValue::classify(beats), RhythmValue::Quarter);
            assert_eq!(RhythmValue::classify(beats).label(), "4");
        }
        let analysis = strip(
            vec![(0., 10., RhythmKind::Hold), (6., 6., RhythmKind::Tap)],
            BpmList::new(vec![(0., 120.), (4., 180.)]),
        );
        assert_eq!(analysis.notes().len(), 2);
        assert!(analysis.notes().iter().all(|note| note.value == RhythmValue::Quarter));
        assert!(analysis.notes().iter().all(|note| note.kind == RhythmKind::Tap && note.end_time == note.time));
        assert!(analysis.visible_notes(1., 10.).next().is_none());
        // Beat guides still query the complete authored musical clock.
        assert_eq!(analysis.bpm_at(3.), 180.);
        assert_eq!(analysis.beat_at(3.), 7.);
    }

    #[test]
    fn rhythm_132_tuplet_counts_follow_actual_fraction_and_binary_subdivision() {
        for count in [5, 7, 9, 11, 13, 17, 25, 53] {
            for numerator in [1., 2., 4.] {
                let beats = numerator / count as f32;
                assert_eq!(RhythmValue::classify(beats), RhythmValue::Tuplet(count));
                assert_eq!(RhythmValue::classify(beats + beats * 0.00001), RhythmValue::Tuplet(count));
                assert_eq!(RhythmValue::classify(beats).label(), count.to_string());
            }
            assert_eq!(RhythmValue::classify(1. / (count as f32 * 2.)), RhythmValue::Tuplet(count));
        }
        assert_eq!(RhythmValue::classify(3. / 16.), RhythmValue::Ratio(3, 16));
        assert_eq!(RhythmValue::classify(3. / 16.).label(), "3/16");
        for invalid in [f32::NAN, f32::INFINITY, 0., -1.] {
            assert_eq!(RhythmValue::classify(invalid).label(), "4");
        }
    }

    #[test]
    fn rhythm_132_time_axis_guides_follow_actual_tempo_density() {
        let strip = strip(Vec::new(), BpmList::new(vec![(0., 120.), (4., 180.), (10., 90.)]));
        let top = 0.1;
        let guide = |beat| strip.time_position_x(strip.tempo.time_beats(beat), 1., top);
        let medium_gap = guide(2.) - guide(1.);
        let dense_gap = guide(6.) - guide(5.);
        let sparse_gap = guide(12.) - guide(11.);
        assert!((medium_gap / dense_gap - 1.5).abs() < 0.00001);
        assert!((sparse_gap / medium_gap - 4. / 3.).abs() < 0.00001);
        assert!(dense_gap < medium_gap && medium_gap < sparse_gap);
        assert_eq!(strip.bpm_at(0.), 120.);
        assert_eq!(strip.bpm_at(2.001), 180.);
        assert_eq!(strip.bpm_at(4.001), 90.);
        let judge = -1. + 2.4 * top;
        for beat in 0..20 {
            let time = strip.tempo.time_beats(beat as f32);
            assert!((strip.time_position_x(time, time, top) - judge).abs() < 0.000001);
        }
    }

    #[test]
    fn rhythm_numeric_onset_only_heads_cross_tempo_nodes_continuously() {
        let strip = strip(
            vec![(1.5, 5.5, RhythmKind::Hold), (3., 3., RhythmKind::Tap), (4.5, 4.5, RhythmKind::Tap)],
            BpmList::new(vec![(0., 120.), (4., 180.), (10., 90.)]),
        );
        let top = 0.1;
        let epsilon = 0.001;
        assert_eq!(strip.notes()[0].end_time, 1.5);
        for boundary in [2., 4.] {
            for note in strip.notes() {
                let before = strip.position_x(note, boundary - epsilon, top);
                let at = strip.position_x(note, boundary, top);
                let after = strip.position_x(note, boundary + epsilon, top);
                assert!((before - at - epsilon * strip.time_pitch(top)).abs() < 0.000001);
                assert!((at - after - epsilon * strip.time_pitch(top)).abs() < 0.000001);
                assert_eq!(note.end_time, note.time);
                assert_eq!(note.end_beat, note.beat);
                let first_time = boundary - 0.2;
                let last_time = boundary + 0.2;
                let visible: Vec<_> = strip.visible_notes(strip.beat_at(first_time), strip.beat_at(last_time))
                    .map(|note| note.time).collect();
                assert!(!visible.contains(&1.5)); // authored tail never extends the onset
            }
        }
        for note in strip.notes() {
            assert!((strip.position_x(note, note.time, top) - (-1. + 2.4 * top)).abs() < 0.000001);
        }
    }

    #[test]
    fn rhythm_132_time_axis_constant_tempo_preserves_previous_default_spacing() {
        for bpm in [90., 120., 180.] {
            let mut strip = strip(vec![(1., 1., RhythmKind::Tap)], BpmList::new(vec![(0., bpm)]));
            for speed in [0.1, 1., 10.] {
                strip.set_speed(speed);
                let note = &strip.notes()[0];
                let time = 0.4;
                let top = 0.1;
                let previous = -1. + 2.4 * top + (note.beat - strip.beat_at(time)) * spacing(top) * speed;
                assert!((strip.position_x(note, time, top) - previous).abs() < 0.000001);
            }
        }
    }

    #[test]
    fn rhythm_132_time_axis_every_authored_beat_has_the_same_pulse_peak() {
        let tempo = BpmList::new(vec![(0., 120.), (4., 180.), (10., 90.)]);
        for beat in -4..80 {
            let time = tempo.time_beats(beat as f32);
            assert_eq!(metronome_pulse(tempo.beat(time)), 1., "beat {beat}, time {time}");
            for phase in [0.125, 0.25, 0.5] {
                let time = tempo.time_beats(beat as f32 + phase);
                assert!((metronome_pulse(tempo.beat(time)) - (-phase * 20.).exp()).abs() < 0.00001);
            }
        }
        assert!(metronome_pulse(3.9999) < 0.00001); // snap only the tiny float quantization interval
        let slow_period = tempo.time_beats(12.) - tempo.time_beats(11.);
        let fast_period = tempo.time_beats(6.) - tempo.time_beats(5.);
        assert!((slow_period / fast_period - 2.).abs() < 0.00001);
    }

    #[test]
    fn rhythm_sampled_metronome_every_137_and_variable_tempo_beat_reaches_one() {
        for tempo in [BpmList::new(vec![(0., 137.)]), BpmList::new(vec![(0., 120.), (4., 180.), (10., 90.)])] {
            for fps in [30., 60.] {
                let mut previous_time = -0.125;
                let mut previous_beat = tempo.beat(previous_time);
                let mut crossings = 0;
                for frame in 1..=(16. * fps) as usize {
                    let time = frame as f32 / fps - 0.125;
                    let beat = tempo.beat(time);
                    let pulse = metronome_pulse_sampled(beat, Some(previous_beat), time - previous_time);
                    if beat.floor() > previous_beat.floor() {
                        assert_eq!(pulse, 1., "{fps}fps at beat {beat}");
                        crossings += 1;
                    } else {
                        assert_eq!(pulse, metronome_pulse(beat));
                    }
                    previous_time = time;
                    previous_beat = beat;
                }
                assert!(crossings > 20);
            }
        }
    }

    #[test]
    fn rhythm_sampled_metronome_snapshots_seeks_and_repeat_use_static_phase() {
        for elapsed in [0., -0.02, 0.101, 1., f32::NAN, f32::INFINITY] {
            assert_eq!(metronome_pulse_sampled(5.125, Some(4.99), elapsed), metronome_pulse(5.125));
        }
        assert_eq!(metronome_pulse_sampled(5.125, None, 0.02), metronome_pulse(5.125));
        assert_eq!(metronome_pulse_sampled(5.125, Some(f32::NAN), 0.02), metronome_pulse(5.125));
        assert_eq!(metronome_pulse_sampled(4.99, Some(5.125), 0.02), metronome_pulse(4.99));
        assert_eq!(metronome_pulse_sampled(5.125, Some(4.99), 0.1), 1.);
    }

    #[test]
    fn rhythm_numeric_tuplet_onsets_and_hold_starts_survive_large_offsets() {
        for offset in [0., 16., 128., 1000.] {
            for count in [5, 7, 13] {
                let duration = 1. / count as f32;
                let strip = strip(
                    vec![
                        (offset, offset + duration, RhythmKind::Hold),
                        (offset, offset, RhythmKind::Tap),
                        (offset + duration * 0.5, offset + duration * 0.5, RhythmKind::Tap),
                    ],
                    BpmList::new(vec![(0., 120.)]),
                );
                // Only the next global start determines the value, even when
                // a coincident hold lasts longer than that onset interval.
                assert_eq!(strip.notes()[0].value, RhythmValue::Tuplet(count), "hold {offset}/{count}");
                assert_eq!(strip.notes().len(), 2);
                assert_eq!(strip.notes()[0].value.label(), count.to_string());
                let taps = super::tests::strip(
                    vec![
                        (offset, offset, RhythmKind::Tap),
                        (offset + duration * 0.5, offset + duration * 0.5, RhythmKind::Tap),
                    ],
                    BpmList::new(vec![(0., 120.)]),
                );
                assert_eq!(taps.notes()[0].value, RhythmValue::Tuplet(count), "tap {offset}/{count}");
            }
        }
    }

    #[test]
    fn rhythm_132_beat_dashes_cover_the_notes_band_with_visible_gaps() {
        for h in [0.125, 0.2, 0.4] {
            let bands = RhythmBands::new(-1. + h, h);
            let dashes: Vec<_> = beat_dashes(bands.notes, 0.2, 0.01).collect();
            assert_eq!(dashes.len(), 8);
            assert_eq!(dashes[0].y, bands.notes.y);
            assert!(dashes.last().unwrap().y + dashes.last().unwrap().h >= bands.notes.y + bands.notes.h * 0.9);
            for dash in &dashes {
                assert!(dash.h > 0. && dash.w > 0.);
                assert!(dash.y >= bands.notes.y && dash.y + dash.h <= bands.notes.y + bands.notes.h);
                assert!(dash.y + dash.h < bands.labels.y);
            }
            for pair in dashes.windows(2) {
                assert!(pair[0].y + pair[0].h < pair[1].y);
                assert_eq!(pair[0].w, pair[1].w);
            }
        }
    }

    #[test]
    fn rhythm_numeric_chart_lines_share_onsets_and_ignore_hold_tails_independent_of_line_order() {
        use serde_json::json;
        let note = |kind, time, hold| json!({"type":kind,"time":time,"positionX":0.,
            "holdTime":hold,"speed":1.,"floorPosition":0.});
        let line = |notes| json!({"bpm":120.,"judgeLineDisappearEvents":[],
            "judgeLineRotateEvents":[],"judgeLineMoveEvents":[],
            "speedEvents":[{"startTime":0.,"endTime":1000.,"value":1.}],
            "notesAbove":notes,"notesBelow":[]});
        let a = line(json!([note(1,32.,0.),note(2,32.,0.),note(3,64.,32.),note(1,96.,0.)]));
        let b = line(json!([note(4,32.,0.),note(3,64.,96.),note(1,96.,0.),note(2,112.,0.)]));
        let selection = RhythmNoteSelection { tap: true, drag: true, hold: true, flick: true };
        let analyse = |lines| {
            let source = json!({"offset":0.75,"judgeLineList":lines}).to_string();
            let chart = crate::parse::parse_phigros(&source, super::super::ChartExtra::default()).unwrap();
            let strip = RhythmStrip::from_chart(&chart, selection, None);
            strip.notes().iter().map(|note| (note.time, note.beat, note.end_beat, note.kind, note.value)).collect::<Vec<_>>()
        };
        let merged = analyse(json!([a.clone(), b.clone()]));
        assert_eq!(merged, analyse(json!([b, a])));
        assert_eq!(merged.len(), 4);
        assert_eq!(merged[0], (0.5, 1., 1., RhythmKind::Flick, RhythmValue::Quarter));
        assert_eq!(merged[1], (1., 2., 2., RhythmKind::Tap, RhythmValue::Quarter));
        assert_eq!(merged[2], (1.5, 3., 3., RhythmKind::Tap, RhythmValue::Eighth));
        assert_eq!(merged[3], (1.75, 3.5, 3.5, RhythmKind::Drag, RhythmValue::Quarter));
    }

    #[test]
    fn rhythm_132_global_same_line_groups_merge_but_distinct_dense_onsets_stay_separate() {
        let events = vec![
            (1., 1., RhythmKind::Tap),
            (1., 1., RhythmKind::Drag),
            (1.00003, 1.00003, RhythmKind::Flick),
            (1.00004, 1.00002, RhythmKind::Hold), // inverted tail is not a valid hold
            (1.0003, 1.0003, RhythmKind::Tap),
            (1.0006, 1.0006, RhythmKind::Tap),
        ];
        let summary = |events| strip(events, BpmList::default()).notes().iter()
            .map(|note| (note.time, note.end_beat, note.kind, note.value)).collect::<Vec<_>>();
        let merged = summary(events.clone());
        assert_eq!(merged.len(), 3);
        assert_eq!(merged[0].0, 1.);
        assert_eq!(merged[0].2, RhythmKind::Flick);
        assert_eq!(merged[0].1, 2.);
        assert_eq!(merged[1].0, 1.0003);
        assert_eq!(merged[2].0, 1.0006);
        let mut reversed = events;
        reversed.reverse();
        assert_eq!(merged, summary(reversed));
    }

    #[test]
    fn cover_accents_are_bounded_and_come_from_the_actual_image() {
        let warm = RgbaImage::from_pixel(20, 20, image::Rgba([180, 45, 20, 255]));
        let cool = RgbaImage::from_pixel(20, 20, image::Rgba([20, 45, 180, 255]));
        let a = cover_palette(Some(&warm));
        let b = cover_palette(Some(&cool));
        assert!(a[0].r > a[0].b);
        assert!(b[0].b > b[0].r);
        assert_ne!((a[0].r, a[0].b), (b[0].r, b[0].b));
        for c in a.into_iter().chain(b) {
            assert!([c.r, c.g, c.b, c.a].into_iter().all(|v| v.is_finite() && (0. ..=1.).contains(&v)));
        }
        let transparent = RgbaImage::from_pixel(1, 1, image::Rgba([255, 0, 0, 0]));
        assert_eq!(cover_palette(Some(&transparent)), cover_palette(None));
        for gray in [0, 80, 200, 255] {
            let image = RgbaImage::from_pixel(20, 20, image::Rgba([gray, gray, gray, 255]));
            let colors = cover_palette(Some(&image));
            assert!(distance(colors[0], colors[1]) > 0.01);
            assert!(distance(colors[0], colors[2]) > 0.01);
            assert!(distance(colors[1], colors[2]) > 0.01);
        }
        let mut two_tone = RgbaImage::from_pixel(20, 20, image::Rgba([156, 156, 156, 255]));
        for x in 0..10 {
            for y in 0..20 {
                two_tone.put_pixel(x, y, image::Rgba([232, 232, 232, 255]));
            }
        }
        let colors = cover_palette(Some(&two_tone));
        assert!(distance(colors[0], colors[1]) >= 0.01);
        assert!(distance(colors[0], colors[2]) >= 0.01);
        assert!(distance(colors[1], colors[2]) >= 0.01);
    }

    #[test]
    fn legacy_config_and_metronome_only_selection_are_supported() {
        let config: Config = serde_json::from_str("{}").unwrap();
        assert!(!config.rhythm_analysis);
        assert_eq!(RhythmNoteSelection::from(&config), RhythmNoteSelection::default());
        let strip = strip(Vec::new(), BpmList::default());
        assert!(strip.notes().is_empty());
        assert_eq!(strip.bpm_at(2.), 120.);
        assert_eq!(strip.beat_at(2.), 4.);
        assert!(!RhythmNoteSelection {
            tap: false,
            drag: false,
            hold: false,
            flick: false
        }
        .includes(&NoteKind::Click));
    }

    #[test]
    fn parsed_pgr_uses_populated_line_tempo_and_filters_actual_note_kinds() {
        use serde_json::json;
        let note = |kind, time, hold| {
            json!({"type":kind,"time":time,"positionX":0.,
            "holdTime":hold,"speed":1.,"floorPosition":0.})
        };
        let line = |bpm, notes| {
            json!({"bpm":bpm,"judgeLineDisappearEvents":[],
            "judgeLineRotateEvents":[],"judgeLineMoveEvents":[],
            "speedEvents":[{"startTime":0.,"endTime":1000.,"value":1.}],
            "notesAbove":notes,"notesBelow":[]})
        };
        let source = json!({"offset":0.75,"judgeLineList":[line(240.,json!([])),
            line(120.,json!([note(1,32.,0.),note(2,48.,0.),note(3,64.,32.),note(4,96.,0.)]))]})
        .to_string();
        let mut chart = crate::parse::parse_phigros(&source, super::super::ChartExtra::default()).unwrap();
        // Ordinary note effects still see the original line-index BPM table.
        assert_eq!(chart.bpm_list.borrow().now_bpm(0.), 240.);
        assert_eq!(chart.bpm_list.borrow().now_bpm(1.), 120.);
        let selected = RhythmStrip::from_chart(&chart, RhythmNoteSelection::default(), None);
        assert_eq!(selected.notes().len(), 2);
        assert_eq!(selected.bpm_at(0.), 120.);
        assert_eq!(selected.bpm_at(0.9), 120.);
        assert_eq!(selected.notes()[0].time, 0.5); // chart offset is applied only by GameScene
        assert_eq!(selected.notes()[0].beat, 1.);
        assert_eq!(selected.notes()[1].kind, RhythmKind::Tap);
        let only_holds = RhythmStrip::from_chart(&chart, RhythmNoteSelection {
            tap: false, drag: false, hold: true, flick: false,
        }, None);
        assert_eq!(only_holds.notes().len(), 1);
        assert_eq!(only_holds.notes()[0].time, 1.);
        assert_eq!(only_holds.notes()[0].kind, RhythmKind::Tap);
        assert_eq!(only_holds.notes()[0].end_time, only_holds.notes()[0].time);
        assert!(chart.lines[1].notes.iter().any(|note| matches!(note.kind, NoteKind::Hold { .. })));
        let all = RhythmNoteSelection {
            tap: true,
            drag: true,
            hold: true,
            flick: true,
        };
        let full = RhythmStrip::from_chart(&chart, all, None);
        assert_eq!(full.notes().len(), 4);
        assert_eq!(full.notes()[0].value, RhythmValue::Eighth);
        assert_eq!(full.notes()[3].value, RhythmValue::Quarter);
        chart.lines[1]
            .notes
            .iter_mut()
            .find(|note| matches!(note.kind, NoteKind::Drag))
            .unwrap()
            .fake = true;
        assert_eq!(RhythmStrip::from_chart(&chart, all, None).notes().len(), 3);
        let none = RhythmNoteSelection {
            tap: false,
            drag: false,
            hold: false,
            flick: false,
        };
        let only_bpm = RhythmStrip::from_chart(&chart, none, None);
        assert!(only_bpm.notes().is_empty());
        assert_eq!(only_bpm.bpm_at(8.), 120.);
    }
    #[test]
    fn cover_accents_use_visible_perceptually_separated_colors() {
        let mut images: Vec<_> = [0, 80, 156, 200, 255].into_iter()
            .map(|gray| RgbaImage::from_pixel(24, 24, image::Rgba([gray, gray, gray, 255])))
            .collect();
        images.push(RgbaImage::from_pixel(24, 24, image::Rgba([180, 45, 20, 255])));
        images.push(RgbaImage::from_pixel(24, 24, image::Rgba([20, 45, 180, 255])));
        images.push(RgbaImage::from_fn(48, 24, |x, _| {
            image::Rgba(if x < 24 { [156, 156, 156, 255] } else { [232, 232, 232, 255] })
        }));
        images.push(RgbaImage::from_fn(48, 24, |x, _| {
            image::Rgba(if x < 24 { [180, 45, 20, 255] } else { [185, 49, 24, 255] })
        }));
        for image in images {
            let accents = cover_palette(Some(&image));
            assert_eq!(accents, cover_palette(Some(&image)));
            for color in accents {
                assert!([color.r, color.g, color.b, color.a].into_iter().all(|value| value.is_finite() && (0. ..=1.).contains(&value)));
                assert!(oklab(color)[0] >= 0.295);
            }
            for i in 1..3 { for j in 0..i {
                assert!(distance(accents[i], accents[j]) >= MIN_ACCENT_DISTANCE, "near-identical cover accents: {accents:?}");
            } }
        }
        let accents = cover_palette(None);
        for i in 1..3 { for j in 0..i { assert!(distance(accents[i], accents[j]) >= MIN_ACCENT_DISTANCE); } }
    }

    #[test]
    fn cover_palette_ignores_hidden_rgb_during_thumbnail_reduction() {
        let image = |hidden| RgbaImage::from_fn(256, 256, |x, y| {
            image::Rgba(if (x + y) % 2 == 0 { [180, 45, 20, 255] } else { hidden })
        });
        let red = RgbaImage::from_pixel(20, 20, image::Rgba([180, 45, 20, 255]));
        assert_eq!(cover_palette(Some(&image([0, 255, 0, 0]))), cover_palette(Some(&image([0, 0, 255, 0]))));
        assert_eq!(cover_palette(Some(&image([0, 255, 0, 0]))), cover_palette(Some(&red)));
    }

    #[test]
    fn rhythm_colors_automatic_palette_preserves_three_real_source_colors() {
        for source_colors in [
            [[60, 61, 83], [226, 222, 233], [140, 134, 160]],
            [[170, 34, 34], [34, 153, 68], [34, 51, 187]],
        ] {
            let image = RgbaImage::from_fn(96, 32, |x, _| {
                let [r, g, b] = source_colors[(x / 32) as usize];
                image::Rgba([r, g, b, 255])
            });
            let palette = cover_palette(Some(&image));
            let actual: std::collections::HashSet<_> = palette.into_iter().map(color_hex).collect();
            let expected: std::collections::HashSet<_> = source_colors.into_iter()
                .map(|[r, g, b]| format!("#{r:02X}{g:02X}{b:02X}")).collect();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn rhythm_colors_neutral_sources_keep_their_shade_and_generate_only_gray_variants() {
        let mut first_colors = Vec::new();
        for gray in [0, 80, 156, 200, 255] {
            let image = RgbaImage::from_pixel(24, 24, image::Rgba([gray, gray, gray, 255]));
            let palette = cover_palette(Some(&image));
            if gray > 0 {
                assert_eq!(palette[0], Color::new(gray as f32 / 255., gray as f32 / 255., gray as f32 / 255., 1.));
            }
            for color in palette {
                assert!((color.r - color.g).abs() <= 1. / 255.);
                assert!((color.g - color.b).abs() <= 1. / 255.);
            }
            first_colors.push(color_hex(palette[0]));
        }
        first_colors.sort();
        first_colors.dedup();
        assert_eq!(first_colors.len(), 5);
    }

    #[test]
    fn rhythm_colors_hex_extraction_matches_automatic_and_manual_renderer_palette_exactly() {
        use std::io::Cursor;
        let image = RgbaImage::from_fn(96, 32, |x, _| {
            image::Rgba([[60, 61, 83, 255], [226, 222, 233, 255], [140, 134, 160, 255]][(x / 32) as usize])
        });
        let mut encoded = Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(image.clone()).write_to(&mut encoded, image::ImageOutputFormat::Png).unwrap();
        let colors = extract_cover_palette(encoded.get_ref()).unwrap();
        let auto = RhythmStrip::from_events(BpmList::default(), Vec::new(), cover_palette(Some(&image)));
        let automatic = auto.palette();
        assert_eq!(colors, automatic.map(color_hex));
        assert_eq!(auto.with_colors(Some(&colors)).palette(), automatic);
        assert!(extract_cover_palette(b"not an image").is_err());
    }

    #[test]
    fn rhythm_color_sampling_returns_exact_pixels_and_rejects_invalid_coordinates_or_data() {
        use std::io::Cursor;
        let image = RgbaImage::from_fn(96, 16, |x, _| {
            image::Rgba(if x < 32 { [170, 34, 34, 255] } else if x < 64 { [34, 153, 68, 255] } else { [34, 51, 187, 255] })
        });
        let mut encoded = Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(image).write_to(&mut encoded, image::ImageOutputFormat::Png).unwrap();
        assert_eq!(sample_cover_color(encoded.get_ref(), 10, 8).unwrap(), "#AA2222");
        assert_eq!(sample_cover_color(encoded.get_ref(), 40, 8).unwrap(), "#229944");
        assert_eq!(sample_cover_color(encoded.get_ref(), 80, 8).unwrap(), "#2233BB");
        let error = sample_cover_color(encoded.get_ref(), 96, 8).unwrap_err().to_string();
        assert!(error.contains("outside the image"), "unexpected error: {error}");
        let error = sample_cover_color(b"not an image", 0, 0).unwrap_err().to_string();
        assert!(error.contains("decode chart illustration"), "unexpected error: {error}");
    }

    #[test]
    fn rhythm_colors_config_defaults_and_exact_overrides_round_trip_with_safe_reset() {
        let legacy: Config = serde_json::from_str("{}").unwrap();
        assert_eq!(legacy.rhythm_colors, None);
        let colors = ["#010203".to_string(), "#aAbBcC".to_string(), "#000000".to_string()];
        let mut config = legacy;
        config.rhythm_colors = Some(colors.clone());
        let saved = serde_json::to_string(&config).unwrap();
        let restored: Config = serde_json::from_str(&saved).unwrap();
        assert_eq!(restored.rhythm_colors, Some(colors.clone()));
        let initial = || RhythmStrip::from_events(BpmList::default(), Vec::new(), cover_palette(None));
        let automatic = initial().palette();
        let overridden = initial().with_colors(restored.rhythm_colors.as_ref());
        assert_eq!(overridden.palette(), [
            Color::new(1. / 255., 2. / 255., 3. / 255., 1.),
            Color::new(170. / 255., 187. / 255., 204. / 255., 1.),
            Color::new(0., 0., 0., 1.),
        ]);
        assert_eq!(overridden.with_colors(None).palette(), automatic);
        for invalid in ["#123", "112233", "#12345678", "#GGHHII", "#é0000"] {
            let bad = [colors[0].clone(), invalid.to_string(), colors[2].clone()];
            assert_eq!(initial().with_colors(Some(&bad)).palette(), automatic);
        }
    }

}
