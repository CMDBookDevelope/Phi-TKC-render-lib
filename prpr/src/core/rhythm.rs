//! Musical analysis independent of chart geometry and GPU readback. The caller
//! gives this strip the exact `Scene::chart_time()` used for note judgement.
use super::{BpmList, Chart, NoteKind};
use crate::{config::Config, ui::Ui};
use anyhow::{ensure, Context, Result};
use image::RgbaImage;
use lyon::{math::point, path::Path};
use macroquad::prelude::{Color, Rect, WHITE};
use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, TAU};

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

/// Durations are inferred from successive distinct selected onsets; holds use
/// their authored duration. Simultaneous notes never create a zero duration.
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
    Other,
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
        Self::Other,
    ];

    pub fn classify(beats: f32) -> Self {
        if !beats.is_finite() || beats <= 0. {
            return Self::Other;
        }
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
        candidates
            .into_iter()
            .find(|(value, _)| (beats - value).abs() <= (value * 0.012).max(0.003))
            .map_or(Self::Other, |(_, value)| value)
    }

    /// Every value has a unique (shape, cover-color index), even for a
    /// monochrome cover where the three extracted accents may be similar.
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
            Self::Other => (Star, 2),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Whole => "1",
            Self::DottedHalf => "2.",
            Self::Half => "2",
            Self::DottedQuarter => "4.",
            Self::Quarter => "4",
            Self::DottedEighth => "8.",
            Self::Eighth => "8",
            Self::DottedSixteenth => "16.",
            Self::Sixteenth => "16",
            Self::ThirtySecond => "32",
            Self::SixtyFourth => "64",
            Self::HalfTriplet => "2T",
            Self::QuarterTriplet => "4T",
            Self::EighthTriplet => "8T",
            Self::SixteenthTriplet => "16T",
            Self::ThirtySecondTriplet => "32T",
            Self::Other => "~",
        }
    }
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
    pub beat: f32,
    pub end_beat: f32,
    pub kind: RhythmKind,
    pub value: RhythmValue,
}

pub struct RhythmStrip {
    tempo: BpmList,
    notes: Vec<RhythmNote>,
    holds: Vec<usize>,
    /// Max endpoint of preceding holds, for bounded sustained-note searches.
    hold_max_end: Vec<f32>,
    palette: [Color; 3],
    speed: f32,
}

impl RhythmStrip {
    pub fn from_chart_bytes(chart: &Chart, selection: RhythmNoteSelection, cover: &[u8]) -> Result<Self> {
        let cover = image::load_from_memory(cover)
            .context("failed to decode rhythm analysis cover")?
            .into_rgba8();
        Ok(Self::from_chart(chart, selection, Some(&cover)))
    }

    pub fn from_chart(chart: &Chart, selection: RhythmNoteSelection, cover: Option<&RgbaImage>) -> Self {
        let tempo = chart.rhythm_bpm_list.clone().unwrap_or_else(|| chart.bpm_list.borrow().clone());
        let events = chart
            .lines
            .iter()
            .flat_map(|line| &line.notes)
            .filter(|note| !note.fake && selection.includes(&note.kind) && note.time.is_finite())
            .map(|note| match note.kind {
                NoteKind::Click => (note.time, note.time, RhythmKind::Tap),
                NoteKind::Drag => (note.time, note.time, RhythmKind::Drag),
                NoteKind::Flick => (note.time, note.time, RhythmKind::Flick),
                NoteKind::Hold { end_time, .. } => (note.time, end_time, RhythmKind::Hold),
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
            // RPE / PGR f32 conversion can place authored chords a few ulps
            // apart. Group only within 0.1ms, retaining each exact hit time.
            let mut end = begin + 1;
            while end < events.len() && (events[end].0 - time).abs() < 0.0001 {
                end += 1;
            }
            let beat = tempo.beat(time);
            let gap = events.get(end).map(|event| tempo.beat(event.0) - beat);
            for &(time, end_time, kind) in &events[begin..end] {
                let beat = tempo.beat(time);
                let end_beat = if kind == RhythmKind::Hold && end_time.is_finite() {
                    tempo.beat(end_time).max(beat)
                } else {
                    beat
                };
                let value = if end_beat > beat {
                    RhythmValue::classify(end_beat - beat)
                } else {
                    // No following onset can establish the last tap's value.
                    // A quarter-note marker is the explicit terminal fallback.
                    gap.map_or(RhythmValue::Quarter, RhythmValue::classify)
                };
                notes.push(RhythmNote {
                    time,
                    beat,
                    end_beat,
                    kind,
                    value,
                });
            }
            begin = end;
        }
        let holds: Vec<_> = notes
            .iter()
            .enumerate()
            .filter(|(_, note)| note.end_beat > note.beat)
            .map(|(index, _)| index)
            .collect();
        let mut max = f32::NEG_INFINITY;
        let hold_max_end = holds
            .iter()
            .map(|&index| {
                let note = &notes[index];
                max = max.max(note.end_beat);
                max
            })
            .collect();
        Self {
            tempo,
            notes,
            holds,
            hold_max_end,
            palette,
            speed: 1.,
        }
    }

    pub fn notes(&self) -> &[RhythmNote] {
        &self.notes
    }
    pub fn palette(&self) -> [Color; 3] {
        self.palette
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

    fn pitch(&self, top: f32) -> f32 {
        spacing(top) * self.speed
    }

    // Later judgements paint first, so an earlier note remains above any
    // overlapping future note. Stable source order breaks exact-time ties.
    fn visible_notes(&self, first: f32, last: f32) -> impl Iterator<Item = &RhythmNote> {
        let begin = self.notes.partition_point(|note| note.beat < first);
        let end = self.notes.partition_point(|note| note.beat <= last);
        let mut heads = (begin.min(end)..end).rev().peekable();
        let hold_begin = self.hold_max_end.partition_point(|&end| end < first);
        let hold_end = self.holds.partition_point(|&index| self.notes[index].beat <= last);
        let mut holds = self.holds[hold_begin.min(hold_end)..hold_end].iter().rev().copied().peekable();
        // Merge the two descending index streams without allocating or
        // scanning expired tap notes behind a still-active long hold.
        std::iter::from_fn(move || {
            let index = match (heads.peek().copied(), holds.peek().copied()) {
                (Some(head), Some(hold)) if head >= hold => {
                    heads.next();
                    if head == hold {
                        holds.next();
                    }
                    head
                }
                (Some(_), Some(hold)) | (None, Some(hold)) => {
                    holds.next();
                    hold
                }
                (Some(head), None) => {
                    heads.next();
                    head
                }
                (None, None) => return None,
            };
            Some(&self.notes[index])
        })
        .filter(move |note| note.beat >= first || note.end_beat >= first)
    }

    /// x is normalized to the strip's -1..1 camera. At `note.time`, x is
    /// precisely the same judgement line used for the beat reference marks.
    pub fn position_x(&self, note: &RhythmNote, time: f32, top: f32) -> f32 {
        let judge = -1. + 2.4 * top;
        judge + (note.beat - self.beat_at(time)) * self.pitch(top)
    }

    /// Draw inside a dedicated Ui/camera viewport whose size is W × W/10.
    /// No music offset, playback speed, or preroll is applied here: the caller
    /// has already resolved those in the scene's judgement clock.
    pub fn draw(&self, chart_time: f32, ui: &mut Ui) {
        let time = if chart_time.is_finite() { chart_time } else { 0. };
        let top = ui.top;
        let h = 2. * top;
        let left = -1. + h;
        let judge = left + 0.2 * h;
        let pitch = self.pitch(top);
        let now = self.beat_at(time);
        let phase = now.rem_euclid(1.);
        let pulse = (-phase * 20.).exp();
        let bg = mix(self.palette[0], Color::new(0.035, 0.043, 0.06, 1.), 0.88);
        ui.fill_rect(ui.screen_rect(), (mix(bg, self.palette[1], 0.10), (-1., -top), bg, (1., top)));
        ui.fill_rect(Rect::new(-1., -top, 2., h * 0.014), alpha(WHITE, 0.19));
        ui.fill_rect(Rect::new(left, -top, 2. - h, h), (alpha(WHITE, 0.065), (left, -top), alpha(WHITE, 0.), (left, top)));

        // The square metronome has a quietly tinted face, a persistent ring,
        // and a quick luminous attack followed by an expanding soft decay.
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
                radius + phase.min(0.25) * h * 0.11 + extra,
                h * 0.01 + extra * 0.45,
                alpha(self.palette[0], pulse * (0.28 - layer as f32 * 0.055)),
            );
        }
        ui.stroke_circle(cx, 0., radius, h * (0.008 + pulse * 0.012), alpha(mix(self.palette[0], WHITE, 0.65), 0.2 + pulse * 0.75));
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

        let bands = RhythmBands::new(left, h);
        ui.fill_rect(bands.labels, alpha(Color::new(0.02, 0.025, 0.035, 1.), 0.58));
        ui.fill_rect(Rect::new(left, bands.labels.y, 1. - left, h * 0.003), alpha(WHITE, 0.08));
        // Note heads, their glows, and hold tails are physically clipped above
        // the label band, including when future notes overlap each other.
        ui.scissor(Some(bands.notes));
        // Guide beats sit behind both sustain bands and note heads.
        let first_beat = (now + (left - judge) / pitch).floor() as i64;
        let last_beat = (now + (1. - judge) / pitch).ceil() as i64;
        for beat in first_beat..=last_beat {
            let x = judge + (beat as f32 - now) * pitch;
            let strong = beat.rem_euclid(4) == 0;
            ui.fill_rect(Rect::new(x - h * 0.0015, -h * 0.27, h * 0.003, h * 0.42), alpha(WHITE, if strong { 0.13 } else { 0.075 }));
            ui.fill_circle(x, -h * 0.30, h * 0.006, alpha(WHITE, if strong { 0.25 } else { 0.10 }));
        }
        ui.fill_rect(Rect::new(left, -h * 0.003, 1. - left, h * 0.006), alpha(WHITE, 0.05));

        let first = now + (left - judge - h * 0.15) / pitch;
        let last = now + (1. - judge + h * 0.15) / pitch;
        // Soft line glow never obscures the note glyph, which is drawn last.
        for layer in 0..3 {
            let w = h * (0.023 - layer as f32 * 0.007);
            ui.fill_rect(Rect::new(judge - w * 0.5, -h * 0.31, w, h * 0.46), alpha(self.palette[0], 0.07 + pulse * 0.055));
        }
        ui.fill_rect(Rect::new(judge - h * 0.0025, -h * 0.33, h * 0.005, h * 0.49), alpha(WHITE, 0.72));
        ui.fill_circle(judge, -h * 0.36, h * 0.010, alpha(WHITE, 0.68));

        for note in self.visible_notes(first, last) {
            let x = judge + (note.beat - now) * pitch;
            let (y, radius) = note_geometry(h);
            let (shape, color) = note.value.style();
            let c = self.palette[color];
            // Paint each whole note together (tail, then head), so an earlier
            // hold also stays above every later note it overlaps.
            if note.end_beat > note.beat {
                let start = x.max(left - radius);
                let tail = (judge + (note.end_beat - now) * pitch).min(1. + radius);
                if tail > start {
                    ui.fill_rect(
                        Rect::new(start, y - radius * 0.32, tail - start, radius * 0.64),
                        (alpha(c, 0.32), (start, y - radius * 0.32), alpha(c, 0.12), (start, y + radius * 0.32)),
                    );
                    ui.fill_rect(Rect::new(start, y - radius * 0.05, tail - start, radius * 0.10), alpha(c, 0.62));
                    ui.fill_circle(tail, y, radius * 0.27, alpha(c, 0.7));
                }
            }
            if x + radius < left || x - radius > 1. {
                continue;
            }
            let hit = (1. - (time - note.time).abs() / 0.07).max(0.);
            glyph(ui, shape, x, y + radius * 0.07, radius * 1.07, alpha(Color::new(0., 0., 0., 1.), 0.35));
            glyph(ui, shape, x, y, radius * (1.14 + hit * 0.13), alpha(c, 0.12 + hit * 0.18));
            glyph(ui, shape, x, y, radius, mix(c, WHITE, 0.12));
            glyph(ui, shape, x, y, radius * 0.83, c);
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
                RhythmKind::Hold => ui.fill_rect(Rect::new(x - radius * 0.24, y - radius * 0.06, radius * 0.48, radius * 0.12), alpha(WHITE, 0.72)),
                RhythmKind::Tap => {}
            }
        }
        text(ui, "RHYTHM", left + h * 0.12, -h * 0.40, h * 0.84, alpha(WHITE, 0.36));
        ui.scissor(Some(bands.labels));
        // Labels are painted after every note, in a dedicated larger band.
        // Identical onsets need one readable label, not repeated overprinting.
        let begin = self.notes.partition_point(|note| note.beat < first);
        let end = self.notes.partition_point(|note| note.beat <= last);
        for index in (begin.min(end)..end).rev() {
            let note = &self.notes[index];
            if index > 0 && (note.time - self.notes[index - 1].time).abs() < 0.0001 {
                continue;
            }
            let x = judge + (note.beat - now) * pitch;
            text(ui, note.value.label(), x, bands.label_y, bands.label_size, alpha(WHITE, 0.86));
        }
        ui.scissor(None);
    }
}

fn spacing(top: f32) -> f32 {
    ((2. - 2.4 * top - 0.08) / 6.).max(0.01)
}

fn note_geometry(h: f32) -> (f32, f32) {
    (-h * 0.02, h * 0.112)
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
            label_size: h * 1.8,
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
    let mut path = Path::builder();
    for i in 0..vertices {
        let a = angle + TAU * i as f32 / vertices as f32;
        let scale = if shape == RhythmShape::Star && i % 2 == 1 {
            0.48
        } else if shape == RhythmShape::Cross && i % 3 == 1 {
            0.45
        } else {
            1.
        };
        let pt = point(x + a.cos() * radius * scale, y + a.sin() * radius * scale);
        if i == 0 {
            path.begin(pt);
        } else {
            path.line_to(pt);
        }
    }
    path.end(true);
    path.build()
}

/// Quantize a bounded cover thumbnail once, then choose three separated
/// accents. Tints remain tied to the cover even for grayscale images.
pub fn cover_palette(cover: Option<&RgbaImage>) -> [Color; 3] {
    let fallback = [
        Color::new(0.56, 0.75, 0.96, 1.),
        Color::new(0.75, 0.66, 0.93, 1.),
        Color::new(0.90, 0.72, 0.63, 1.),
    ];
    let Some(cover) = cover.filter(|image| image.width() > 0 && image.height() > 0) else {
        return fallback;
    };
    let small = image::imageops::thumbnail(cover, 128, 128);
    let colors = color_thief::get_palette(small.as_raw(), color_thief::ColorFormat::Rgba, 1, 10).unwrap_or_default();
    let candidates: Vec<Color> = colors
        .into_iter()
        .map(|c| Color::new(c.r as f32 / 255., c.g as f32 / 255., c.b as f32 / 255., 1.))
        .collect();
    let mut average = [0.; 3];
    let mut weight = 0.;
    for pixel in small.pixels() {
        let a = pixel[3] as f32 / 255.;
        weight += a;
        for i in 0..3 {
            average[i] += pixel[i] as f32 / 255. * a;
        }
    }
    if weight <= 0. {
        return fallback;
    }
    let average = Color::new(average[0] / weight, average[1] / weight, average[2] / weight, 1.);
    let first = candidates.first().copied().unwrap_or(average);
    let second = candidates
        .iter()
        .copied()
        .max_by(|a, b| distance(*a, first).total_cmp(&distance(*b, first)))
        .unwrap_or(first);
    let third = candidates
        .iter()
        .copied()
        .max_by(|a, b| {
            distance(*a, first)
                .min(distance(*a, second))
                .total_cmp(&distance(*b, first).min(distance(*b, second)))
        })
        .unwrap_or(first);
    let accents = [first, second, third];
    let mut result = std::array::from_fn(|i| {
        let c = accents[i];
        let light = 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
        let tint = ((0.57 - light) / (1. - light).max(0.01)).clamp(0., 0.7);
        let c = mix(c, WHITE, tint);
        // A flat monochrome source still produces distinguishable luminance
        // accents; shapes retain independent identification in every case.
        if distance(first, second) < 0.025 && distance(first, third) < 0.025 {
            with_luminance(c, [0.57, 0.72, 0.87][i])
        } else {
            c
        }
    });
    for i in 1..3 {
        if result[..i].iter().any(|&other| distance(other, result[i]) < 0.01) {
            // Color quantization can yield duplicates even on a multicolor
            // cover. Try spaced shades of that same hue until every earlier
            // accent differs. Three colors require at most three such shades.
            for target in [0.91, 0.74, 0.57, 0.40, 0.98] {
                let candidate = with_luminance(result[i], target);
                if result[..i].iter().all(|&other| distance(other, candidate) >= 0.01) {
                    result[i] = candidate;
                    break;
                }
            }
        }
    }
    result
}

fn with_luminance(color: Color, target: f32) -> Color {
    let light = 0.2126 * color.r + 0.7152 * color.g + 0.0722 * color.b;
    if light > target {
        mix(color, Color::new(0., 0., 0., 1.), 1. - target / light.max(0.001))
    } else {
        mix(color, WHITE, ((target - light) / (1. - light).max(0.001)).clamp(0., 1.))
    }
}

fn distance(a: Color, b: Color) -> f32 {
    (a.r - b.r).powi(2) + (a.g - b.g).powi(2) + (a.b - b.b).powi(2)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strip(events: Vec<(f32, f32, RhythmKind)>, tempo: BpmList) -> RhythmStrip {
        RhythmStrip::from_events(tempo, events, cover_palette(None))
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
    fn rhythm_adjust_simultaneous_notes_keep_single_onset_values_and_hold_span() {
        let tempo = BpmList::new(vec![(0., 120.), (4., 240.)]);
        let strip = strip(
            vec![
                (1.75, 2.25, RhythmKind::Hold),
                (1.75, 1.75, RhythmKind::Tap),
                (1.75, 1.75, RhythmKind::Flick),
                (2., 2., RhythmKind::Tap),
            ],
            tempo,
        );
        assert_eq!(strip.notes().len(), 4);
        let hold = &strip.notes()[0];
        assert_eq!(hold.value, RhythmValue::DottedQuarter);
        assert!((hold.end_beat - hold.beat - 1.5).abs() < 0.00001);
        assert_eq!(strip.notes()[1].value, RhythmValue::Eighth);
        assert_eq!(strip.notes()[2].value, RhythmValue::Eighth);
        // Duplicate starts are ordinary overlapping single notes: their tap
        // values still use the next distinct onset, never a zero-length gap.
        assert_eq!(strip.notes()[1].value, strip.notes()[2].value);
        assert_eq!(strip.notes()[0].beat, strip.notes()[1].beat);
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
            assert!(strip.pitch(0.1).is_finite() && strip.pitch(0.1) > 0.);
        }
    }

    #[test]
    fn rhythm_adjust_earlier_judgements_paint_last_on_the_shared_track() {
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
                (2., RhythmKind::Hold),
                (1., RhythmKind::Flick),
                (0.5, RhythmKind::Drag),
                (0.5, RhythmKind::Tap)
            ]
        );
        let clipped: Vec<_> = strip.visible_notes(2., 4.).map(|note| note.time).collect();
        assert_eq!(clipped, vec![2., 1.]);
        assert!(strip.visible_notes(8., 9.).next().is_none());
    }

    #[test]
    fn rhythm_adjust_active_hold_and_future_heads_share_earliest_on_top_order() {
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
        // The first hold's head has left the visible region, but its tail must
        // paint last above later heads; expired short holds stay absent.
        let order: Vec<_> = strip.visible_notes(3., 8.).map(|note| note.time).collect();
        assert_eq!(order, vec![3., 2., 0.]);
        // A visible hold is merged once even though its head and tail both
        // qualify for the visible region.
        let order: Vec<_> = strip.visible_notes(0., 8.).map(|note| note.time).collect();
        assert_eq!(order, vec![3., 2., 1., 0.5, 0.]);
    }

    #[test]
    fn rhythm_adjust_all_shapes_glows_and_sustains_leave_label_band_clear() {
        for h in [0.125, 0.2, 0.4] {
            let bands = RhythmBands::new(-1. + h, h);
            let (y, radius) = note_geometry(h);
            assert!(bands.notes.y + bands.notes.h < bands.labels.y);
            // All vector glyphs stay inside their radius. The active hit glow
            // (1.27r) exceeds both the shadow and each type's inner decoration.
            for value in RhythmValue::ALL {
                let shape = value.style().0;
                if shape == RhythmShape::Circle {
                    assert!(y + radius * 1.27 < bands.notes.y + bands.notes.h);
                    assert!(y - radius * 1.27 > bands.notes.y);
                } else {
                    // Inspect vertices of the actual paths submitted to Ui,
                    // including stars/crosses and the largest active glow.
                    for event in glyph_path(shape, 0., y, radius * 1.27).iter() {
                        let at = match event {
                            lyon::path::PathEvent::Begin { at } => at,
                            lyon::path::PathEvent::Line { to, .. } => to,
                            _ => continue,
                        };
                        assert!(at.y >= bands.notes.y && at.y < bands.notes.y + bands.notes.h);
                        assert!(at.y < bands.labels.y);
                    }
                }
            }
            assert!(y + radius * 0.32 < bands.labels.y); // sustain width
            assert!(bands.label_y > bands.labels.y);
            assert!(bands.label_y < bands.labels.y + bands.labels.h);
            assert!(bands.label_size > h * 0.9); // larger than the previous text
        }
    }

    #[test]
    fn conventional_subdivisions_and_triplets_keep_distinct_visual_codes() {
        assert_eq!(RhythmValue::classify(1.), RhythmValue::Quarter);
        assert_eq!(RhythmValue::classify(0.5), RhythmValue::Eighth);
        assert_eq!(RhythmValue::classify(0.25), RhythmValue::Sixteenth);
        assert_eq!(RhythmValue::classify(1. / 3.), RhythmValue::EighthTriplet);
        assert_eq!(RhythmValue::classify(0.75), RhythmValue::DottedEighth);
        assert_eq!(RhythmValue::classify(0.42), RhythmValue::Other);
        let styles: std::collections::HashSet<_> = RhythmValue::ALL.into_iter().map(RhythmValue::style).collect();
        assert_eq!(styles.len(), RhythmValue::ALL.len());
        assert_eq!(RhythmValue::Quarter.style().0, RhythmShape::Circle);
        assert_eq!(RhythmValue::EighthTriplet.style().0, RhythmShape::Triangle);
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
        assert_eq!(selected.notes()[1].kind, RhythmKind::Hold);
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
}
