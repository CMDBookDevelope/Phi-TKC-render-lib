//! A small, immutable layout/time model. This is a deterministic ergonomic
//! planner, not trained weights for the legacy experimental ViT/PPO networks.

use crate::config::Config;
use crate::core::{note::Hand, AnimFloat, JudgeLine, Note, NoteKind, Vector};
use once_cell::sync::Lazy;
use std::path::Path;

const MODEL_BYTES: &[u8] = include_bytes!("../../assets/phitk_ai_model.bin");
const MAGIC: &[u8; 8] = b"PHITKHP\0";
const PARAMS: usize = 10;
const MODEL_LEN: usize = 16 + PARAMS * 4 + 4;

pub(super) fn bundled_model_bytes() -> &'static [u8] {
    MODEL_BYTES
}

#[derive(Clone, Debug, PartialEq)]
struct Model {
    home: f32,
    spatial: f32,
    travel: f32,
    reuse: f32,
    reuse_window: f32,
    relax_time: f32,
    chord_epsilon: f32,
    hold_reach: f32,
    hold_penalty: f32,
    chord_extra: f32,
}

impl Default for Model {
    fn default() -> Self {
        Self {
            home: 0.45,
            spatial: 2.0,
            travel: 0.35,
            reuse: 0.9,
            reuse_window: 0.18,
            relax_time: 1.0,
            chord_epsilon: 0.003,
            hold_reach: 0.30,
            hold_penalty: 8.0,
            chord_extra: 0.6,
        }
    }
}

fn checksum(data: &[u8]) -> u32 {
    data.iter()
        .fold(0x811c9dc5u32, |hash, byte| (hash ^ *byte as u32).wrapping_mul(0x01000193))
}

impl Model {
    fn decode(data: &[u8]) -> Option<Self> {
        if data.len() != MODEL_LEN || data[..8] != MAGIC[..] {
            return None;
        }
        let word = |at| u32::from_le_bytes(data[at..at + 4].try_into().unwrap());
        if word(8) != 1 || word(12) != PARAMS as u32 || word(MODEL_LEN - 4) != checksum(&data[..MODEL_LEN - 4]) {
            return None;
        }
        let values: [f32; PARAMS] = std::array::from_fn(|id| f32::from_bits(word(16 + id * 4)));
        if values.iter().any(|value| !value.is_finite() || *value <= 0.0 || *value > 100.0)
            || !(0.1..=1.0).contains(&values[0])
            || !(0.01..=1.0).contains(&values[4])
            || !(0.1..=10.0).contains(&values[5])
            || !(0.0001..=0.02).contains(&values[6])
            || !(0.05..=1.0).contains(&values[7])
        {
            return None;
        }
        let [home, spatial, travel, reuse, reuse_window, relax_time, chord_epsilon, hold_reach, hold_penalty, chord_extra] = values;
        Some(Self {
            home,
            spatial,
            travel,
            reuse,
            reuse_window,
            relax_time,
            chord_epsilon,
            hold_reach,
            hold_penalty,
            chord_extra,
        })
    }

    fn bundled() -> Self {
        Self::decode(MODEL_BYTES).unwrap_or_default()
    }

    fn load(path: &Path) -> Self {
        // An old, very large PPO file is not this model. Reject it without
        // loading megabytes or creating a GPU, and never overwrite user data.
        if std::fs::metadata(path).map_or(false, |metadata| metadata.len() == MODEL_LEN as u64) {
            if let Some(model) = std::fs::read(path).ok().and_then(|bytes| Self::decode(&bytes)) {
                return model;
            }
        }
        tracing::warn!("Hand model unavailable or incompatible at {}; using bundled layout/time model", path.display());
        Self::bundled()
    }
}

static MODEL: Lazy<Model> = Lazy::new(|| std::env::var_os("PHITK_HAND_MODEL").map_or_else(Model::bundled, |path| Model::load(Path::new(&path))));

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PlanKey {
    aspect: u32,
    ratio: u32,
    flip_x: bool,
    lines: usize,
    notes: usize,
}

impl PlanKey {
    pub(crate) fn new(lines: &[JudgeLine], config: &Config, aspect: f32) -> Self {
        Self {
            aspect: aspect.to_bits(),
            ratio: config.chart_ratio.to_bits(),
            flip_x: config.flip_x(),
            lines: lines.len(),
            notes: lines.iter().map(|line| line.notes.len()).sum(),
        }
    }
}

#[derive(Clone, Copy)]
struct Target {
    time: f32,
    end_time: f32,
    pos: Vector,
    fake: bool,
}

fn finite(value: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

fn target(note: &Note, pos: Vector) -> Target {
    let time = finite(note.time);
    Target {
        time,
        end_time: match note.kind {
            NoteKind::Hold { end_time, .. } => finite(end_time).max(time),
            _ => time,
        },
        pos: Vector::new(finite(pos.x).clamp(-4.0, 4.0), finite(pos.y).clamp(-4.0, 4.0)),
        fake: note.fake,
    }
}

#[derive(Clone, Copy)]
struct Hold {
    end: f32,
    pos: Vector,
}

struct HandState {
    sign: f32,
    pos: Vector,
    time: f32,
    holds: Vec<Hold>,
}

impl HandState {
    fn new(sign: f32, model: &Model) -> Self {
        Self {
            sign,
            pos: Vector::new(sign * model.home, 0.0),
            time: f32::NEG_INFINITY,
            holds: Vec::new(),
        }
    }

    fn cost(&self, note: &Target, model: &Model) -> f32 {
        let dt = (note.time - self.time).max(0.0);
        let home = Vector::new(self.sign * model.home, 0.0);
        let recent = (1.0 - dt / model.relax_time).clamp(0.0, 1.0);
        let from = home + (self.pos - home) * recent;
        let mut cost = model.spatial * (note.pos.x - home.x).powi(2)
            + model.travel * distance_sq(note.pos, from) * recent
            + model.reuse * (1.0 - dt / model.reuse_window).clamp(0.0, 1.0);
        for hold in &self.holds {
            // Nearby extra fingers remain possible; reaching away from a held
            // note is expensive. Holds reserve their hand until their end.
            let reach = distance_sq(note.pos, hold.pos).sqrt();
            cost += 0.2 + model.hold_penalty * (reach - model.hold_reach).max(0.0).powi(2);
        }
        cost
    }

    fn accept(&mut self, notes: impl Iterator<Item = Target>) {
        let mut sum = Vector::zeros();
        let mut count = 0;
        for note in notes {
            sum += note.pos;
            count += 1;
            self.time = note.time;
            if note.end_time > note.time {
                self.holds.push(Hold {
                    end: note.end_time,
                    pos: note.pos,
                });
            }
        }
        if count > 0 {
            self.pos = sum / count as f32;
        }
    }
}

fn distance_sq(a: Vector, b: Vector) -> f32 {
    let d = a - b;
    d.x * d.x + 0.25 * d.y * d.y
}

fn plan(notes: &[Target], model: &Model) -> Vec<Hand> {
    // Fake decoration notes have a stable spatial colour, but cannot occupy a
    // hand or change a real note's timing/alternation.
    let mut hands: Vec<_> = notes.iter().map(|note| if note.pos.x > 0.0 { Hand::Right } else { Hand::Left }).collect();
    let mut order: Vec<_> = notes.iter().enumerate().filter_map(|(id, note)| (!note.fake).then_some(id)).collect();
    order.sort_by(|&a, &b| notes[a].time.total_cmp(&notes[b].time).then(a.cmp(&b)));
    let mut left = HandState::new(-1.0, model);
    let mut right = HandState::new(1.0, model);
    let mut prefix_left = Vec::new();
    let mut prefix_right = Vec::new();
    let mut at = 0;
    while at < order.len() {
        let time = notes[order[at]].time;
        let mut end = at + 1;
        while end < order.len() && notes[order[end]].time - time <= model.chord_epsilon {
            end += 1;
        }
        let group = &mut order[at..end];
        group.sort_by(|&a, &b| {
            notes[a]
                .pos
                .x
                .total_cmp(&notes[b].pos.x)
                .then(notes[a].pos.y.total_cmp(&notes[b].pos.y))
                .then(a.cmp(&b))
        });
        left.holds.retain(|hold| hold.end > time);
        right.holds.retain(|hold| hold.end > time);
        prefix_left.clear();
        prefix_right.clear();
        prefix_left.push(0.0);
        prefix_right.push(0.0);
        for &id in group.iter() {
            prefix_left.push(prefix_left.last().unwrap() + left.cost(&notes[id], model));
            prefix_right.push(prefix_right.last().unwrap() + right.cost(&notes[id], model));
        }
        let n = group.len();
        // With both hands free, a chord uses both hands. With active holds a
        // free hand may use multiple fingers for the remaining notes.
        let bilateral = n >= 2 && left.holds.is_empty() && right.holds.is_empty();
        let candidates = if bilateral { 1..=n - 1 } else { 0..=n };
        let mut best = (f32::INFINITY, 0usize);
        for split in candidates {
            let extra = split.saturating_sub(1).pow(2) + (n - split).saturating_sub(1).pow(2);
            let cost = prefix_left[split] + prefix_right[n] - prefix_right[split] + extra as f32 * model.chord_extra;
            let middle = n.div_ceil(2);
            if cost < best.0 - 0.00001 || ((cost - best.0).abs() <= 0.00001 && split.abs_diff(middle) <= best.1.abs_diff(middle)) {
                best = (cost, split);
            }
        }
        let split = best.1;
        for (index, &id) in group.iter().enumerate() {
            hands[id] = if index < split { Hand::Left } else { Hand::Right };
        }
        left.accept(group[..split].iter().map(|&id| notes[id]));
        right.accept(group[split..].iter().map(|&id| notes[id]));
        at = end;
    }
    hands
}

/// Sample without changing the renderer's animation time/cursor or cloning
/// keyframes for every note. Chained RPE animation layers are additive.
fn sample(anim: &AnimFloat, time: f32, default: f32) -> f32 {
    if anim.keyframes.is_empty() {
        return default;
    }
    let at = anim.keyframes.partition_point(|keyframe| keyframe.time <= time).saturating_sub(1);
    let first = &anim.keyframes[at];
    let value = if let Some(second) = anim.keyframes.get(at + 1) {
        let t = (time - first.time) / (second.time - first.time);
        first.value + (second.value - first.value) * first.tween.y(t)
    } else {
        first.value
    };
    value + anim.next.as_ref().map_or(0.0, |next| sample(next, time, 0.0))
}

#[derive(Clone, Copy)]
struct LinePosition {
    time: f32,
    pos: Vector,
    rotation: f32,
}

struct Projection {
    cache: Vec<Option<LinePosition>>,
    ancestry: Vec<usize>,
    visited: Vec<u64>,
    stamp: u64,
    aspect: f32,
}

impl Projection {
    fn new(lines: usize, aspect: f32) -> Self {
        Self {
            cache: vec![None; lines],
            ancestry: Vec::new(),
            visited: vec![0; lines],
            stamp: 0,
            aspect,
        }
    }

    fn line(&mut self, lines: &[JudgeLine], id: usize, time: f32) -> LinePosition {
        self.stamp += 1;
        self.ancestry.clear();
        let mut current = id;
        let mut parent = None;
        loop {
            if let Some(cached) = self.cache[current].filter(|cached| cached.time == time) {
                parent = Some(cached);
                break;
            }
            if self.visited[current] == self.stamp {
                // Malformed parent cycles cannot recurse or block rendering.
                break;
            }
            self.visited[current] = self.stamp;
            self.ancestry.push(current);
            match lines[current].parent.filter(|&next| next < lines.len() && next != current) {
                Some(next) => current = next,
                None => break,
            }
        }
        for &current in self.ancestry.iter().rev() {
            let object = &lines[current].object;
            let local = Vector::new(sample(&object.translation.0, time, 0.0), sample(&object.translation.1, time, 0.0) / self.aspect);
            let pos = parent.map_or(local, |parent: LinePosition| parent.pos + rotate(local, parent.rotation));
            let result = LinePosition {
                time,
                pos,
                rotation: sample(&object.rotation, time, 0.0),
            };
            self.cache[current] = Some(result);
            parent = Some(result);
        }
        self.cache[id].unwrap()
    }

    fn note(&mut self, lines: &[JudgeLine], line_id: usize, note: &Note, config: &Config) -> Vector {
        let time = finite(note.time);
        let line_pos = self.line(lines, line_id, time);
        let line = &lines[line_id];
        let y = sample(&note.object.translation.1, time, 0.0);
        let height = note.height - sample(&line.height, time, 0.0) + if note.speed == 0.0 { 0.0 } else { y / note.speed };
        let ctrl_pos = sample(&line.ctrl_obj.borrow().pos, height * crate::parse::RPE_HEIGHT / 2.0, 1.0);
        let incline = sample(&line.incline, time, 0.0).to_radians().sin();
        let x = sample(&note.object.translation.0, time, 0.0) * ctrl_pos * (1.0 - incline * y * crate::parse::RPE_HEIGHT / 720.0);
        // Rotate the local note first, then add the already resolved world
        // translation. Rotating the translation again moved notes across hands.
        let world = line_pos.pos + rotate(Vector::new(x, y / self.aspect), line_pos.rotation);
        let ratio = config.chart_ratio.max(0.001);
        Vector::new(world.x * ratio * if config.flip_x() { -1.0 } else { 1.0 }, -world.y * ratio)
    }
}

fn rotate(pos: Vector, rotation: f32) -> Vector {
    let (sin, cos) = rotation.to_radians().sin_cos();
    Vector::new(pos.x * cos - pos.y * sin, pos.x * sin + pos.y * cos)
}

pub(crate) fn assign_chart(lines: &mut [JudgeLine], config: &Config, aspect: f32) {
    let started = std::time::Instant::now();
    let mut projection = Projection::new(lines.len(), aspect.max(0.001));
    let mut targets = Vec::with_capacity(lines.iter().map(|line| line.notes.len()).sum());
    for (line_id, line) in lines.iter().enumerate() {
        for note in &line.notes {
            targets.push(target(note, projection.note(lines, line_id, note, config)));
        }
    }
    let hands = plan(&targets, &MODEL);
    let left = hands.iter().filter(|&&hand| hand == Hand::Left).count();
    let right = hands.len() - left;
    for (note, hand) in lines.iter_mut().flat_map(|line| &mut line.notes).zip(hands) {
        note.hand = hand;
    }
    tracing::info!(
        "Hand layout/time plan: {left} left, {right} right; {:.2} ms; CPU model, no framebuffer readback",
        started.elapsed().as_secs_f64() * 1000.0
    );
}

pub(super) fn assign_notes(notes: &mut [Note], positions: &[Vector], flip_x: bool) {
    let targets: Vec<_> = notes
        .iter()
        .enumerate()
        .map(|(id, note)| {
            let mut pos = positions.get(id).copied().unwrap_or_default();
            if flip_x {
                pos.x = -pos.x;
            }
            target(note, pos)
        })
        .collect();
    for (note, hand) in notes.iter_mut().zip(plan(&targets, &MODEL)) {
        note.hand = hand;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Anim, AnimVector, JudgeLineCache, JudgeLineKind, Keyframe, Object};
    use crate::judge::JudgeStatus;
    use std::{cell::RefCell, rc::Rc};

    fn click(time: f32, x: f32) -> Note {
        Note {
            time,
            kind: NoteKind::Click,
            height: time,
            object: Object {
                translation: AnimVector::fixed(Vector::new(x, 0.0)),
                ..Default::default()
            },
            speed: 1.0,
            end_speed: 1.0,
            start_height: 0.0,
            hand: Hand::Left,
            above: true,
            multiple_hint: false,
            fake: false,
            judge: JudgeStatus::NotJudged,
            format: false,
        }
    }

    fn line(mut notes: Vec<Note>) -> JudgeLine {
        let cache = JudgeLineCache::new(&mut notes);
        JudgeLine {
            object: Object::default(),
            ctrl_obj: Rc::new(RefCell::default()),
            kind: JudgeLineKind::Normal,
            height: Anim::default(),
            incline: Anim::default(),
            notes,
            color: Anim::default(),
            parent: None,
            z_index: 0,
            show_below: true,
            attach_ui: None,
            cache,
            cached_world_pos: None,
        }
    }

    fn hands(notes: &[Note]) -> Vec<Hand> {
        let targets: Vec<_> = notes.iter().map(|note| target(note, note.object.translation.now())).collect();
        plan(&targets, &Model::bundled())
    }

    #[test]
    fn packaged_model_has_valid_parameters_and_rejects_corruption() {
        assert_eq!(Model::decode(MODEL_BYTES), Some(Model::default()));
        let mut corrupt = MODEL_BYTES.to_vec();
        corrupt[20] ^= 1;
        assert!(Model::decode(&corrupt).is_none());
        assert!(Model::decode(&MODEL_BYTES[..MODEL_LEN - 1]).is_none());
        assert!(Model::decode(b"PHITKAI\0\x03\0\0\0").is_none());
        // A checksum-correct NaN or out-of-range policy must also be rejected.
        corrupt = MODEL_BYTES.to_vec();
        corrupt[16..20].copy_from_slice(&f32::NAN.to_le_bytes());
        let check = checksum(&corrupt[..MODEL_LEN - 4]);
        corrupt[MODEL_LEN - 4..].copy_from_slice(&check.to_le_bytes());
        assert!(Model::decode(&corrupt).is_none());
    }

    #[test]
    fn missing_invalid_or_legacy_external_model_falls_back_without_writing() {
        let directory = tempfile::tempdir().unwrap();
        let file = directory.path().join("phitk_ai_model.bin");
        assert_eq!(Model::load(&file), Model::bundled());
        assert!(!file.exists());
        for bytes in [vec![0u8; MODEL_LEN], b"PHITKAI\0\x03\0\0\0".to_vec(), MODEL_BYTES.to_vec()] {
            std::fs::write(&file, &bytes).unwrap();
            assert_eq!(Model::load(&file), Model::bundled());
            assert_eq!(std::fs::read(&file).unwrap(), bytes);
        }
    }

    #[test]
    fn separated_notes_use_their_side_without_forced_distant_alternation() {
        let notes: Vec<_> = (0..12).map(|id| click(id as f32 * 0.1, if id < 6 { -0.8 } else { 0.8 })).collect();
        assert_eq!(hands(&notes), [vec![Hand::Left; 6], vec![Hand::Right; 6]].concat());
    }

    #[test]
    fn rapid_center_stream_alternates_and_repeated_plans_are_identical() {
        let notes: Vec<_> = (0..24).map(|id| click(id as f32 * 0.1, 0.0)).collect();
        let expected: Vec<_> = (0..24).map(|id| if id % 2 == 0 { Hand::Left } else { Hand::Right }).collect();
        assert_eq!(hands(&notes), expected);
        for _ in 0..100 {
            assert_eq!(hands(&notes), expected);
        }
    }

    #[test]
    fn simultaneous_notes_share_both_hands_across_lines() {
        let mut lines = vec![line(vec![click(1.0, 0.7)]), line(vec![click(1.0, -0.7)])];
        assign_chart(&mut lines, &Config::default(), 16.0 / 9.0);
        assert_eq!(lines[0].notes[0].hand, Hand::Right);
        assert_eq!(lines[1].notes[0].hand, Hand::Left);
        let notes = vec![click(1.0, 0.0), click(1.0, 0.01), click(1.0, -0.01), click(1.0, 0.02)];
        let assignments = hands(&notes);
        assert_eq!(assignments.iter().filter(|&&hand| hand == Hand::Left).count(), 2);
    }

    #[test]
    fn hold_reserves_its_hand_and_releases_at_its_end() {
        let mut hold = click(0.0, -0.8);
        hold.kind = NoteKind::Hold {
            end_time: 1.0,
            end_height: 1.0,
        };
        let notes = vec![hold, click(0.4, 0.0), click(0.6, 0.8), click(1.0, -0.8)];
        assert_eq!(hands(&notes), vec![Hand::Left, Hand::Right, Hand::Right, Hand::Left]);
    }

    #[test]
    fn decoration_notes_do_not_change_real_hands() {
        let mut notes: Vec<_> = (0..10).map(|id| click(id as f32 * 0.1, 0.0)).collect();
        let expected = hands(&notes);
        let mut fake = click(0.05, -0.8);
        fake.fake = true;
        fake.kind = NoteKind::Hold {
            end_time: 10.0,
            end_height: 10.0,
        };
        notes.push(fake);
        assert_eq!(&hands(&notes)[..10], &expected);
    }

    #[test]
    fn hit_time_positions_rotate_local_notes_before_world_translation() {
        let mut moving = line(vec![click(2.0, 0.2)]);
        moving.object.translation.0 = Anim::new(vec![Keyframe::new(0.0, -0.8, 2), Keyframe::new(2.0, 0.8, 2)]);
        moving.object.rotation = Anim::fixed(180.0);
        let original = moving.object.clone();
        let mut lines = vec![moving];
        let mut projection = Projection::new(1, 16.0 / 9.0);
        let pos = projection.note(&lines, 0, &lines[0].notes[0], &Config::default());
        assert!((pos.x - 0.6).abs() < 0.0001, "world x={}", pos.x);
        assign_chart(&mut lines, &Config::default(), 16.0 / 9.0);
        assert_eq!(lines[0].notes[0].hand, Hand::Right);
        assert_eq!(lines[0].object, original, "planning must not change render animation cursors");
    }

    #[test]
    fn animated_parent_coordinates_and_horizontal_flip_match_rendered_sides() {
        let mut parent = line(Vec::new());
        parent.object.translation = AnimVector::fixed(Vector::new(-0.8, 0.0));
        parent.object.rotation = Anim::fixed(90.0);
        let mut child = line(vec![click(1.0, 0.2)]);
        child.parent = Some(0);
        child.object.translation = AnimVector::fixed(Vector::new(0.0, -0.4));
        let mut lines = vec![parent, child];
        let mut config = Config::default();
        assign_chart(&mut lines, &config, 1.0);
        assert_eq!(lines[1].notes[0].hand, Hand::Left);
        config.mods.insert(crate::config::Mods::FLIP_X);
        assign_chart(&mut lines, &config, 1.0);
        assert_eq!(lines[1].notes[0].hand, Hand::Right);
    }

    #[test]
    fn animation_sampling_handles_additive_layers_and_does_not_mutate() {
        let mut anim = Anim::new(vec![Keyframe::new(0.0, -1.0, 2), Keyframe::new(2.0, 1.0, 2)]);
        anim.next = Some(Box::new(Anim::fixed(0.25)));
        let original = anim.clone();
        for time in [2.0, 0.5, 0.0, 4.0, 1.0] {
            let mut reference = anim.clone();
            reference.set_time(time);
            assert!((sample(&anim, time, 0.0) - reference.now()).abs() < 0.00001);
        }
        assert_eq!(anim, original);
    }

    #[test]
    fn separate_charts_do_not_share_state_and_malformed_parent_cycles_terminate() {
        let mut first = vec![line(vec![click(0.0, 0.8)])];
        let mut second = vec![line(vec![click(0.0, -0.8)])];
        let config = Config::default();
        assign_chart(&mut first, &config, 1.0);
        assign_chart(&mut second, &config, 1.0);
        assert_eq!(first[0].notes[0].hand, Hand::Right);
        assert_eq!(second[0].notes[0].hand, Hand::Left);
        first.push(line(vec![click(1.0, 0.2)]));
        first[0].parent = Some(1);
        first[1].parent = Some(0);
        assign_chart(&mut first, &config, 1.0);
    }

    #[test]
    fn compatibility_assignment_is_immediate_and_never_reads_a_framebuffer() {
        let mut notes = vec![click(1.0, -0.8), click(1.0, 0.8)];
        let config = Config {
            hand_split: true,
            ..Config::default()
        };
        crate::hand::assign_hands_unified_perspective(&mut notes, &config, 0, 0.0, 0.0, &[Vector::new(-0.8, 0.0), Vector::new(0.8, 0.0)], || {
            panic!("hand planning attempted GPU readback")
        });
        assert_eq!(notes.iter().map(|note| note.hand).collect::<Vec<_>>(), vec![Hand::Left, Hand::Right]);
    }
}
