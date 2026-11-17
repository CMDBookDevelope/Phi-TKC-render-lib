crate::tl_file!("parser" ptl);

use super::process_lines;

use crate::{
    core::{
        Anim, AnimFloat, AnimVector, BlockArea, BlockMoveEvent, BlockRotateEvent, BlockScaleEvent, BpmList, Chart, ChartExtra, ChartSettings,
        JudgeLine, JudgeLineCache, JudgeLineKind, Keyframe, Note, NoteKind, Object, Triple, Vector, HEIGHT_RATIO,
    },
    ext::NotNanExt,
    judge::JudgeStatus,
};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize};
use tracing::warn;
use anyhow::bail;
use crate::core::note::Hand;
use std::rc::Rc;
use std::cell::RefCell;
use crate::core::CtrlObject;

fn smart_assign_hand(
    position_x: f32,
    time: f32,
    previous_notes: &[(f32, f32, Hand)],
    switch_threshold: f32,
) -> Hand {
    if previous_notes.is_empty() {
        return if position_x < 0.5 { Hand::Left } else { Hand::Right };
    }

    const TEMPORAL_WINDOW: f32 = 2.0;
    const POSITION_THRESHOLD: f32 = 0.2;
    const CLOSE_TIME_THRESHOLD: f32 = TEMPORAL_WINDOW * 0.3;

    let len = previous_notes.len();
    let mut best_idx = 0;
    let mut min_time_diff = f32::INFINITY;
    for i in (0..len).rev() {
        let (note_time, note_pos, note_hand) = previous_notes[i];
        let time_diff = time - note_time;
        let abs_time_diff = time_diff.abs();

        if abs_time_diff > TEMPORAL_WINDOW {
            if time_diff > TEMPORAL_WINDOW {
                break;
            }
            continue;
        }
        if abs_time_diff < CLOSE_TIME_THRESHOLD
            || (position_x - note_pos).abs() < POSITION_THRESHOLD {
            return note_hand;
        }
        if abs_time_diff < min_time_diff {
            min_time_diff = abs_time_diff;
            best_idx = i;
        }
    }

    if min_time_diff < TEMPORAL_WINDOW {
        return previous_notes[best_idx].2;
    }
    if position_x < switch_threshold {
        Hand::Left
    } else {
        Hand::Right
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PgrEvent {
    pub start_time: f32,
    pub end_time: f32,
    pub start: f32,
    pub end: f32,
    #[serde(default)]
    pub start2: f32,
    #[serde(default)]
    pub end2: f32,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PgrSpeedEvent {
    pub start_time: f32,
    pub end_time: f32,
    pub value: f32,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PgrNote {
    #[serde(rename = "type")]
    kind: u8,
    time: f32,
    position_x: f32,
    hold_time: f32,
    speed: f32,
    floor_position: f32,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PgrJudgeLine {
    bpm: f32,
    #[serde(rename = "judgeLineDisappearEvents")]
    alpha_events: Vec<PgrEvent>,
    #[serde(rename = "judgeLineRotateEvents")]
    rotate_events: Vec<PgrEvent>,
    #[serde(rename = "judgeLineMoveEvents")]
    move_events: Vec<PgrEvent>,
    speed_events: Vec<PgrSpeedEvent>,
    notes_above: Vec<PgrNote>,
    notes_below: Vec<PgrNote>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PgrChart {
    /// Optional authored musical tempo for the rhythm strip only.
    #[serde(default, rename = "BPMList", alias = "bpmList", alias = "bpmlist")]
    rhythm_bpm_list: Option<Vec<PgrRhythmBpmItem>>,
    offset: f32,
    judge_line_list: Vec<PgrJudgeLine>,
    /// Phigros 9th-chapter "噪域" / block areas. Absent in every older chart.
    #[serde(default)]
    block_area_list: Vec<PgrBlockArea>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PgrVector2 {
    x: f32,
    y: f32,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PgrBlockRotateEvent {
    anchor: PgrVector2,
    time: f64,
    #[serde(default)]
    ease_type: i32,
    #[serde(default)]
    rotation: f32,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PgrBlockMoveEvent {
    end_position: PgrVector2,
    time: f64,
    #[serde(default)]
    ease_type_x: i32,
    #[serde(default)]
    ease_type_y: i32,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PgrBlockScaleEvent {
    anchor: PgrVector2,
    time: f64,
    #[serde(default)]
    ease_type_x: i32,
    #[serde(default)]
    ease_type_y: i32,
    scale: PgrVector2,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PgrBlockArea {
    top_right_percentage: PgrVector2,
    bottom_left_percentage: PgrVector2,
    #[serde(default)]
    appear_time: f64,
    #[serde(default)]
    enable_time: f64,
    #[serde(default)]
    disable_time: f64,
    #[serde(default)]
    disappear_time: f64,
    #[serde(default)]
    is_subtract: bool,
    #[serde(default)]
    rotate_events: Vec<PgrBlockRotateEvent>,
    #[serde(default)]
    move_events: Vec<PgrBlockMoveEvent>,
    #[serde(default)]
    scale_events: Vec<PgrBlockScaleEvent>,
}

fn v2(p: PgrVector2) -> Vector {
    Vector::new(p.x, p.y)
}

/// Block-area times are authored in seconds (the official runtime compares them
/// directly against the song time), so no beat conversion is applied.
fn parse_block_areas(list: Vec<PgrBlockArea>) -> Vec<BlockArea> {
    list.into_iter()
        .map(|b| BlockArea {
            animation: None,
            top_right: v2(b.top_right_percentage),
            bottom_left: v2(b.bottom_left_percentage),
            appear_time: b.appear_time,
            enable_time: b.enable_time,
            disable_time: b.disable_time,
            disappear_time: b.disappear_time,
            is_subtract: b.is_subtract,
            rotate_events: b
                .rotate_events
                .into_iter()
                .map(|e| BlockRotateEvent {
                    anchor: v2(e.anchor),
                    time: e.time,
                    ease: e.ease_type,
                    rotation: e.rotation,
                })
                .collect(),
            move_events: b
                .move_events
                .into_iter()
                .map(|e| BlockMoveEvent {
                    end: v2(e.end_position),
                    time: e.time,
                    ease_x: e.ease_type_x,
                    ease_y: e.ease_type_y,
                })
                .collect(),
            scale_events: b
                .scale_events
                .into_iter()
                .map(|e| BlockScaleEvent {
                    anchor: v2(e.anchor),
                    time: e.time,
                    ease_x: e.ease_type_x,
                    ease_y: e.ease_type_y,
                    scale: v2(e.scale),
                })
                .collect(),
        })
        .collect()
}

macro_rules! validate_events {
    ($pgr:expr) => {
        // 保留有效的事件
        $pgr.retain(|it| {
            if it.start_time > it.end_time {
                warn!("invalid time range, ignoring");
                false
            } else {
                true
            }
        });

        /* Uncomment if needed
        Official music should be continuous, so it is useless
        for i in 0..($pgr.len() - 1) {
            if $pgr[i].end_time != $pgr[i + 1].start_time {
                ptl!(bail "event-not-contiguous");
            }
        }
        */

        // Uncomment this check if needed
        // if $pgr.last().unwrap().end_time <= 900000000.0 {
        //     bail!("End time is not great enough ({})", $pgr.last().unwrap().end_time);
        // }
    };
}

fn parse_speed_events(r: f32, mut pgr: Vec<PgrSpeedEvent>, max_time: f32) -> Result<(AnimFloat, AnimFloat)> {
    validate_events!(pgr);
    assert_eq!(pgr[0].start_time, 0.0);
    let mut kfs = Vec::with_capacity(pgr.len() + 1); // 预分配 kfs 的内存容量
    let mut pos = 0.0;
    for it in pgr.iter().take(pgr.len() - 1) {
        let from_pos = pos;
        pos += (it.end_time - it.start_time) * r * it.value;
        kfs.push(Keyframe::new(it.start_time * r, from_pos, 2));
    }
    let last = pgr.last().unwrap();
    kfs.push(Keyframe::new(last.start_time * r, pos, 2));
    kfs.push(Keyframe::new(max_time, pos + (max_time - last.start_time * r) * last.value, 0));
    for kf in &mut kfs {
        kf.value /= HEIGHT_RATIO;
    }
    Ok((AnimFloat::new(pgr.iter().map(|it| Keyframe::new(it.start_time * r, it.value, 0)).collect()), AnimFloat::new(kfs)))
}

fn parse_float_events(r: f32, mut pgr: Vec<PgrEvent>) -> Result<AnimFloat> {
    validate_events!(pgr);
    let mut kfs = Vec::<Keyframe<f32>>::new();
    for e in pgr {
        let st = (e.start_time * r).max(0.); // 处理开始时间
        let en = (e.end_time * r).max(0.); // 处理结束时间
        if kfs.is_empty() || kfs.last().unwrap().value != e.start {
            kfs.push(Keyframe::new(st, e.start, 2));
        }
        kfs.push(Keyframe::new(en, e.end, 2));
    }
    // 只在非空情况下移除最后一个关键帧
    //if !kfs.is_empty() {
    //    kfs.pop();
    //}
    Ok(AnimFloat::new(kfs))
}

fn parse_move_events(r: f32, mut pgr: Vec<PgrEvent>) -> Result<AnimVector> {
    validate_events!(pgr);
    let mut kf1 = Vec::<Keyframe<f32>>::new();
    let mut kf2 = Vec::<Keyframe<f32>>::new();
    for e in pgr {
        let st = (e.start_time * r).max(0.);
        let en = e.end_time * r;
        if !kf1.last().map_or(false, |it| it.value == e.start) {
            kf1.push(Keyframe::new(st, e.start, 2));
        }
        if !kf2.last().map_or(false, |it| it.value == e.start2) {
            kf2.push(Keyframe::new(st, e.start2, 2));
        }
        kf1.push(Keyframe::new(en, e.end, 2));
        kf2.push(Keyframe::new(en, e.end2, 2));
    }
    //kf1.pop();
    //kf2.pop();
    for kf in &mut kf1 {
        kf.value = -1. + kf.value * 2.;
    }
    for kf in &mut kf2 {
        kf.value = -1. + kf.value * 2.;
    }
    Ok(AnimVector(AnimFloat::new(kf1), AnimFloat::new(kf2)))
}

fn parse_notes(r: f32, mut pgr: Vec<PgrNote>, speed: &mut AnimFloat, height: &mut AnimFloat, above: bool) -> Result<Vec<Note>> {
    // is_sorted is unstable...
    if pgr.is_empty() {
        return Ok(Vec::new());
    }
    pgr.sort_by(|a, b| a.time.partial_cmp(&b.time).expect("Invalid note time"));

    // 用于智能手部分配的追踪
    let previous_notes: Vec<(f32, f32, Hand)> = Vec::new();

    pgr.into_iter()
        .map(|pgr| {
            let time = pgr.time * r;
            Ok(Note {
                object: Object {
                    translation: AnimVector(AnimFloat::fixed(pgr.position_x * (2. * 9. / 160.)), AnimFloat::default()),
                    ..Default::default()
                },
                kind: match pgr.kind {
                    1 => NoteKind::Click,
                    2 => NoteKind::Drag,
                    3 => {
                        let end_time = (pgr.time + pgr.hold_time) * r;
                        height.set_time(time);
                        let start_height = height.now();
                        let end_height = start_height + (pgr.hold_time * pgr.speed * r / HEIGHT_RATIO);
                        NoteKind::Hold { end_time, end_height }
                    }
                    4 => NoteKind::Flick,
                    _ => ptl!(bail "unknown-note-type", "type" => pgr.kind),
                },
                time,
                speed: if pgr.kind == 3 {
                    speed.set_time(time);
                    1.
                } else {
                    pgr.speed
                },
                end_speed: pgr.speed,
                height: pgr.floor_position / HEIGHT_RATIO,
                start_height: {
                    height.set_time(time);
                    height.now()
                },
                hand: smart_assign_hand(pgr.position_x, time, &previous_notes, 0.3),
                above,
                multiple_hint: false,
                fake: false,
                judge: JudgeStatus::NotJudged,
                format: true
            })
        })
        .collect::<Result<Vec<_>>>()
}

fn parse_judge_line(pgr: PgrJudgeLine, max_time: f32) -> Result<JudgeLine> {
    if pgr.bpm <= 0.0 {
        bail!("Invalid BPM: {}", pgr.bpm);
    }
    let r = 60. / pgr.bpm / 32.;
    let (mut speed, mut height) = parse_speed_events(r, pgr.speed_events, max_time).context("Failed to parse speed events")?;
    let notes_above = parse_notes(r, pgr.notes_above, &mut speed, &mut height, true).context("Failed to parse notes above")?;
    let mut notes_below = parse_notes(r, pgr.notes_below, &mut speed, &mut height, false).context("Failed to parse notes below")?;
    let mut notes = notes_above;
    notes.append(&mut notes_below);
    let cache = JudgeLineCache::new(&mut notes);
    Ok(JudgeLine {
        object: Object {
            alpha: parse_float_events(r, pgr.alpha_events).with_context(|| ptl!("alpha-events-parse-failed"))?,
            rotation: parse_float_events(r, pgr.rotate_events).with_context(|| ptl!("rotate-events-parse-failed"))?,
            translation: parse_move_events(r, pgr.move_events).with_context(|| ptl!("move-events-parse-failed"))?,
            ..Default::default()
        },
        ctrl_obj: Rc::new(RefCell::new(CtrlObject::default())),
        kind: JudgeLineKind::Normal,
        height,
        incline: AnimFloat::default(),
        notes,
        color: Anim::default(),
        parent: None,
        z_index: 0,
        show_below: false,
        attach_ui: None,

        cache,
        cached_world_pos: None,
    })
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PgrRhythmBpmItem {
    start_time: Triple,
    bpm: f32,
}

fn parse_rhythm_bpm_list(items: Option<Vec<PgrRhythmBpmItem>>, fallback: f32) -> Result<BpmList> {
    let Some(items) = items.filter(|items| !items.is_empty()) else {
        return Ok(BpmList::new(vec![(0., fallback)]));
    };
    let mut ranges = Vec::with_capacity(items.len());
    for (index, item) in items.into_iter().enumerate() {
        let beat = item.start_time.beats();
        ensure!(beat.is_finite(), "invalid PGR BPMList entry {index}: beat must be finite");
        ensure!(
            item.bpm.is_finite() && item.bpm > 0. && (60. / item.bpm).is_finite(),
            "invalid PGR BPMList entry {index}: BPM must be finite and positive"
        );
        ranges.push((beat, item.bpm));
    }
    // Stable sorting preserves the last-authored BPM at a duplicated beat.
    ranges.sort_by(|a, b| a.0.total_cmp(&b.0));
    ensure!(ranges[0].0 == 0., "PGR BPMList must begin at beat zero");
    let tempo = BpmList::new(ranges);
    ensure!(
        tempo.points().iter().all(|point| point.1.is_finite()),
        "PGR BPMList produces a non-finite timestamp"
    );
    Ok(tempo)
}

pub fn parse_phigros(source: &str, extra: ChartExtra) -> Result<Chart> {
    let pgr: PgrChart = serde_json::from_str(source).with_context(|| ptl!("json-parse-failed"))?;
    let mut bpm_values = Vec::new();
    for (index, judge_line) in pgr.judge_line_list.iter().enumerate() {
        bpm_values.push((index as f32, judge_line.bpm));
    }
    // PGR has independent constant BPM values per line, not a global tempo
    // track. Use the first populated line as the analysis grid, while keeping
    // the line-index BPM table used by the original renderer unchanged.
    let rhythm_bpm = pgr.judge_line_list.iter()
        .find(|line| !line.notes_above.is_empty() || !line.notes_below.is_empty())
        .or_else(|| pgr.judge_line_list.first())
        .map_or(120., |line| line.bpm);
    let rhythm_bpm_authored = pgr.rhythm_bpm_list.as_ref().is_some_and(|items| !items.is_empty());
    let rhythm_bpm_list = parse_rhythm_bpm_list(pgr.rhythm_bpm_list, rhythm_bpm)?;

    let max_time = *pgr
        .judge_line_list
        .iter()
        .map(|line| {
            line.notes_above
                .iter()
                .chain(line.notes_below.iter())
                .map(|note| note.time.not_nan())
                .max()
                .unwrap_or_default()
                * (60. / line.bpm / 32.)
        })
        .max()
        .unwrap_or_default()
        + 1.;
    let mut lines = pgr
        .judge_line_list
        .into_iter()
        .enumerate()
        .map(|(id, pgr)| parse_judge_line(pgr, max_time).with_context(|| ptl!("judge-line-location", "jlid" => id)))
        .collect::<Result<Vec<_>>>()?;

    process_lines(&mut lines);
    let mut chart = Chart::new(pgr.offset, lines, BpmList::new_time(bpm_values), ChartSettings::default(), extra);
    chart.rhythm_bpm_list = Some(rhythm_bpm_list);
    chart.rhythm_bpm_authored = rhythm_bpm_authored;
    // Block-area times are authored in seconds (the official runtime compares
    // them directly against the song time), so no beat conversion is applied.
    chart.block_areas = parse_block_areas(pgr.block_area_list);
    Ok(chart)
}

#[cfg(test)]
mod explicit_rhythm_bpm_tests {
    use super::*;
    use crate::core::{RhythmNoteSelection, RhythmStrip};
    use serde_json::{json, Value};

    const FIXTURE: &str = r#"{"formatVersion":3,"offset":0.125,"BPMList":[{"startTime":[0,0,1],"bpm":120},{"startTime":[4,0,1],"bpm":180},{"startTime":[10,0,1],"bpm":90}],"judgeLineList":[{"bpm":120,"judgeLineDisappearEvents":[{"startTime":0,"endTime":512,"start":0,"end":1}],"judgeLineRotateEvents":[{"startTime":0,"endTime":512,"start":0,"end":90}],"judgeLineMoveEvents":[{"startTime":0,"endTime":512,"start":0.25,"end":0.75,"start2":0.5,"end2":0.5}],"speedEvents":[{"startTime":0,"endTime":512,"value":1}],"notesAbove":[{"type":1,"time":32,"positionX":0,"holdTime":0,"speed":1,"floorPosition":0.5},{"type":3,"time":224,"positionX":0,"holdTime":96,"speed":1,"floorPosition":3.5}],"notesBelow":[]}]}"#;

    fn fixture() -> Value {
        serde_json::from_str(FIXTURE).unwrap()
    }

    fn parse(value: &Value) -> Result<Chart> {
        parse_phigros(&value.to_string(), ChartExtra::default())
    }

    #[test]
    fn pgr_rhythm_explicit_bpm_list_updates_all_aliases_without_retiming_game() {
        let mut baseline_source = fixture();
        baseline_source.as_object_mut().unwrap().remove("BPMList");
        let baseline = parse(&baseline_source).unwrap();
        for alias in ["BPMList", "bpmList", "bpmlist"] {
            let mut value = fixture();
            let table = value.as_object_mut().unwrap().remove("BPMList").unwrap();
            value[alias] = table;
            let chart = parse(&value).unwrap();
            assert!(chart.rhythm_bpm_authored);
            let tempo = chart.rhythm_bpm_list.as_ref().unwrap();
            assert_eq!(tempo.now_bpm(0.), 120., "{alias}");
            // The optional table changes rhythm analysis, preserving PGR judgement seconds.
            assert_eq!(tempo.now_bpm(2.001), 180., "{alias}");
            assert_eq!(tempo.now_bpm(4.001), 90., "{alias}");
            assert_eq!(tempo.points(), &[(0., 0., 120.), (4., 2., 180.), (10., 4., 90.)]);
            assert_eq!(tempo.beat(3.), 7.);
            assert_eq!(tempo.time_beats(7.), 3.);

            assert_eq!(chart.offset, baseline.offset);
            assert_eq!(chart.bpm_list.borrow().points(), baseline.bpm_list.borrow().points());
            assert_eq!(chart.lines[0].notes, baseline.lines[0].notes);
            assert_eq!(chart.lines[0].object, baseline.lines[0].object);
            assert_eq!(chart.lines[0].height, baseline.lines[0].height);
            let tap = chart.lines[0].notes.iter()
                .find(|note| matches!(note.kind, NoteKind::Click)).unwrap();
            let hold = chart.lines[0].notes.iter()
                .find(|note| matches!(note.kind, NoteKind::Hold { .. })).unwrap();
            assert_eq!(tap.time, 0.5);
            assert_eq!(hold.time, 3.5);
            assert!(matches!(
                hold.kind,
                NoteKind::Hold { end_time: 5., .. }
            ));
            assert_eq!(chart.lines[0].object.alpha.keyframes[1].time, 8.);
            assert_eq!(chart.lines[0].object.rotation.keyframes[1].time, 8.);

            let strip = RhythmStrip::from_chart(&chart, RhythmNoteSelection::default(), None);
            assert_eq!(strip.bpm_at(2.001), 180., "{alias}");
            assert_eq!(strip.bpm_at(4.001), 90., "{alias}");
            assert_eq!(strip.notes()[1].time, 3.5);
            assert_eq!(strip.notes()[1].end_time, 5.);
            assert_eq!(strip.notes()[1].beat, 8.5);
            assert_eq!(strip.notes()[1].end_beat, 11.5);
        }
    }

    #[test]
    fn pgr_rhythm_explicit_bpm_list_keeps_legacy_fallback_for_missing_null_empty() {
        for table in [None, Some(Value::Null), Some(json!([]))] {
            let mut value = fixture();
            value.as_object_mut().unwrap().remove("BPMList");
            if let Some(table) = table {
                value["BPMList"] = table;
            }
            let chart = parse(&value).unwrap();
            assert!(!chart.rhythm_bpm_authored);
            assert_eq!(chart.rhythm_bpm_list.as_ref().unwrap().points(), &[(0., 0., 120.)]);
            let tap = chart.lines[0].notes.iter()
                .find(|note| matches!(note.kind, NoteKind::Click)).unwrap();
            assert_eq!(tap.time, 0.5);
        }
    }

    #[test]
    fn pgr_rhythm_explicit_bpm_list_stably_sorts_and_last_duplicate_wins() {
        let mut value = fixture();
        value["BPMList"] = json!([
            {"startTime":[10,0,1],"bpm":90},
            {"startTime":[4,0,1],"bpm":150},
            {"startTime":[0,0,1],"bpm":120},
            {"startTime":[3,1,1],"bpm":180}
        ]);
        let chart = parse(&value).unwrap();
        let tempo = chart.rhythm_bpm_list.as_ref().unwrap();
        assert_eq!(tempo.points(), &[
            (0., 0., 120.), (4., 2., 150.), (4., 2., 180.), (10., 4., 90.)
        ]);
        assert_eq!(tempo.now_bpm(2.), 180.);
        assert_eq!(tempo.now_bpm(4.), 90.);
        assert_eq!(tempo.time_beats(7.), 3.);
    }

    #[test]
    fn pgr_rhythm_explicit_bpm_list_rejects_invalid_authored_tables() {
        let invalid = [
            json!([{"startTime":[0,0,1],"bpm":0}]),
            json!([{"startTime":[0,0,1],"bpm":-120}]),
            json!([{"startTime":[0,0,1],"bpm":"120"}]),
            json!([{"startTime":[0,0,1]}]),
            json!([{"startTime":[0,0,0],"bpm":120}]),
            json!([{"startTime":[1,0,1],"bpm":120}]),
            json!([{"startTime":[-1,0,1],"bpm":120}]),
            json!([{"startTime":[0,0,1],"bpm":1e-40}]),
            json!([
                {"startTime":[0,0,1],"bpm":1e-35},
                {"startTime":[2147483647,0,1],"bpm":120}
            ]),
        ];
        for table in invalid {
            let mut value = fixture();
            value["BPMList"] = table;
            assert!(parse(&value).is_err(), "accepted invalid table: {}", value["BPMList"]);
        }
        let mut ambiguous = fixture();
        ambiguous["bpmList"] = ambiguous["BPMList"].clone();
        assert!(parse(&ambiguous).is_err(), "accepted duplicate aliases");
    }
}
