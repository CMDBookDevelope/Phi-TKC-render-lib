crate::tl_file!("parser" ptl);

use super::{process_lines, RPE_TWEEN_MAP};
use crate::{
    core::{
        Anim, AnimFloat, AnimVector, BezierTween, BpmList, Chart, ChartExtra, ChartSettings, ClampedTween, CtrlObject, GifFrames,
        JudgeLine, JudgeLineCache, JudgeLineKind, Keyframe, Note, NoteKind, Object, StaticTween, Triple, TweenFunction, Tweenable, UIElement, EPS,
        HEIGHT_RATIO,
    },
    ext::NotNanExt,
    fs::FileSystem,
    judge::JudgeStatus,
};
use anyhow::{Context, Result};
use macroquad::prelude::Color;
use serde::Deserialize;
mod noise;
use image::{codecs::gif, AnimationDecoder, DynamicImage};
use std::{collections::HashMap, time::Duration};
use crate::ext::SafeTexture;
use crate::core::note::Hand;
use std::sync::{Arc, Mutex};
use std::rc::Rc;
use std::cell::RefCell;

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

pub const RPE_WIDTH: f32 = 1350.;
pub const RPE_HEIGHT: f32 = 900.;
const SPEED_RATIO: f32 = 10. / 45. / HEIGHT_RATIO;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RPEBpmItem {
    bpm: f32,
    start_time: Triple,
}

// serde is weird...
fn f32_zero() -> f32 {
    0.
}

fn f32_one() -> f32 {
    1.
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RPEEvent<T = f32> {
    /// RPE 的绑定组 ID：相同非 0 值的事件在编辑器中被关联（链式编辑）。
    /// 它只用于 RPE 标记，对谱面读取与播放没有任何影响，因此这里仅解析存档。
    #[serde(default)]
    #[allow(dead_code)]
    linkgroup: i32,
    #[serde(default = "f32_zero")]
    easing_left: f32,
    #[serde(default = "f32_one")]
    easing_right: f32,
    #[serde(default)]
    bezier: u8,
    #[serde(default)]
    bezier_points: [f32; 4],
    easing_type: i32,
    start: T,
    end: T,
    start_time: Triple,
    end_time: Triple,
}

impl<T> RPEEvent<T> {
    /// The tween governing the segment that starts at this event's `startTime`.
    ///
    /// Factored out of `parse_events` so that `parse_gif_events` can reuse it.
    fn tween(&self, bezier_map: &BezierMap) -> Arc<dyn TweenFunction> {
        let tween = RPE_TWEEN_MAP.get(self.easing_type.max(1) as usize)
            .copied()
            .unwrap_or(RPE_TWEEN_MAP[0]);
        if self.bezier != 0 {
            Arc::clone(&bezier_map[&bezier_key(self)])
        } else if self.easing_left.abs() < EPS && (self.easing_right - 1.0).abs() < EPS {
            StaticTween::get_arc(tween)
        } else {
            Arc::new(ClampedTween::new(tween, self.easing_left..self.easing_right))
        }
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RPECtrlEvent {
    easing: u8,
    x: f32,
    #[serde(flatten)]
    value: HashMap<String, f32>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RPESpeedEvent {
    /// 见 `RPEEvent::linkgroup`：仅 RPE 编辑器标记用，读取时不使用。
    #[serde(default)]
    #[allow(dead_code)]
    linkgroup: i32,
    start_time: Triple,
    end_time: Triple,
    start: f32,
    end: f32,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RPEEventLayer {
    alpha_events: Option<Vec<RPEEvent>>,
    move_x_events: Option<Vec<RPEEvent>>,
    move_y_events: Option<Vec<RPEEvent>>,
    rotate_events: Option<Vec<RPEEvent>>,
    speed_events: Option<Vec<RPESpeedEvent>>,
}

#[derive(Clone, Deserialize)]
struct RGBColor(u8, u8, u8);
impl From<RGBColor> for Color {
    fn from(RGBColor(r, g, b): RGBColor) -> Self {
        Self::from_rgba(r, g, b, 255)
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RPEExtendedEvents {
    color_events: Option<Vec<RPEEvent<RGBColor>>>,
    text_events: Option<Vec<RPEEvent<String>>>,
    scale_x_events: Option<Vec<RPEEvent>>,
    scale_y_events: Option<Vec<RPEEvent>>,
    incline_events: Option<Vec<RPEEvent>>,
    paint_events: Option<Vec<RPEEvent>>,
    gif_events: Option<Vec<RPEEvent>>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RPENote {
    #[serde(rename = "type")]
    kind: u8,
    /// `1` = 音符从判定线的正面下落；其余数值（含 `0`）= 从背面下落。
    /// 这里读取时统一按 `above == 1` 判定为正面。
    above: u8,
    start_time: Triple,
    end_time: Triple,
    position_x: f32,
    y_offset: f32,
    alpha: u16, // some alpha has 256...
    size: f32,
    speed: f32,
    is_fake: u8,
    visible_time: f32,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RPEJudgeLine {
    /// 判定线所属的组索引（对应谱面根级的 `judgeLineGroup` 字符串数组）。
    /// 谱面读取时不会使用这个属性，仅为 RPE 标记，因此这里只解析存档。
    #[serde(rename = "Group", default)]
    #[allow(dead_code)]
    group: i32,
    /// 本线的 BPM 因子，默认 `1.0`（RPE 中无法编辑该字段）。
    /// 本线的当前 BPM 为 `nowBpm / bpmfactor`，见 [`BpmList::scaled`]。
    #[serde(default = "f32_one")]
    bpmfactor: f32,
    #[serde(rename = "Name")]
    name: String,
    #[serde(rename = "Texture")]
    texture: String,
    #[serde(rename = "father")]
    parent: Option<isize>,
    event_layers: Vec<Option<RPEEventLayer>>,
    extended: Option<RPEExtendedEvents>,
    notes: Option<Vec<RPENote>>,
    is_cover: u8,
    #[serde(default)]
    z_order: i32,
    #[serde(rename = "attachUI")]
    attach_ui: Option<UIElement>,

    #[serde(default)]
    pos_control: Vec<RPECtrlEvent>,
    #[serde(default)]
    size_control: Vec<RPECtrlEvent>,
    #[serde(default)]
    alpha_control: Vec<RPECtrlEvent>,
    #[serde(default)]
    y_control: Vec<RPECtrlEvent>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RPEMetadata {
    offset: i32,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RPEChart {
    #[serde(rename = "META")]
    meta: RPEMetadata,
    #[serde(rename = "BPMList")]
    bpm_list: Vec<RPEBpmItem>,
    judge_line_list: Vec<RPEJudgeLine>,
    #[serde(default)]
    block_area_list: Option<Vec<noise::RpeBlockArea>>,
    #[serde(default)]
    noise_area_layers: Option<Vec<noise::RpeNoiseLayer>>,
}

type BezierMap = HashMap<(u16, i16, i16), Arc<dyn TweenFunction>>;

fn bezier_key<T>(event: &RPEEvent<T>) -> (u16, i16, i16) {
    let p = &event.bezier_points;
    let int = |p: f32| (p * 100.).round() as i16;
    ((int(p[0]) * 100 + int(p[1])) as u16, int(p[2]), int(p[3]))
}

fn parse_events<T: Tweenable, V: Clone + Into<T>>(
    r: &mut BpmList,
    rpe: &[RPEEvent<V>],
    default: Option<T>,
    bezier_map: &BezierMap,
) -> Result<Anim<T>> {
    let mut kfs = Vec::new();
    if rpe.len() > 0{
        if let Some(default) = default {
            if rpe[0].start_time.beats() != 0.0 {
                kfs.push(Keyframe::new(0.0, default, 0));
            }
        }
    }
    for e in rpe {
        kfs.push(Keyframe {
            time: r.time(&e.start_time),
            value: e.start.clone().into(),
            tween: e.tween(bezier_map),
        });
        kfs.push(Keyframe::new(r.time(&e.end_time), e.end.clone().into(), 0));
    }
    Ok(Anim::new(kfs))
}

/// RPE 的 `gifEvents` 是把 GIF 的**播放进度**（`0.0..=1.0`）当成事件值来动画的，
/// 而不是普通的关键帧数值：事件之间 GIF 必须按自己的时长自动循环播放，
/// 事件接管的那一刻则要先停在循环已经走到的进度上，再跳到事件给定的值。
/// 因此这里除了事件本身，还要显式插入"回卷"关键帧（进度 `1.0` 立刻回到 `0.0`），
/// 事件结束后也要让进度继续以 GIF 的循环速率推进——不能直接用 [`parse_events`]。
fn parse_gif_events(
    r: &mut BpmList,
    rpe: &[RPEEvent],
    bezier_map: &BezierMap,
    gif: &GifFrames,
) -> Result<Anim<f32>> {
    let total_time = gif.total_time();
    // 循环周期为 0 的 GIF 无从推进，渲染侧会一直回退到最后一帧。
    if total_time == 0 {
        return Ok(Anim::default());
    }
    // 首尾两处"空转"关键帧的时间上限，足以覆盖任何实际谱面。
    const GIF_MAX_TIME: f32 = 2000.;
    // 同时限制关键帧总量：周期极短的 GIF 反复回卷会产生海量关键帧。
    const GIF_MAX_KEYFRAMES: usize = 1 << 20;
    let mut kfs = vec![Keyframe::new(0.0, 0.0, 2)];
    let mut next_rep_time: u128 = 0;
    for e in rpe {
        let start = r.time(&e.start_time);
        let end = r.time(&e.end_time);
        // 事件开始前，先把 GIF 空转过的每一圈补上回卷关键帧。
        while start > next_rep_time as f32 / 1000.
            && next_rep_time as f32 / 1000. < GIF_MAX_TIME
            && kfs.len() < GIF_MAX_KEYFRAMES
        {
            kfs.push(Keyframe::new(next_rep_time as f32 / 1000., 1.0, 0));
            kfs.push(Keyframe::new(next_rep_time as f32 / 1000., 0.0, 2));
            next_rep_time += total_time;
        }
        // 事件接管时，GIF 正好停在本圈循环的这个进度上。
        let raw_stop = 1. - (next_rep_time as f32 - start * 1000.) / total_time as f32;
        let stop_prog = if raw_stop.is_finite() { raw_stop.clamp(0., 1.) } else { 0. };
        let end_val = e.end;
        kfs.push(Keyframe::new(start, stop_prog, 0));
        kfs.push(Keyframe {
            time: start,
            value: e.start,
            tween: e.tween(bezier_map),
        });
        kfs.push(Keyframe::new(end, end_val, 2));
        // 事件结束后进度仍按 `1 / total_time` 的速率推进，直到走满这一圈。
        let end_ms = (end * 1000.).max(0.);
        let next = (end_ms + total_time as f32 * (1. - end_val)).round().max(end_ms);
        // 不得回退，否则后续回卷关键帧会早于刚刚写入的关键帧。
        next_rep_time = (next as u128).max(next_rep_time);
    }

    // 最后一个事件之后 GIF 继续自动循环（"若当前播放进度没有 gifEvents 时，GIF 会自动循环播放"）。
    while GIF_MAX_TIME > next_rep_time as f32 / 1000. && kfs.len() < GIF_MAX_KEYFRAMES {
        kfs.push(Keyframe::new(next_rep_time as f32 / 1000., 1.0, 0));
        kfs.push(Keyframe::new(next_rep_time as f32 / 1000., 0.0, 2));
        next_rep_time += total_time;
    }
    // 事件按时间递增且互不重叠；万一手工修改过的谱面出现乱序或重叠，
    // 稳定排序保证 `Anim::set_time` 依赖的关键帧有序这一前提（等序相对顺序不变）。
    kfs.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap_or(std::cmp::Ordering::Equal));
    Ok(Anim::new(kfs))
}

fn parse_speed_events(r: &mut BpmList, rpe: &[RPEEventLayer], max_time: f32) -> Result<AnimFloat> {
    let rpe: Vec<_> = rpe.iter().filter_map(|it| it.speed_events.as_ref()).collect();
    if rpe.is_empty() {
        return Ok(AnimFloat::default());
    };
    let anis: Vec<_> = rpe
        .into_iter()
        .map(|it| {
            let mut kfs = Vec::new();
            for e in it {
                kfs.push(Keyframe::new(r.time(&e.start_time), e.start, 2));
                kfs.push(Keyframe::new(r.time(&e.end_time), e.end, 0));
            }
            AnimFloat::new(kfs)
        })
        .collect();
    let mut pts: Vec<_> = anis.iter().flat_map(|it| it.keyframes.iter().map(|it| it.time.not_nan())).collect();
    pts.push(max_time.not_nan());
    pts.sort();
    pts.dedup();
    let mut sani = AnimFloat::chain(anis);
    sani.map_value(|v| v * SPEED_RATIO);
    for i in 0..(pts.len() - 1) {
        let now_time = *pts[i];
        let end_time = *pts[i + 1];
        sani.set_time(now_time);
        let start_speed = sani.now();
        sani.set_time(end_time - 1e-4);
        let end_speed = sani.now();
        let duration = end_time - now_time;
        if duration > EPS && (start_speed - end_speed).abs() > EPS {
            let acceleration = (end_speed - start_speed) / duration;
            if acceleration.abs() > EPS {
                if start_speed.signum() != end_speed.signum() {
                    let zero_time = now_time - start_speed / acceleration;
                    if zero_time > now_time && zero_time < end_time {
                        pts.push(zero_time.not_nan());
                    }
                }
                let mid_time = (now_time + end_time) / 2.0;
                pts.push(mid_time.not_nan());
            }
        }
    }
    pts.sort();
    pts.dedup();
    let mut kfs = Vec::new();
    let mut height = 0.0;

    if *pts[0] > 0.0 {
        kfs.push(Keyframe::new(0.0, height, 2));
    }

    for i in 0..(pts.len() - 1) {
        let now_time = *pts[i];
        let end_time = *pts[i + 1];
        let duration = end_time - now_time;

        sani.set_time(now_time);
        let start_speed = sani.now();
        sani.set_time(end_time - 1e-4);
        let end_speed = sani.now();
        let delta_height = if duration < EPS {
            0.0
        } else if (start_speed - end_speed).abs() < EPS {
            start_speed * duration
        } else {
            (start_speed + end_speed) * duration / 2.0
        };

        // 添加当前时间点的关键帧
        kfs.push(Keyframe::new(now_time, height, 2));
        height += delta_height;
    }

    // 添加最终关键帧
    kfs.push(Keyframe::new(max_time, height, 0));

    Ok(AnimFloat::new(kfs))
}

fn parse_notes(r: &mut BpmList, rpe: Vec<RPENote>, height: &mut AnimFloat) -> Result<Vec<Note>> {
    let mut previous_notes: Vec<(f32, f32, Hand)> = Vec::new();

    let notes = rpe.into_iter()
        .map(|note| {
            let time = r.time(&note.start_time);
            let y_offset = note.y_offset * 2. / RPE_HEIGHT * note.speed;

            // Set time and get height after calculating y_offset
            height.set_time(time);
            let note_height = height.now() + y_offset;
            let normalized_x = note.position_x / (RPE_WIDTH / 2.0) - 1.0;

            Ok(Note {
                object: Object {
                    alpha: if note.visible_time >= time {
                        if note.alpha >= 255 {
                            AnimFloat::default()
                        } else {
                            AnimFloat::fixed(note.alpha as f32 / 255.)
                        }
                    } else {
                        let alpha = note.alpha.min(255) as f32 / 255.;
                        AnimFloat::new(vec![Keyframe::new(0.0, 0.0, 0), Keyframe::new(time - note.visible_time, alpha, 0)])
                    },
                    translation: AnimVector(AnimFloat::fixed(note.position_x / (RPE_WIDTH / 2.)), AnimFloat::fixed(y_offset)),
                    scale: AnimVector(
                        if note.size == 1.0 {
                            AnimFloat::default()
                        } else {
                            AnimFloat::fixed(note.size)
                        },
                        AnimFloat::default(),
                    ),
                    ..Default::default()
                },
                kind: match note.kind {
                    1 => NoteKind::Click,
                    2 => {
                        let end_time = r.time(&note.end_time);
                        height.set_time(end_time);
                        NoteKind::Hold {
                            end_time,
                            end_height: height.now() + y_offset,
                        }
                    }
                    3 => NoteKind::Flick,
                    4 => NoteKind::Drag,
                    _ => ptl!(bail "unknown-note-type", "type" => note.kind),
                },
                time,
                height: note_height,
                speed: note.speed,
                end_speed: note.speed,
                start_height: {
                    height.set_time(r.time(&note.start_time));
                    height.now() + y_offset
                },
                hand: smart_assign_hand(normalized_x, time, &previous_notes, 0.3),
                above: note.above == 1,
                multiple_hint: false,
                fake: note.is_fake != 0,
                judge: JudgeStatus::NotJudged,
                format: false,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    for note in &notes {
        let pos = note.object.translation.0.now();
        previous_notes.push((note.time, pos, note.hand));
    }

    Ok(notes)
}

fn parse_ctrl_events(rpe: &[RPECtrlEvent], key: &str) -> AnimFloat {
    let vals: Vec<_> = rpe
        .iter()
        .filter_map(|it| it.value.get(key).copied())
        .collect();

    if rpe.is_empty() || (rpe.len() == 2 && rpe[0].easing == 1 && (vals[0] - 1.).abs() < 1e-4) {
        return AnimFloat::default();
    }
    let mut kfs = Vec::new();


    if let Some(&first_val) = vals.first() {
        kfs.push(Keyframe::new(0.0, first_val, 0));
    }

    // 添加其他关键帧
    for (it, val) in rpe.iter().zip(vals.iter().copied()) {
        kfs.push(Keyframe::new(
            it.x,
            val,
            RPE_TWEEN_MAP.get(it.easing.max(1) as usize).copied().unwrap_or(RPE_TWEEN_MAP[0])
        ));
    }

    AnimFloat::new(kfs)
}

async fn parse_judge_line(
    r: &mut BpmList,
    rpe: RPEJudgeLine,
    max_time: f32,
    fs: &mut dyn FileSystem,
    bezier_map: &BezierMap,
    texture_cache: &mut std::collections::HashMap<String, SafeTexture>,
) -> Result<JudgeLine> {
    // 每条判定线可带 `bpmfactor`（默认 1.0，RPE 中不可编辑）：本线的当前 BPM 为
    // `nowBpm / bpmfactor`，因此本线的 beat→秒换算整体是谱面级换算的 `bpmfactor` 倍。
    // 换成按线缩放的 BpmList 后，事件、音符与本线的速度事件都会自然随之缩放。
    let bpmfactor = if rpe.bpmfactor.is_finite() && rpe.bpmfactor != 1.0 {
        rpe.bpmfactor
    } else {
        1.0
    };
    // 速度事件的时间上限属于本线时间轴，需一并缩放；
    // `bpmfactor == 1.0` 时 `x * 1.0` 恒等，行为与改动前完全一致。
    let max_time = max_time * bpmfactor;
    let mut scaled = (bpmfactor != 1.0).then(|| r.scaled(bpmfactor));
    let r = match scaled.as_mut() {
        Some(it) => it,
        None => r,
    };

    let event_layers: Vec<_> = rpe.event_layers.into_iter().flatten().collect();

    fn events_with_factor(
        r: &mut BpmList,
        event_layers: &[RPEEventLayer],
        get: impl Fn(&RPEEventLayer) -> &Option<Vec<RPEEvent>>,
        factor: f32,
        desc: &str,
        bezier_map: &BezierMap,
    ) -> Result<AnimFloat> {
        let anis: Vec<_> = event_layers
            .iter()
            // Empty layers add zero. An empty Anim at the head of a chain
            // hides later layers; at its tail it has no value to add.
            .filter_map(|it| get(it).as_ref().filter(|es| !es.is_empty()).map(|es| parse_events(r, es, None, bezier_map)))
            .collect::<Result<_>>()
            .with_context(|| ptl!("type-events-parse-failed", "type" => desc))?;
        let mut res = AnimFloat::chain(anis);
        res.map_value(|v| v * factor);
        Ok(res)
    }

    let mut height = parse_speed_events(r, &event_layers, max_time)?;
    let mut notes = parse_notes(r, rpe.notes.unwrap_or_default(), &mut height)?;
    let cache = JudgeLineCache::new(&mut notes);

    // `gifEvents` 存在时 `Texture` 指向的就是 GIF 文件，因此必须先于纹理名判定，
    // 否则这类判定线会被当成静态纹理，`gifEvents` 整条被忽略。
    let kind = if let Some(events) = rpe.extended.as_ref().and_then(|e| e.gif_events.as_ref()) {
        let data = fs
            .load_file(&rpe.texture)
            .await
            .with_context(|| ptl!("gif-load-failed", "path" => rpe.texture.clone()))?;
        let decoder = gif::GifDecoder::new(&data[..])?;
        let frames = GifFrames::new(
            decoder
                .into_frames()
                .map(|frame| -> (u128, SafeTexture) {
                    let frame = frame.unwrap();
                    let delay: Duration = frame.delay().into();
                    (delay.as_millis(), SafeTexture::from(DynamicImage::ImageRgba8(frame.into_buffer())))
                })
                .collect(),
        );
        let events = parse_gif_events(r, events, bezier_map, &frames).with_context(|| ptl!("gif-events-parse-failed"))?;
        JudgeLineKind::TextureGif(events, frames, rpe.texture.clone())
    } else if rpe.texture == "line.png" {
        if let Some(events) = rpe.extended.as_ref().and_then(|e| e.paint_events.as_ref()) {
            JudgeLineKind::Paint(
                parse_events(r, events, Some(-1.), bezier_map).with_context(|| ptl!("paint-events-parse-failed"))?,
                Arc::new(Mutex::new((None, false))),
            )
        } else if let Some(events) = rpe.extended.as_ref().and_then(|e| e.text_events.as_ref()) {
            JudgeLineKind::Text(
                parse_events(r, events, Some(String::new()), bezier_map)
                    .with_context(|| ptl!("text-events-parse-failed"))?,
            )
        } else {
            JudgeLineKind::Normal
        }
    } else {
        match texture_cache.get(&rpe.texture) {
            Some(texture) => JudgeLineKind::Texture(texture.clone(), rpe.texture.clone()),
            None => {
                let img_data = fs
                    .load_file(&rpe.texture)
                    .await
                    .with_context(|| ptl!("illustration-load-failed", "path" => rpe.texture.clone()))?;
                let img = image::load_from_memory(&img_data)?;
                let texture = SafeTexture::from_image(&img).with_mipmap();
                texture_cache.insert(rpe.texture.clone(), texture.clone());

                JudgeLineKind::Texture(texture, rpe.texture.clone())
            }
        }
    };

    Ok(JudgeLine {
        object: Object {
            alpha: events_with_factor(r, &event_layers, |it| &it.alpha_events, 1. / 255., "alpha", bezier_map)?,
            rotation: events_with_factor(r, &event_layers, |it| &it.rotate_events, -1., "rotate", bezier_map)?,
            translation: AnimVector(
                events_with_factor(r, &event_layers, |it| &it.move_x_events, 2. / RPE_WIDTH, "move X", bezier_map)?,
                events_with_factor(r, &event_layers, |it| &it.move_y_events, 2. / RPE_HEIGHT, "move Y", bezier_map)?,
            ),
            scale: {
                fn parse(r: &mut BpmList, opt: &Option<Vec<RPEEvent>>, factor: f32, bezier_map: &BezierMap) -> Result<AnimFloat> {
                    let mut res = opt
                        .as_ref()
                        .map(|it| parse_events(r, it, None, bezier_map))
                        .transpose()?
                        .unwrap_or_default();
                    res.map_value(|v| v * factor);
                    Ok(res)
                }
                let factor = if rpe.texture == "line.png" {
                    1.
                } else {
                    // `line.png` 的缩放以 1.0 为基准；其他纹理沿用 moveX 的同一个
                    // 换算系数（RPE 画布宽 `RPE_WIDTH` = 1350px ↔ 引擎单位 2.0），
                    // 属于单位换算，不是经验调参值。
                    2. / RPE_WIDTH
                };
                rpe.extended
                    .as_ref()
                    .map(|e| -> Result<_> {
                        Ok(AnimVector(
                            parse(
                                r,
                                &e.scale_x_events,
                                factor
                                    * if rpe.texture == "line.png"
                                    && rpe
                                    .extended
                                    .as_ref()
                                    .map_or(true, |it| it.text_events.as_ref().map_or(true, |it| it.is_empty()))
                                    && rpe.attach_ui.is_none()
                                {
                                    0.5
                                } else {
                                    1.
                                },
                                bezier_map,
                            )?,
                            parse(r, &e.scale_y_events, factor, bezier_map)?,
                        ))
                    })
                    .transpose()?
                    .unwrap_or_default()
            },
        },
        ctrl_obj: Rc::new(RefCell::new(CtrlObject {
            alpha: parse_ctrl_events(&rpe.alpha_control, "alpha"),
            size: parse_ctrl_events(&rpe.size_control, "size"),
            pos: parse_ctrl_events(&rpe.pos_control, "pos"),
            y: parse_ctrl_events(&rpe.y_control, "y"),
        })),
        height,
        incline: if let Some(events) = rpe.extended.as_ref().and_then(|e| e.incline_events.as_ref()) {
            parse_events(r, events, Some(0.), bezier_map).with_context(|| ptl!("incline-events-parse-failed"))?
        } else {
            AnimFloat::default()
        },
        notes,
        kind,
        color: if let Some(events) = rpe.extended.as_ref().and_then(|e| e.color_events.as_ref()) {
            parse_events(r, events, Some(Color::new(0.0, 0.0, 0.0, 0.0)), bezier_map).with_context(|| ptl!("color-events-parse-failed"))?
        } else {
            Anim::default()
        },
        parent: {
            let parent = rpe.parent.unwrap_or(-1);
            if parent == -1 {
                None
            } else {
                Some(parent as usize)
            }
        },
        z_index: rpe.z_order,
        show_below: rpe.is_cover != 1,
        attach_ui: rpe.attach_ui,
        cache,
        cached_world_pos: None,
    })
}

fn add_bezier<T>(map: &mut BezierMap, event: &RPEEvent<T>) {
    if event.bezier != 0 {
        let p = &event.bezier_points;
        let int = |p: f32| (p * 100.).round() as i16;
        map.entry(((int(p[0]) * 100 + int(p[1])) as u16, int(p[2]), int(p[3])))
            .or_insert_with(|| Arc::new(BezierTween::new((p[0], p[1]), (p[2], p[3]))));
    }
}

fn get_bezier_map(rpe: &RPEChart) -> BezierMap {
    let mut map = HashMap::new();
    for line in &rpe.judge_line_list {
        for event_layer in line.event_layers.iter().flatten() {
            for event in event_layer
                .alpha_events
                .iter()
                .chain(event_layer.move_x_events.iter())
                .chain(event_layer.move_y_events.iter())
                .chain(event_layer.rotate_events.iter())
                .flatten()
            {
                add_bezier(&mut map, event);
            }
        }
    }
    map
}

pub async fn parse_rpe(source: &str, fs: &mut dyn FileSystem, extra: ChartExtra) -> Result<Chart> {
    let rpe: RPEChart = serde_json::from_str(source).with_context(|| ptl!("json-parse-failed"))?;
    let bezier_map = get_bezier_map(&rpe);
    let mut r = BpmList::new(rpe.bpm_list.into_iter().map(|it| (it.start_time.beats(), it.bpm)).collect());
    let (block_areas, noise_skins) = noise::parse(&r, rpe.block_area_list.unwrap_or_default(), rpe.noise_area_layers, fs).await?;
    let mut texture_cache = std::collections::HashMap::new();
    fn vec<T>(v: &Option<Vec<T>>) -> impl Iterator<Item = &T> {
        v.iter().flat_map(|it| it.iter())
    }

    #[rustfmt::skip]
    let max_time = *rpe
        .judge_line_list
        .iter()
        .map(|line| {
            line.notes.as_ref().map(|notes| {
                notes
                    .iter()
                    .map(|note| {
                        // 修复：在这里处理Hold音符的结束时间
                        let time = if note.kind == 2 { // Hold类型
                            r.time(&note.end_time)
                        } else {
                            r.time(&note.start_time)
                        };
                        time.not_nan()
                    })
                    .max()
                    .unwrap_or_default()
            }).unwrap_or_default().max(
                line.event_layers.iter().filter_map(|it| it.as_ref().map(|layer| {
                    vec(&layer.alpha_events)
                        .chain(vec(&layer.move_x_events))
                        .chain(vec(&layer.move_y_events))
                        .chain(vec(&layer.rotate_events))
                        .map(|it| r.time(&it.end_time).not_nan())
                        .max().unwrap_or_default()
                })).max().unwrap_or_default()
            ).max(
                line.extended.as_ref().map(|e| {
                    vec(&e.scale_x_events)
                        .chain(vec(&e.scale_y_events))
                        .map(|it| r.time(&it.end_time).not_nan())
                        .max().unwrap_or_default()
                        .max(vec(&e.text_events).map(|it| r.time(&it.end_time).not_nan()).max().unwrap_or_default())
                }).unwrap_or_default()
            )
        })
        .max().unwrap_or_default() + 1.;

    let mut lines = Vec::new();
    for (id, rpe) in rpe.judge_line_list.into_iter().enumerate() {
        let name = rpe.name.clone();
        lines.push(
            parse_judge_line(&mut r, rpe, max_time, fs, &bezier_map, &mut texture_cache)
                .await
                .with_context(move || ptl!("judge-line-location-name", "jlid" => id, "name" => name))?,
        );
    }
    process_lines(&mut lines);
    let mut chart = Chart::new(rpe.meta.offset as f32 / 1000.0, lines, r, ChartSettings::default(), extra);
    chart.block_areas = block_areas;
    chart.noise_skins = std::cell::RefCell::new(noise_skins);
    Ok(chart)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line_with_layers(layers: serde_json::Value) -> JudgeLine {
        let line = serde_json::from_value(serde_json::json!({
            "Name": "Empty event layers", "Texture": "line.png", "isCover": 1,
            "eventLayers": layers, "notes": []
        })).unwrap();
        let mut bpm = BpmList::new(vec![(0., 120.)]);
        let mut fs = crate::fs::fs_from_assets("").unwrap();
        futures::executor::block_on(parse_judge_line(
            &mut bpm, line, 10., fs.as_mut(), &HashMap::new(), &mut HashMap::new(),
        )).unwrap()
    }

    fn constant(value: f32) -> serde_json::Value {
        serde_json::json!([{
            "startTime": [0, 0, 1], "endTime": [2, 0, 1], "start": value,
            "end": value, "easingType": 1
        }])
    }

    #[test]
    fn empty_first_alpha_layer_keeps_explicit_transparency() {
        let mut line = line_with_layers(serde_json::json!([
            { "alphaEvents": [] }, { "alphaEvents": constant(0.) },
            { "alphaEvents": constant(0.) }, { "alphaEvents": constant(0.) }
        ]));
        for time in [0., 0.5, 2., 60.] {
            line.object.alpha.set_time(time);
            assert_eq!(line.object.alpha.now_opt(), Some(0.));
        }
    }

    #[test]
    fn empty_middle_and_last_layers_do_not_break_additive_tracks() {
        let mut line = line_with_layers(serde_json::json!([
            { "alphaEvents": constant(51.), "moveXEvents": constant(100.),
              "rotateEvents": constant(30.) },
            { "alphaEvents": [], "moveXEvents": [], "rotateEvents": [] },
            { "alphaEvents": constant(102.), "moveXEvents": constant(200.),
              "rotateEvents": constant(60.) },
            { "alphaEvents": [], "moveXEvents": [], "rotateEvents": [] }, null
        ]));
        for time in [0., 0.5, 2., 60.] {
            line.object.alpha.set_time(time);
            line.object.translation.set_time(time);
            line.object.rotation.set_time(time);
            assert!((line.object.alpha.now_opt().unwrap() - 0.6).abs() < 1e-6);
            assert!((line.object.translation.0.now() - 600. / RPE_WIDTH).abs() < 1e-6);
            assert_eq!(line.object.rotation.now(), -90.);
        }
    }
}
