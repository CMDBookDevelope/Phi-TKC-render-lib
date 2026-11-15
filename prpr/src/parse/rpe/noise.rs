//! RPE Next noise areas use beat triples and interval events, unlike the
//! official PGR seconds/keyframe format handled by `parse::pgr`.
use super::{add_bezier, BezierMap, RPEEvent};
use crate::{
    core::{
        block_animation::{BlockActivation, BlockAnimation, BlockEasing, BlockTrack, BlockTrackEvent},
        BlockArea, BpmList, Matrix, StaticTween, Triple, Vector,
    },
    fs::FileSystem,
};
use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Clone, Copy, Deserialize)]
pub(super) struct Position {
    x: f32,
    y: f32,
}
impl Position {
    fn vector(self) -> Vector {
        Vector::new(self.x, self.y)
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Event {
    #[serde(flatten)]
    value: RPEEvent,
    anchor: Option<Position>,
    transform_start: Option<f32>,
    transform_end: Option<f32>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Activation {
    start_time: Triple,
    end_time: Triple,
    ready_flash: Option<serde_json::Value>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct RpeBlockArea {
    bottom_left: Position,
    top_right: Position,
    appear_time: Triple,
    disappear_time: Triple,
    #[serde(default)]
    is_invert: bool,
    #[serde(default)]
    active_intervals: Vec<Activation>,
    #[serde(default)]
    move_x_events: Vec<Event>,
    #[serde(default)]
    move_y_events: Vec<Event>,
    #[serde(default)]
    rotate_events: Vec<Event>,
    #[serde(default)]
    scale_x_events: Vec<Event>,
    #[serde(default)]
    scale_y_events: Vec<Event>,
    layer_id: Option<String>,
}

#[derive(Deserialize)]
pub(super) struct RpeNoiseLayer {
    pub id: String,
    #[serde(default)]
    pub skin: String,
    #[serde(default)]
    pub channel: String,
    #[serde(default)]
    pub parameters: HashMap<String, serde_json::Value>,
}

fn track(events: Vec<Event>, bpm: &BpmList, center: Vector, axis: Option<usize>) -> Result<BlockTrack> {
    let mut bezier_map = BezierMap::new();
    for e in &events {
        add_bezier(&mut bezier_map, &e.value);
    }
    let mut track = BlockTrack(
        events
            .into_iter()
            .map(|e| {
                let value = e.value;
                let start_time = bpm.time(&value.start_time) as f64;
                let end_time = bpm.time(&value.end_time) as f64;
                if !start_time.is_finite() || !end_time.is_finite() || end_time < start_time {
                    bail!("Invalid RPE Next noise event time");
                }
                Ok(BlockTrackEvent {
                    start_time,
                    end_time,
                    start: value.start,
                    end: value.end,
                    tween: if value.bezier != 0 {
                        value.tween(&bezier_map)
                    } else {
                        let id = crate::parse::RPE_TWEEN_MAP.get(value.easing_type.max(1) as usize).copied().unwrap_or(2);
                        std::sync::Arc::new(BlockEasing::new(StaticTween::get_arc(id), value.easing_left, value.easing_right))
                    },
                    anchor: e.anchor.map(Position::vector).unwrap_or(center),
                    transform_start: e.transform_start.unwrap_or(value.start),
                    transform_end: e.transform_end.unwrap_or(value.end),
                    prefix: Matrix::identity(),
                })
            })
            .collect::<Result<_>>()?,
    );
    track.0.sort_by(|a, b| a.start_time.total_cmp(&b.start_time));
    if let Some(axis) = axis {
        track.prepare_anchors(axis);
    }
    Ok(track)
}

fn ready_seconds(value: Option<serde_json::Value>) -> f64 {
    match value {
        Some(serde_json::Value::Bool(v)) => {
            if v {
                0.5
            } else {
                0.
            }
        }
        Some(serde_json::Value::Number(v)) => (v.as_f64().unwrap_or(0.5).max(0.) * 1000.).round() / 1000.,
        _ => 0.5,
    }
}

fn area(b: RpeBlockArea, bpm: &BpmList, layer: usize) -> Result<BlockArea> {
    let bottom_left = b.bottom_left.vector();
    let top_right = b.top_right.vector();
    let center = (bottom_left + top_right) * 0.5;
    let appear = bpm.time(&b.appear_time) as f64;
    let disappear = bpm.time(&b.disappear_time) as f64;
    if !appear.is_finite() || !disappear.is_finite() || disappear < appear {
        bail!("Invalid RPE Next noise area lifetime");
    }
    let mut intervals = Vec::<BlockActivation>::new();
    for i in b.active_intervals {
        let original_start = bpm.time(&i.start_time) as f64;
        let original_end = bpm.time(&i.end_time) as f64;
        let start = original_start.max(appear);
        let end = original_end.min(disappear);
        if end > start || original_start == original_end && start == end {
            intervals.push(BlockActivation {
                start,
                end,
                ready_flash: ready_seconds(i.ready_flash),
            });
        }
    }
    intervals.sort_by(|a, b| a.start.total_cmp(&b.start));
    let mut merged = Vec::<BlockActivation>::new();
    for interval in intervals {
        if let Some(previous) = merged.last_mut().filter(|p| p.start == interval.start) {
            previous.end = previous.end.max(interval.end);
            previous.ready_flash = previous.ready_flash.max(interval.ready_flash);
        } else {
            merged.push(interval);
        }
    }
    Ok(BlockArea {
        bottom_left: bottom_left,
        top_right: top_right,
        appear_time: appear,
        disappear_time: disappear,
        enable_time: 0.,
        disable_time: 0.,
        is_subtract: b.is_invert,
        rotate_events: vec![],
        move_events: vec![],
        scale_events: vec![],
        animation: Some(Box::new(BlockAnimation {
            bottom_left,
            top_right,
            move_x: track(b.move_x_events, bpm, center, None)?,
            move_y: track(b.move_y_events, bpm, center, None)?,
            scale_x: track(b.scale_x_events, bpm, center, Some(0))?,
            scale_y: track(b.scale_y_events, bpm, center, Some(1))?,
            rotate: track(b.rotate_events, bpm, center, Some(2))?,
            intervals: merged,
            layer,
        })),
    })
}

pub(super) async fn parse(
    bpm: &BpmList,
    blocks: Vec<RpeBlockArea>,
    layers: Option<Vec<RpeNoiseLayer>>,
    fs: &mut dyn FileSystem,
) -> Result<(Vec<BlockArea>, Vec<crate::core::block_skin::NoiseSkin>)> {
    if blocks.is_empty() {
        return Ok((vec![], vec![]));
    }
    let layers = layers.unwrap_or_else(|| {
        vec![RpeNoiseLayer {
            id: "default".into(),
            skin: String::new(),
            channel: String::new(),
            parameters: HashMap::new(),
        }]
    });
    let mut areas = Vec::with_capacity(blocks.len());
    for (index, block) in blocks.into_iter().enumerate() {
        let id = block.layer_id.as_deref().unwrap_or("default");
        let layer = layers
            .iter()
            .position(|l| l.id == id)
            .context("RPE Next noise area references a missing layer")?;
        areas.push(area(block, bpm, layer).with_context(|| format!("RPE Next noise area {index}"))?);
    }
    let mut skins = Vec::new();
    for layer in layers {
        skins.push(
            crate::core::block_skin::NoiseSkin::load(&layer.skin, &layer.channel, layer.parameters, fs)
                .await
                .with_context(|| format!("RPE Next noise layer {}", layer.id))?,
        );
    }
    Ok((areas, skins))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::BlockPhase;
    use serde_json::{json, Value};

    fn fixture() -> Value {
        json!({"bottomLeft":{"x":-120,"y":-60},"topRight":{"x":120,"y":60},"appearTime":[0,0,1],"disappearTime":[16,0,1]})
    }
    fn event(start: f32, end: f32, a: i32, b: i32) -> Value {
        json!({"start":start,"end":end,"startTime":[a,0,1],"endTime":[b,0,1],"easingType":1})
    }
    fn load(value: Value, bpm: &BpmList) -> BlockArea {
        area(serde_json::from_value(value).unwrap(), bpm, 0).unwrap()
    }
    fn bpm() -> BpmList {
        BpmList::new(vec![(0., 120.)])
    }
    fn near(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-5, "{a} != {b}");
    }

    #[test]
    fn rpe_noise_uses_chart_bpm_and_multiple_activation_intervals() {
        let mut f = fixture();
        f["appearTime"] = json!([2, 0, 1]);
        f["disappearTime"] = json!([10, 0, 1]);
        f["activeIntervals"] = json!([{"startTime":[0,0,1],"endTime":[3,0,1],"readyFlash":false},{"startTime":[5,0,1],"endTime":[7,0,1],"readyFlash":false},{"startTime":[9,0,1],"endTime":[12,0,1],"readyFlash":false}]);
        let b = load(f, &BpmList::new(vec![(0., 120.), (4., 60.)]));
        assert_eq!(b.appear_time, 1.);
        assert_eq!(b.disappear_time, 8.);
        for (t, phase) in [
            (0.9, BlockPhase::Hidden),
            (1., BlockPhase::Active),
            (1.5, BlockPhase::Disabled),
            (3., BlockPhase::Active),
            (5., BlockPhase::Disabled),
            (7., BlockPhase::Active),
            (8., BlockPhase::Hidden),
        ] {
            assert_eq!(b.phase(t), phase);
        }
        assert_eq!(b.animation.as_ref().unwrap().intervals.len(), 3);
    }

    #[test]
    fn rpe_noise_preparation_is_seconds_and_zero_length_keeps_only_flash() {
        let mut f = fixture();
        f["activeIntervals"] =
            json!([{"startTime":[5,0,1],"endTime":[5,0,1],"readyFlash":1.25},{"startTime":[8,0,1],"endTime":[10,0,1],"readyFlash":true}]);
        let b = load(f, &bpm());
        let a = b.animation.as_ref().unwrap();
        assert!(!a.is_ready(1.24));
        assert!(a.is_ready(1.25));
        assert!(a.is_ready(2.49));
        assert!(!a.is_ready(2.5));
        assert!(!a.is_active(2.5));
        assert!(a.is_ready(3.5));
        assert!(a.is_active(4.));
        assert!(!a.is_active(5.));
        assert_eq!(ready_seconds(None), 0.5);
        assert_eq!(ready_seconds(Some(json!(false))), 0.);
        assert_eq!(ready_seconds(Some(json!(0.23456))), 0.235);
    }

    #[test]
    fn rpe_noise_canvas_is_cropped_isotropically_for_output_aspect() {
        let b = load(fixture(), &bpm());
        for aspect in [1., 1.5, 16. / 9., 9. / 16.] {
            let tr = b.transform(0., aspect);
            let unit = 2. / 1350f32.min(900. * aspect);
            near(tr.size.x, 240. * unit);
            near(tr.size.y, 120. * unit);
            near(tr.center.x, 0.);
            near(tr.center.y, 0.);
        }
    }

    #[test]
    fn rpe_noise_independent_moves_hold_endpoint_across_gaps_and_seeks() {
        let mut f = fixture();
        f["moveXEvents"] = json!([event(10., 50., 2, 4), event(-20., -40., 8, 10)]);
        f["moveYEvents"] = json!([event(100., 200., 0, 6)]);
        let b = load(f, &bpm());
        let unit = 2. / 1350.;
        for (t, x, y) in [
            (0., 0., 100.),
            (1., 10., 100. + 100. / 3.),
            (1.5, 30., 150.),
            (3., 50., 200.),
            (4., -20., 200.),
            (5., -40., 200.),
            (1.5, 30., 150.),
        ] {
            let tr = b.transform(t, 1.5);
            near(tr.center.x, x * unit);
            near(tr.center.y, y * unit);
        }
    }

    #[test]
    fn rpe_noise_separate_scale_anchors_and_clockwise_rotation() {
        let mut f = fixture();
        let mut x = event(1., 2., 0, 4);
        x["anchor"] = json!({"x":100.,"y":50.});
        let mut y = event(1., 3., 2, 6);
        y["anchor"] = json!({"x":-30.,"y":80.});
        let mut r = event(0., 90., 0, 4);
        r["anchor"] = json!({"x":0.,"y":0.});
        f["scaleXEvents"] = json!([x]);
        f["scaleYEvents"] = json!([y]);
        f["rotateEvents"] = json!([r]);
        let b = load(f, &bpm());
        let tr = b.transform(2., 1.5);
        let unit = 2. / 1350.;
        near(tr.center.x, -80. * unit);
        near(tr.center.y, 100. * unit);
        near(tr.size.x, 480. * unit);
        near(tr.size.y, 240. * unit);
        near(tr.rotation, -90.);
    }

    #[test]
    fn rpe_noise_anchor_prefix_handles_gap_baseline_jump_and_metadata() {
        let mut f = fixture();
        let mut first = event(1., 2., 0, 2);
        first["anchor"] = json!({"x":100.,"y":0.});
        let mut second = event(4., 6., 4, 6);
        second["anchor"] = json!({"x":-50.,"y":0.});
        f["scaleXEvents"] = json!([first, second]);
        let b = load(f, &bpm());
        let unit = 2. / 1350.;
        near(b.transform(1., 1.5).center.x, -100. * unit);
        near(b.transform(1.5, 1.5).center.x, -100. * unit);
        near(b.transform(2., 1.5).center.x, -300. * unit);
        near(b.transform(3., 1.5).center.x, -425. * unit);
        let mut f = fixture();
        let mut step = event(3., 3., 0, 2);
        step["anchor"] = json!({"x":100.,"y":0.});
        step["transformStart"] = json!(1.);
        step["transformEnd"] = json!(3.);
        f["scaleXEvents"] = json!([step]);
        let b = load(f, &bpm());
        near(b.transform(0., 1.5).center.x, -200. * unit);
        near(b.transform(1.5, 1.5).center.x, -200. * unit);
    }

    #[test]
    fn rpe_noise_uses_rpe_easing_and_clipping_without_pgr_enum() {
        let mut f = fixture();
        let mut e = event(0., 100., 0, 4);
        e["easingType"] = json!(5);
        f["moveXEvents"] = json!([e.clone()]);
        let b = load(f.clone(), &bpm());
        near(b.transform(1., 1.5).center.x, 25. * 2. / 1350.);
        e["easingLeft"] = json!(0.25);
        e["easingRight"] = json!(0.75);
        f["moveXEvents"] = json!([e]);
        let b = load(f, &bpm());
        near(b.transform(1., 1.5).center.x, 37.5 * 2. / 1350.);
    }

    #[test]
    fn rpe_noise_zero_and_negative_scale_are_finite_and_first_event_has_no_orbit() {
        let mut f = fixture();
        let mut e = event(0., -2., 2, 4);
        e["anchor"] = json!({"x":300.,"y":0.});
        f["scaleXEvents"] = json!([e]);
        f["rotateEvents"] = json!([event(30., 30., 2, 4)]);
        let b = load(f, &bpm());
        for t in [0., 1., 1.5, 2., 3., 1.] {
            let tr = b.transform(t, 1.5);
            assert!(tr.center.iter().all(|n| n.is_finite()));
            near(tr.center.x, 0.);
            near(tr.center.y, 0.);
        }
        near(b.transform(0., 1.5).size.x, 240. * 2. / 1350.);
        near(b.transform(2., 1.5).size.x, 480. * 2. / 1350.);
        near(b.transform(1., 1.5).rotation, -30.);
    }

    #[derive(Clone)]
    struct UnusedFs;
    #[async_trait::async_trait]
    impl FileSystem for UnusedFs {
        async fn load_file(&mut self, _: &str) -> Result<Vec<u8>> {
            bail!("Unexpected resource read")
        }
        async fn exists(&mut self, _: &str) -> Result<bool> {
            bail!("Unexpected resource read")
        }
        fn list_root(&self) -> Result<Vec<String>> {
            Ok(vec![])
        }
        fn clone_box(&self) -> Box<dyn FileSystem> {
            Box::new(self.clone())
        }
        fn as_any(&mut self) -> &mut dyn std::any::Any {
            self
        }
    }

    #[test]
    fn rpe_noise_empty_legacy_chart_needs_no_resources_or_layers() {
        let result = futures::executor::block_on(parse(&bpm(), vec![], None, &mut UnusedFs)).unwrap();
        assert!(result.0.is_empty());
        assert!(result.1.is_empty());
    }

    #[test]
    fn rpe_noise_default_layer_and_named_layers_are_resolved_without_assets() {
        let block = serde_json::from_value(fixture()).unwrap();
        let (areas, skins) = futures::executor::block_on(parse(&bpm(), vec![block], None, &mut UnusedFs)).unwrap();
        assert_eq!(skins.len(), 1);
        assert_eq!(areas[0].animation.as_ref().unwrap().layer, 0);
        let mut f = fixture();
        f["layerId"] = json!("second");
        let layers = serde_json::from_value(json!([{"id":"first"},{"id":"second"}])).unwrap();
        let (areas, skins) =
            futures::executor::block_on(parse(&bpm(), vec![serde_json::from_value(f).unwrap()], Some(layers), &mut UnusedFs)).unwrap();
        assert_eq!(skins.len(), 2);
        assert_eq!(areas[0].animation.as_ref().unwrap().layer, 1);
    }

    #[test]
    fn rpe_noise_invalid_layer_and_lifetime_fail_before_gpu_allocation() {
        let mut f = fixture();
        f["layerId"] = json!("missing");
        assert!(futures::executor::block_on(parse(&bpm(), vec![serde_json::from_value(f).unwrap()], None, &mut UnusedFs)).is_err());
        let mut f = fixture();
        f["disappearTime"] = json!([-1, 0, 1]);
        assert!(area(serde_json::from_value(f).unwrap(), &bpm(), 0).is_err());
    }
}
