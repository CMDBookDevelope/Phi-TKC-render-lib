//! Runtime for RPE Next's interval tracks. Coordinates are pixels on its
//! centered 1350 x 900 canvas; rotation is clockwise and movement is absolute.
use super::{BlockTransform, Matrix, Point, TweenFunction, Vector};
use nalgebra::Rotation2;
use std::sync::Arc;

pub struct BlockEasing {
    base: Arc<dyn TweenFunction>,
    left: f32,
    right: f32,
    begin: f32,
    range: f32,
}
impl BlockEasing {
    pub fn new(base: Arc<dyn TweenFunction>, left: f32, right: f32) -> Self {
        let begin = base.y(left);
        let range = base.y(right) - begin;
        Self {
            base,
            left,
            right,
            begin,
            range,
        }
    }
}
impl TweenFunction for BlockEasing {
    fn y(&self, x: f32) -> f32 {
        if self.range.abs() < 1e-12 {
            x
        } else {
            (self.base.y(self.left + (self.right - self.left) * x) - self.begin) / self.range
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

pub struct BlockTrackEvent {
    pub start_time: f64,
    pub end_time: f64,
    pub start: f32,
    pub end: f32,
    pub tween: Arc<dyn TweenFunction>,
    pub anchor: Vector,
    pub transform_start: f32,
    pub transform_end: f32,
    pub prefix: Matrix,
}

#[derive(Default)]
pub struct BlockTrack(pub Vec<BlockTrackEvent>);

impl BlockTrack {
    pub fn sample(&self, t: f64, fallback: f32) -> (f32, Option<&BlockTrackEvent>) {
        let Some(index) = self.0.partition_point(|e| e.start_time <= t + 1e-9).checked_sub(1) else {
            return (fallback, None);
        };
        let e = &self.0[index];
        let p = if e.end_time <= e.start_time {
            1.
        } else {
            ((t - e.start_time) / (e.end_time - e.start_time)).clamp(0., 1.) as f32
        };
        let value = if e.start == e.end {
            e.start
        } else {
            e.start + (e.end - e.start) * e.tween.y(p)
        };
        (value, Some(e))
    }

    pub fn prepare_anchors(&mut self, axis: usize) {
        let mut prefix = Matrix::identity();
        for i in 0..self.0.len() {
            if i > 0 {
                let previous = &self.0[i - 1];
                prefix = relative(axis, previous.transform_start, previous.transform_end, previous.anchor) * prefix;
                prefix = relative(axis, previous.transform_end, self.0[i].transform_start, previous.anchor) * prefix;
            }
            self.0[i].prefix = prefix;
        }
    }

    fn center(&self, t: f64, axis: usize, center: Vector) -> Vector {
        let (value, event) = self.sample(t, if axis == 2 { 0. } else { 1. });
        match event {
            Some(e) => {
                let matrix = relative(axis, e.transform_start, value, e.anchor) * e.prefix;
                matrix.transform_point(&Point::from(center)).coords
            }
            None => center,
        }
    }
}

fn relative(axis: usize, start: f32, value: f32, anchor: Vector) -> Matrix {
    let transform = if axis == 2 {
        Rotation2::new(-(value - start).to_radians()).to_homogeneous()
    } else {
        let ratio = if start == 0. { 1. } else { value / start };
        let ratio = if ratio.is_finite() { ratio } else { 1. };
        Matrix::identity().append_nonuniform_scaling(&if axis == 0 { Vector::new(ratio, 1.) } else { Vector::new(1., ratio) })
    };
    Matrix::new_translation(&anchor) * transform * Matrix::new_translation(&-anchor)
}

pub struct BlockActivation {
    pub start: f64,
    pub end: f64,
    /// Seconds, independent of chart BPM.
    pub ready_flash: f64,
}

pub struct BlockAnimation {
    pub bottom_left: Vector,
    pub top_right: Vector,
    pub move_x: BlockTrack,
    pub move_y: BlockTrack,
    pub scale_x: BlockTrack,
    pub scale_y: BlockTrack,
    pub rotate: BlockTrack,
    pub intervals: Vec<BlockActivation>,
    pub layer: usize,
}

impl BlockAnimation {
    pub fn is_active(&self, t: f64) -> bool {
        self.intervals
            .iter()
            .rev()
            .find(|e| e.end > e.start && e.start <= t + 1e-9)
            .is_some_and(|e| t + 1e-9 < e.end)
    }

    pub fn is_ready(&self, t: f64) -> bool {
        !self.is_active(t)
            && self
                .intervals
                .iter()
                .find(|e| e.start > t + 1e-9)
                .is_some_and(|e| e.ready_flash > 0. && t + 1e-9 >= e.start - e.ready_flash)
    }

    pub fn transform(&self, t: f64, aspect: f32) -> BlockTransform {
        let center = (self.bottom_left + self.top_right) * 0.5;
        let mut moved = self.scale_x.center(t, 0, center);
        moved = self.scale_y.center(t, 1, moved);
        moved = self.rotate.center(t, 2, moved);
        moved += Vector::new(self.move_x.sample(t, center.x).0 - center.x, self.move_y.sample(t, center.y).0 - center.y);
        // The editor crops the logical canvas to the output aspect instead of
        // independently stretching its two axes.
        let logical_width = 1350f32.min(900. * aspect);
        let unit = 2. / logical_width;
        let size = self.top_right - self.bottom_left;
        BlockTransform {
            center: moved * unit,
            size: Vector::new((size.x * self.scale_x.sample(t, 1.).0).abs(), (size.y * self.scale_y.sample(t, 1.).0).abs()) * unit,
            rotation: -self.rotate.sample(t, 0.).0,
        }
    }
}
