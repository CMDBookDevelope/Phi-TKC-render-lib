//! 噪域 (block area) renderer, ported from `ziheng18/block-area-edit`
//! (GPL-3.0-only) to Rust / macroquad.
//!
//! The reference draws two different things and we follow its **stage view**
//! (`renderStage`, the default `colorPreview = true` branch) line by line:
//!   * every visible block is drawn on its own, in list order, so overlapping
//!     blocks stack their translucency instead of being merged into one shape;
//!   * fill = the block colour at `0.34` while active and `0.12` while
//!     disabled (`#FF5454` is the editor's `NOISE_DEFAULT_COLOR`);
//!   * border = the same colour at `0.98` / `0.45`, `1.6px` / `1px` wide;
//!   * subtract blocks keep the fill but get a `5,3` dashed border.
//!
//! The editor's other branch ("实机贴合") collapses everything into a single
//! `U XOR S` mask at `0.72` / `0.15`; it is *not* what this port renders.
//!
//! Differences forced by the medium: the editor works in screen pixels, while
//! this runs in chart space inside the chart camera, so pixel-sized borders and
//! dashes are converted with the caller-provided `units_per_px`.
use super::{BlockArea, BlockPhase, Vector};
use macroquad::prelude::*;

/// The editor's `NOISE_DEFAULT_COLOR`: `[255, 84, 84]` (#FF5454).
pub const DEFAULT_COLOR: [f32; 3] = [255. / 255., 84. / 255., 84. / 255.];
/// `CORE.colorCss(q.color, q.active ? 0.34 : 0.12)`.
pub const ACTIVE_FILL: f32 = 0.34;
pub const DISABLED_FILL: f32 = 0.12;
/// `sctx.strokeStyle = …(q.active ? 0.98 : 0.45)`.
pub const ACTIVE_STROKE: f32 = 0.98;
pub const DISABLED_STROKE: f32 = 0.45;
/// `sctx.lineWidth = q.active ? 1.6 : 1`.
pub const ACTIVE_BORDER_PX: f32 = 1.6;
pub const DISABLED_BORDER_PX: f32 = 1.;
/// `sctx.setLineDash(q.subtract ? [5, 3] : [])`.
const DASH_ON_PX: f32 = 5.;
const DASH_OFF_PX: f32 = 3.;

/// A block area resolved at one instant: a rotated rectangle in chart space
/// (x in `[-1, 1]`, y in `[-1/aspect, 1/aspect]`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Zone {
    pub center: Vector,
    pub half: Vector,
    pub angle: f32,
    pub invert: bool,
    pub active: bool,
}

impl Zone {
    pub fn from_area(area: &BlockArea, time: f64, aspect: f32) -> Option<Self> {
        let phase = area.phase(time);
        if phase == BlockPhase::Hidden {
            return None;
        }
        let tr = area.transform(time, aspect);
        let half = tr.size.map(|v| v.abs() * 0.5);
        if half.x == 0. || half.y == 0. {
            return None;
        }
        Some(Self {
            center: tr.center,
            half,
            angle: tr.rotation.to_radians(),
            invert: area.is_subtract,
            active: phase == BlockPhase::Active,
        })
    }

    /// The four corners in the same order as the editor's `quadCorners`:
    /// bottom-left, bottom-right, top-right, top-left.
    fn corners(&self) -> [Vec2; 4] {
        let (s, c) = self.angle.sin_cos();
        [(-1., -1.), (1., -1.), (1., 1.), (-1., 1.)].map(|(x, y)| {
            let (x, y) = (x * self.half.x, y * self.half.y);
            vec2(self.center.x + c * x - s * y, self.center.y + s * x + c * y)
        })
    }
}

/// Draw every visible block like the editor's default stage view.
///
/// `units_per_px` is how many chart units one pixel covers, used to keep the
/// border width and the dash pattern looking the same at any render size.
pub fn draw(aspect: f32, zones: &[Zone], units_per_px: f32) {
    if !aspect.is_finite() || aspect <= 0. || !units_per_px.is_finite() || units_per_px <= 0. {
        return;
    }
    gl_use_default_material();
    let base = Color::new(DEFAULT_COLOR[0], DEFAULT_COLOR[1], DEFAULT_COLOR[2], 1.);
    for z in zones {
        let cs = z.corners();
        if cs.iter().any(|p| !p.x.is_finite() || !p.y.is_finite()) {
            continue;
        }
        let (fill_alpha, stroke_alpha, width_px) = if z.active {
            (ACTIVE_FILL, ACTIVE_STROKE, ACTIVE_BORDER_PX)
        } else {
            (DISABLED_FILL, DISABLED_STROKE, DISABLED_BORDER_PX)
        };
        let fill = Color { a: fill_alpha, ..base };
        draw_triangle(cs[0], cs[1], cs[2], fill);
        draw_triangle(cs[0], cs[2], cs[3], fill);

        let stroke = Color { a: stroke_alpha, ..base };
        let width = width_px * units_per_px;
        for i in 0..4 {
            edge(cs[i], cs[(i + 1) % 4], width, stroke, z.invert, units_per_px);
        }
    }
}

fn edge(a: Vec2, b: Vec2, width: f32, color: Color, dashed: bool, units_per_px: f32) {
    if !dashed {
        draw_line(a.x, a.y, b.x, b.y, width, color);
        return;
    }
    let delta = b - a;
    let len = delta.length();
    if !(len > 1e-6) {
        return;
    }
    let dir = delta / len;
    let on = DASH_ON_PX * units_per_px;
    let period = (DASH_ON_PX + DASH_OFF_PX) * units_per_px;
    let mut t = 0.;
    while t < len {
        let end = (t + on).min(len);
        let p = a + dir * t;
        let q = a + dir * end;
        draw_line(p.x, p.y, q.x, q.y, width, color);
        t += period;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zone(x: f32, y: f32, hx: f32, hy: f32, angle: f32, invert: bool, active: bool) -> Zone {
        Zone { center: Vector::new(x, y), half: Vector::new(hx, hy), angle, invert, active }
    }

    #[test]
    fn corners_follow_the_rotated_rectangle() {
        let z = zone(0.25, -0.5, 0.2, 0.1, 0., false, true);
        let cs = z.corners();
        assert!((cs[0].x - 0.05).abs() < 1e-6 && (cs[0].y + 0.6).abs() < 1e-6);
        assert!((cs[1].x - 0.45).abs() < 1e-6 && (cs[1].y + 0.6).abs() < 1e-6);
        assert!((cs[2].x - 0.45).abs() < 1e-6 && (cs[2].y + 0.4).abs() < 1e-6);
        assert!((cs[3].x - 0.05).abs() < 1e-6 && (cs[3].y + 0.4).abs() < 1e-6);

        // A quarter turn swaps the extents around the same centre.
        let r = zone(0., 0., 0.2, 0.1, std::f32::consts::FRAC_PI_2, false, true);
        let cs = r.corners();
        assert!((cs[0].x - 0.1).abs() < 1e-6 && (cs[0].y + 0.2).abs() < 1e-6);
    }

    #[test]
    fn zone_phase_comes_from_the_block_timeline() {
        let area = BlockArea {
            top_right: Vector::new(1., 1.),
            bottom_left: Vector::zeros(),
            appear_time: 1.,
            enable_time: 2.,
            disable_time: 3.,
            disappear_time: 4.,
            is_subtract: false,
            rotate_events: vec![],
            move_events: vec![],
            scale_events: vec![],
        };
        assert!(Zone::from_area(&area, 0.5, 1.).is_none());
        assert!(!Zone::from_area(&area, 1.5, 1.).unwrap().active);
        assert!(Zone::from_area(&area, 2.5, 1.).unwrap().active);
        assert!(!Zone::from_area(&area, 3.5, 1.).unwrap().active);
        assert!(Zone::from_area(&area, 4.5, 1.).is_none());
    }
}
