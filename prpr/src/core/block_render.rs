//! Flat 噪域 (block area) renderer, ported from `ziheng18/block-area-edit`
//! (GPL-3.0-only) to Rust / macroquad.
//!
//! The editor's "实机贴合" (machine-truth) view is the reference: the visible
//! region is `U XOR S` — the union of normal blocks, inverted by an **odd**
//! number of subtract blocks — filled with the block colour (default noise red
//! `#FF5454`) at `0.72` alpha while active and `0.15` alpha while disabled.
//! A disabled region is not drawn underneath an active one (the editor cuts it
//! out with `destination-out`, so the two alphas never stack).
//!
//! Differences from the editor's canvas implementation that come from this
//! being a per-frame GPU renderer rather than an editing view:
//!   * the mask is evaluated analytically (planar sweep over edge crossings)
//!     instead of compositing temporary canvases, so it costs nothing at 4K;
//!   * only the machine-truth fill is implemented — the editor's lighter
//!     "colour preview" view (0.34/0.12 + outlines) is an editing aid, not
//!     something that ends up in a rendered video;
//!   * official `blockAreaList` has no colour channel, so every block uses the
//!     editor's default red. `Zone` keeps the geometry only.
use super::{BlockArea, BlockPhase, Vector};
use macroquad::prelude::*;
use std::cell::RefCell;

/// The editor's `NOISE_DEFAULT_COLOR`: `[255, 84, 84]` (#FF5454).
pub const DEFAULT_COLOR: [f32; 3] = [255. / 255., 84. / 255., 84. / 255.];
/// `globalAlpha` the editor uses for the active mask in machine-truth mode.
pub const ACTIVE_ALPHA: f32 = 0.72;
/// `globalAlpha` the editor uses for the disabled mask.
pub const DISABLED_ALPHA: f32 = 0.15;

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
}

#[derive(Clone, Copy)]
struct Boundary {
    slope: f64,
    intercept: f64,
    bottom: f64,
    top: f64,
    /// `+1` for a left edge, `-1` for a right edge; summing them over a span
    /// yields how many rectangles cover it.
    delta: f64,
    invert: bool,
    active: bool,
}

impl Boundary {
    fn x(self, y: f64) -> f64 {
        self.slope * y + self.intercept
    }
}

struct Bands {
    zones: Vec<Zone>,
    aspect: f32,
    quads: Vec<([Vec2; 4], f32)>,
}

thread_local! {
    static BANDS: RefCell<Bands> = RefCell::new(Bands {
        zones: Vec::new(),
        aspect: f32::NAN,
        quads: Vec::new(),
    });
}

pub fn draw(aspect: f32, zones: &[Zone]) {
    BANDS.with(|cached| {
        let mut cached = cached.borrow_mut();
        if cached.aspect != aspect || cached.zones != zones {
            cached.aspect = aspect;
            cached.zones.clear();
            cached.zones.extend_from_slice(zones);
            cached.quads = tessellate(aspect, zones);
        }
        gl_use_default_material();
        for &(p, alpha) in &cached.quads {
            let fill = Color::new(DEFAULT_COLOR[0], DEFAULT_COLOR[1], DEFAULT_COLOR[2], alpha);
            draw_triangle(p[0], p[1], p[2], fill);
            draw_triangle(p[0], p[2], p[3], fill);
        }
    });
}

/// Split the viewport into vertical bands whose left-to-right edge order is
/// constant, then emit one quad per visible span.
fn tessellate(aspect: f32, zones: &[Zone]) -> Vec<([Vec2; 4], f32)> {
    if !aspect.is_finite() || aspect <= 0. {
        return Vec::new();
    }
    let half_height = 1. / aspect as f64;
    let mut cuts = vec![-half_height, half_height];
    let mut edges = Vec::new();
    for z in zones {
        let (s, c) = (z.angle as f64).sin_cos();
        let corners = [(-1., -1.), (1., -1.), (1., 1.), (-1., 1.)].map(|(x, y)| {
            let (x, y) = (x * z.half.x as f64, y * z.half.y as f64);
            (z.center.x as f64 + c * x - s * y, z.center.y as f64 + s * x + c * y)
        });
        if corners.iter().any(|&(x, y)| !x.is_finite() || !y.is_finite()) {
            continue;
        }
        for i in 0..4 {
            let (a, b) = (corners[i], corners[(i + 1) % 4]);
            if a.1 == b.1 {
                continue;
            }
            let (bottom, top) = (a.1.min(b.1).max(-half_height), a.1.max(b.1).min(half_height));
            if bottom >= top {
                continue;
            }
            let slope = (b.0 - a.0) / (b.1 - a.1);
            let edge = Boundary {
                slope,
                intercept: a.0 - slope * a.1,
                bottom,
                top,
                delta: if b.1 > a.1 { -1. } else { 1. },
                invert: z.invert,
                active: z.active,
            };
            cuts.extend([bottom, top]);
            // A slanted side can leave/re-enter the viewport horizontally, so
            // the band split has to include those crossings too.
            if slope != 0. {
                for x in [-1., 1.] {
                    let y = (x - edge.intercept) / slope;
                    if y > bottom && y < top {
                        cuts.push(y);
                    }
                }
            }
            edges.push(edge);
        }
    }
    // Split wherever two edges cross, so inside a band their x order is fixed.
    for (i, a) in edges.iter().enumerate() {
        for b in &edges[..i] {
            let divisor = a.slope - b.slope;
            if divisor == 0. {
                continue;
            }
            let y = (b.intercept - a.intercept) / divisor;
            if y > a.bottom.max(b.bottom) && y < a.top.min(b.top) {
                cuts.push(y);
            }
        }
    }
    cuts.sort_unstable_by(f64::total_cmp);
    cuts.dedup();
    let mut result = Vec::new();
    let mut crossings = Vec::new();
    for band in cuts.windows(2) {
        let (lo, hi) = (band[0], band[1]);
        let mid = (lo + hi) * 0.5;
        if mid == lo || mid == hi {
            continue;
        }
        crossings.clear();
        crossings.extend(edges.iter().copied().filter(|e| e.bottom < mid && mid < e.top));
        crossings.sort_unstable_by(|a, b| a.x(mid).total_cmp(&b.x(mid)));
        let (mut active_normal, mut active_subtract) = (0_i64, 0_i64);
        let (mut disabled_normal, mut disabled_subtract) = (0_i64, 0_i64);
        for pair in crossings.windows(2) {
            let (left, right) = (pair[0], pair[1]);
            if left.active {
                if left.invert {
                    active_subtract += left.delta as i64;
                } else {
                    active_normal += left.delta as i64;
                }
            } else if left.invert {
                disabled_subtract += left.delta as i64;
            } else {
                disabled_normal += left.delta as i64;
            }
            // `U XOR S`: a span is visible when it is covered by a normal block
            // or by an odd number of subtract blocks, but not both.
            let active_visible = (active_normal > 0) != (active_subtract % 2 != 0);
            let disabled_visible = (disabled_normal > 0) != (disabled_subtract % 2 != 0);
            // An active span wins; the editor cuts the disabled mask out from
            // under it instead of letting 0.15 stack on top of 0.72.
            let alpha = if active_visible {
                ACTIVE_ALPHA
            } else if disabled_visible {
                DISABLED_ALPHA
            } else {
                continue;
            };
            if left.x(mid).max(-1.) >= right.x(mid).min(1.) {
                continue;
            }
            let p = [
                vec2(left.x(lo).clamp(-1., 1.) as f32, lo as f32),
                vec2(right.x(lo).clamp(-1., 1.) as f32, lo as f32),
                vec2(right.x(hi).clamp(-1., 1.) as f32, hi as f32),
                vec2(left.x(hi).clamp(-1., 1.) as f32, hi as f32),
            ];
            result.push((p, alpha));
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zone(x: f32, y: f32, hx: f32, hy: f32, angle: f32, invert: bool, active: bool) -> Zone {
        Zone { center: Vector::new(x, y), half: Vector::new(hx, hy), angle, invert, active }
    }

    fn coverage(ranges: &[([Vec2; 4], f32)], p: Vector) -> Option<f32> {
        ranges
            .iter()
            .find(|(q, _)| {
                if p.y < q[0].y || p.y >= q[2].y {
                    return false;
                }
                let t = (p.y - q[0].y) / (q[2].y - q[0].y);
                let left = q[0].x + (q[3].x - q[0].x) * t;
                let right = q[1].x + (q[2].x - q[1].x) * t;
                p.x > left && p.x < right
            })
            .map(|(_, alpha)| *alpha)
    }

    #[test]
    fn machine_mask_matches_xor_of_normal_and_subtract_coverage() {
        let zones = [
            zone(-0.3, 0.05, 0.42, 0.15, 0.2, false, true),
            zone(0.1, -0.1, 0.34, 0.29, -0.9, false, true),
            zone(0., 0., 1.5, 1., 0., true, true),
            zone(0.6, 0.2, 0.35, 0.13, 1.2, true, true),
            zone(0.6, 0.2, 0.15, 0.09, -0.4, true, true),
        ];
        let aspect = 16. / 9.;
        let ranges = tessellate(aspect, &zones);
        for y in 0..53 {
            for x in 0..97 {
                let p = Vector::new((x as f32 + 0.37) * 2. / 97. - 1., ((y as f32 + 0.41) * 2. / 53. - 1.) / aspect);
                let (mut normal, mut subtract) = (false, 0);
                for z in &zones {
                    let q = nalgebra::Rotation2::new(-z.angle) * (p - z.center);
                    if q.x.abs() < z.half.x && q.y.abs() < z.half.y {
                        if z.invert {
                            subtract += 1;
                        } else {
                            normal = true;
                        }
                    }
                }
                let visible = normal ^ (subtract % 2 == 1);
                assert_eq!(coverage(&ranges, p).is_some(), visible, "({x},{y})");
                if visible {
                    assert!((coverage(&ranges, p).unwrap() - ACTIVE_ALPHA).abs() < 1e-6);
                }
            }
        }
    }

    #[test]
    fn disabled_regions_use_the_faint_alpha_and_never_stack_under_active_ones() {
        let disabled = zone(0., 0., 0.5, 0.25, 0., false, false);
        let ranges = tessellate(1., &[disabled]);
        assert_eq!(ranges.len(), 1);
        assert!((ranges[0].1 - DISABLED_ALPHA).abs() < 1e-6);

        // Same rectangle, once disabled and once active → one quad at 0.72.
        let active = zone(0., 0., 0.5, 0.25, 0., false, true);
        let ranges = tessellate(1., &[disabled, active]);
        assert_eq!(ranges.len(), 1);
        assert!((ranges[0].1 - ACTIVE_ALPHA).abs() < 1e-6);
    }

    #[test]
    fn odd_subtract_counts_are_visible_and_even_ones_cancel() {
        let s = zone(0., 0., 1., 1., 0., true, true);
        assert_eq!(tessellate(1., &[s]).len(), 1, "one subtract inverts the empty region");
        assert!(tessellate(1., &[s, s]).is_empty(), "two subtracts cancel out");
        assert_eq!(tessellate(1., &[s, s, s]).len(), 1, "three subtracts invert again");
        // A normal block flips the other way round.
        let n = zone(0., 0., 1., 1., 0., false, true);
        assert!(tessellate(1., &[n, s]).is_empty(), "normal plus one subtract hides");
        assert_eq!(tessellate(1., &[n, s, s]).len(), 1, "normal plus two subtracts shows");
    }
}
