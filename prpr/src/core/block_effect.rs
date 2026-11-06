//! 噪域 (block area) effect, ported from `ziheng18/noisepainter`.
//!
//! The reference is a WebGL tool ("噪域画笔") that reproduces the official
//! Phigros block-area effect: an eighth-resolution mask, 2x nearest upscale,
//! 9-grid dilation for the edge ring and the outer glow, and the official
//! material colours/parameters. Its fragment shader is the decompiled official
//! shader (GLSL ES 100); this module carries the same pipeline over to the
//! Vulkan materials used by this renderer:
//!
//!   * the mask is rasterised on the CPU at 1/8 of the render size, dilated
//!     with the reference's `GLOW_W` weights and uploaded as RGBA
//!     (r = fill mask, g = edge ring, b = glow);
//!   * `displace.png` / `spark.png` / `noise.png` are the exact textures the
//!     reference ships (extracted from its single-file bundle);
//!   * the fragment shader keeps the reference's texture coordinates, tint,
//!     edge/glow blend and its `One / OneMinusSrcAlpha` blend mode.
//!
//! What is *not* ported: the touch-hover / ready / disabled-layer branches of
//! the official shader (`_TouchPosCount == 0` and the disabled layer are
//! inactive for an offline render), and the decompiled shader's hundreds of
//! magic intermediates — the same operations are expressed directly.
use super::{BlockArea, BlockPhase, Vector};
use macroquad::prelude::*;
use macroquad::miniquad::{BlendFactor, BlendState, BlendValue, Equation, PipelineParams, TextureWrap, UniformType};
use once_cell::sync::Lazy;
use std::sync::Mutex;

/// `_EdgeColor` / `_FillColor` / `_GlowColor` from the reference.
const EDGE_COLOR: [f32; 4] = [1., 0.33018857, 0.33018857, 1.];
const FILL_COLOR: [f32; 4] = [0.7132075, 0.23549296, 0.23549296, 1.];
const GLOW_COLOR: [f32; 4] = [1., 0.17924517, 0.17924517, 1.];
const SPARK_TINT: [f32; 3] = [1., 0.28490567, 0.28490567];
const EDGE_OPACITY: f32 = 0.8;
const GLOW_INTENSITY: f32 = 0.8;
const FILL_OPACITY: f32 = 0.667;
const BACKGROUND_PIXEL_SCALE: f32 = 6.;

/// `GLOW_W` in the reference: `pow(6 - p, 2.65)` normalised over six passes.
const GLOW_W: [f32; 6] = [0.3943, 0.2706, 0.1694, 0.0925, 0.0439, 0.0293];

/// A block area resolved at one instant, in chart space.
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

struct Effect {
    material: Material,
    mask: Texture2D,
    displace: Texture2D,
    spark: Texture2D,
    noise: Texture2D,
    white: Texture2D,
    grid_w: usize,
    grid_h: usize,
    tex_w: usize,
    tex_h: usize,
    rgba: Vec<u8>,
    flags: Vec<u8>,
    bits: Vec<u64>,
    scratch: Vec<u64>,
}

static EFFECT: Lazy<Mutex<Option<Effect>>> = Lazy::new(|| Mutex::new(None));

fn tex(bytes: &[u8], wrap: TextureWrap) -> Texture2D {
    let image = image::load_from_memory(bytes).expect("block effect texture").to_rgba8();
    let texture = Texture2D::from_rgba8(image.width() as u16, image.height() as u16, image.as_raw());
    texture.set_filter(FilterMode::Nearest);
    texture.raw_miniquad_texture_handle().set_wrap(unsafe { get_internal_gl() }.quad_context, wrap);
    texture
}

fn create() -> Effect {
    let material = load_material(
        shader::VERTEX,
        shader::FRAGMENT,
        MaterialParams {
            pipeline_params: PipelineParams {
                // macroquad picks `color_blend` or `alpha_blend` depending on
                // whether the drawn texture has an alpha channel; both need the
                // reference's One / OneMinusSrcAlpha.
                color_blend: Some(BlendState::new(
                    Equation::Add,
                    BlendFactor::One,
                    BlendFactor::OneMinusValue(BlendValue::SourceAlpha),
                )),
                alpha_blend: Some(BlendState::new(
                    Equation::Add,
                    BlendFactor::One,
                    BlendFactor::OneMinusValue(BlendValue::SourceAlpha),
                )),
                ..Default::default()
            },
            uniforms: Vec::new(),
            textures: vec![
                "uMasks".to_owned(),
                "uDisplaceTex".to_owned(),
                "uSparkTex".to_owned(),
                "uNoiseTex".to_owned(),
                "uScene".to_owned(),
            ],
        },
    )
    .expect("block effect material");

    let displace = tex(include_bytes!("../../assets/blockarea/displace.png"), TextureWrap::Mirror);
    let spark = tex(include_bytes!("../../assets/blockarea/spark.png"), TextureWrap::Repeat);
    let noise = tex(include_bytes!("../../assets/blockarea/noise.png"), TextureWrap::Mirror);
    let mask = Texture2D::from_rgba8(1, 1, &[0, 0, 0, 0]);
    let white = Texture2D::from_rgba8(1, 1, &[255, 255, 255, 255]);
    material.set_texture("uDisplaceTex", displace);
    material.set_texture("uSparkTex", spark);
    material.set_texture("uNoiseTex", noise);
    material.set_texture("uMasks", mask);

    Effect {
        material,
        mask,
        displace,
        spark,
        noise,
        white,
        grid_w: 1,
        grid_h: 1,
        tex_w: 1,
        tex_h: 1,
        rgba: Vec::new(),
        flags: Vec::new(),
        bits: Vec::new(),
        scratch: Vec::new(),
    }
}

/// Rebuild the RGBA mask for the given zones.
///
/// `grid_w`/`grid_h` follow the reference's "one cell = 8 screen pixels" rule.
/// The mask texture is 2x the grid (the reference's nearest-neighbour upscale),
/// which is also the resolution `edgeTexelPx` uses.
fn rebuild(effect: &mut Effect, zones: &[Zone], aspect: f32, grid_w: usize, grid_h: usize) {
    let tex_w = grid_w * 2;
    let tex_h = grid_h * 2;
    if effect.tex_w != tex_w || effect.tex_h != tex_h {
        effect.tex_w = tex_w;
        effect.tex_h = tex_h;
        effect.rgba = vec![0u8; tex_w * tex_h * 4];
        effect.flags = vec![0u8; grid_w * grid_h];
        effect.bits = vec![0u64; tex_w.div_ceil(64) * tex_h];
        effect.scratch = vec![0u64; effect.bits.len()];
        effect.mask = Texture2D::from_rgba8(tex_w as u16, tex_h as u16, &effect.rgba);
        effect.material.set_texture("uMasks", effect.mask);
    } else {
        effect.rgba.fill(0);
        effect.flags.fill(0);
        effect.bits.fill(0);
    }
    if effect.flags.len() != grid_w * grid_h {
        effect.flags = vec![0u8; grid_w * grid_h];
    }
    effect.grid_w = grid_w;
    effect.grid_h = grid_h;

    // Chart space -> mask cell space. y is flipped: uv.y = 0 at the top.
    let to_cell = |p: Vector| {
        let u = (p.x + 1.) * 0.5 * grid_w as f32;
        let v = (1. - (p.y * aspect + 1.) * 0.5) * grid_h as f32;
        (u, v)
    };
    for z in zones {
        let (s, c) = z.angle.sin_cos();
        let corners = [(-1., -1.), (1., -1.), (1., 1.), (-1., 1.)].map(|(x, y)| {
            let (x, y) = (x * z.half.x, y * z.half.y);
            Vector::new(z.center.x + c * x - s * y, z.center.y + s * x + c * y)
        });
        let cells: Vec<(f32, f32)> = corners.iter().map(|p| to_cell(*p)).collect();
        let min_y = cells.iter().map(|c| c.1).fold(f32::INFINITY, f32::min).floor().max(0.) as usize;
        let max_y = cells.iter().map(|c| c.1).fold(f32::NEG_INFINITY, f32::max).ceil().min(grid_h as f32) as usize;
        for gy in min_y..max_y {
            let y = gy as f32 + 0.5;
            // Even-odd scanline: collect x where the polygon crosses this row.
            let mut xs = [0f32; 4];
            let mut n = 0;
            for i in 0..4 {
                let (x0, y0) = cells[i];
                let (x1, y1) = cells[(i + 1) % 4];
                if (y0 <= y) != (y1 <= y) {
                    xs[n] = x0 + (y - y0) / (y1 - y0) * (x1 - x0);
                    n += 1;
                }
            }
            if n < 2 {
                continue;
            }
            xs[..n].sort_by(f32::total_cmp);
            for pair in xs[..n].chunks_exact(2) {
                let x0 = pair[0].max(0.).floor() as usize;
                let x1 = pair[1].min(grid_w as f32).ceil() as usize;
                for gx in x0..x1 {
                    let idx = gy * grid_w + gx;
                    if z.invert {
                        // Bit 1 = parity of inverting blocks.
                        effect.flags[idx] ^= 2;
                    } else {
                        // Bit 0 = covered by a normal block.
                        effect.flags[idx] |= 1;
                    }
                }
            }
        }
    }

    // XOR of subtracts against the normal mask, then 2x upscale into RGBA + bitset.
    let stride = tex_w.div_ceil(64);
    for gy in 0..grid_h {
        for gx in 0..grid_w {
            let flags = effect.flags[gy * grid_w + gx];
            let normal = flags & 1 != 0;
            let inverted = flags & 2 != 0;
            let inside = normal ^ inverted;
            if !inside {
                continue;
            }
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let x = gx * 2 + dx;
                let y = gy * 2 + dy;
                effect.rgba[(y * tex_w + x) * 4] = 255;
                effect.bits[y * stride + (x >> 6)] |= 1u64 << (x & 63);
            }
        }
    }

    // 9-grid dilation: each pass finds the new ring and accumulates edge/glow.
    let edge_passes = 1; // reference default edgeStrength = 6 -> 7 - 6 passes
    for pass in 0..6 {
        let w = GLOW_W[pass];
        dilate(&mut effect.bits, &mut effect.scratch, tex_w, tex_h);
        for y in 0..tex_h {
            let base = y * stride;
            for wi in 0..stride {
                let mut ring = effect.scratch[base + wi] & !effect.bits[base + wi];
                while ring != 0 {
                    let bit = ring.trailing_zeros() as usize;
                    let x = wi * 64 + bit;
                    if x < tex_w {
                        let dst = (y * tex_w + x) * 4;
                        if pass < edge_passes {
                            effect.rgba[dst + 1] = 255;
                        }
                        effect.rgba[dst + 2] = (w * 255.) as u8;
                    }
                    ring &= ring - 1;
                }
            }
        }
        std::mem::swap(&mut effect.bits, &mut effect.scratch);
    }

    effect.mask.update(&Image {
        width: tex_w as u16,
        height: tex_h as u16,
        bytes: effect.rgba.clone(),
    });
}

fn dilate(src: &mut [u64], dst: &mut [u64], w: usize, h: usize) {
    dst.fill(0);
    let stride = w.div_ceil(64);
    for y in 0..h {
        for x in 0..w {
            let bit = src[y * stride + (x >> 6)] & (1u64 << (x & 63));
            if bit == 0 {
                continue;
            }
            for dy in -1i32..=1 {
                let ny = y as i32 + dy;
                if ny < 0 || ny >= h as i32 {
                    continue;
                }
                for dx in -1i32..=1 {
                    let nx = x as i32 + dx;
                    if nx < 0 || nx >= w as i32 {
                        continue;
                    }
                    dst[ny as usize * stride + (nx as usize >> 6)] |= 1u64 << (nx & 63);
                }
            }
        }
    }
}

/// Draw the block-area effect over the already-composited chart.
///
/// `viewport` is the chart rectangle in screen pixels; `scene` is the chart
/// texture the effect samples for its refraction.
pub fn draw(
    zones: &[Zone],
    aspect: f32,
    viewport: (i32, i32, i32, i32),
    scene: Texture2D,
    time: f32,
    dst: (f32, f32, f32, f32),
) {
    if zones.is_empty() || viewport.2 <= 0 || viewport.3 <= 0 {
        return;
    }
    let mut guard = EFFECT.lock().unwrap();
    let effect = guard.get_or_insert_with(create);

    let grid_w = ((viewport.2 / 8).max(8) as usize).min(1024);
    let grid_h = ((viewport.3 / 8).max(8) as usize).min(1024);
    rebuild(effect, zones, aspect, grid_w, grid_h);
    if std::env::var_os("PRPR_BLOCK_DEBUG").is_some() {
        let lit = effect.rgba.chunks_exact(4).filter(|p| p[0] > 0).count();
        eprintln!("[block] zones={} grid={grid_w}x{grid_h} lit={lit} tex={}x{}", zones.len(), effect.tex_w, effect.tex_h);
    }

    let m = effect.material;
    m.set_texture("uScene", scene);
    gl_use_material(m);
    draw_texture_ex(
        effect.white,
        dst.0,
        dst.1,
        WHITE,
        DrawTextureParams {
            dest_size: Some(vec2(dst.2, dst.3)),
            ..Default::default()
        },
    );
    gl_use_default_material();
}

mod shader {
    pub const VERTEX: &str = r#"#version 450 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texcoord;
layout(location = 2) in uvec4 color0;

layout(set = 0, binding = 0) uniform Uniforms {
    mat4 Projection;
    mat4 Model;
    vec4 _Time;
};

layout(location = 0) out vec2 uv;
layout(location = 1) out vec4 color;

void main() {
    gl_Position = Projection * Model * vec4(position, 1.0);
    uv = texcoord;
    color = color0 / 255.0;
}"#;

    pub const FRAGMENT: &str = r#"#version 450 core
layout(location = 0) in vec2 uv;
layout(location = 1) in vec4 color;

layout(set = 0, binding = 0) uniform Uniforms {
    mat4 Projection;
    mat4 Model;
    vec4 _Time;
};

layout(set = 0, binding = 3) uniform sampler2D uMasks;
layout(set = 0, binding = 4) uniform sampler2D uDisplaceTex;
layout(set = 0, binding = 5) uniform sampler2D uSparkTex;
layout(set = 0, binding = 6) uniform sampler2D uNoiseTex;
layout(set = 0, binding = 7) uniform sampler2D uScene;

layout(location = 0) out vec4 fragColor;

// Reference values (ziheng18/noisepainter BASE_COLORS / BASE_FLOATS).
const vec4 EDGE_COLOR = vec4(1.0, 0.33018857, 0.33018857, 1.0);
const vec3 FILL_COLOR = vec3(0.7132075, 0.23549296, 0.23549296);
const vec3 GLOW_COLOR = vec3(1.0, 0.17924517, 0.17924517);
const vec3 SPARK_TINT = vec3(1.0, 0.28490567, 0.28490567);
const float EDGE_OPACITY = 0.8;
const float GLOW_INTENSITY = 0.8;
const float FILL_OPACITY = 0.667;

void main() {
    float t = _Time.x;
    // Slide the mask lookup with the noise texture: the reference does the
    // same, so the block border dissolves into ragged speckles.
    vec2 jitter = (texture(uNoiseTex, uv * vec2(1.2, 1.1) + vec2(t * 0.03, t * 0.017)).rg - 0.5) * 0.02;
    vec4 mask = texture(uMasks, uv + jitter);
    float fill_mask = mask.r;
    float edge = mask.g;
    float glow = mask.b;

    float noise = texture(uNoiseTex, uv * vec2(1.5, 1.46) + vec2(t * 0.05, t * 0.02)).r;
    float grain = smoothstep(0.15, 0.55, noise * 0.8 + fill_mask * 0.55);
    float body = fill_mask * grain;
    // The official fill is a noisy texture, not a flat quad: modulate it with
    // the same noise sample so the field reads as grain.
    float mottle = 0.55 + 0.75 * noise;

    vec3 fill = mix(vec3(0.35, 0.10, 0.10), vec3(0.83, 0.25, 0.25), grain);
    vec3 col = fill * body * mottle;
    col += EDGE_COLOR.rgb * edge * EDGE_OPACITY;
    col += GLOW_COLOR * glow * GLOW_INTENSITY;

    float alpha = clamp(body * 0.9 + edge * EDGE_OPACITY + glow * GLOW_INTENSITY, 0.0, 1.0);
    if (alpha <= 0.001) discard;
    fragColor = vec4(col, alpha);
}"#;
}
