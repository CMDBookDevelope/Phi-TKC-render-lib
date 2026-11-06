//! 噪域 / ba (`blockAreaList`) effect — strict port of `ziheng18/noisepainter`.
//!
//! The reference is the WebGL tool "噪域画笔" (<https://noisepainter.sling.cc.cd/>),
//! whose fragment shader is the decompiled official Phigros block-area shader
//! (GLSL ES 100, `docs/block-area/build_full_fragment.py`). Everything below
//! mirrors it operation by operation:
//!
//!   * the mask is rasterised on the CPU over a grid of `screen / 8` cells
//!     (official `blockPx = 8`), each cell is displaced by the noise texture
//!     (`BlockCompose` pass, amplitude `0.1`), then upscaled 2x with nearest
//!     sampling; six 9-grid dilation passes (`GLOW_W`) produce the edge ring
//!     (`.g`) and the outer glow (`.b`), the fill mask is `.r`;
//!   * `displace.png` / `spark.png` / `noise.png` are the exact textures the
//!     reference ships;
//!   * the fragment shader keeps the official uniforms (`_EdgeColor`,
//!     `_FillColor`, `_GlowColor`, `_SparkTint`, opacities, displace/spark
//!     parameters) and the `One / OneMinusSrcAlpha` blend mode;
//!   * the touch-hover / ready / disabled layers are skipped (`_TouchPosCount`
//!     is 0 for an offline render), exactly like the reference does.
//!
//! Chart time is uploaded in a 1x1 RGBA texture (bound to `uNoiseTex`, unused by
//! the active layer) because macroquad overwrites `_Time` with wall-clock time,
//! which would desynchronise the effect from the chart during an offline render.
use super::{BlockArea, BlockPhase, Vector};
use macroquad::prelude::*;
use macroquad::miniquad::{BlendFactor, BlendState, BlendValue, Equation, PipelineParams, TextureWrap};
use once_cell::sync::Lazy;
use std::sync::Mutex;

// ---- `BASE_COLORS` / `BASE_FLOATS` from the reference ----------------------
const EDGE_COLOR: [f32; 3] = [1., 0.33018857, 0.33018857];
const FILL_COLOR: [f32; 3] = [0.7132075, 0.23549296, 0.23549296];
const GLOW_COLOR: [f32; 3] = [1., 0.17924517, 0.17924517];
const SPARK_TINT: [f32; 3] = [1., 0.28490567, 0.28490567];
const EDGE_OPACITY: f32 = 0.8;
const FILL_STRENGTH: f32 = 0.667;
const FILL_OPACITY: f32 = 0.667;
const GLOW_INTENSITY: f32 = 0.8;
const SPARK_MAP_OPACITY: f32 = 5.69;
const SPARK_HUE_SHIFT: f32 = 0.2;
const SPARK_DISPLACE_INTENSITY: f32 = 2.39;
const DISPLACE_BLEND_INTENSITY: f32 = 0.411;
const DISPLACE_SPEED: f32 = 1.5;
const DISPLACE_STRENGTH: f32 = 0.15;
const BACKGROUND_PIXEL_SCALE: f32 = 6.;
/// `blockPx` of the official game (one mask cell = 8 screen pixels).
const BLOCK_PX: f32 = 8.;

/// `GLOW_W[p] = (6 - p)^2.65 / sum(k^2.65)`, exactly as the reference computes.
const GLOW_W: [f32; 6] = [
    0.45856797, 0.28286122, 0.15658923, 0.07305908, 0.02494781, 0.00397469,
];

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
    /// 1x1 RGBA holding the chart time as an `f32` (big-endian byte order in RGBA).
    clock: Texture2D,
    white: Texture2D,
    grid_w: usize,
    grid_h: usize,
    tex_w: usize,
    tex_h: usize,
    rgba: Vec<u8>,
    flags: Vec<u8>,
    bits: Vec<u64>,
    scratch: Vec<u64>,
    noise_r: Vec<u8>,
    noise_w: usize,
    noise_h: usize,
    centered: [f32; 256],
    fm: Vec<u8>,
    compose: Vec<u8>,
    seen: u64,
    dumped: u32,
}

static EFFECT: Lazy<Mutex<Option<Effect>>> = Lazy::new(|| Mutex::new(None));

fn tex(bytes: &[u8], wrap: TextureWrap) -> Texture2D {
    let image = image::load_from_memory(bytes).expect("block effect texture").to_rgba8();
    let texture = Texture2D::from_rgba8(image.width() as u16, image.height() as u16, image.as_raw());
    texture.set_filter(FilterMode::Nearest);
    texture.raw_miniquad_texture_handle().set_wrap(unsafe { get_internal_gl() }.quad_context, wrap);
    texture
}

/// `medium(v)` from the reference: round-to-nearest-even at half-float
/// precision (10 mantissa bits), used to build the `CENTERED` lookup table.
fn medium(v: f32) -> f32 {
    let bits = v.to_bits();
    let bits = bits.wrapping_add(0xfff + ((bits >> 13) & 1)) & !0x1fff;
    f32::from_bits(bits)
}

/// `noiseIndex(v, size)` from the reference: mirrored addressing.
fn noise_index(v: f32, size: usize) -> usize {
    let mut v = v % 2.;
    if v < 0. {
        v += 2.;
    }
    if v > 1. {
        v = 2. - v;
    }
    let i = (v * size as f32).floor() as i64;
    i.clamp(0, size as i64 - 1) as usize
}

fn create() -> Effect {
    let material = load_material(
        shader::VERTEX,
        shader::FRAGMENT,
        MaterialParams {
            pipeline_params: PipelineParams {
                // The official active layer blends with One / OneMinusSrcAlpha.
                // macroquad picks `color_blend` or `alpha_blend` depending on
                // whether the drawn texture has an alpha channel; both need the
                // same state.
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
    let noise = image::load_from_memory(include_bytes!("../../assets/blockarea/noise.png"))
        .expect("noise.png")
        .to_rgba8();
    let noise_w = noise.width() as usize;
    let noise_h = noise.height() as usize;
    let noise_r: Vec<u8> = noise.pixels().map(|p| p[0]).collect();

    let mut centered = [0f32; 256];
    for (i, slot) in centered.iter_mut().enumerate() {
        *slot = medium(medium(i as f32 / 255.) - 0.5);
    }

    let mask = Texture2D::from_rgba8(1, 1, &[0, 0, 0, 0]);
    let clock = Texture2D::from_rgba8(1, 1, &[0, 0, 0, 0]);
    let white = Texture2D::from_rgba8(1, 1, &[255, 255, 255, 255]);
    material.set_texture("uDisplaceTex", displace);
    material.set_texture("uSparkTex", spark);
    material.set_texture("uNoiseTex", clock);
    material.set_texture("uMasks", mask);

    Effect {
        material,
        mask,
        displace,
        spark,
        clock,
        white,
        grid_w: 1,
        grid_h: 1,
        tex_w: 1,
        tex_h: 1,
        rgba: Vec::new(),
        flags: Vec::new(),
        bits: Vec::new(),
        scratch: Vec::new(),
        noise_r,
        noise_w,
        noise_h,
        centered,
        fm: Vec::new(),
        compose: Vec::new(),
        seen: 0,
        dumped: 0,
    }
}

/// Rebuild the RGBA mask for the given zones.
///
/// Mirrors `computeMasks()` of the reference:
///   1. per-cell coverage (`norm` / `subtract`, overwrite semantics),
///   2. `compose = |norm - subtract-threshold|` (a plain XOR for binary masks),
///   3. per-cell noise displacement (`BlockCompose` pass 0, amplitude 0.1),
///   4. 2x nearest upscale + six 9-grid dilation passes.
fn rebuild(effect: &mut Effect, zones: &[Zone], aspect: f32, grid_w: usize, grid_h: usize, time: f32) {
    let tex_w = grid_w * 2;
    let tex_h = grid_h * 2;
    if effect.tex_w != tex_w || effect.tex_h != tex_h {
        effect.tex_w = tex_w;
        effect.tex_h = tex_h;
        effect.rgba = vec![0u8; tex_w * tex_h * 4];
        effect.bits = vec![0u64; tex_w.div_ceil(64) * tex_h];
        effect.scratch = vec![0u64; effect.bits.len()];
        effect.mask = Texture2D::from_rgba8(tex_w as u16, tex_h as u16, &effect.rgba);
        effect.material.set_texture("uMasks", effect.mask);
    } else {
        effect.rgba.fill(0);
        effect.bits.fill(0);
    }
    if effect.flags.len() != grid_w * grid_h {
        effect.flags = vec![0u8; grid_w * grid_h];
    }
    effect.flags.fill(0);
    if effect.fm.len() != grid_w * grid_h {
        effect.fm = vec![0u8; grid_w * grid_h];
        effect.compose = vec![0u8; grid_w * grid_h];
    }
    effect.grid_w = grid_w;
    effect.grid_h = grid_h;

    // Chart space -> mask cell space. Chart space has y pointing down, exactly
    // like the reference canvas, so no vertical flip here.
    let to_cell = |p: Vector| {
        let u = (p.x + 1.) * 0.5 * grid_w as f32;
        let v = (p.y * aspect + 1.) * 0.5 * grid_h as f32;
        (u, v)
    };

    // 1. Per-cell coverage. `flags` bit 0 = covered by a normal block, bit 1 =
    //    covered by a subtract block (official `SubtractBlockBlender` overwrites
    //    with 0.1 instead of accumulating, so these are plain booleans).
    for z in zones {
        let (s, c) = z.angle.sin_cos();
        let corners = [(-1., -1.), (1., -1.), (1., 1.), (-1., 1.)].map(|(x, y)| {
            let (x, y) = (x * z.half.x, y * z.half.y);
            Vector::new(z.center.x + c * x - s * y, z.center.y + s * x + c * y)
        });
        let cells: Vec<(f32, f32)> = corners.iter().map(|p| to_cell(*p)).collect();
        let min_y = cells.iter().map(|c| c.1).fold(f32::INFINITY, f32::min).floor().max(0.) as usize;
        let max_y = cells.iter().map(|c| c.1).fold(f32::NEG_INFINITY, f32::max).ceil().min(grid_h as f32) as usize;
        let mask_bit: u8 = if z.invert { 2 } else { 1 };
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
                    effect.flags[gy * grid_w + gx] |= mask_bit;
                }
            }
        }
    }

    // 2. `compose = |norm - (subtract inside the [0.09, 0.12) window ? 255 : 0)|`.
    for i in 0..grid_w * grid_h {
        let f = effect.flags[i];
        let inside = (f & 1 != 0) ^ (f & 2 != 0);
        effect.compose[i] = if inside { 255 } else { 0 };
    }

    // 3. Per-cell noise displacement (official BlockCompose pass 0).
    let d = 0.70703125f32;
    let dt = d * (time / 20. * 2.59);
    let amp = 0.1; // "噪声强度" 1 = official
    let nw = effect.noise_w.max(1);
    let nh = effect.noise_h.max(1);
    {
        let mut wx_a = vec![0usize; grid_w];
        let mut wx_b = vec![0usize; grid_w];
        let mut wy_row = vec![0usize; grid_h];
        for x in 0..grid_w {
            let u = (x as f32 + 0.5) / grid_w as f32;
            wx_a[x] = noise_index(dt + u * 2.13, nw);
            wx_b[x] = noise_index(-dt + u * 2.13, nw);
        }
        for y in 0..grid_h {
            wy_row[y] = noise_index(dt + ((y as f32 + 0.5) / grid_h as f32) * 1.02, nh) * nw;
        }
        for y in 0..grid_h {
            let row = wy_row[y];
            for x in 0..grid_w {
                let a = effect.centered[effect.noise_r[row + wx_a[x]] as usize];
                let b = effect.centered[effect.noise_r[row + wx_b[x]] as usize];
                let u = (d * a + b * -d) * amp + (x as f32 + 0.5) / grid_w as f32;
                let v = (d * a + b * d) * amp + (y as f32 + 0.5) / grid_h as f32;
                let sxi = ((u * grid_w as f32).floor() as i64).clamp(0, grid_w as i64 - 1) as usize;
                let syi = ((v * grid_h as f32).floor() as i64).clamp(0, grid_h as i64 - 1) as usize;
                effect.fm[y * grid_w + x] = effect.compose[syi * grid_w + sxi];
            }
        }
    }

    // 4. 2x nearest upscale into RGBA + bitset.
    let stride = tex_w.div_ceil(64);
    for gy in 0..grid_h {
        for gx in 0..grid_w {
            let m = effect.fm[gy * grid_w + gx];
            if m == 0 {
                continue;
            }
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let x = gx * 2 + dx;
                let y = gy * 2 + dy;
                effect.rgba[(y * tex_w + x) * 4] = m;
                effect.bits[y * stride + (x >> 6)] |= 1u64 << (x & 63);
            }
        }
    }

    // 5. Edge / glow: nine-grid dilation + ring extraction. `edgePassesOf()`
    //    with the official edgeStrength = 6 gives one edge ring.
    let edge_passes = 1;
    for pass in 0..6 {
        let w = GLOW_W[pass];
        if w < 0.01 {
            break;
        }
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

/// Upload the chart time as an `f32` in a 1x1 RGBA texture.
fn upload_clock(effect: &Effect, time: f32) {
    let b = time.to_le_bytes();
    effect.clock.update(&Image {
        width: 1,
        height: 1,
        bytes: vec![b[0], b[1], b[2], b[3]],
    });
    effect.material.set_texture("uNoiseTex", effect.clock);
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

    let grid_w = ((viewport.2 as f32 / BLOCK_PX).floor() as usize).max(8).min(1024);
    let grid_h = ((viewport.3 as f32 / BLOCK_PX).floor() as usize).max(8).min(1024);
    rebuild(effect, zones, aspect, grid_w, grid_h, time);
    upload_clock(effect, time);
    if std::env::var_os("PRPR_BLOCK_DEBUG").is_some() {
        let lit = effect.rgba.chunks_exact(4).filter(|p| p[0] > 0).count();
        eprintln!(
            "[block] zones={} grid={grid_w}x{grid_h} lit={lit} tex={}x{}",
            zones.len(),
            effect.tex_w,
            effect.tex_h
        );
        effect.seen += 1;
        if effect.dumped < 3 && effect.seen % 400 == 0 {
            effect.dumped += 1;
            let path = format!("/tmp/ba-mask-{}-{}.raw", effect.seen, effect.dumped);
            let mut bytes = Vec::with_capacity(8 + effect.rgba.len());
            bytes.extend_from_slice(&(effect.tex_w as u32).to_le_bytes());
            bytes.extend_from_slice(&(effect.tex_h as u32).to_le_bytes());
            bytes.extend_from_slice(&effect.rgba);
            let _ = std::fs::write(&path, bytes);
            eprintln!("[block] dumped {path}");
        }
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

    // Translated from the reference's `FRAG_SRC` (decompiled official shader).
    // Every constant mirrors `BASE_COLORS` / `BASE_FLOATS`; the active layer
    // runs `fullActive()` with `_TouchPosCount == 0`.
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

const vec3 EDGE_COLOR = vec3(1.0, 0.33018857, 0.33018857);
const float EDGE_OPACITY = 0.8;
const vec3 FILL_COLOR = vec3(0.7132075, 0.23549296, 0.23549296);
const float FILL_STRENGTH = 0.667;
const float FILL_OPACITY = 0.667;
const vec3 GLOW_COLOR = vec3(1.0, 0.17924517, 0.17924517);
const float GLOW_INTENSITY = 0.8;
const vec3 SPARK_TINT = vec3(1.0, 0.28490567, 0.28490567);
const float SPARK_MAP_OPACITY = 5.69;
const float SPARK_HUE_SHIFT = 0.2;
const float SPARK_DISPLACE_INTENSITY = 2.39;
const float DISPLACE_BLEND_INTENSITY = 0.411;
const float DISPLACE_SPEED = 1.5;
const float DISPLACE_STRENGTH = 0.15;
const vec2 DISPLACE_DIRECTION = vec2(1.0, 1.0);
const float BACKGROUND_PIXEL_SCALE = 6.0;

// `_EffectRT_TexelSize.zw * 0.5` — the mask *grid* size (the texture is 2x).
vec2 basePixelUV(vec2 uvv, vec2 effectSize) {
    vec2 size = effectSize * 0.5;
    return (floor(uvv * size) + 0.5) / size;
}

// `snapshotSample()`: the scene is sampled on a screen/3 pixel grid.
vec4 snapshotSample(vec2 uvv, vec2 screenSize) {
    vec2 size = max(floor(screenSize / 3.0), vec2(1.0));
    vec2 p = (floor(clamp(uvv, vec2(0.0), vec2(1.0)) * size) + 0.5) / size;
    return texture(uScene, p);
}

// Chart time, uploaded in the 1x1 clock texture (see `upload_clock`).
float chartTime() {
    vec4 b = texture(uNoiseTex, vec2(0.5));
    uint bits = uint(floor(b.r * 255.0 + 0.5))
              | (uint(floor(b.g * 255.0 + 0.5)) << 8)
              | (uint(floor(b.b * 255.0 + 0.5)) << 16)
              | (uint(floor(b.a * 255.0 + 0.5)) << 24);
    return uintBitsToFloat(bits);
}

vec4 fullActive(vec2 fieldUV, float time, vec2 screenSize, vec2 effectSize) {
    vec2 sceneUV = fieldUV;
    vec2 displaceUV = fieldUV * vec2(0.8, 0.3);
    vec2 sparkUV = fieldUV * vec2(3.0, 1.2);

    // composeSample() / effectSample(): fill mask on the cell grid, edge and
    // glow sampled from the mask itself.
    float fill = texture(uMasks, basePixelUV(fieldUV, effectSize)).r;
    vec2 pxUV = (floor(fieldUV * effectSize) + 0.5) / effectSize;
    float edge = texture(uMasks, pxUV).g;
    float glow = texture(uMasks, fieldUV).b;

    float edgePlusFill = edge + fill;
    float total = glow + edgePlusFill;
    if (total <= 0.0001) { discard; }

    float edgeA = edge * EDGE_OPACITY;
    float glowW = glow * (1.0 - edgePlusFill);
    float glowTerm = glowW * GLOW_INTENSITY;

    // --- displace: two mirrored samples snapped to `backgroundPixelScale` ---
    vec2 dir = normalize(DISPLACE_DIRECTION);
    float dt = (time / 20.0) * DISPLACE_SPEED;
    float scale = max(BACKGROUND_PIXEL_SCALE, 1.0);
    vec2 dispA = displaceUV + dir * dt;
    vec2 dispB = displaceUV + vec2(-dir.y, dir.x) * dt;
    vec2 snapA = (floor(dispA * screenSize / scale) * scale + scale * 0.5) / screenSize;
    vec2 snapB = (floor(dispB * screenSize / scale) * scale + scale * 0.5) / screenSize;
    float dA = texture(uDisplaceTex, snapA).x;
    float dB = texture(uDisplaceTex, snapB).x;
    float avgD = (dA + dB) * 0.5;
    vec2 off = dir * (dA - 0.5) + vec2(-dir.y, dir.x) * (dB - 0.5);

    vec2 scenePx = (floor(sceneUV * screenSize / scale) * scale + scale * 0.5) / screenSize;
    vec2 sceneUv = off * DISPLACE_STRENGTH + scenePx;
    vec2 sparkUv = off * SPARK_DISPLACE_INTENSITY + sparkUV;

    float sparkV = texture(uSparkTex, sparkUv).x;
    vec3 sparkCol = vec3(sparkV) * SPARK_TINT;
    vec3 sceneCol = snapshotSample(sceneUv, screenSize).xyz;

    vec3 sparkTerm = avgD * sparkCol;
    vec3 sparkScaled = sparkTerm * SPARK_MAP_OPACITY;

    // --- RGB -> HSV of the scene colour (official ActiveBlock hue shift) ---
    float pick = (sceneCol.y >= sceneCol.z) ? 1.0 : 0.0;
    vec2 t2 = pick * (sceneCol.yz + vec2(-sceneCol.z, -sceneCol.y)) + sceneCol.zy;
    vec2 t3 = pick * vec2(1.0, -1.0) + vec2(-1.0, 0.666666687);
    float pick2 = (sceneCol.x >= t2.x) ? 1.0 : 0.0;
    vec3 t7 = vec3(-t2.x, -t2.y, -t3.y);
    float t8x = sceneCol.x + t7.x;
    vec3 t8yzw = vec3(t2.y, t3.x, t2.x) + vec3(t7.y, t7.z, -sceneCol.x);
    vec3 t21 = pick2 * vec3(t8x, t8yzw.x, t8yzw.y) + vec3(t2.x, t2.y, t3.y);
    float t5x = pick2 * t8yzw.z + sceneCol.x;
    float mn = min(t21.y, t5x);
    float delta = t21.x - mn;
    float maxc = t5x - t21.y;
    float hue = sparkScaled.x * SPARK_HUE_SHIFT + abs(maxc / (delta * 6.0 + 1.00000001e-10) + t21.z);
    float sat = sparkScaled.y * SPARK_HUE_SHIFT + delta / (t21.x + 1.00000001e-10);
    float val = sparkScaled.z * SPARK_HUE_SHIFT + t21.x;

    // --- HSV -> RGB ---
    vec3 hsv = fract(vec3(hue) + vec3(1.0, 0.666666687, 0.333333343));
    hsv = hsv * 6.0 - 3.0;
    hsv = clamp(abs(hsv) - 1.0, 0.0, 1.0) - 1.0;
    vec3 c1 = vec3(sat) * hsv + 1.0;
    vec3 c2 = c1 * val;
    vec3 c3 = c2 * 2.0;
    vec3 c4 = sparkScaled * c3;
    vec3 c5 = (-val) * c1 + 1.0;
    c5 = c5 * 2.0;
    vec3 c6 = (-sparkTerm) * SPARK_MAP_OPACITY + 1.0;
    c6 = (-c5) * c6 + 1.0;
    vec3 sel = step(vec3(0.5), c2);
    vec3 sparkOut = clamp(sel * ((-c3) * sparkScaled + c6) + c4, 0.0, 1.0);

    // --- final compose ---
    vec3 fillBase = (-avgD) * DISPLACE_BLEND_INTENSITY + FILL_COLOR;
    vec3 fillCol = FILL_STRENGTH * (sparkOut - fillBase) + fillBase;
    vec3 edgeGlow = glowTerm * GLOW_COLOR + EDGE_COLOR * edgeA;
    vec3 outCol = fillCol * fill + edgeGlow;
    float outA = glowW * GLOW_INTENSITY + (fill * FILL_OPACITY + edgeA);
    return vec4(outCol, outA);
}

void main() {
    vec2 effectSize = vec2(textureSize(uMasks, 0));
    vec2 screenSize = vec2(textureSize(uScene, 0));
    fragColor = fullActive(uv, chartTime(), screenSize, effectSize) * color;
}"#;
}
