use crate::config::HdrLayerSettings;
use macroquad::{
    camera::{set_camera, Camera2D},
    color::WHITE,
    material::{gl_use_default_material, gl_use_material, load_material, Material, MaterialParams},
    math::vec2,
    miniquad::{RenderPass, Texture, TextureFormat, TextureParams, UniformType},
    texture::draw_texture_ex,
    texture::{DrawTextureParams, RenderTarget, Texture2D},
    window::get_internal_gl,
};
use once_cell::unsync::Lazy;
use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};

static C: AtomicU64 = AtomicU64::new(0);

pub fn begin_readback_cycle() {
    C.fetch_add(1, Ordering::Relaxed);
}

struct Rb {
    cache: Option<(u64, usize, Vec<f32>)>,
    target: Option<RenderTarget>,
    staging: Vec<u8>,
}

thread_local! {
    static RB: RefCell<Rb> = RefCell::new(Rb { cache: None, target: None, staging: Vec::new() });
    static HDR_MATERIAL: Lazy<Material> = Lazy::new(|| load_material(HDR_VERTEX, HDR_FRAGMENT, MaterialParams {
        uniforms: vec![
            ("mode".into(), UniformType::Int1), ("peak".into(), UniformType::Float1),
            ("knee".into(), UniformType::Float1), ("power".into(), UniformType::Float1),
            ("shadowGain".into(), UniformType::Float1), ("chromaScale".into(), UniformType::Float1),
            ("softClip".into(), UniformType::Int1),
            ("inputLinear".into(), UniformType::Int1),
            ("compareBase".into(), UniformType::Int1),
        ],
        textures: vec!["BaseTexture".into()],
        ..Default::default()
    }).expect("HDR layer material"));
}

const HDR_VERTEX: &str = r#"#version 450 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texcoord;
layout(set = 0, binding = 0) uniform Uniforms { mat4 Projection; mat4 Model; vec4 _Time; int mode; float peak; float knee; float power; float shadowGain; float chromaScale; int softClip; int inputLinear; int compareBase; };
layout(location = 0) out vec2 uv;
void main() { gl_Position = Projection * Model * vec4(position, 1.0); uv = texcoord; }
"#;

const HDR_FRAGMENT: &str = r#"#version 450 core
layout(location = 0) in vec2 uv;
layout(set = 0, binding = 0) uniform Uniforms { mat4 Projection; mat4 Model; vec4 _Time; int mode; float peak; float knee; float power; float shadowGain; float chromaScale; int softClip; int inputLinear; int compareBase; };
layout(set = 0, binding = 1) uniform sampler2D Texture;
layout(set = 0, binding = 3) uniform sampler2D BaseTexture;
layout(location = 0) out vec4 fragColor;
float eotf(float c) { return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4); }
float mapChannel(float c) {
  if (mode == 0) return max(c, 0.0);
  float x = max(c, 0.0);
  if (mode == 1) return x * peak;
  x *= 1.0 + (shadowGain - 1.0) * (1.0 - clamp(x, 0.0, 1.0));
  if (x > knee) {
    float t = clamp((x - knee) / max(1.0 - knee, 0.00001), 0.0, 1.0);
    x = knee + (1.0 - pow(1.0 - t, power)) * max(peak - knee, 0.0);
  }
  return min(x, peak);
}
vec3 clipGamut(vec3 rgb, float luma) {
  if (softClip == 0) return clamp(rgb, 0.0, peak);
  vec3 chroma = rgb - vec3(luma);
  float upper = max(max(chroma.r, chroma.g), chroma.b);
  float lower = min(min(chroma.r, chroma.g), chroma.b);
  float utilization = max(upper / max(peak - luma, 0.00001), -lower / max(luma, 0.00001));
  float t = max((utilization - 0.8) / 0.2, 0.0);
  float scale = utilization > 0.8 ? (0.8 + 0.2 * (1.0 - exp(-t))) / utilization : 1.0;
  return clamp(vec3(luma) + chroma * scale, 0.0, peak);
}
void main() {
  vec4 src = texture(Texture, uv);
  if (compareBase == 1 && all(lessThanEqual(abs(src - texture(BaseTexture, uv)), vec4(0.0021)))) {
    fragColor = vec4(0.0);
    return;
  }
  vec3 straight = src.a > 0.0 ? src.rgb / src.a : vec3(0.0);
  vec3 rgb = inputLinear == 1 ? straight : vec3(eotf(straight.r), eotf(straight.g), eotf(straight.b));
  rgb = vec3(mapChannel(rgb.r), mapChannel(rgb.g), mapChannel(rgb.b));
  if (inputLinear == 0) rgb = mat3(0.6274,0.0691,0.0164, 0.3293,0.9195,0.0880, 0.0433,0.0114,0.8956) * rgb;
  float y = dot(rgb, vec3(0.2627, 0.6780, 0.0593));
  rgb = vec3(y) + (rgb - vec3(y)) * (mode == 2 ? max(chromaScale, 0.0) : 1.0);
  if (mode == 2) rgb = clipGamut(rgb, y);
  fragColor = vec4(rgb, src.a);
}
"#;

pub struct MSRenderTarget {
    dim: (u32, u32),
    /// Multisampled pass resolving into `out[0]`; absent when the backend
    /// cannot multisample, in which case drawing goes to `out[0]` directly.
    inp: Option<RenderTarget>,
    out: [RenderTarget; 2],
}

pub(super) fn new_rt(w: u32, h: u32, format: TextureFormat) -> RenderTarget {
    let gl = unsafe { get_internal_gl() };
    let t = Texture::new_render_texture(
        gl.quad_context,
        TextureParams {
            width: w,
            height: h,
            format,
            ..Default::default()
        },
    );
    let p = RenderPass::new(gl.quad_context, t, None);
    RenderTarget {
        texture: Texture2D::from_miniquad_texture(t),
        render_pass: p,
    }
}

impl MSRenderTarget {
    pub fn new(dim: (u32, u32), samples: u32) -> Self {
        Self::new_with_format(dim, samples, TextureFormat::RGB8)
    }

    pub fn new_rgba(dim: (u32, u32), samples: u32) -> Self {
        Self::new_with_format(dim, samples, TextureFormat::RGBA8)
    }

    /// Linear-light RGBA16F target. Values are normalized to 100 nits, so 1.0
    /// represents SDR reference white and values above 1.0 preserve HDR headroom.
    pub fn new_hdr(dim: (u32, u32), samples: u32) -> Self {
        Self::new_with_format(dim, samples, TextureFormat::RGBA16F)
    }

    fn new_with_format(dim: (u32, u32), samples: u32, format: TextureFormat) -> Self {
        let gl = unsafe { get_internal_gl() };
        let o0 = new_rt(dim.0, dim.1, format);
        let o1 = new_rt(dim.0, dim.1, format);

        let inp = if samples > 1 {
            let handle = o0.texture.raw_miniquad_texture_handle();
            let pass = RenderPass::new_multisampled(gl.quad_context, handle, samples);
            Some(RenderTarget {
                texture: o0.texture,
                render_pass: pass,
            })
        } else {
            None
        };

        Self { dim, inp, out: [o0, o1] }
    }

    pub fn swap(&mut self) {
        self.out.swap(0, 1);
    }

    /// Start a new scene frame on the texture the multisampled pass resolves
    /// into. An odd number of post effects leaves the ping-pong output on the
    /// other texture; carrying that index into the next frame samples stale
    /// output instead of the freshly resolved scene.
    pub fn begin_frame(&mut self) {
        if let Some(input) = self.inp {
            if self.out[0].texture != input.texture {
                self.swap();
            }
        }
    }

    pub fn input(&self) -> RenderTarget {
        self.inp.unwrap_or(self.out[0])
    }

    pub fn output(&self) -> RenderTarget {
        self.out[0]
    }

    pub fn old(&self) -> RenderTarget {
        self.out[1]
    }

    pub fn dimensions(&self) -> (u32, u32) {
        self.dim
    }

    /// Convert this layer from sRGB to linear BT.2020 and apply its settings.
    /// The mapped result remains in this target's ping-pong pair.
    pub fn map_hdr(&mut self, settings: &HdrLayerSettings) {
        self.map_hdr_with_input(settings, false, None);
    }

    pub fn map_hdr_linear_bt2020(&mut self, settings: &HdrLayerSettings) {
        self.map_hdr_with_input(settings, true, None);
    }

    pub fn map_hdr_changed_pixels(&mut self, settings: &HdrLayerSettings, base: Texture2D) {
        self.map_hdr_with_input(settings, true, Some(base));
    }

    fn map_hdr_with_input(&mut self, settings: &HdrLayerSettings, input_linear: bool, base: Option<Texture2D>) {
        self.swap();
        let source = self.old().texture;
        let destination = self.output();
        let material = HDR_MATERIAL.with(|slot| **slot);
        unsafe { get_internal_gl() }.flush();
        let mode = match settings.mode.as_str() {
            "mapped" => 1,
            "custom" => 2,
            _ => 0,
        };
        material.set_uniform("mode", mode);
        material.set_uniform("peak", (settings.peak_nits.max(100.0) / 100.0).max(1.0));
        material.set_uniform("knee", settings.knee.clamp(0.01, 0.99));
        material.set_uniform("power", settings.power.clamp(0.1, 8.0));
        material.set_uniform("shadowGain", settings.shadow_gain.max(0.0));
        material.set_uniform("chromaScale", settings.chroma_scale.max(0.0));
        material.set_uniform("softClip", if settings.gamut_clip == "soft" { 1 } else { 0 });
        material.set_uniform("inputLinear", if input_linear { 1 } else { 0 });
        material.set_uniform("compareBase", i32::from(base.is_some()));
        material.set_texture("BaseTexture", base.unwrap_or(source));
        set_camera(&Camera2D {
            zoom: vec2(1.0, 1.0),
            render_target: Some(destination),
            ..Default::default()
        });
        gl_use_material(material);
        draw_texture_ex(
            source,
            -1.0,
            -1.0,
            WHITE,
            DrawTextureParams {
                dest_size: Some(vec2(2.0, 2.0)),
                ..Default::default()
            },
        );
        gl_use_default_material();
        unsafe { get_internal_gl() }.flush();
    }

    pub fn read_pixels_resized(&self, target_w: usize, target_h: usize) -> Vec<f32> {
        let cycle = C.load(Ordering::Relaxed);
        let len = target_w * target_h * 3;

        RB.with(|slot| {
            let mut st = slot.borrow_mut();
            if let Some((cc, cl, d)) = st.cache.as_ref() {
                if *cc == cycle && *cl == len {
                    return d.clone();
                }
            }
            let fresh = self.raw(&mut st, target_w, target_h);
            st.cache = Some((cycle, len, fresh.clone()));
            fresh
        })
    }

    fn raw(&self, st: &mut Rb, tw: usize, th: usize) -> Vec<f32> {
        let gl = unsafe { get_internal_gl() };
        let rgba = tw * th * 4;

        let size_matches = st
            .target
            .as_ref()
            .map_or(false, |target| (target.texture.width(), target.texture.height()) == (tw as f32, th as f32));
        if !size_matches {
            if let Some(target) = st.target.take() {
                target.delete();
            }
            let texture = Texture::new_render_texture(
                gl.quad_context,
                TextureParams {
                    width: tw as u32,
                    height: th as u32,
                    format: TextureFormat::RGBA8,
                    ..Default::default()
                },
            );
            let render_pass = RenderPass::new(gl.quad_context, texture, None);
            st.target = Some(RenderTarget {
                texture: Texture2D::from_miniquad_texture(texture),
                render_pass,
            });
        }

        let target = st.target.as_ref().unwrap();
        gl.quad_context.copy_render_target(self.output().render_pass, target.render_pass);

        st.staging.clear();
        st.staging.resize(rgba, 0);
        target.texture.raw_miniquad_texture_handle().read_pixels(gl.quad_context, &mut st.staging);

        st.staging
            .chunks_exact(4)
            .flat_map(|pixel| [pixel[0] as f32 / 255.0, pixel[1] as f32 / 255.0, pixel[2] as f32 / 255.0])
            .collect()
    }
}

impl Drop for MSRenderTarget {
    fn drop(&mut self) {
        self.out[0].delete();
        self.out[1].delete();
    }
}
