//! RPE Next noise-skin v1 resource runner. Skins remain chart assets: this module
//! loads their pass graph, GLSL and textures without bundling a named skin.
use super::{BlockArea, BlockPhase, Vector};
use crate::fs::FileSystem;
use anyhow::{bail, Context, Result};
use macroquad::{
    miniquad::{PipelineParams, TextureFormat, TextureWrap, UniformType},
    prelude::*,
};
use once_cell::sync::Lazy;
use regex::Regex;
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;

fn one() -> f32 {
    1.
}

#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TargetSpec {
    #[serde(default = "one")]
    scale_x: f32,
    #[serde(default = "one")]
    scale_y: f32,
    #[serde(default)]
    format: String,
    #[serde(default)]
    filter: String,
    #[serde(default)]
    no_clear: bool,
}
impl Default for TargetSpec {
    fn default() -> Self {
        Self {
            scale_x: 1.,
            scale_y: 1.,
            format: String::new(),
            filter: String::new(),
            no_clear: false,
        }
    }
}

#[derive(Deserialize)]
struct PassSpec {
    shader: String,
    output: String,
    #[serde(default)]
    inputs: HashMap<String, String>,
    #[serde(default)]
    constants: HashMap<String, Value>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Graph {
    #[serde(default)]
    targets: HashMap<String, TargetSpec>,
    passes: Vec<PassSpec>,
    #[serde(default)]
    block_mask: TargetSpec,
    #[serde(default)]
    padding: Option<Value>,
}

#[derive(Deserialize)]
struct Config {
    version: u32,
    #[serde(flatten)]
    normal: Graph,
    #[serde(rename = "lowPerformance")]
    low: Option<Graph>,
}

struct Pass {
    material: Material,
    uniforms: Vec<(String, UniformType)>,
    samplers: Vec<String>,
    images: HashMap<String, Texture2D>,
    spec: PassSpec,
}
impl Drop for Pass {
    fn drop(&mut self) {
        self.material.delete();
        for image in self.images.values() {
            image.delete();
        }
    }
}

struct Frame {
    target: RenderTarget,
    logical: Vec2,
    origin: Vec2,
}
impl Frame {
    fn new(width: usize, height: usize, spec: &TargetSpec, padding: Vec2, hdr: bool) -> Self {
        let logical = vec2((width as f32 * spec.scale_x).floor().max(1.), (height as f32 * spec.scale_y).floor().max(1.));
        let origin = vec2((padding.x * logical.x / width as f32).ceil(), (padding.y * logical.y / height as f32).ceil());
        let width = (logical.x + 2. * origin.x) as u32;
        let height = (logical.y + 2. * origin.y) as u32;
        let target = if hdr {
            super::render::new_rt(width, height, TextureFormat::RGBA16F)
        } else {
            render_target(width, height)
        };
        target.texture.set_filter(if spec.filter == "linear" {
            FilterMode::Linear
        } else {
            FilterMode::Nearest
        });
        Self { target, logical, origin }
    }
    fn rect(&self) -> [f32; 4] {
        let width = self.target.texture.width();
        let height = self.target.texture.height();
        [
            self.origin.x / width,
            self.origin.y / height,
            self.logical.x / width,
            self.logical.y / height,
        ]
    }
}
impl Drop for Frame {
    fn drop(&mut self) {
        self.target.delete();
    }
}

struct Pair {
    front: Frame,
    back: Frame,
    history: Option<Frame>,
}
struct Instance {
    width: usize,
    height: usize,
    hdr: bool,
    screens: Pair,
    targets: HashMap<String, Pair>,
    mask: Frame,
    mask_image: Image,
    previous_mask: Vec<u8>,
    time: Option<f64>,
}

pub struct NoiseSkin {
    graph: Option<Graph>,
    passes: Vec<Pass>,
    parameters: HashMap<String, Value>,
    copy: Option<Material>,
    instance: Option<Instance>,
}
impl Drop for NoiseSkin {
    fn drop(&mut self) {
        if let Some(mut copy) = self.copy.take() {
            copy.delete();
        }
    }
}

fn asset_path(root: &str, path: &str) -> Result<String> {
    for part in [root, path] {
        if part.is_empty()
            || part.starts_with('/')
            || part.contains(['\\', ':', '\0'])
            || part.split('/').any(|p| p.is_empty() || p == "." || p == "..")
        {
            bail!("Invalid noise-skin asset path");
        }
    }
    Ok(format!("{root}/{path}"))
}

static UNIFORM: Lazy<Regex> = Lazy::new(|| Regex::new(r"\buniform\s+(?:(?:highp|mediump|lowp)\s+)?(\w+)\s+(\w+)\s*(\[\s*\d+\s*\])?\s*;").unwrap());
static DEFAULT_VALUE: Lazy<Regex> = Lazy::new(|| Regex::new(r"uniform\s+(float|vec[234])\s+(\w+)\s*;[^\r\n]*?//\s*%([^%\r\n]+)%").unwrap());

fn uniform_type(name: &str) -> Result<UniformType> {
    Ok(match name {
        "float" => UniformType::Float1,
        "vec2" => UniformType::Float2,
        "vec3" => UniformType::Float3,
        "vec4" => UniformType::Float4,
        "int" | "bool" => UniformType::Int1,
        "ivec2" => UniformType::Int2,
        "ivec3" => UniformType::Int3,
        "ivec4" => UniformType::Int4,
        "mat4" => UniformType::Mat4,
        _ => bail!("Unsupported noise-skin uniform type {name}"),
    })
}

/// GLSL100 resources are normalized once, then share macroquad's portable
/// GLSL450 material path on OpenGL, Vulkan and Metal.
fn shaders(source: &str, component: Option<char>, format: &str) -> Result<(String, String, Vec<(String, UniformType)>, Vec<String>)> {
    let mut uniforms = vec![("tkOrigin".into(), UniformType::Float2), ("tkLogical".into(), UniformType::Float2)];
    let mut samplers = Vec::new();
    let mut arrays = String::new();
    for caps in UNIFORM.captures_iter(source) {
        let ty = &caps[1];
        let name = &caps[2];
        if let Some(array) = caps.get(3) {
            if name != "touchPoints" || ty != "vec4" {
                bail!("Unsupported noise-skin uniform array {name}");
            }
            let count: usize = array.as_str().trim_matches(['[', ']', ' ']).parse()?;
            if count > 1024 {
                bail!("Noise-skin touch array too large");
            }
            arrays.push_str(&format!("const vec4 {name}[{count}]=vec4[{count}]({});\n", vec!["vec4(0.0)"; count].join(",")));
        } else if ty == "sampler2D" {
            samplers.push(name.to_owned());
            uniforms.push((format!("tkRect_{name}"), UniformType::Float4));
        } else {
            uniforms.push((name.to_owned(), uniform_type(ty)?));
        }
    }
    let block = super::effect::uniform_block(&uniforms);
    let vertex = format!("#version 450 core\nlayout(location=0) in vec3 position;\nlayout(location=1) in vec2 texcoord;\n{block}void main(){{gl_Position=Projection*Model*vec4(position,1.0);}}\n");
    let mut body = UNIFORM.replace_all(source, "").into_owned();
    static VERSION: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?m)^\s*#version[^\n]*\n?").unwrap());
    static PRECISION: Lazy<Regex> = Lazy::new(|| Regex::new(r"\bprecision\s+\w+\s+\w+\s*;|\b(?:lowp|mediump|highp)\s+").unwrap());
    static VARYING: Lazy<Regex> = Lazy::new(|| Regex::new(r"\bvarying\s+vec2\s+uv\s*;").unwrap());
    static MAIN: Lazy<Regex> = Lazy::new(|| Regex::new(r"\bvoid\s+main\s*\(\s*(?:void\s*)?\)").unwrap());
    body = VERSION.replace_all(&body, "").into_owned();
    body = PRECISION.replace_all(&body, "").into_owned();
    body = VARYING.replace_all(&body, "").into_owned();
    body = MAIN.replace_all(&body, "void tkMain()").into_owned();
    body = body.replace("gl_FragColor", "fragColor");
    let mut declarations = String::new();
    for (i, name) in samplers.iter().enumerate() {
        declarations.push_str(&format!("layout(set=0,binding={}) uniform sampler2D {name};\nvec4 tkSample_{name}(vec2 p){{return texture({name},tkRect_{name}.xy+p*tkRect_{name}.zw);}}\n", i + 3));
        let sample = Regex::new(&format!(r"\b(?:texture2D|texture)\s*\(\s*{}\s*,", regex::escape(name)))?;
        body = sample.replace_all(&body, format!("tkSample_{name}(")).into_owned();
    }
    // Emulate absent channels of R8/RG8 with RGBA8 targets supported by every
    // backend, while retaining the original eight-bit quantization.
    let format_fix = if component.is_none() && format == "r8" {
        "fragColor.gba=vec3(0.0,0.0,1.0);"
    } else if component.is_none() && format == "rg8" {
        "fragColor.ba=vec2(0.0,1.0);"
    } else {
        ""
    };
    let fragment = format!("#version 450 core\nlayout(location=0) out vec4 fragColor;\n{block}{declarations}{arrays}vec2 uv;\n{body}\nvoid main(){{uv=(gl_FragCoord.xy-tkOrigin)/tkLogical;tkMain();{format_fix}}}\n");
    Ok((vertex, fragment, uniforms, samplers))
}

const COPY: &str = "varying vec2 uv; uniform sampler2D screenTexture; void main(){gl_FragColor=texture2D(screenTexture,uv);}";
fn material(source: &str, component: Option<char>, format: &str) -> Result<(Material, Vec<(String, UniformType)>, Vec<String>)> {
    let (vertex, fragment, uniforms, samplers) = shaders(source, component, format)?;
    let color_write = component.map_or((true, true, true, true), |c| (c == 'r', c == 'g', c == 'b', c == 'a'));
    let m = load_material(
        &vertex,
        &fragment,
        MaterialParams {
            uniforms: uniforms.clone(),
            textures: samplers.clone(),
            pipeline_params: PipelineParams {
                color_write,
                ..Default::default()
            },
        },
    )?;
    Ok((m, uniforms, samplers))
}

impl NoiseSkin {
    pub async fn load(root: &str, channel: &str, mut parameters: HashMap<String, Value>, fs: &mut dyn FileSystem) -> Result<Self> {
        if root.is_empty() {
            return Ok(Self {
                graph: None,
                passes: vec![],
                parameters,
                copy: None,
                instance: None,
            });
        }
        let config_path = asset_path(root, "config.json")?;
        let config: Config = if fs.exists(&config_path).await? {
            serde_json::from_slice(&fs.load_file(&config_path).await?)?
        } else {
            Config {
                version: 1,
                normal: Graph {
                    targets: HashMap::new(),
                    passes: vec![PassSpec {
                        shader: "main.glsl".into(),
                        output: "$screen".into(),
                        inputs: HashMap::new(),
                        constants: HashMap::new(),
                    }],
                    block_mask: TargetSpec::default(),
                    padding: None,
                },
                low: None,
            }
        };
        if config.version != 1 {
            bail!("Unsupported noise-skin version {}", config.version);
        }
        let main = String::from_utf8(fs.load_file(&asset_path(root, "main.glsl")?).await?)?;
        for caps in DEFAULT_VALUE.captures_iter(&main) {
            let values = caps[3]
                .split(',')
                .map(|v| v.trim().parse::<f32>().map(|v| Value::from(v)))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let value = if values.len() == 1 { values[0].clone() } else { Value::Array(values) };
            parameters.entry(caps[2].to_owned()).or_insert(value);
        }
        let graph = if channel == "low" {
            config.low.unwrap_or(config.normal)
        } else {
            config.normal
        };
        let mut skin = Self {
            graph: None,
            passes: vec![],
            parameters,
            copy: Some(material(COPY, None, "rgba8")?.0),
            instance: None,
        };
        for spec in graph.targets.values().chain(std::iter::once(&graph.block_mask)) {
            if !spec.scale_x.is_finite() || !spec.scale_y.is_finite() || spec.scale_x <= 0. || spec.scale_y <= 0. {
                bail!("Invalid noise-skin target scale");
            }
            if !["", "r8", "rg8", "rgba8"].contains(&spec.format.as_str()) {
                bail!("Unsupported noise-skin target format");
            }
        }
        for spec in graph.passes.iter() {
            let (output, component) = split_output(&spec.output)?;
            let format = if output == "$screen" {
                "rgba8"
            } else {
                &graph.targets.get(output).context("Missing noise-skin output target")?.format
            };
            let source = String::from_utf8(fs.load_file(&asset_path(root, &spec.shader)?).await?)?;
            let (material, uniforms, samplers) =
                material(&source, component, format).with_context(|| format!("Noise-skin shader {}", spec.shader))?;
            let mut pass = Pass {
                material,
                uniforms,
                samplers,
                images: HashMap::new(),
                spec: PassSpec {
                    shader: spec.shader.clone(),
                    output: spec.output.clone(),
                    inputs: spec.inputs.clone(),
                    constants: spec.constants.clone(),
                },
            };
            static TEXTURE: Lazy<Regex> = Lazy::new(|| Regex::new(r"uniform\s+sampler2D\s+(\w+)\s*;\s*//\s*@texture\s+(\S+)([^\r\n]*)").unwrap());
            for caps in TEXTURE.captures_iter(&source) {
                let directory = spec.shader.rsplit_once('/').map_or("", |(d, _)| d);
                let relative = if directory.is_empty() {
                    caps[2].to_owned()
                } else {
                    format!("{directory}/{}", &caps[2])
                };
                let bytes = fs.load_file(&asset_path(root, &relative)?).await?;
                let mut image = image::load_from_memory(&bytes)?.to_rgba8();
                // RPE Next uploads skin image bitmaps with imageOrientation=flipY.
                image::imageops::flip_vertical_in_place(&mut image);
                let options = &caps[3];
                for p in image.pixels_mut() {
                    if options.contains("colorspace=srgb") {
                        for c in &mut p.0[..3] {
                            let v = *c as f32 / 255.;
                            *c = ((if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }) * 255.).round() as u8;
                        }
                    }
                    if options.contains("alphaMode=premult") {
                        for i in 0..3 {
                            p.0[i] = (p.0[i] as f32 * p.0[3] as f32 / 255.).round() as u8;
                        }
                    }
                }
                let texture = Texture2D::from_rgba8(image.width() as u16, image.height() as u16, image.as_raw());
                texture.set_filter(if options.contains("filter=nearest") {
                    FilterMode::Nearest
                } else {
                    FilterMode::Linear
                });
                let wrap = if options.contains("wrap=mirror") {
                    TextureWrap::Mirror
                } else if options.contains("wrap=repeat") {
                    TextureWrap::Repeat
                } else {
                    TextureWrap::Clamp
                };
                texture
                    .raw_miniquad_texture_handle()
                    .set_wrap(unsafe { get_internal_gl() }.quad_context, wrap);
                pass.images.insert(caps[1].to_owned(), texture);
            }
            for input in pass.spec.inputs.values() {
                if input != "$screen" && !graph.targets.contains_key(input) {
                    bail!("Missing noise-skin input target {input}");
                }
            }
            for name in &pass.samplers {
                if name != "screenTexture" && name != "blockMask" && !pass.images.contains_key(name) && !pass.spec.inputs.contains_key(name) {
                    bail!("Missing noise-skin sampler input {name}");
                }
            }
            skin.passes.push(pass);
        }
        skin.graph = Some(graph);
        Ok(skin)
    }

    pub fn draw(
        &mut self,
        areas: &[BlockArea],
        layer: usize,
        scene: Texture2D,
        time: f64,
        aspect: f32,
        dst: (f32, f32, f32, f32),
    ) -> Option<Texture2D> {
        if std::env::var_os("PRPR_BLOCK_OFF").is_some() {
            return None;
        }
        let visible: Vec<_> = areas
            .iter()
            .filter(|a| a.animation.as_ref().is_some_and(|n| n.layer == layer) && a.phase(time) != BlockPhase::Hidden)
            .collect();
        if visible.is_empty() {
            return None;
        }
        let Some(graph) = &self.graph else {
            draw_flat(&visible, time, aspect, dst);
            return None;
        };
        let width = scene.width() as usize;
        let height = scene.height() as usize;
        let hdr = scene.raw_miniquad_texture_handle().format == TextureFormat::RGBA16F;
        if self.instance.as_ref().is_none_or(|i| i.width != width || i.height != height || i.hdr != hdr) {
            let padding = padding(graph.padding.as_ref(), width, height);
            let screen_padding = if graph.padding.as_ref().and_then(|p| p.get("screen")).and_then(Value::as_bool) == Some(false) {
                Vec2::ZERO
            } else {
                padding
            };
            let screens = Pair {
                front: Frame::new(
                    width,
                    height,
                    &TargetSpec {
                        filter: "linear".into(),
                        ..Default::default()
                    },
                    screen_padding,
                    hdr,
                ),
                back: Frame::new(
                    width,
                    height,
                    &TargetSpec {
                        filter: "linear".into(),
                        ..Default::default()
                    },
                    screen_padding,
                    hdr,
                ),
                history: None,
            };
            let targets = graph
                .targets
                .iter()
                .map(|(name, spec)| {
                    (
                        name.clone(),
                        Pair {
                            front: Frame::new(width, height, spec, padding, false),
                            back: Frame::new(width, height, spec, padding, false),
                            history: spec.no_clear.then(|| Frame::new(width, height, spec, padding, false)),
                        },
                    )
                })
                .collect();
            let mask = Frame::new(width, height, &graph.block_mask, padding, false);
            let mw = mask.target.texture.width() as u16;
            let mh = mask.target.texture.height() as u16;
            self.instance = Some(Instance {
                width,
                height,
                hdr,
                screens,
                targets,
                mask,
                mask_image: Image {
                    width: mw,
                    height: mh,
                    bytes: vec![0; mw as usize * mh as usize * 4],
                },
                previous_mask: vec![],
                time: None,
            });
        }
        push_camera_state();
        let copy = self.copy.unwrap();
        let instance = self.instance.as_mut().unwrap();
        blit(copy, scene, [0., 0., 1., 1.], &instance.screens.front);
        raster_mask(&visible, time, aspect, instance);
        if instance.previous_mask != instance.mask_image.bytes {
            instance.mask.target.texture.update(&instance.mask_image);
            instance.previous_mask.clone_from(&instance.mask_image.bytes);
        }
        for (name, pair) in &mut instance.targets {
            if !graph.targets[name].no_clear || instance.time.is_none_or(|t| time < t || time - t > 1.) {
                clear_frame(&pair.front);
            }
            if let Some(history) = &pair.history {
                if instance.time == Some(time) {
                    blit(copy, history.target.texture, history.rect(), &pair.front);
                } else {
                    blit(copy, pair.front.target.texture, pair.front.rect(), history);
                }
            }
        }
        for pass in &self.passes {
            let (output, component) = split_output(&pass.spec.output).unwrap();
            let frame = if output == "$screen" {
                &instance.screens.back
            } else {
                &instance.targets[output].back
            };
            if component.is_some() {
                let front = &instance.targets[output].front;
                blit(copy, front.target.texture, front.rect(), frame);
            }
            begin_frame(frame);
            let m = pass.material;
            m.set_uniform("tkOrigin", frame.origin);
            m.set_uniform("tkLogical", frame.logical);
            for (name, ty) in &pass.uniforms {
                match name.as_str() {
                    "screenSize" => m.set_uniform(name, vec2(width as f32, height as f32)),
                    "targetSize" => m.set_uniform(name, frame.logical),
                    "targetTexelSize" => m.set_uniform(name, Vec2::ONE / frame.logical),
                    "time" => m.set_uniform(name, time as f32),
                    "touchCount" => m.set_uniform(name, 0i32),
                    _ => {
                        if let Some(value) = pass.spec.constants.get(name).or_else(|| self.parameters.get(name)) {
                            set_value(m, name, *ty, value);
                        }
                    }
                }
            }
            for name in &pass.samplers {
                let (texture, rect, logical) = if let Some(image) = pass.images.get(name) {
                    (*image, [0., 0., 1., 1.], vec2(image.width(), image.height()))
                } else if name == "blockMask" {
                    (instance.mask.target.texture, instance.mask.rect(), instance.mask.logical)
                } else if let Some(input) = pass.spec.inputs.get(name).filter(|i| i.as_str() != "$screen") {
                    let t = &instance.targets[input].front;
                    (t.target.texture, t.rect(), t.logical)
                } else {
                    let t = &instance.screens.front;
                    (t.target.texture, t.rect(), t.logical)
                };
                m.set_texture(name, texture);
                m.set_uniform(&format!("tkRect_{name}"), rect);
                let size_name = format!("{name}Size");
                if pass.uniforms.iter().any(|(n, _)| n == &size_name) {
                    m.set_uniform(&size_name, logical);
                }
            }
            gl_use_material(m);
            draw_rectangle(-1., -1., 2., 2., WHITE);
            gl_use_default_material();
            if output == "$screen" {
                std::mem::swap(&mut instance.screens.front, &mut instance.screens.back);
            } else {
                let pair = instance.targets.get_mut(output).unwrap();
                std::mem::swap(&mut pair.front, &mut pair.back);
            }
        }
        instance.time = Some(time);
        pop_camera_state();
        // Draw the complete result over the already-composited chart, using the
        // same viewport and destination as the chart blit.
        // Final presentation uses interpolated UV, as gl_FragCoord is in the
        // enclosing viewport rather than this texture's local pixel coordinates.
        draw_texture_ex(
            instance.screens.front.target.texture,
            dst.0,
            dst.1,
            WHITE,
            DrawTextureParams {
                dest_size: Some(vec2(dst.2, dst.3)),
                source: Some(Rect::new(instance.screens.front.origin.x, instance.screens.front.origin.y, width as f32, height as f32)),
                ..Default::default()
            },
        );
        Some(instance.screens.front.target.texture)
    }
}

fn padding(value: Option<&Value>, width: usize, height: usize) -> Vec2 {
    match value {
        Some(Value::Number(n)) => Vec2::splat(n.as_f64().unwrap_or(32.) as f32),
        Some(v) => {
            let pixels = v.get("pixels").and_then(Value::as_f64).unwrap_or(0.) as f32;
            vec2(
                (pixels + width as f32 * v.get("screenWidth").and_then(Value::as_f64).unwrap_or(0.) as f32).ceil(),
                (pixels + height as f32 * v.get("screenHeight").and_then(Value::as_f64).unwrap_or(0.) as f32).ceil(),
            )
        }
        None => Vec2::splat(32.),
    }
}
fn split_output(output: &str) -> Result<(&str, Option<char>)> {
    match output.split_once('.') {
        Some((name, component)) if component.len() == 1 && "rgba".contains(component) && name != "$screen" => Ok((name, component.chars().next())),
        Some(_) => bail!("Invalid noise-skin output component"),
        None => Ok((output, None)),
    }
}
fn begin_frame(frame: &Frame) {
    set_camera(&Camera2D {
        render_target: Some(frame.target),
        zoom: vec2(1., 1.),
        viewport: None,
        ..Default::default()
    });
}
fn clear_frame(frame: &Frame) {
    begin_frame(frame);
    clear_background(Color::new(0., 0., 0., 1.));
}
fn blit(copy: Material, source: Texture2D, rect: [f32; 4], target: &Frame) {
    begin_frame(target);
    copy.set_texture("screenTexture", source);
    copy.set_uniform("tkRect_screenTexture", rect);
    copy.set_uniform("tkOrigin", target.origin);
    copy.set_uniform("tkLogical", target.logical);
    gl_use_material(copy);
    draw_rectangle(-1., -1., 2., 2., WHITE);
    gl_use_default_material();
}
fn set_value(material: Material, name: &str, ty: UniformType, value: &Value) {
    if let Some(value) = value.as_f64() {
        if matches!(ty, UniformType::Int1) {
            material.set_uniform(name, value as i32);
        } else {
            material.set_uniform(name, value as f32);
        }
    } else if let Some(values) = value.as_array() {
        let v: Vec<f32> = values.iter().filter_map(|v| v.as_f64().map(|v| v as f32)).collect();
        match v.len() {
            2 => material.set_uniform(name, [v[0], v[1]]),
            3 => material.set_uniform(name, [v[0], v[1], v[2]]),
            4 => material.set_uniform(name, [v[0], v[1], v[2], v[3]]),
            _ => {}
        }
    }
}

fn raster_mask(areas: &[&BlockArea], time: f64, aspect: f32, instance: &mut Instance) {
    let transforms: Vec<_> = areas
        .iter()
        .map(|a| {
            let tr = a.transform(time, aspect);
            let angle = -tr.rotation.to_radians();
            (tr, angle.cos(), angle.sin(), a.is_subtract, a.is_active(time), a.animation.as_ref().unwrap().is_ready(time))
        })
        .collect();
    let mw = instance.mask_image.width as usize;
    let mh = instance.mask_image.height as usize;
    for y in 0..mh {
        for x in 0..mw {
            // Downsample the packed full-resolution geometry at texel centres.
            let u = (x as f32 + 0.5 - instance.mask.origin.x) / instance.mask.logical.x;
            let v = (y as f32 + 0.5 - instance.mask.origin.y) / instance.mask.logical.y;
            let u = ((u * instance.width as f32).floor() + 0.5) / instance.width as f32;
            let v = ((v * instance.height as f32).floor() + 0.5) / instance.height as f32;
            let p = Vector::new(u * 2. - 1., (v * 2. - 1.) / aspect);
            let mut counts = [0u32; 6];
            for (tr, cos, sin, inverted, active, ready) in &transforms {
                let d = p - tr.center;
                let local = Vector::new(cos * d.x - sin * d.y, sin * d.x + cos * d.y);
                if local.x.abs() <= tr.size.x * 0.5 && local.y.abs() <= tr.size.y * 0.5 {
                    let bit = if *active { 0 } else { 2 } + usize::from(*inverted);
                    counts[bit] += 1;
                    if *ready {
                        counts[4 + usize::from(*inverted)] += 1;
                    }
                }
            }
            let word = pack_masks(counts);
            let p = (y * mw + x) * 4;
            instance.mask_image.bytes[p..p + 4].copy_from_slice(&[word, 0, 0, 255]);
        }
    }
}

fn pack_masks(counts: [u32; 6]) -> u8 {
    counts.iter().enumerate().fold(0u8, |m, (bit, count)| {
        if if bit % 2 == 0 { *count > 0 } else { *count == 1 } {
            m | (1 << bit)
        } else {
            m
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noise_skin_shader_preserves_sampler_order_uniform_layout_and_padding() {
        let source = "#version 100\nprecision highp float; varying highp vec2 uv; uniform sampler2D blockMask; uniform sampler2D image; uniform float gain; void main(){gl_FragColor=texture2D(image,uv)*texture2D(blockMask,uv).r*gain;}";
        let (vertex, fragment, uniforms, samplers) = shaders(source, None, "rg8").unwrap();
        assert_eq!(samplers, ["blockMask", "image"]);
        assert!(vertex.contains("mat4 Projection;"));
        assert_eq!(
            uniforms.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(),
            ["tkOrigin", "tkLogical", "tkRect_blockMask", "tkRect_image", "gain"]
        );
        assert!(fragment.contains("binding=3) uniform sampler2D blockMask;"));
        assert!(fragment.contains("binding=4) uniform sampler2D image;"));
        assert!(fragment.contains("tkSample_image(uv)"));
        assert!(fragment.contains("uv=(gl_FragCoord.xy-tkOrigin)/tkLogical"));
        assert!(fragment.contains("fragColor.ba=vec2(0.0,1.0)"));
        assert!(!fragment.contains("#version 100"));
    }

    #[test]
    fn noise_skin_component_pass_preserves_other_channels_and_no_touch_arrays() {
        let source="varying vec2 uv; uniform int touchCount; uniform vec4 touchPoints[32]; void main(){gl_FragColor=vec4(float(touchCount))+touchPoints[0];}";
        let (_, fragment, uniforms, _) = shaders(source, Some('r'), "rg8").unwrap();
        assert!(fragment.contains("const vec4 touchPoints[32]"));
        assert!(!uniforms.iter().any(|(n, _)| n == "touchPoints"));
        assert!(!fragment.contains("fragColor.ba="));
    }

    #[test]
    fn noise_skin_six_masks_distinguish_active_disabled_ready_and_single_subtract() {
        assert_eq!(pack_masks([1, 0, 0, 0, 0, 0]), 1);
        assert_eq!(pack_masks([10, 1, 2, 1, 1, 1]), 63);
        assert_eq!(pack_masks([0, 2, 1, 3, 0, 4]), 4);
        assert_eq!(pack_masks([0, 1, 0, 0, 0, 1]), 34);
    }

    #[test]
    fn noise_skin_schema_keeps_low_channel_history_and_padding_defaults() {
        let config:Config=serde_json::from_value(serde_json::json!({"version":1,"targets":{},"passes":[{"shader":"main.glsl","output":"$screen"}],"lowPerformance":{"targets":{"history":{"noClear":true,"scaleX":0.5}},"passes":[{"shader":"main.glsl","output":"history.r"}]}})).unwrap();
        let low = config.low.unwrap();
        assert!(low.targets["history"].no_clear);
        assert_eq!(low.targets["history"].scale_y, 1.);
        assert_eq!(low.block_mask.scale_x, 1.);
        assert_eq!(padding(None, 640, 360), vec2(32., 32.));
        assert_eq!(padding(Some(&serde_json::json!({"pixels":32,"screenWidth":0.071,"screenHeight":0.071})), 640, 360), vec2(78., 58.));
        assert_eq!(split_output("history.r").unwrap(), ("history", Some('r')));
        assert!(split_output("$screen.r").is_err());
    }

    #[test]
    fn noise_skin_asset_paths_stay_relative_and_missing_channels_are_not_silently_bound() {
        assert_eq!(asset_path("shader/Skin", "tex/noise.png").unwrap(), "shader/Skin/tex/noise.png");
        for (root, path) in [
            ("../Skin", "main.glsl"),
            ("shader/Skin", "../noise.png"),
            ("/Skin", "main.glsl"),
            ("shader/Skin", "/noise.png"),
        ] {
            assert!(asset_path(root, path).is_err());
        }
        assert!(shaders("uniform vec4 unsupported[2];void main(){gl_FragColor=unsupported[0];}", None, "rgba8").is_err());
    }
}

fn draw_flat(areas: &[&BlockArea], time: f64, aspect: f32, dst: (f32, f32, f32, f32)) {
    for area in areas {
        let tr = area.transform(time, aspect);
        let a = tr.rotation.to_radians();
        let points = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)].map(|(x, y)| {
            let p = Vector::new(x * tr.size.x, y * tr.size.y);
            let p = Vector::new(a.cos() * p.x - a.sin() * p.y, a.sin() * p.x + a.cos() * p.y) + tr.center;
            vec2(dst.0 + (p.x + 1.) * dst.2 * 0.5, dst.1 + (p.y * aspect + 1.) * dst.3 * 0.5)
        });
        let color = Color::new(182. / 255., 60. / 255., 60. / 255., if area.is_active(time) { 0.667 } else { 0.4 });
        draw_triangle(points[0], points[1], points[2], color);
        draw_triangle(points[0], points[2], points[3], color);
    }
}
