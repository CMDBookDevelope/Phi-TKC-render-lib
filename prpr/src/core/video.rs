use super::{Anim, Resource};
use crate::ext::{source_of_image, ScaleType};
use anyhow::{ensure, Context, Result};
use macroquad::prelude::*;
use macroquad::miniquad::{FilterMode, MipmapFilter, Texture, TextureAccess, TextureFormat, TextureParams, TextureWrap};
use prpr_avc::AVPixelFormat;
use std::collections::VecDeque;
use std::io::Write;
use std::path::PathBuf;
use tempfile::NamedTempFile;

#[derive(Debug, Clone)]
struct FrameData {
    frame_index: usize,
    timestamp: f64,
    buf_y: Vec<u8>,
    buf_u: Vec<u8>,
    buf_v: Vec<u8>,
}

pub struct Video {
    video: prpr_avc::Video,
    source: PathBuf,
    _video_file: Option<NamedTempFile>,

    material: Material,
    tex_y: Texture2D,
    tex_u: Texture2D,
    tex_v: Texture2D,

    start_time: f32,
    scale_type: ScaleType,
    alpha: Anim<f32>,
    dim: Anim<f32>,
    frame_delta: f64,

    // 帧缓存系统
    frame_cache: VecDeque<FrameData>,
    current_video_frame: usize,
    pub next_frame: usize,
    ended: bool,
    first_timestamp: Option<f64>,
    last_timestamp: Option<f64>,
    displayed_timestamp: Option<f64>,
    background_visible: bool,
    last_background_time: Option<f64>,
    last_cycle: u64,
    duration: Option<f64>,

    // 异步解码配置
    max_cache_size: usize,
    decode_ahead_frames: usize,

    width: usize,
    height: usize,

    // 缓冲区重用
    buf_y: Vec<u8>,
    buf_u: Vec<u8>,
    buf_v: Vec<u8>,
}

fn new_tex(w: u32, h: u32) -> Texture2D {
    Texture2D::from_miniquad_texture(Texture::new(
        unsafe { get_internal_gl() }.quad_context,
        TextureAccess::Static,
        None,
        TextureParams {
            width: w,
            height: h,
            format: TextureFormat::Alpha,
            filter: FilterMode::Linear,
            wrap: TextureWrap::Clamp,
            mipmap_filter: MipmapFilter::None,
        },
    ))
}

impl Video {
    pub fn new(
        data: Vec<u8>,
        start_time: f32,
        scale_type: ScaleType,
        alpha: Anim<f32>,
        dim: Anim<f32>,
    ) -> Result<Self> {
        let mut video_file = NamedTempFile::new()?;
        video_file.write_all(&data)?;
        drop(data);

        Self::open(video_file.path().to_path_buf(), Some(video_file), start_time, scale_type, alpha, dim)
    }

    pub(crate) fn open_background(path: PathBuf) -> Result<Self> {
        Self::open(path, None, 0., ScaleType::CropCenter, Anim::fixed(1.), Anim::fixed(0.))
    }

    fn open(
        source: PathBuf,
        video_file: Option<NamedTempFile>,
        start_time: f32,
        scale_type: ScaleType,
        alpha: Anim<f32>,
        dim: Anim<f32>,
    ) -> Result<Self> {
        let video = Self::open_decoder(&source)?;

        let frame_delta = video.frame_rate().to_f64_inv();
        let frame_delta = if frame_delta.is_finite() && frame_delta > 0. {
            frame_delta
        } else {
            1. / 30.
        };
        let duration = video.duration();
        let format = video.stream_format();
        ensure!(format.width > 0 && format.height > 0, "background video has invalid dimensions");
        let w = format.width as u32;
        let h = format.height as u32;

        let buf_size_y = w as usize * h as usize;
        let buf_size_uv = w.div_ceil(2) as usize * h.div_ceil(2) as usize;
        let buf_y = vec![0; buf_size_y];
        let buf_u = vec![0; buf_size_uv];
        let buf_v = vec![0; buf_size_uv];

        let material = load_material(
            shader::VERTEX,
            shader::FRAGMENT,
            MaterialParams {
                pipeline_params: PipelineParams::default(),
                uniforms: Vec::new(),
                textures: vec!["tex_y".to_owned(), "tex_u".to_owned(), "tex_v".to_owned()],
            },
        )?;
        let tex_y = new_tex(w, h);
        let tex_u = new_tex(w.div_ceil(2), h.div_ceil(2));
        let tex_v = new_tex(w.div_ceil(2), h.div_ceil(2));
        material.set_texture("tex_y", tex_y);
        material.set_texture("tex_u", tex_u);
        material.set_texture("tex_v", tex_v);

        Ok(Self {
            video,
            source,
            _video_file: video_file,
            material,
            tex_y,
            tex_u,
            tex_v,
            start_time,
            scale_type,
            alpha,
            dim,
            frame_delta,
            frame_cache: VecDeque::new(),
            current_video_frame: 0,
            next_frame: 0,
            ended: false,
            first_timestamp: None,
            last_timestamp: None,
            displayed_timestamp: None,
            background_visible: false,
            last_background_time: None,
            last_cycle: 0,
            duration,
            max_cache_size: 10, // 最多缓存10帧
            decode_ahead_frames: 3, // 提前解码3帧
            width: w as usize,
            height: h as usize,
            buf_y,
            buf_u,
            buf_v,
        })
    }

    fn open_decoder(source: &std::path::Path) -> Result<prpr_avc::Video> {
        let path = source.to_str().context("video path is not valid UTF-8")?;
        prpr_avc::Video::open(path, AVPixelFormat::YUV420P)
    }

    fn decode_frame_to_cache(&mut self) -> Result<bool> {
        if self.ended {
            return Ok(false);
        }

        let frame_data = self.video.with_timed_frame(|frame, timestamp| {
            let fallback = self.last_timestamp.map_or(0., |last| last + self.frame_delta);
            let timestamp = timestamp.map_or(fallback, |pts| {
                let first = self.first_timestamp.get_or_insert(pts - fallback);
                (pts - *first).max(self.last_timestamp.unwrap_or(0.))
            });
            self.last_timestamp = Some(timestamp);
            frame.copy_plane(0, self.width, self.height, &mut self.buf_y);
            for (plane, buffer) in [(1, &mut self.buf_u), (2, &mut self.buf_v)] {
                frame.copy_plane(plane, self.width.div_ceil(2), self.height.div_ceil(2), buffer);
            }

            FrameData {
                frame_index: self.current_video_frame,
                timestamp,
                buf_y: self.buf_y.clone(),
                buf_u: self.buf_u.clone(),
                buf_v: self.buf_v.clone(),
            }
        });

        if let Some(data) = frame_data {
            self.frame_cache.push_back(data);
            self.current_video_frame += 1;
            Ok(true)
        } else {
            self.ended = true;
            Ok(false)
        }
    }

    fn ensure_frame_available(&mut self, target_frame: usize) -> Result<()> {
        // 清理过期的缓存帧
        while let Some(front) = self.frame_cache.front() {
            if front.frame_index < target_frame.saturating_sub(2) {
                self.frame_cache.pop_front();
            } else {
                break;
            }
        }

        // 跳过不需要的帧（如果目标帧远超当前帧）
        while self.current_video_frame < target_frame && self.frame_cache.is_empty() {
            if !self.decode_frame_to_cache()? {
                return Ok(());
            }
            // 如果跳跃太大，直接丢弃中间帧
            if target_frame > self.current_video_frame + 5 {
                self.frame_cache.clear();
            }
        }

        // 预解码帧以保持流畅播放
        let decode_target = target_frame + self.decode_ahead_frames;
        while self.current_video_frame <= decode_target
            && self.frame_cache.len() < self.max_cache_size
            && !self.ended {
            if !self.decode_frame_to_cache()? {
                break;
            }
        }

        Ok(())
    }

    pub fn update(&mut self, t: f32) -> Result<()> {
        if t < self.start_time || self.ended {
            return Ok(());
        }

        self.alpha.set_time(t);
        self.dim.set_time(t);

        let target_frame = ((t - self.start_time) as f64 / self.frame_delta) as usize;

        // 确保目标帧可用
        self.ensure_frame_available(target_frame)?;

        // 查找并应用目标帧
        if let Some(frame_data) = self.frame_cache.iter().rfind(|f| f.frame_index <= target_frame) {
            if frame_data.frame_index >= self.next_frame {
                self.upload_frame(frame_data);
                self.next_frame = frame_data.frame_index + 1;
            }
        }

        Ok(())
    }

    fn upload_frame(&self, frame: &FrameData) {
        let ctx = unsafe { get_internal_gl() }.quad_context;
        for (texture, buffer) in [
            (self.tex_y, &frame.buf_y), (self.tex_u, &frame.buf_u), (self.tex_v, &frame.buf_v),
        ] {
            texture.raw_miniquad_texture_handle().update(ctx, buffer);
        }
    }

    fn rewind(&mut self) -> Result<()> {
        self.video = Self::open_decoder(&self.source)?;
        self.frame_cache.clear();
        self.current_video_frame = 0;
        self.next_frame = 0;
        self.ended = false;
        self.last_timestamp = None;
        self.first_timestamp = None;
        self.displayed_timestamp = None;
        Ok(())
    }

    pub(crate) fn duration(&self) -> Option<f64> {
        self.duration
    }

    pub(crate) fn update_background(&mut self, elapsed: f64, repeat: bool) -> Result<bool> {
        self.background_visible = false;
        if elapsed < 0. {
            return Ok(false);
        }
        let (time, cycle) = if let Some(duration) = self.duration {
            if repeat {
                (elapsed.rem_euclid(duration), (elapsed / duration) as u64)
            } else if elapsed >= duration {
                return Ok(false);
            } else {
                (elapsed, 0)
            }
        } else {
            (elapsed, 0)
        };
        if cycle != self.last_cycle || self.last_background_time.is_some_and(|old| old > time) {
            self.rewind()?;
        }
        self.last_cycle = cycle;
        self.last_background_time = Some(time);

        let mut latest = None;
        loop {
            if self.frame_cache.is_empty() && !self.decode_frame_to_cache()? {
                break;
            }
            if self.frame_cache.front().is_some_and(|frame| frame.timestamp > time) {
                break;
            }
            latest = self.frame_cache.pop_front();
        }
        if self.ended && self.duration.is_none() {
            if let Some(last) = self.last_timestamp {
                self.duration = Some(last + self.frame_delta);
                if repeat && elapsed >= self.duration.unwrap() {
                    return self.update_background(elapsed, true);
                }
            }
        }
        if self.ended && self.duration.is_some_and(|end| time >= end) {
            return Ok(false);
        }
        if let Some(frame) = latest {
            self.upload_frame(&frame);
            self.displayed_timestamp = Some(frame.timestamp);
        }
        self.background_visible = self.displayed_timestamp.is_some();
        Ok(self.background_visible)
    }

    pub(crate) fn render_background(&self, aspect_ratio: f32) {
        if self.background_visible {
            self.draw(aspect_ratio, 1., 0.);
        }
    }

    pub fn render(&self, res: &Resource) {
        if res.time < self.start_time || self.ended {
            return;
        }

        let alpha = self.alpha.now_opt().unwrap_or(1.0);
        if alpha <= 0.0 {
            return;
        }

        self.draw(res.aspect_ratio, alpha, self.dim.now());
    }

    fn draw(&self, aspect_ratio: f32, alpha: f32, dim: f32) {
        let top = 1. / aspect_ratio;
        let r = Rect::new(-1., -top, 2., top * 2.);

        gl_use_material(self.material);
        let s = source_of_image(&self.tex_y, r, self.scale_type).unwrap_or_else(|| Rect::new(0., 0., 1., 1.));
        let dim = 1. - dim;
        let color = Color::new(dim, dim, dim, alpha);
        let vertices = [
            Vertex::new(r.x, r.y, 0., s.x, s.y, color),
            Vertex::new(r.right(), r.y, 0., s.right(), s.y, color),
            Vertex::new(r.x, r.bottom(), 0., s.x, s.bottom(), color),
            Vertex::new(r.right(), r.bottom(), 0., s.right(), s.bottom(), color),
        ];
        let gl = unsafe { get_internal_gl() }.quad_gl;
        gl.draw_mode(DrawMode::Triangles);
        gl.geometry(&vertices, &[0, 2, 3, 0, 1, 3]);
        gl_use_default_material();
    }

    // 添加一些性能调优方法
    pub fn set_cache_size(&mut self, size: usize) {
        self.max_cache_size = size;
    }

    pub fn set_decode_ahead_frames(&mut self, frames: usize) {
        self.decode_ahead_frames = frames;
    }
}

impl Drop for Video {
    fn drop(&mut self) {
        self.material.delete();
        self.tex_y.delete();
        self.tex_u.delete();
        self.tex_v.delete();
    }
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
    color = color0 / 255.0;
    uv = texcoord;
}"#;

    pub const FRAGMENT: &str = r#"#version 450 core
layout(location = 0) in vec2 uv;
layout(location = 1) in vec4 color;

layout(set = 0, binding = 0) uniform Uniforms {
    mat4 Projection;
    mat4 Model;
    vec4 _Time;
};

layout(set = 0, binding = 3) uniform sampler2D tex_y;
layout(set = 0, binding = 4) uniform sampler2D tex_u;
layout(set = 0, binding = 5) uniform sampler2D tex_v;

layout(location = 0) out vec4 FragColor;

void main() {
    float y = (texture(tex_y, uv).a * 255.0 - 16.0) / 219.0;
    float u = (texture(tex_u, uv).a * 255.0 - 128.0) / 224.0;
    float v = (texture(tex_v, uv).a * 255.0 - 128.0) / 224.0;
    vec3 rgb = vec3(y + 1.402 * v, y - 0.344136 * u - 0.714136 * v, y + 1.772 * u);
    FragColor = vec4(clamp(rgb, 0.0, 1.0), 1.0) * color;
}"#;
}
