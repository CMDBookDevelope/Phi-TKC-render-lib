//! wgpu compute executor for `hand::vit`. Uploads the packed weights once and
//! runs the `embed` / `encode` / `classify` kernels, falling back to the CPU
//! path in `hand::vit` whenever anything is missing.

use crate::hand::vit::{
    HandVisionTransformer, BLOCK_COUNT, MODEL_DIM, NOTE_FEATURE_DIM, PATCH_COUNT, PATCH_DIM,
};
use std::sync::atomic::{AtomicBool, Ordering};
use wgpu::util::DeviceExt;

const FFN_DIM: usize = MODEL_DIM * 4;
const FUSED_DIM: usize = MODEL_DIM * 2;

/// Latched once the GPU path is rejected, so later frames stay on the CPU.
static GPU_REJECTED: AtomicBool = AtomicBool::new(false);

enum GpuPolicy {
    Auto,
    /// Also accept software/virtual adapters (`PRPR_VIT_GPU=force`).
    Force,
    Off,
}

fn gpu_policy() -> GpuPolicy {
    match std::env::var("PRPR_VIT_GPU").ok().as_deref().map(str::trim) {
        Some("0") | Some("off") | Some("false") | Some("no") => GpuPolicy::Off,
        Some("1") | Some("on") | Some("true") | Some("yes") | Some("force") => GpuPolicy::Force,
        _ => GpuPolicy::Auto,
    }
}

/// Packed weight layout, mirrored by the constants in `gpu_shader/vit.wgsl`.
pub mod offs {
    use super::*;

    pub const OFF_PATCH_WEIGHTS: usize = 0;
    pub const OFF_PATCH_BIAS: usize = OFF_PATCH_WEIGHTS + MODEL_DIM * PATCH_DIM;
    pub const OFF_POSITIONAL: usize = OFF_PATCH_BIAS + MODEL_DIM;
    pub const OFF_NOTE_WEIGHTS: usize = OFF_POSITIONAL + PATCH_COUNT * MODEL_DIM;
    pub const OFF_NOTE_BIAS: usize = OFF_NOTE_WEIGHTS + MODEL_DIM * NOTE_FEATURE_DIM;
    pub const BLOCK_BASE: usize = OFF_NOTE_BIAS + MODEL_DIM;

    pub const B_ATTN_GAIN: usize = 0;
    pub const B_QUERY_WEIGHTS: usize = B_ATTN_GAIN + MODEL_DIM;
    pub const B_QUERY_BIAS: usize = B_QUERY_WEIGHTS + MODEL_DIM * MODEL_DIM;
    pub const B_KEY_WEIGHTS: usize = B_QUERY_BIAS + MODEL_DIM;
    pub const B_KEY_BIAS: usize = B_KEY_WEIGHTS + MODEL_DIM * MODEL_DIM;
    pub const B_VALUE_WEIGHTS: usize = B_KEY_BIAS + MODEL_DIM;
    pub const B_VALUE_BIAS: usize = B_VALUE_WEIGHTS + MODEL_DIM * MODEL_DIM;
    pub const B_OUT_WEIGHTS: usize = B_VALUE_BIAS + MODEL_DIM;
    pub const B_OUT_BIAS: usize = B_OUT_WEIGHTS + MODEL_DIM * MODEL_DIM;
    pub const B_FFN_GAIN: usize = B_OUT_BIAS + MODEL_DIM;
    pub const B_FFN_IN_WEIGHTS: usize = B_FFN_GAIN + MODEL_DIM;
    pub const B_FFN_IN_BIAS: usize = B_FFN_IN_WEIGHTS + FFN_DIM * MODEL_DIM;
    pub const B_FFN_OUT_WEIGHTS: usize = B_FFN_IN_BIAS + FFN_DIM;
    pub const B_FFN_OUT_BIAS: usize = B_FFN_OUT_WEIGHTS + MODEL_DIM * FFN_DIM;
    pub const BLOCK_SIZE: usize = B_FFN_OUT_BIAS + MODEL_DIM;

    pub const OFF_CLASSIFIER: usize = BLOCK_BASE + BLOCK_COUNT * BLOCK_SIZE;
    pub const WEIGHT_COUNT: usize = OFF_CLASSIFIER + FUSED_DIM + 1;
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct VitParams {
    query_count: u32,
    layer: u32,
    /// `1` when the encoder left its final tokens in the second ping-pong buffer.
    final_in_second: u32,
    _pad: u32,
}

pub fn pack(model: &HandVisionTransformer) -> Vec<f32> {
    use offs::*;

    let mut buffer = vec![0.0f32; WEIGHT_COUNT];
    buffer[OFF_PATCH_WEIGHTS..OFF_PATCH_WEIGHTS + MODEL_DIM * PATCH_DIM]
        .copy_from_slice(&model.patch_embed.weights);
    buffer[OFF_PATCH_BIAS..OFF_PATCH_BIAS + MODEL_DIM].copy_from_slice(&model.patch_embed.bias);
    buffer[OFF_POSITIONAL..OFF_POSITIONAL + PATCH_COUNT * MODEL_DIM].copy_from_slice(&model.positional);
    buffer[OFF_NOTE_WEIGHTS..OFF_NOTE_WEIGHTS + MODEL_DIM * NOTE_FEATURE_DIM]
        .copy_from_slice(&model.note_embed.weights);
    buffer[OFF_NOTE_BIAS..OFF_NOTE_BIAS + MODEL_DIM].copy_from_slice(&model.note_embed.bias);

    for (index, block) in model.blocks.iter().enumerate() {
        let base = BLOCK_BASE + index * BLOCK_SIZE;
        buffer[base + B_ATTN_GAIN..base + B_ATTN_GAIN + MODEL_DIM].copy_from_slice(&block.norm_attention.gain);
        buffer[base + B_QUERY_WEIGHTS..base + B_QUERY_WEIGHTS + MODEL_DIM * MODEL_DIM]
            .copy_from_slice(&block.query.weights);
        buffer[base + B_QUERY_BIAS..base + B_QUERY_BIAS + MODEL_DIM].copy_from_slice(&block.query.bias);
        buffer[base + B_KEY_WEIGHTS..base + B_KEY_WEIGHTS + MODEL_DIM * MODEL_DIM]
            .copy_from_slice(&block.key.weights);
        buffer[base + B_KEY_BIAS..base + B_KEY_BIAS + MODEL_DIM].copy_from_slice(&block.key.bias);
        buffer[base + B_VALUE_WEIGHTS..base + B_VALUE_WEIGHTS + MODEL_DIM * MODEL_DIM]
            .copy_from_slice(&block.value.weights);
        buffer[base + B_VALUE_BIAS..base + B_VALUE_BIAS + MODEL_DIM].copy_from_slice(&block.value.bias);
        buffer[base + B_OUT_WEIGHTS..base + B_OUT_WEIGHTS + MODEL_DIM * MODEL_DIM]
            .copy_from_slice(&block.out.weights);
        buffer[base + B_OUT_BIAS..base + B_OUT_BIAS + MODEL_DIM].copy_from_slice(&block.out.bias);
        buffer[base + B_FFN_GAIN..base + B_FFN_GAIN + MODEL_DIM].copy_from_slice(&block.norm_ffn.gain);
        buffer[base + B_FFN_IN_WEIGHTS..base + B_FFN_IN_WEIGHTS + FFN_DIM * MODEL_DIM]
            .copy_from_slice(&block.ffn_in.weights);
        buffer[base + B_FFN_IN_BIAS..base + B_FFN_IN_BIAS + FFN_DIM].copy_from_slice(&block.ffn_in.bias);
        buffer[base + B_FFN_OUT_WEIGHTS..base + B_FFN_OUT_WEIGHTS + MODEL_DIM * FFN_DIM]
            .copy_from_slice(&block.ffn_out.weights);
        buffer[base + B_FFN_OUT_BIAS..base + B_FFN_OUT_BIAS + MODEL_DIM].copy_from_slice(&block.ffn_out.bias);
    }

    buffer[OFF_CLASSIFIER..OFF_CLASSIFIER + FUSED_DIM].copy_from_slice(&model.classifier.weights);
    buffer[OFF_CLASSIFIER + FUSED_DIM] = model.classifier.bias[0];
    buffer
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn load_shader(device: &wgpu::Device) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("vit"),
        source: wgpu::ShaderSource::Wgsl(include_str!("gpu_shader/vit.wgsl").into()),
    })
}

fn compute_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    entry_point: &str,
    shader: &wgpu::ShaderModule,
) -> wgpu::ComputePipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("vit"),
        bind_group_layouts: &[layout],
        push_constant_ranges: &[],
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry_point),
        layout: Some(&pipeline_layout),
        module: shader,
        entry_point: Some(entry_point),
        cache: None,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
    })
}

fn storage_buffer(device: &wgpu::Device, label: &str, bytes: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

struct Inner {
    device: wgpu::Device,
    queue: wgpu::Queue,
    layout: wgpu::BindGroupLayout,
    embed_pipeline: wgpu::ComputePipeline,
    encode_pipeline: wgpu::ComputePipeline,
    classify_pipeline: wgpu::ComputePipeline,
    params: wgpu::Buffer,
    frame: wgpu::Buffer,
    tokens_a: wgpu::Buffer,
    tokens_b: wgpu::Buffer,
    weights: wgpu::Buffer,
    query_features: wgpu::Buffer,
    query_patches: wgpu::Buffer,
    logits: wgpu::Buffer,
    query_capacity: usize,
}

/// GPU handle for the ViT forward pass. Cheap to clone: the buffers are shared.
#[derive(Clone)]
pub struct VitGpu {
    inner: std::sync::Arc<Inner>,
}

impl std::fmt::Debug for VitGpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VitGpu").finish_non_exhaustive()
    }
}

async fn try_init(model: &HandVisionTransformer, force: bool) -> Option<Inner> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        // GL is included so machines without Vulkan/DX12 still reach a real GPU.
        backends: wgpu::Backends::VULKAN
            | wgpu::Backends::METAL
            | wgpu::Backends::DX12
            | wgpu::Backends::GL,
        ..Default::default()
    });
    let adapter = match instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await
    {
        Ok(adapter) => adapter,
        Err(error) => {
            log::info!("[hand AI] no wgpu adapter on Vulkan/DX12/Metal/OpenGL: {error}");
            return None;
        }
    };

    let info = adapter.get_info();
    log::info!(
        "[hand AI] wgpu adapter: {} [{:?}, {:?}]",
        info.name, info.device_type, info.backend
    );
    let name = info.name.to_lowercase();
    let software = matches!(info.device_type, wgpu::DeviceType::Cpu)
        || name.contains("llvmpipe")
        || name.contains("swiftshader")
        || name.contains("software")
        || name.contains("virtual")
        || name.contains("warp")
        || name.contains("basic render")
        || info.vendor == 0x10005
        || (info.vendor == 0x8086 && name.contains("haswell"));
    if software && !force {
        log::info!(
            "[hand AI] software/virtual adapter, using the CPU path (PRPR_VIT_GPU=force overrides)"
        );
        return None;
    }

    let (device, queue) = match adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("vit"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        })
        .await
    {
        Ok(pair) => pair,
        Err(error) => {
            log::info!("[hand AI] adapter {} rejected the device request: {error}", info.name);
            return None;
        }
    };

    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("vit"),
        entries: &[
            uniform_entry(0),
            storage_entry(1, true),
            storage_entry(2, false),
            storage_entry(3, true),
            storage_entry(4, true),
            storage_entry(5, false),
            storage_entry(6, true),
            storage_entry(7, false),
        ],
    });

    let shader = load_shader(&device);
    let embed_pipeline = compute_pipeline(&device, &layout, "embed", &shader);
    let encode_pipeline = compute_pipeline(&device, &layout, "encode", &shader);
    let classify_pipeline = compute_pipeline(&device, &layout, "classify", &shader);

    let params = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vit.params"),
        size: std::mem::size_of::<VitParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let packed = pack(model);
    let weights = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("vit.weights"),
        contents: bytemuck::cast_slice(&packed),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let query_capacity = 1024usize;
    let frame = storage_buffer(&device, "vit.frame", (crate::hand::AI_IMAGE_SIZE * 4) as u64);
    let tokens_a = storage_buffer(&device, "vit.tokens_a", ((PATCH_COUNT * MODEL_DIM) * 4) as u64);
    let tokens_b = storage_buffer(&device, "vit.tokens_b", ((PATCH_COUNT * MODEL_DIM) * 4) as u64);
    let query_features = storage_buffer(&device, "vit.query_features", (query_capacity * NOTE_FEATURE_DIM * 4) as u64);
    let query_patches = storage_buffer(&device, "vit.query_patches", (query_capacity * 4) as u64);
    let logits = storage_buffer(&device, "vit.logits", (query_capacity * 4) as u64);

    log::info!("[hand AI] ViT running on the GPU ({})", info.name);

    Some(Inner {
        device,
        queue,
        layout,
        embed_pipeline,
        encode_pipeline,
        classify_pipeline,
        params,
        frame,
        tokens_a,
        tokens_b,
        weights,
        query_features,
        query_patches,
        logits,
        query_capacity,
    })
}

impl VitGpu {
    /// Blocking one-shot init; `None` means the caller must stay on the CPU path.
    /// The first rejection is latched for the process, and a driver that panics
    /// on the shader or pipelines never takes the AI worker thread down.
    pub fn try_new(model: &HandVisionTransformer) -> Option<Self> {
        if GPU_REJECTED.load(Ordering::Relaxed) {
            return None;
        }
        let force = match gpu_policy() {
            GpuPolicy::Off => {
                GPU_REJECTED.store(true, Ordering::Relaxed);
                log::info!("[hand AI] PRPR_VIT_GPU=off, using the CPU path");
                return None;
            }
            GpuPolicy::Force => true,
            GpuPolicy::Auto => false,
        };
        if !model.validate() {
            return None;
        }
        let runtime = tokio::runtime::Runtime::new().ok()?;
        let inner = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            runtime.block_on(try_init(model, force))
        })) {
            Ok(Some(inner)) => inner,
            Ok(None) => {
                GPU_REJECTED.store(true, Ordering::Relaxed);
                return None;
            }
            Err(_) => {
                GPU_REJECTED.store(true, Ordering::Relaxed);
                log::info!("[hand AI] ViT shader/pipeline creation failed, using the CPU path");
                return None;
            }
        };
        Some(Self {
            inner: std::sync::Arc::new(inner),
        })
    }

    /// Number of queries one `forward` call accepts; larger sets are chunked by
    /// the caller.
    pub fn query_capacity(&self) -> usize {
        self.inner.query_capacity
    }

    /// `frame` plus per-note features `[query_count, NOTE_FEATURE_DIM]` and patch
    /// indices -> one right-hand logit per note.
    pub fn forward(&self, frame: &[f32], features: &[f32], patches: &[u32]) -> Option<Vec<f32>> {
        let query_count = patches.len();
        if query_count == 0 {
            return Some(Vec::new());
        }
        let inner = &self.inner;
        if query_count > inner.query_capacity || features.len() < query_count * NOTE_FEATURE_DIM {
            return None;
        }

        let mut params = VitParams {
            query_count: query_count as u32,
            layer: 0,
            final_in_second: 0,
            _pad: 0,
        };
        inner.queue.write_buffer(&inner.params, 0, bytemuck::bytes_of(&params));
        inner.queue.write_buffer(
            &inner.frame,
            0,
            bytemuck::cast_slice(&frame[..frame.len().min(crate::hand::AI_IMAGE_SIZE)]),
        );
        inner.queue.write_buffer(&inner.query_features, 0, bytemuck::cast_slice(features));
        inner.queue.write_buffer(&inner.query_patches, 0, bytemuck::cast_slice(patches));

        // One bind group serves every pass: `query` reads the token buffers via
        // bindings 2 and 7 and never touches the frame.
        let bind_group = inner.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &inner.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: inner.params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: inner.frame.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: inner.tokens_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: inner.weights.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: inner.query_features.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: inner.logits.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: inner.query_patches.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: inner.tokens_b.as_entire_binding() },
            ],
            label: Some("vit.bind_group"),
        });

        let groups = |count: u32| count.div_ceil(64);

        // The encoder reads a uniform written just before each pass, so the
        // passes are submitted separately.
        {
            let mut encoder = inner.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("vit.embed"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("embed"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&inner.embed_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(groups(PATCH_COUNT as u32), 1, 1);
            }
            inner.queue.submit(Some(encoder.finish()));
        }

        for layer in 0..BLOCK_COUNT {
            params.layer = layer as u32;
            inner.queue.write_buffer(&inner.params, 0, bytemuck::bytes_of(&params));
            let mut encoder = inner.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("vit.encode"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("encode"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&inner.encode_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(groups(PATCH_COUNT as u32), 1, 1);
            }
            inner.queue.submit(Some(encoder.finish()));
        }

        params.layer = 0;
        params.final_in_second = (BLOCK_COUNT as u32) & 1;
        inner.queue.write_buffer(&inner.params, 0, bytemuck::bytes_of(&params));
        {
            let mut encoder = inner.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("vit.classify"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("classify"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&inner.classify_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(groups(query_count as u32), 1, 1);
            }
            inner.queue.submit(Some(encoder.finish()));
        }

        let readback = inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vit.readback"),
            size: (query_count * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = inner.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vit.readback"),
        });
        encoder.copy_buffer_to_buffer(&inner.logits, 0, &readback, 0, (query_count * 4) as u64);
        inner.queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        let _ = inner.device.poll(wgpu::PollType::Wait);
        if receiver.recv().ok().and_then(|result| result.ok()).is_none() {
            return None;
        }
        let mut logits = {
            let data = slice.get_mapped_range();
            let values: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            values
        };
        readback.unmap();
        logits.resize(query_count, 0.0);
        if logits.iter().any(|value| value.is_nan()) {
            return None;
        }
        Some(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_length_matches_layout() {
        let model = HandVisionTransformer::new();
        let packed = pack(&model);
        assert_eq!(packed.len(), offs::WEIGHT_COUNT);
        assert_eq!(packed.len(), 115009);
    }

    #[test]
    fn vit_wgsl_parses_and_validates() {
        match naga::front::wgsl::parse_str(include_str!("gpu_shader/vit.wgsl")) {
            Ok(module) => {
                let mut validator = naga::valid::Validator::new(
                    naga::valid::ValidationFlags::all(),
                    naga::valid::Capabilities::all(),
                );
                if let Err(error) = validator.validate(&module) {
                    panic!("vit.wgsl validation failed: {error}");
                }
            }
            Err(error) => {
                panic!(
                    "vit.wgsl parse failed:\n{}",
                    error.emit_to_string(include_str!("gpu_shader/vit.wgsl"))
                );
            }
        }
    }

    /// Skipped when no hardware adapter is available.
    #[test]
    fn gpu_forward_runs() {
        let model = HandVisionTransformer::new();
        let Some(gpu) = VitGpu::try_new(&model) else {
            log::warn!("skip: no GPU adapter");
            return;
        };
        let frame = vec![0.5f32; crate::hand::AI_IMAGE_SIZE];
        let query_count = 3usize;
        let features = vec![0.1f32; query_count * NOTE_FEATURE_DIM];
        let patches = vec![0u32, 70, 143];
        let logits = gpu
            .forward(&frame, &features, &patches)
            .expect("gpu forward failed (validation error or NaN)");
        assert_eq!(logits.len(), query_count);
        assert!(logits.iter().all(|value| value.is_finite()));
    }
}
