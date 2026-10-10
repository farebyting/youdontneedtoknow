// flowfield-rs — GPU flow-field particle painter
//
// Every particle lives in one storage buffer on the GPU. A compute shader
// advances them each frame; the same buffer is then read directly by the
// vertex shader. The CPU never touches particle data in the frame loop.
//
// The particle buffer is allocated once at startup and never resized, so
// raising the particle count at runtime cannot fail, fragment, or OOM —
// it just makes the dispatch bigger.
//
// Usage:
//   flowfield                       windowed, 1M particles
//   flowfield --max 4000000         bigger buffer (up to GPU limit)
//   flowfield --count 2000000       start with 2M active
//   flowfield --points              start in 1px point mode
//   flowfield --bench               headless speed test, prints a table

use std::sync::Arc;
use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt as _;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

const WORKGROUP: u32 = 256;
const PARTICLE_SIZE: u64 = 32; // must match the WGSL struct
const COUNT_STEP: u32 = 250_000;

// ------------------------------------------------------------------ data

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Particle {
    pos: [f32; 2],
    vel: [f32; 2],
    seed: f32,
    life: f32,
    max_life: f32,
    _pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Uniforms {
    res: [f32; 2],
    mouse: [f32; 2],
    dt: f32,
    time: f32,
    hue: f32,
    field_scale: f32,
    swirl_radius: f32,
    sprite_size: f32,
    count: u32,
    palette: u32,
    frame: u32,
    point_mode: u32,
    _pad: [u32; 2],
    cols: [[f32; 4]; 4],
}

const PALETTE_NAMES: [&str; 4] = ["aurora", "ember", "ocean", "bloom"];

const PALETTES: [[[f32; 4]; 4]; 4] = [
    // aurora
    [
        [0.25, 1.00, 0.82, 1.0],
        [0.25, 0.63, 1.00, 1.0],
        [0.63, 0.38, 1.00, 1.0],
        [1.00, 0.43, 0.86, 1.0],
    ],
    // ember
    [
        [1.00, 0.84, 0.40, 1.0],
        [1.00, 0.55, 0.26, 1.0],
        [1.00, 0.28, 0.28, 1.0],
        [0.75, 0.16, 0.47, 1.0],
    ],
    // ocean
    [
        [0.13, 0.83, 0.93, 1.0],
        [0.23, 0.51, 0.96, 1.0],
        [0.39, 0.40, 0.95, 1.0],
        [0.06, 0.73, 0.51, 1.0],
    ],
    // bloom
    [
        [1.00, 0.43, 0.71, 1.0],
        [1.00, 0.75, 0.47, 1.0],
        [0.75, 1.00, 0.63, 1.0],
        [0.59, 0.71, 1.00, 1.0],
    ],
];

fn fmt_thousands(n: u32) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + 4);
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

// ------------------------------------------------------------------ gpu

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,

    uniform_buf: wgpu::Buffer,
    particle_buf: wgpu::Buffer,
    uniform_bg: wgpu::BindGroup,
    particle_bg_compute: wgpu::BindGroup,
    particle_bg_render: wgpu::BindGroup,

    compute_pipeline: wgpu::ComputePipeline,
    sprite_pipeline: wgpu::RenderPipeline,
    point_pipeline: wgpu::RenderPipeline,
    fade_pipeline: wgpu::RenderPipeline,
    blit_pipeline: wgpu::RenderPipeline,

    canvas_format: wgpu::TextureFormat,
    canvas_view: wgpu::TextureView,
    canvas_bg: wgpu::BindGroup,
    bg2_layout: wgpu::BindGroupLayout,

    width: u32,
    height: u32,
    capacity: u32,
    count: u32,

    time: f32,
    hue: f32,
    palette: u32,
    sprite_size: f32,
    mouse: [f32; 2],
    mouse_active: bool,
    paused: bool,
    point_mode: bool,
    clear_requested: bool,
    frame_no: u32,
}

impl Gpu {
    async fn new(
        adapter: &wgpu::Adapter,
        device: wgpu::Device,
        queue: wgpu::Queue,
        output_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        capacity: u32,
        count: u32,
        point_mode: bool,
    ) -> Self {
        // Canvas accumulates additive light. A float format keeps bright
        // clusters from clipping to flat white; fall back if unblendable.
        let hdr_ok = adapter
            .get_texture_format_features(wgpu::TextureFormat::Rgba16Float)
            .flags
            .contains(wgpu::TextureFormatFeatureFlags::BLENDABLE);
        let canvas_format = if hdr_ok {
            wgpu::TextureFormat::Rgba16Float
        } else {
            wgpu::TextureFormat::Rgba8Unorm
        };

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("flowfield"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        // ---- bind group layouts
        let bg0_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("uniforms"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE
                    | wgpu::ShaderStages::VERTEX
                    | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bg1_compute_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("particles-rw"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let bg1_render_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("particles-ro"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let bg2_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("canvas"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // ---- pipelines
        let compute_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("compute-layout"),
            bind_group_layouts: &[Some(&bg0_layout), Some(&bg1_compute_layout)],
            immediate_size: 0,
        });
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("update"),
            layout: Some(&compute_layout),
            module: &shader,
            entry_point: Some("update"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Canvas passes (fade/sprite/point) read uniforms + particles. They must
        // NOT bind the canvas texture: it is the render target in that pass, and
        // sampling a texture while rendering into it is invalid.
        let draw_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("draw-layout"),
            bind_group_layouts: &[Some(&bg0_layout), Some(&bg1_render_layout)],
            immediate_size: 0,
        });

        // The blit pass only samples the canvas, so its layout is group 2 alone.
        // Groups 0 and 1 stay as None to keep the shader's group indices aligned.
        let blit_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("blit-layout"),
            bind_group_layouts: &[None, None, Some(&bg2_layout)],
            immediate_size: 0,
        });

        let add_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let veil_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let replace_blend = wgpu::BlendState::REPLACE;

        let make_render =
            |label: &str, vs: &str, fs: &str, blend: wgpu::BlendState, topology: wgpu::PrimitiveTopology| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&draw_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(vs),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState {
                    topology,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: canvas_format,
                        blend: Some(blend),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };

        let sprite_pipeline = make_render(
            "sprite",
            "vs_sprite",
            "fs_sprite",
            add_blend,
            wgpu::PrimitiveTopology::TriangleStrip,
        );
        let point_pipeline = make_render(
            "point",
            "vs_point",
            "fs_point",
            add_blend,
            wgpu::PrimitiveTopology::PointList,
        );
        let fade_pipeline = make_render(
            "fade",
            "vs_fullscreen",
            "fs_fade",
            veil_blend,
            wgpu::PrimitiveTopology::TriangleStrip,
        );

        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blit"),
            layout: Some(&blit_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_fullscreen"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_blit"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: output_format,
                    blend: Some(replace_blend),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        // ---- buffers
        let uniforms = Uniforms {
            res: [width as f32, height as f32],
            mouse: [-9999.0, -9999.0],
            dt: 0.0,
            time: 0.0,
            hue: 0.0,
            field_scale: 0.002,
            swirl_radius: 170.0,
            sprite_size: 2.2,
            count,
            palette: 0,
            frame: 0,
            point_mode: point_mode as u32,
            _pad: [0, 0],
            cols: PALETTES[0],
        };
        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Zeroed: life = 0 makes the compute shader respawn every particle on
        // frame 1. Uninitialized memory could hold NaN, which would poison the
        // position math, so this matters.
        let zeros = vec![0u8; capacity as usize * PARTICLE_SIZE as usize];
        let particle_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("particles"),
            contents: &zeros,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let uniform_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("uniforms"),
            layout: &bg0_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let particle_bg_compute = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("particles-rw"),
            layout: &bg1_compute_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: particle_buf.as_entire_binding(),
            }],
        });

        let particle_bg_render = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("particles-ro"),
            layout: &bg1_render_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: particle_buf.as_entire_binding(),
            }],
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("canvas-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let (canvas_view, canvas_bg) =
            make_canvas(&device, &bg2_layout, &sampler, canvas_format, width, height);

        Self {
            device,
            queue,
            uniform_buf,
            particle_buf,
            uniform_bg,
            particle_bg_compute,
            particle_bg_render,
            compute_pipeline,
            sprite_pipeline,
            point_pipeline,
            fade_pipeline,
            blit_pipeline,
            canvas_format,
            canvas_view,
            canvas_bg,
            bg2_layout,
            width,
            height,
            capacity,
            count: count.min(capacity),
            time: 0.0,
            hue: 0.0,
            palette: 0,
            sprite_size: 2.2,
            mouse: [-9999.0, -9999.0],
            mouse_active: false,
            paused: false,
            point_mode,
            clear_requested: false,
            frame_no: 0,
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 || (width == self.width && height == self.height) {
            return;
        }
        self.width = width;
        self.height = height;
        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("canvas-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let (view, bg) = make_canvas(
            &self.device,
            &self.bg2_layout,
            &sampler,
            self.canvas_format,
            width,
            height,
        );
        self.canvas_view = view;
        self.canvas_bg = bg;
    }

    /// Encode and submit one frame: advance particles, fade the canvas, draw
    /// them additively, then blit the canvas to `target`.
    fn frame(&mut self, target: &wgpu::TextureView, dt: f32) {
        if !self.paused {
            self.time += dt;
            self.hue += dt * 0.03;
            self.frame_no = self.frame_no.wrapping_add(1);
        }

        let uniforms = Uniforms {
            res: [self.width as f32, self.height as f32],
            mouse: if self.mouse_active {
                self.mouse
            } else {
                [-9999.0, -9999.0]
            },
            dt: if self.paused { 0.0 } else { dt },
            time: self.time,
            hue: self.hue,
            field_scale: 0.002,
            swirl_radius: 170.0,
            sprite_size: self.sprite_size,
            count: self.count,
            palette: self.palette,
            frame: self.frame_no,
            point_mode: self.point_mode as u32,
            _pad: [0, 0],
            cols: PALETTES[self.palette as usize % PALETTES.len()],
        };
        self.queue
            .write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });

        // 1. advance every particle on the GPU
        if self.count > 0 && !self.paused {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("update"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_bind_group(0, &self.uniform_bg, &[]);
            cpass.set_bind_group(1, &self.particle_bg_compute, &[]);
            let groups = self.count.div_ceil(WORKGROUP);
            cpass.dispatch_workgroups(groups, 1, 1);
        }

        // 2. fade the canvas, then draw the particles into it
        {
            let load = if self.clear_requested {
                wgpu::LoadOp::Clear(wgpu::Color::BLACK)
            } else {
                wgpu::LoadOp::Load
            };
            let mut rpass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("canvas"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.canvas_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            if self.count > 0 {
                rpass.set_bind_group(0, &self.uniform_bg, &[]);
                rpass.set_bind_group(1, &self.particle_bg_render, &[]);

                rpass.set_pipeline(&self.fade_pipeline);
                rpass.draw(0..3, 0..1);

                if self.point_mode {
                    rpass.set_pipeline(&self.point_pipeline);
                    rpass.draw(0..self.count, 0..1);
                } else {
                    rpass.set_pipeline(&self.sprite_pipeline);
                    rpass.draw(0..4, 0..self.count);
                }
            }
        }
        self.clear_requested = false;

        // 3. present the canvas
        {
            let mut rpass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("blit"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rpass.set_pipeline(&self.blit_pipeline);
            rpass.set_bind_group(2, &self.canvas_bg, &[]);
            rpass.draw(0..3, 0..1);
        }

        self.queue.submit([enc.finish()]);
    }

    fn buffer_mb(&self) -> f64 {
        // Read the real allocation rather than recomputing it, so the number
        // reported always matches what the GPU actually holds.
        self.particle_buf.size() as f64 / (1024.0 * 1024.0)
    }

    fn set_count(&mut self, n: u32) {
        self.count = n.min(self.capacity);
    }
}

fn make_canvas(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
) -> (wgpu::TextureView, wgpu::BindGroup) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("canvas"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("canvas"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    });
    (view, bg)
}

// ------------------------------------------------------------------ setup

async fn init(
    backends: wgpu::Backends,
) -> (wgpu::Instance, wgpu::Adapter, wgpu::Device, wgpu::Queue) {
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = backends;
    let instance = wgpu::Instance::new(desc);

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            apply_limit_buckets: false,
        })
        .await
        .expect("no suitable GPU adapter found");

    let info = adapter.get_info();
    eprintln!(
        "gpu: {} ({:?}, {:?})",
        info.name, info.backend, info.device_type
    );

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("flowfield"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        })
        .await
        .expect("failed to create device");

    (instance, adapter, device, queue)
}

fn max_capacity(device: &wgpu::Device) -> u32 {
    let limit = device.limits().max_storage_buffer_binding_size as u64;
    let by_limit = (limit / PARTICLE_SIZE).min(u32::MAX as u64) as u32;
    // Also respect the storage-buffer-per-stage limit indirectly: one buffer only.
    by_limit.min(8_000_000)
}

// ------------------------------------------------------------------ app

struct App {
    gpu: Option<Gpu>,
    surface: Option<wgpu::Surface<'static>>,
    surface_config: Option<wgpu::SurfaceConfiguration>,
    window: Option<Arc<Window>>,
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    capacity: u32,
    count: u32,
    point_mode: bool,
    last_frame: Instant,
    fps_accum: f32,
    fps_frames: u32,
    title_timer: Instant,
}

impl App {
    fn new(instance: wgpu::Instance, adapter: wgpu::Adapter, device: wgpu::Device, queue: wgpu::Queue, capacity: u32, count: u32, point_mode: bool) -> Self {
        Self {
            gpu: None,
            surface: None,
            surface_config: None,
            window: None,
            instance,
            adapter,
            device,
            queue,
            capacity,
            count,
            point_mode,
            last_frame: Instant::now(),
            fps_accum: 0.0,
            fps_frames: 0,
            title_timer: Instant::now(),
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gpu.is_some() {
            return;
        }

        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("flowfield-rs")
                        .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 800.0)),
                )
                .expect("failed to create window"),
        );

        let surface = self
            .instance
            .create_surface(window.clone())
            .expect("failed to create surface");

        let size = window.inner_size();
        let caps = surface.get_capabilities(&self.adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            color_space: wgpu::SurfaceColorSpace::Auto,
        };
        surface.configure(&self.device, &config);

        let gpu = pollster::block_on(Gpu::new(
            &self.adapter,
            self.device.clone(),
            self.queue.clone(),
            format,
            config.width,
            config.height,
            self.capacity,
            self.count,
            self.point_mode,
        ));

        println!(
            "flowfield-rs — {} particles active, {} capacity ({:.0} MB vram)",
            fmt_thousands(gpu.count),
            fmt_thousands(gpu.capacity),
            gpu.buffer_mb()
        );
        print_help();

        self.surface = Some(surface);
        self.surface_config = Some(config);
        self.window = Some(window);
        self.gpu = Some(gpu);
        self.last_frame = Instant::now();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        let (Some(gpu), Some(surface), Some(config), Some(window)) = (
            self.gpu.as_mut(),
            self.surface.as_ref(),
            self.surface_config.as_mut(),
            self.window.as_ref(),
        ) else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(size) => {
                if size.width > 0 && size.height > 0 {
                    config.width = size.width;
                    config.height = size.height;
                    surface.configure(&gpu.device, config);
                    gpu.resize(size.width, size.height);
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                gpu.mouse = [position.x as f32, position.y as f32];
                gpu.mouse_active = true;
            }

            WindowEvent::CursorLeft { .. } => {
                gpu.mouse_active = false;
            }

            WindowEvent::KeyboardInput { event, .. } => {
                if event.state != ElementState::Pressed {
                    return;
                }
                match event.logical_key.as_ref() {
                    Key::Named(NamedKey::Escape) => event_loop.exit(),
                    Key::Named(NamedKey::Space) => {
                        gpu.paused = !gpu.paused;
                        println!("{}", if gpu.paused { "paused" } else { "running" });
                    }
                    Key::Named(NamedKey::ArrowUp) => {
                        let n = (gpu.count + COUNT_STEP).min(gpu.capacity);
                        gpu.set_count(n);
                        println!("particles: {}", fmt_thousands(gpu.count));
                    }
                    Key::Named(NamedKey::ArrowDown) => {
                        let n = gpu.count.saturating_sub(COUNT_STEP);
                        gpu.set_count(n);
                        println!("particles: {}", fmt_thousands(gpu.count));
                    }
                    Key::Named(NamedKey::Tab) => {
                        gpu.point_mode = !gpu.point_mode;
                        println!(
                            "mode: {}",
                            if gpu.point_mode { "points (max speed)" } else { "sprites (pretty)" }
                        );
                    }
                    Key::Character(c) => match c {
                        "c" | "C" => {
                            gpu.clear_requested = true;
                            println!("canvas cleared");
                        }
                        "p" | "P" => {
                            gpu.palette = (gpu.palette + 1) % PALETTES.len() as u32;
                            println!("palette: {}", PALETTE_NAMES[gpu.palette as usize]);
                        }
                        "[" => {
                            gpu.sprite_size = (gpu.sprite_size - 0.2).max(0.6);
                            println!("sprite size: {:.1}", gpu.sprite_size);
                        }
                        "]" => {
                            gpu.sprite_size = (gpu.sprite_size + 0.2).min(12.0);
                            println!("sprite size: {:.1}", gpu.sprite_size);
                        }
                        _ => {}
                    },
                    _ => {}
                }
            }

            WindowEvent::RedrawRequested => {
                let st = match surface.get_current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(t) => Some(t),
                    wgpu::CurrentSurfaceTexture::Suboptimal(t) => {
                        surface.configure(&gpu.device, config);
                        Some(t)
                    }
                    wgpu::CurrentSurfaceTexture::Outdated => {
                        surface.configure(&gpu.device, config);
                        None
                    }
                    wgpu::CurrentSurfaceTexture::Lost => {
                        surface.configure(&gpu.device, config);
                        None
                    }
                    _ => None,
                };

                if let Some(st) = st {
                    let view = st
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default());
                    let now = Instant::now();
                    let dt = (now - self.last_frame).as_secs_f32().min(1.0 / 30.0);
                    self.last_frame = now;

                    gpu.frame(&view, dt);
                    gpu.queue.present(st);

                    self.fps_accum += dt;
                    self.fps_frames += 1;
                    if self.title_timer.elapsed().as_millis() > 500 && self.fps_frames > 0 {
                        let fps = self.fps_frames as f32 / self.fps_accum;
                        let _ = window.set_title(&format!(
                            "flowfield-rs — {} particles — {:.0} fps{}",
                            fmt_thousands(gpu.count),
                            fps,
                            if gpu.paused { " — paused" } else { "" }
                        ));
                        self.fps_accum = 0.0;
                        self.fps_frames = 0;
                        self.title_timer = Instant::now();
                    }
                }
            }

            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }
}

fn print_help() {
    println!(
        "\n  move mouse    stir the field\n  \
         up / down     particle count +/- {}\n  \
         tab           sprites <-> points\n  \
         [ / ]         sprite size\n  \
         p             palette\n  \
         space         pause\n  \
         c             clear canvas\n  \
         esc           quit\n",
        fmt_thousands(COUNT_STEP)
    );
}

// ------------------------------------------------------------------ bench

fn bench(
    instance: &wgpu::Instance,
    adapter: &wgpu::Adapter,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    capacity: u32,
) {
    const W: u32 = 1920;
    const H: u32 = 1080;
    const WARMUP: u32 = 30;
    const FRAMES: u32 = 200;

    println!(
        "\nbench: {}x{}, {} warmup + {} timed frames per step, point mode\n",
        W, H, WARMUP, FRAMES
    );

    let mut gpu = pollster::block_on(Gpu::new(
        adapter,
        device.clone(),
        queue.clone(),
        wgpu::TextureFormat::Rgba8Unorm,
        W,
        H,
        capacity,
        capacity,
        true,
    ));

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("bench-target"),
        size: wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());

    println!(
        "{:>12}  {:>9}  {:>10}  {:>12}",
        "particles", "fps", "ms/frame", "gpu buffer"
    );
    println!("{}", "-".repeat(50));

    let steps = [100_000u32, 250_000, 500_000, 1_000_000, 2_000_000, 4_000_000];
    for &n in &steps {
        if n > capacity {
            continue;
        }
        gpu.set_count(n);
        gpu.point_mode = true;

        for _ in 0..WARMUP {
            gpu.frame(&view, 1.0 / 60.0);
        }
        let _ = instance.poll_all(true);

        let t0 = Instant::now();
        for _ in 0..FRAMES {
            gpu.frame(&view, 1.0 / 60.0);
        }
        let _ = instance.poll_all(true);
        let elapsed = t0.elapsed().as_secs_f64();

        let fps = FRAMES as f64 / elapsed;
        let ms = elapsed * 1000.0 / FRAMES as f64;
        let mb = (n as u64 * PARTICLE_SIZE) as f64 / (1024.0 * 1024.0);

        println!(
            "{:>12}  {:>9.1}  {:>10.3}  {:>10.0} MB",
            fmt_thousands(n),
            fps,
            ms,
            mb
        );
    }
    println!();
}

// ------------------------------------------------------------------ main

fn parse_arg(args: &[String], name: &str) -> Option<u32> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.replace('_', "").replace(',', "").parse().ok())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!(
            "flowfield-rs — GPU flow-field particle painter\n\n\
             USAGE:\n  flowfield [OPTIONS]\n\n\
             OPTIONS:\n  \
             --count N    start with N active particles (default 1000000)\n  \
             --max N      preallocate buffer for N particles (default 2000000)\n  \
             --points     start in 1px point mode instead of sprites\n  \
             --bench      headless speed test, prints a table, then exits\n  \
             -h, --help   this text\n\n\
             KEYS:\n  \
             mouse        stir the field\n  \
             up/down      particle count +-250000\n  \
             tab          sprites <-> points\n  \
             [ / ]        sprite size\n  \
             p            palette\n  \
             space        pause\n  \
             c            clear canvas\n  \
             esc          quit\n"
        );
        return;
    }

    let bench_mode = args.iter().any(|a| a == "--bench");
    let want_count = parse_arg(&args, "--count");
    let want_max = parse_arg(&args, "--max");
    let point_mode = args.iter().any(|a| a == "--points");

    let (instance, adapter, device, queue) = pollster::block_on(init(wgpu::Backends::all()));

    let hard_max = max_capacity(&device);
    let mut capacity = want_max.unwrap_or(2_000_000).min(hard_max);
    if let Some(w) = want_max {
        if w > hard_max {
            eprintln!(
                "warning: --max {} exceeds this GPU's limit; clamped to {}",
                fmt_thousands(w),
                fmt_thousands(hard_max)
            );
        }
    }
    capacity = capacity.max(1);

    if bench_mode {
        bench(&instance, &adapter, &device, &queue, capacity);
        return;
    }

    let count = want_count.unwrap_or(1_000_000).min(capacity);

    let event_loop = EventLoop::new().expect("failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new(instance, adapter, device, queue, capacity, count, point_mode);
    event_loop.run_app(&mut app).expect("event loop failed");
}
