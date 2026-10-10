// flowfield-rs — GPU particle simulation + rendering
//
// Particle state lives in one storage buffer. A compute shader advances every
// particle each frame; the same buffer is bound as a vertex source so the
// render pass reads it directly. The CPU never touches particle data during
// the frame loop — it only uploads a small uniform block.
//
// Bind group map:
//   group(0) binding(0)  uniform
//   group(1) binding(0)  particle storage (compute rw / vertex ro)
//   group(2) binding(0)  persistent canvas texture
//   group(2) binding(1)  sampler

// ---------------------------------------------------------------- structs

struct Particle {
    pos: vec2<f32>,
    vel: vec2<f32>,
    seed: f32,       // stable per-particle random, also drives lifetime
    life: f32,       // counts down; <= 0 means respawn
    max_life: f32,
    _pad: f32,       // keeps the struct 32B, the friendliest array stride
};

struct Uniforms {
    res: vec2<f32>,
    mouse: vec2<f32>,
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
    _pad: vec2<u32>,
    cols: array<vec4<f32>, 4>,   // active palette, uploaded from the CPU
};

struct VSOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) local: vec2<f32>,   // -1..1 inside the sprite quad
    @location(1) tint: vec3<f32>,
    @location(2) alpha: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

@group(1) @binding(0) var<storage, read_write> particles_rw: array<Particle>;
@group(1) @binding(0) var<storage, read>       particles_ro: array<Particle>;

@group(2) @binding(0) var canvas_tex: texture_2d<f32>;
@group(2) @binding(1) var canvas_smp: sampler;

// ---------------------------------------------------------------- helpers

const TAU: f32 = 6.28318530718;

fn hash21(p: vec2<f32>) -> f32 {
    var q = fract(p * vec2<f32>(123.34, 456.21));
    q += dot(q, q + 45.32);
    return fract(q.x * q.y);
}

// value noise, 2D
fn noise2(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let w = f * f * (3.0 - 2.0 * f);   // smoothstep interpolation
    let a = hash21(i);
    let b = hash21(i + vec2<f32>(1.0, 0.0));
    let c = hash21(i + vec2<f32>(0.0, 1.0));
    let d = hash21(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, w.x), mix(c, d, w.x), w.y);
}

fn pal(t: f32) -> vec3<f32> {
    let x = fract(t) * 4.0;
    let i = u32(floor(x));
    let f = fract(x);
    let a = u.cols[i % 4u].rgb;
    let b = u.cols[(i + 1u) % 4u].rgb;
    return mix(a, b, f);
}

fn max_life_for(seed: f32) -> f32 {
    return 2.2 + seed * 2.2;
}

// ---------------------------------------------------------------- compute

@compute @workgroup_size(256)
fn update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.count) { return; }

    var p = particles_rw[i];

    // Respawn expired particles. Staggering by seed keeps the field from
    // popping in and out in visible waves.
    if (p.life <= 0.0) {
        let r1 = hash21(vec2<f32>(f32(i) * 1.7, 3.3));
        let r2 = hash21(vec2<f32>(f32(i) * 0.37, 7.7));
        p.pos = vec2<f32>(r1 * u.res.x, r2 * u.res.y);
        p.vel = vec2<f32>(0.0);
        p.seed = fract(r1 * 91.7);
        p.max_life = max_life_for(p.seed);
        p.life = p.max_life;
    }

    // ---- flow field: two octaves of drifting value noise
    let s = p.pos * u.field_scale;
    let drift = vec2<f32>(u.time * 0.05, u.time * 0.12);
    let n = noise2(s + drift) * 0.65
          + noise2(s * 2.7 + vec2<f32>(40.0, -17.0)) * 0.35;
    let ang = n * TAU * 2.0;
    let speed = 70.0 + 60.0 * p.seed;
    var desired = vec2<f32>(cos(ang), sin(ang)) * speed;

    // ---- cursor is a stirring stick: tangential force inside the radius
    let d = p.pos - u.mouse;
    let dist = length(d);
    if (dist < u.swirl_radius) {
        let falloff = 1.0 - dist / u.swirl_radius;
        let tangent = vec2<f32>(-d.y, d.x) / max(dist, 0.001);
        desired += tangent * falloff * 280.0;
    }

    // ---- steer toward the desired velocity, then integrate
    p.vel += (desired - p.vel) * min(1.0, 2.6 * u.dt);
    p.pos += p.vel * u.dt;
    p.life -= u.dt;

    // Off-screen particles die immediately and get recycled next frame.
    if (p.pos.x < -40.0 || p.pos.x > u.res.x + 40.0 ||
        p.pos.y < -40.0 || p.pos.y > u.res.y + 40.0) {
        p.life = 0.0;
    }

    particles_rw[i] = p;
}

// ---------------------------------------------------------------- sprites

@vertex
fn vs_sprite(@builtin(vertex_index) vi: u32,
             @builtin(instance_index) ii: u32) -> VSOut {
    var p = particles_ro[ii];

    // triangle-strip corners: (-1,-1) (1,-1) (-1,1) (1,1)
    let corners = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0,  1.0), vec2<f32>(1.0,  1.0),
    );
    let c = corners[vi];

    let lf = clamp(p.life / max(p.max_life, 0.001), 0.0, 1.0);
    let alpha = min(1.0, lf * 2.2) * 0.5;

    let half_size = u.sprite_size * 0.5;
    let world = p.pos + c * half_size;

    var out: VSOut;
    // pixels -> clip space
    out.clip = vec4<f32>(
        world.x / u.res.x * 2.0 - 1.0,
        1.0 - world.y / u.res.y * 2.0,
        0.0, 1.0,
    );
    out.local = c;
    out.tint = pal(p.seed * 0.30 + u.hue);
    out.alpha = alpha;
    return out;
}

@fragment
fn fs_sprite(in: VSOut) -> @location(0) vec4<f32> {
    // soft radial falloff -> premultiplied for additive blending
    let d = length(in.local);
    let a = smoothstep(1.0, 0.0, d) * in.alpha;
    return vec4<f32>(in.tint * a, a);
}

// ---------------------------------------------------------------- points
// 1px points, no overdraw, for the "how many can it take" mode.

@vertex
fn vs_point(@builtin(vertex_index) vi: u32) -> VSOut {
    var p = particles_ro[vi];
    let lf = clamp(p.life / max(p.max_life, 0.001), 0.0, 1.0);
    let alpha = min(1.0, lf * 2.2) * 0.5;

    var out: VSOut;
    out.clip = vec4<f32>(
        p.pos.x / u.res.x * 2.0 - 1.0,
        1.0 - p.pos.y / u.res.y * 2.0,
        0.0, 1.0,
    );
    out.local = vec2<f32>(0.0);
    out.tint = pal(p.seed * 0.30 + u.hue);
    out.alpha = alpha;
    return out;
}

@fragment
fn fs_point(in: VSOut) -> @location(0) vec4<f32> {
    return vec4<f32>(in.tint * in.alpha, in.alpha);
}

// ---------------------------------------------------------------- fade + blit
// The trail effect: every frame we darken the previous frame slightly instead
// of clearing it, so particle paths decay into silk.

@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    // one oversized triangle covering the viewport
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0),
    );
    return vec4<f32>(pos[vi], 0.0, 1.0);
}

@fragment
fn fs_fade() -> @location(0) vec4<f32> {
    // alpha-blended dark veil, premultiplied
    let a = 0.075;
    return vec4<f32>(0.016 * a, 0.020 * a, 0.039 * a, a);
}

@fragment
fn fs_blit(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(canvas_tex, 0));
    let uv = frag.xy / dims;
    return textureSample(canvas_tex, canvas_smp, uv);
}
