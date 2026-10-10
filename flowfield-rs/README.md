# flowfield-rs

A GPU flow-field particle painter. Millions of particles, all simulated on the
GPU, no CPU involvement in the frame loop.

Part of [youdontneedtoknow](..) — small creative coding toys.

## What it does

Thousands to millions of particles drift along an invisible wind field built from
layered drifting noise. Your cursor acts as a stirring stick: it drags a vortex
through the field. Particles leave trails, so the screen slowly fills with silk.

## Why it's fast

- **The simulation is a compute shader.** Each frame, a WGSL compute pass
  advances every particle in parallel on the GPU. The CPU never reads or writes
  particle data.
- **Zero-copy rendering.** The same storage buffer the compute shader writes is
  bound directly as a vertex source. There is no staging, no readback, no
  per-frame upload of particle state.
- **One upload per frame.** The CPU only writes a ~100 byte uniform block
  (time, mouse, count, palette).
- **No allocation during play.** The particle buffer is allocated once at
  startup. Changing the particle count just changes the dispatch size and draw
  count, so raising the count at runtime cannot fail or fragment memory.

Particle struct is 32 bytes (two `vec2`, four `f32`), matching the WGSL layout.

## Build

Needs a Rust toolchain and, on Windows, the MSVC build tools.

```
cd flowfield-rs
cargo build --release
```

## Run

```
cargo run --release
```

Or run the binary directly:

```
flowfield                      # windowed, 1M particles
flowfield --count 4000000      # start with 4M active
flowfield --max 6000000        # preallocate for 6M
flowfield --points             # start in 1px point mode
flowfield --bench              # headless speed test, prints a table
```

### Options

| Flag | Meaning |
| --- | --- |
| `--count N` | Active particles at startup (default 1,000,000) |
| `--max N` | Buffer capacity (default 2,000,000). Clamped to the GPU's storage buffer limit. |
| `--points` | Start in 1px point mode instead of soft sprites |
| `--bench` | Headless benchmark, prints fps per particle count, then exits |

### Keys

| Key | Action |
| --- | --- |
| mouse | Stir the field |
| up / down | Particle count +/- 250,000 |
| tab | Sprites <-> points |
| `[` / `]` | Sprite size |
| `p` | Palette (aurora / ember / ocean / bloom) |
| space | Pause |
| `c` | Clear canvas |
| esc | Quit |

## Bench

`--bench` runs 30 warmup frames plus 200 timed frames at each particle count and
prints the result. Run it yourself to see what your GPU does:

```
flowfield --bench --max 4000000
```

Note: `--bench` renders to an offscreen texture at 1920x1080 in point mode, so
the numbers reflect simulation + draw cost without windowing overhead. Vsync is
not involved.

## Design notes

- **Two render modes.** Sprite mode draws a soft radial quad per particle
  (pretty, more fill cost). Point mode draws 1px points (ugly, maximum count).
- **Additive blending into a float canvas.** Particles accumulate light into an
  `Rgba16Float` canvas rather than drawing straight to the swapchain, so dense
  regions glow instead of clipping to flat white. Falls back to `Rgba8Unorm` if
  the adapter can't blend float targets.
- **Trails via fade.** Instead of clearing the canvas each frame, a fullscreen
  pass darkens it by ~7%. Particle paths decay into streaks. The fade pass is
  skipped while paused so a paused image doesn't slowly fade to black.
- **Particles respawn on expiry**, staggered by a per-particle seed so the field
  doesn't pulse in visible waves.
- **Buffer starts zeroed.** `life = 0` makes every particle respawn on frame 1.
  Uninitialized memory could contain NaN, which would poison the position math.

## Limits

`--max` is clamped to `max_storage_buffer_binding_size / 32`. On most desktop
GPUs that's hundreds of millions; the default cap in the code is 8M to keep
startup allocation sane. On this machine (Intel UHD, 8GB RAM) the practical
comfortable ceiling is far lower — see the bench output.

## License

MIT
