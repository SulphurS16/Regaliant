# Comparing play-session rendering

Use an immutable save snapshot at the affected scene, preferably the Nexus for the large
flat surfaces. It must contain `saves/`. The checker copies it and the shader store into a
private run directory; the player files and game folder remain untouched. A tutorial save
cannot establish whether Nexus streaming is correct.

Build `dsport` and the FSR bridge as described in [linux.md](linux.md). Export the three paths:

```sh
export DS_GAME_DIR=/path/to/PPSA01341-app0
export DS_RELAY_SAVE_SEED=/path/to/immutable/save-snapshot
export DS_RELAY_SHADER_SEED=/path/to/immutable/prepared-shaders
```

The following is **six runs total** of the same saved camera. Each opens a visible 960×540
logical-pixel window, silently, rendering at native 3840×2160. This keeps the desktop usable;
it is not a measurement of fullscreen compositor cost. All use uffd, safe sparse binds,
no whole-run recording, the CPU-heavy lock, nice 10, limited cores/workers and thermal guard.
Each stops after flips 3600/3900/4200, usually about four minutes; a timeout bounds failure.

```sh
for case in off fg fg-real-only depth-off cpu-detile vertices; do
  python3 scripts/linux-exposure-check.py --binary target/release/dsport \
    --save-seed "$DS_RELAY_SAVE_SEED" --shader-seed "$DS_RELAY_SHADER_SEED" \
    --play-case "$case" --label "play-$case" \
    || echo "Inspect failed case: $case"
done
```

| Case | Difference and evidence |
| --- | --- |
| `off` | Real frames only; reference for geometry, texture appearance and fps. |
| `fg` | Normal FSR3, generated/real PNG pairs and exact input dumps at relay checkpoints. |
| `fg-real-only` | Still runs FSR and history, but skips the generated presentation and both pacing sleeps (`DS_FG_REAL_ONLY=1`). Separates interpolation computation/capture from two-present FIFO backpressure. |
| `depth-off` | FG off, metadata claims 0 instead of Linux default 2. Tests the remaining depth handoff; colour handoff stays off. |
| `cpu-detile` | FG off, `DS_GPU_DETILE=0`. Tests GPU vs CPU texture detiling without treating unmapped pages as readable. |
| `vertices` | FG off, vertex-position census plus one frame's commands, shader binaries and target dumps. Finds large/near-zero-W clip positions and target NaNs; substantially slower, not an fps measurement. |

Review `local-data/linux-port/exposure/run-*/summary.json` (`play_metrics`), `summary.txt`,
`user/linux-runs/*/dsport.log`, fixed-flip screenshots and `generated-*` / `window-*` PNGs.
FG dumps live in `fg-inputs-*`; inspect `real.png`, `generated.png`, `hudless.png`, depth and
motion together. Census output is `outliers.txt`; capture output is `capture-*.txt`,
`targets-*` and `shaders/`. These are local diagnostic artifacts, not files to publish.
The `vertices` arm recompiles instrumented shaders. Capture stalls and those compiles must
be excluded from performance comparisons. The other arms can still compile newly seen pipelines;
check `pb_source_calls` before calling a window warm.

F9 requests a scene probe if a defect depends on the camera angle; use it once at that angle.
Moving the camera makes that arm unsuitable for a fixed-camera comparison. Keep screenshots
of the same angle with FG off/on rather than comparing unrelated frames.

With FG enabled, the current presenter displays two images per real frame. Its two pacing
steps at 60 Hz cap real throughput near 30 fps, even when simulation could run faster.
This is not fixed by more shader preparation. The real-only case diagnoses the cost; it
is not a new player-facing frame-generation mode. Adaptive generated-frame selection for
60 Hz remains work. At 120 Hz there is room for two presentations per real frame at up to 60.

For a normal play session's optional one-second counters, add `DS_PLAY_TELEMETRY=1` to the
usual launch command. `Linux play:` reports cumulative pipeline creates/reuse, FG work/pacing,
retirements, texture-upload/remap reports, and sampled uffd mapping/mutex/work durations.
Uffd durations sum across threads and include waiting; they are not frame critical-path CPU
milliseconds. Counts of zero-filled subresources are reports, not unique missing textures.
Analyze a saved log without launching anything:

```sh
python3 scripts/linux_play_metrics.py /path/to/dsport.log > /path/to/analysis.json
```

Movie audio and video use different paths: the guest's Bink workers drive video/shader uploads;
audio goes through AudioOut and SDL. The two startup movie files are 30 fps. Check the early
heartbeat/FG/pipeline counters in every arm before changing audio buffering or guest clocks.
