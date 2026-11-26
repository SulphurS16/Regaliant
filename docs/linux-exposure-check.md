# Comparing an exposure regression

The checker runs an existing binary; it never builds, switches revisions or modifies seeds.
It loads the saved scene, keeps the character stationary, and captures absolute guest video
flips 3600, 3900 and 4200. All runs use hidden offscreen native 3840×2160, TAA, frame generation
off, silent audio, a 30 fps cap, private settings/caches and the CPU/temperature guard.
This tests the title's real display buffer before swapchain presentation, not the FSR output.
It is an image-correctness test, not a frame-rate measurement.

```sh
export DS_GAME_DIR=/path/to/PPSA01341-app0
python3 scripts/linux-exposure-check.py --binary target/release/dsport \
  --save-seed local-data/linux-port/save-seed \
  --shader-seed /path/to/immutable/shader-store \
  --label current-uffd --env DS_LINUX_WRITE_WATCH=uffd
```

NumPy and Pillow are required. `--dry-run` prints the complete plan without launching.
The supplied executable is hashed and run directly; it is not replaced by this checkout's
binary. The shared CPU-heavy lock covers the run. The default wall bound is 600 seconds,
with an outer timeout and TERM/KILL fallback. Outputs stay under
`local-data/linux-port/exposure/`; `latest.txt` names the last run. No player saves or driver
caches are used. Shader seeds are optional and copied; an omitted seed starts an empty store.

`summary.txt` reports full-frame and central-half mean RGB. `summary.json` also contains
encoded luma, sRGB-decoded luminance, near-white/black fractions, image hashes, active tracking,
translation policy, pipeline reuse and thermal evidence. Mean RGB uses stored SDR code values
in [0,1], matching the original screenshot comparison; it is not physical luminance. Review
the PNGs for matching camera/pose, loading screens and missing draws.

The fixed flips are absolute, not offsets from a wall-timed load. A conservative first
heartbeat after the stationary gameplay marker must precede every capture. Missing samples,
a tracker fallback, a crash, or reaching the end of the stationary window fails the run.
A deliberate completed-capture stop uses runtime status 75 on existing binaries; it is
accepted only with the completed-shutdown log, every valid capture, and no thermal abort.
The raw exit status remains in the summary. Other status-75 exits still fail.
If any revision loads too late, choose a later **common** `--flips` list and rerun all arms.
The old binaries allow only 180 seconds in this stationary phase; the checker stops before
scripted walking and fails rather than substituting different frames. Shutdown can overshoot
the final requested flip because the bootstrap polls its stop file; the captures themselves
are fixed-flip. Exposure adaptation has run for hundreds of frames by the default samples,
but different load histories and async compilation can still affect the scene.

## Reviewer build and run matrix

These worktree commands are for the reviewer, outside the file-edit-only agent sandbox.
Run from the main checkout. Build the three requested revisions without changing the main
branch, with repository CARGO_HOME and one compiler worker. Existing worktrees can be reused;
do not delete or reset one to make this command succeed.

```sh
repo="$PWD"
build_cpu="$(python3 -c 'import os; print(min(os.sched_getaffinity(0)))')"
for rev in 2efc3a7 8d0c5fc d91a8b0; do
  wt="$repo/local-data/linux-port/exposure-builds/$rev"
  git worktree add --detach "$wt" "$rev" || break
  flock -n "$HOME/.cache/cpu-heavy.lock" \
    env CARGO_HOME="$repo/.cargo-home" CARGO_BUILD_JOBS=1 \
    timeout -k 5s 1800s taskset -c "$build_cpu" nice -n 10 cargo build --release -p dsport \
      --manifest-path "$wt/Cargo.toml" --target-dir "$wt/target" || break
done
```

Use the same immutable shader seed in every arm. The recent prepared store is useful, but
incompatible translation entries will be rebuilt in each private copy. Keep that rebuilding
separate from any performance claim. Inspect pipeline-reuse evidence and images; if missing
cold pipelines affect the scene, repeat using that arm's private `shaders` directory as seed.
Do not compare one cold incomplete image with another fully compiled image.

```sh
export DS_RELAY_SHADER_SEED=/path/to/immutable/shader-store
for rev in 2efc3a7 8d0c5fc d91a8b0; do
  for watch in mprotect uffd; do
    python3 scripts/linux-exposure-check.py \
      --binary "local-data/linux-port/exposure-builds/$rev/target/release/dsport" \
      --shader-seed "$DS_RELAY_SHADER_SEED" --label "$rev-$watch" \
      --env "DS_LINUX_WRITE_WATCH=$watch" || echo "Inspect failed arm: $rev-$watch"
  done
done
# Isolate the new float-preservation policy from both tracking backends.
for watch in mprotect uffd; do
  python3 scripts/linux-exposure-check.py \
    --binary local-data/linux-port/exposure-builds/d91a8b0/target/release/dsport \
    --shader-seed "$DS_RELAY_SHADER_SEED" --label "d91a8b0-$watch-nan-off" \
    --env "DS_LINUX_WRITE_WATCH=$watch" --env DS_NAN_PRESERVE=0 \
    || echo "Inspect failed control: $watch"
done
```

Uffd exists in all three revisions; the default change is not its introduction. Each arm
requires the log to confirm the requested backend. The first six runs separate tracking
from intervening code; the last two isolate float preservation. Preserve subgroup/reconvergence
defaults for this matrix. The checker also accepts explicit `DS_PLAIN_MODES` and
`DS_MAX_RECONV` controls, but changing several policies together defeats the comparison.

Interpretation:

- Only uffd is bright within the same revision: investigate CPU dirty tracking and buffer
  upload/write-back ordering before touching shader semantics.
- Both backends change between the old revision and 8d0c5fc: test 9bb44b5 (FG input changes)
  and 35739da (the earlier upstream merge) next; the final merge alone is not isolated.
- Disabling float preservation restores the dark scene on both backends: inspect shader
  NaN/Inf control-flow semantics and driver lowering. Do not promote the override from a
  still image alone; it was introduced to fix other Blackwell rendering failures.
- FG-off images are correct in every arm: repeat a current arm with the established FG relay;
  this points toward the capture/presenter path or a changed test setup rather than a general
  real-frame regression.

## Exposure-buffer diagnostic

After the image-only matrix, repeat one dark and one bright arm with `--tap-exposure`.
It enables the existing cross-version `DS_TAP_CS=100893aa00:0:16` diagnostic beginning
120 flips before the first capture. The runtime logs GPU buffer words beside the guest
copy; the checker decodes the first float and collects the series in `summary.json`.
No taps is a failed diagnostic, not evidence that exposure is zero.

This tap adds barriers/copies and uses a 32-tap delayed ring without a per-slot fence.
It can perturb a race and cannot alone prove GPU/CPU coherence loss. Keep its images
separate from the uninstrumented matrix. The TAA resource-0 buffer descriptor lives at
pass constants +0xf0; its first value multiplies scene RGB before TAA. It is **not**
`fg-inputs/inputs.json`'s `taa_words[0]`, which belongs to the containing pass-constant block.

A shader-bundle audit finds a GPU path: luminance histogram/downsample -> buffer reductions
and bitonic sort -> exposure-control buffers -> TAA. Exposure-control variants read previous
state and write the exposure buffer (including dword 0); the adaptive variant uses log/exp
and history values. TAA loads dword 0 and multiplies RGB at PCs 0xe0–0xe8. This establishes a
GPU producer/consumer path; it does not establish which optional CPU consumers execute in
this saved scene. Resource addresses and the active exposure-control variant still need
runtime evidence.

Uffd tracks CPU writes through registered guest mappings, not GPU DMA through Vulkan aliases.
GPU writes instead have shader dirty masks and FE2 forwarding. FE2 records a GPU
`mirror_scatter` write-back before GPU CP writes and at batch ends; later uploads execute
after earlier batches and their write-backs. This is distinct from the optional resident
mirror mode's CPU write-back. File reads/decompression have explicit host-write guards and an uffd dirty
journal spanning the write. A GPU-to-GPU exposure read should not depend on a CPU write
fault. Missing alias announcements, stale forwarding, or unsynchronized overlapping CPU/GPU
stores remain possible bugs; a difference in tracker alone is evidence to investigate them,
not proof that uffd must track device DMA.

The earlier upstream merge also changes fast-clear metadata ownership between aliased
lighting/HDR targets. Keep that in the next revision bisect if both trackers are bright and
the float-preservation control does not explain the difference. The final ultrawide render
cap does not change the requested 3840×2160 (16:9) dimensions.

## Metadata-ownership comparison in one build

The revision comparison isolated the regression to the first upstream merge: both trackers
are dark before it and bright after it, even with float preservation disabled. The new
main-view filter cannot run in this Linux FG-off test: frame-generation input capture returns
before it, and the Linux vendor-upscaler path is inactive. Metadata handoff remains a live
suspect; the following controls leave the renderer's normal defaults unchanged.

Build this checkout once, then run the baseline and handoff-disabled control with the same
immutable shader seed. All arms use uffd (already excluded as the exposure cause).

```sh
export DS_GAME_DIR=/path/to/PPSA01341-app0
export DS_RELAY_SHADER_SEED=/path/to/immutable/shader-store
for mask in 3 0; do
  python3 scripts/linux-exposure-check.py --binary target/release/dsport \
    --shader-seed "$DS_RELAY_SHADER_SEED" --label "claims-$mask" --trace-clears \
    --env DS_LINUX_WRITE_WATCH=uffd --env "DS_META_CLAIMS=$mask" \
    || echo "Inspect invalid arm: claims-$mask"
done
```

`DS_META_CLAIMS` controls only the upstream `Exec::claim_meta` handoff (including held
recorders). It does not disable ordinary metadata fills, fast clears or the earlier
attachment-rebind fix. Values: `3` = colour and depth (default), `0` = neither,
`1` = colour only, `2` = depth only. The runtime logs the policy; the checker rejects a
binary which ignores the requested switches. These recording switches do not change shader
translation or pipeline identities.

If disabling handoff restores approximately 0.07 mean RGB (baseline approximately 0.60),
run the two partial controls to locate the affected target type. Run the rebind candidate
as well: it retains handoff, but invalidates attachment caches when a claim changes the
metadata binding, using the existing `bind_clear_metadata` protocol. Upstream's direct
map insertion updates the write-note generation alone. A unit test demonstrates the stale
attachment memo; its effect on this scene still needs the image comparison.

```sh
for mask in 1 2; do
  python3 scripts/linux-exposure-check.py --binary target/release/dsport \
    --shader-seed "$DS_RELAY_SHADER_SEED" --label "claims-$mask" --trace-clears \
    --env DS_LINUX_WRITE_WATCH=uffd --env "DS_META_CLAIMS=$mask" \
    || echo "Inspect invalid arm: claims-$mask"
done
python3 scripts/linux-exposure-check.py --binary target/release/dsport \
  --shader-seed "$DS_RELAY_SHADER_SEED" --label claims-rebind --trace-clears \
  --env DS_LINUX_WRITE_WATCH=uffd --env DS_META_CLAIMS=3 --env DS_META_CLAIM_REBIND=1
```

`--trace-clears` sets `DS_TRACE_CLEARS_FLIPS=3598:3600` for the default capture list.
`clear-trace.txt` collects fills, ownership transfers (source/destination image keys, pending
metadata code and clear words), and resolutions with the reading programs. The window uses
the queued guest flip, so it describes work around the first captured image; asynchronous
queues can be ahead of presentation. It adds logging, not GPU read-backs. Later captures are
outside that logging window. Missing trace output fails a requested trace run.

If the handoff-disabled control stays bright, skip the partial controls; retain the traces
and try the rebind candidate. Do not keep handoff disabled as a player workaround without
checking HUD trails, depth/shadows and moving scenes. A dark image alone does not establish
that either rollback or the rebind candidate is the final fix. Follow a successful candidate
with an untraced exposure check and the frame-generation relay. Optional `--tap-exposure`
remains a separate diagnostic for comparing TAA's exposure scalar with the scene brightness.
