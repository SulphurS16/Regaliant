# Linux maintainer notes

These are constraints to preserve when merging shared renderer changes. Player instructions
are in [linux.md](linux.md); frame-interpolation details are in
[linux-frame-generation.md](linux-frame-generation.md).

## Platform and memory

The shared entry point dispatches to `windows_main.rs` or `linux_main.rs`; host APIs live
under `platform/`. Keep upstream Windows entry-point edits in the Windows module rather
than restoring a second entry point. The Linux host keeps libc/Rust FS intact; supported
guest FS accesses are patched to GS. Fault handlers use alternate stacks and avoid allocation,
logging, mutexes and Vulkan calls on direct recovery. Shutdown persists completed saves and
shader records before ending process-lifetime workers with `_exit`; this is not an embeddable
renderer teardown.

Async uffd write protection and `PAGEMAP_SCAN` are the default dirty tracker. The full shader
tour completed with this backend after repeated corrupted-pointer failures with mprotect.
The latter remains an explicit diagnostic/fallback backend. Re-arming must preserve dirty
notifications across concurrent writers, and FE2 write-back must not erase newer CPU writes.
Tests cover these races, but mprotect's repaired path still lacks equivalent full-tour proof.
Neither tracker makes unsynchronized CPU/GPU writes to the same word well defined. C4 held
pages remain disabled; Linux uses eager GPU-to-CPU copies.

Host file reads use staging/copy handling for tracked guest destinations: a kernel copy into
an mprotect page can return EFAULT or a short count without a userspace signal. Completion
must follow the final byte copy. Preserve alias tracking and the write announcements around
host writes when adding I/O paths. A read of reserved guest memory is not permission to commit
arbitrary pages: a stale stream cursor or missing producer needs diagnosis, not zero filling.

Label waits must publish promises before waking address waiters; otherwise a consumer can
miss visibility and fall back to the millisecond timeout. Keep batch/queue ordering intact.
A previous attempt to wait only for a narrower input batch hung the GPU. Hang reports collect
HLE wait objects and active labels alongside signal samples and `/proc` states. Repeated
reports may reuse only completed sample slots; pending signals must never write into a
reassigned slot. Ctrl+F10 and the watchdog share this reporter.

## Shader correctness and caches

The volumetric scatter shader (FNV `cb83739fea041817`) writes NaN sentinels outside valid
slices and must reject NaN history inside them. Ordinary floating-point comparisons can
misclassify NaNs under driver optimization. `v_cmp_class_f32` therefore classifies integer
bit patterns, preserving ABS/NEG modifiers; med3/min3 semantics explicitly handle NaNs.
These are instruction semantics, not a fog-specific colour clamp. Tests cover all 1,024 class
masks and NaN operand permutations. An isolated 8×8×72 scatter fixture with NaN history
checks that each column contains exactly `depth_limit + 1` finite slices.

Linux NVIDIA retains the subgroup translation modes, with maximal reconvergence off by
default: the portable path produced rainbow fragments on tested Blackwell hardware.
Windows retains upstream's device policy. `DS_PLAIN_MODES` and `DS_MAX_RECONV` remain explicit
overrides. The NaN fixes are needed independently of that policy; keep them when merging
translator changes. Translation semantics, enabled features and mode bits participate in
cache compatibility. Never compare a newly changed translation against an old "warm" store.

Integer cube-array texel fetches use face-array views rather than unsupported cube fetch
forms, retaining integer values and coordinates exactly. This avoids an NVIDIA compiler
`cannot select` trap. Record risky pipeline creation before entering the driver, clearing it
only on success; quarantine prevents the next startup from repeating a known fatal compile.
A successful launch that skipped a quarantined pipeline is not proof of shader correctness.

The pipeline store holds Vulkan pipeline binaries, not just `VkPipelineCache` data. Retain
per-pipeline keys and shared binary data consistently across capture, store, load and replay.
The compatibility identity includes device/driver information, enabled features and translation
fingerprint; source keys describe the creation inputs. Private NVIDIA/Mesa caches supplement
it. Compare actual binary hits, driver refusals and creation time across two processes;
"source pipeline created" alone does not establish a costly compile. Preparation and play
must create devices with matching features even when preparation pins TAA and disables FG.

## Recording and frame-generation boundaries

`DS_PAR_WHOLE_RUN=1` is off and must remain off: moving scenes showed flicker, green shapes
and black pops despite plausible boundary PNGs and some faster timings. The experiment hands
the first synchronization-free PM4 chunk to a worker and records primary batches concurrently
in submission order. It still has an unresolved state/resource handoff error. Command-coverage
tests alone do not establish correctness. Before revisiting it, audit pending clears, prior
label-only batches, FE2 visibility, end-state validation and speculative replay, then inspect
moving gameplay with validation. FG rejects this mode.

Deferred upscaler and FG closures must record into the callback's command buffer, never a
command buffer captured from speculative recording. Snapshots follow guest command order
and are paired with the actual displayed image. Main-view filtering keeps menu/preview TAA
passes out of the player camera history.

Render and display sizes are separate. Upstream caps render height to 1512 for aspect ratios
wider than 16:9; it does not shrink the output or HUDless colour. FG uses the actual depth/MV
extent for render dimensions and motion scaling, and the final colour extent for display
size. Dimension changes recreate the context and reset history. Unit tests exercise 21:9,
32:9 and unchanged 16:9/16:10 sizing; Vulkan image correctness still requires a GPU relay.
