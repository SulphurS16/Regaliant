# Fog history and RDNA NaN semantics

The tutorial's black world was traced to the fog-scatter shader (program FNV-1a
`0xcb83739fea041817`). It intentionally writes NaN sentinels to voxels beyond a depth limit.
Those sentinels can appear in the next frame's history, so the shader must reject invalid
history before blending it. Slice zero cannot be excluded by the clamped depth limit.

An isolated 8x8x72 fixture with depth=1, zero media and NaN history reproduced the failure:
all valid slices became NaN. Instruction probes identified the history class comparison at
PC `0x294` as accepting NaN. A CPU replay that deliberately accepted history reproduced the
first poisoned multiply-accumulate at PC `0x4508`. The ordinary arithmetic correctly propagates
NaN; replacing all NaNs with zero would hide the faulty classification.

`v_cmp_class_f32` now classifies integer sign/exponent/mantissa bits, including abs/neg sign
modifiers, without relying on floating comparisons or denormal handling. Tests evaluate the
emitted expression for all 1,024 class masks over signed zeros, subnormals, normals, infinities,
and signaling/quiet NaN payloads. An independent audit fixed `v_med3_f32`: any NaN selects
min3 rather than the old sorting-network result. Its emitted-expression test covers 729 triples.
Legacy zero-multiply handling, NMin/NMax and clamp's existing NaN behavior remain intact;
ordinary MUL/MAC still propagate NaN.

The isolated GPU check then produced exactly `limit + 1` finite slices per column, and full
game captures showed the lit tutorial world and fog. Captured code and resources stay outside
source control. `rdna/examples/scatter_replay.rs` creates the fixture; Linux `--scatter-check`
runs it independently of game execution and pipeline stores. Probe registers are snapshots
while execution continues, not instruction cutoffs.

These arithmetic fixes remain necessary with upstream 0.1.2's portable Blackwell modes and
maximal reconvergence. Reconvergence governs control flow, not the class test's arithmetic
contract. There is no separate Windows classifier or image-initialization policy: the shared
renderer clears new GPU-written images, and later history legitimately contains sentinels.
A Windows GPU reproduction was not obtained, so this is not claimed to be a Linux-only bug.

The translation store is keyed by `rdna::SOURCE_HASH` and serialized specialization modes.
Pipeline binaries also validate the driver key and translated SPIR-V fingerprint. The latter
now salts maximal reconvergence with `0x4d524331`, preventing reuse across declaration changes.
Windows Blackwell, AMD and Intel use upstream's portable compute modes by default; Linux
NVIDIA now retains the checked subgroup specializations on all designs. Linux preparation checkpoints include this policy revision,
so an old completed tour does not incorrectly skip rebuilding changed pipelines.

## Portable-mode artifacts on Linux

A later Blackwell Linux play test reported rainbow geometry with portable modes and maximal
reconvergence enabled; earlier subgroup-mode captures rendered the tutorial correctly. This
is distinct from the NaN classifier bug. The reported device has fixed 32-lane subgroups, so
an unexpectedly selected 64-lane Vulkan subgroup does not explain that report. No
Windows-only translation branches were found. Portable mode instead selects the wave-mask /
workgroup-LDS synchronization paths described in `rdna/src/translate/mod.rs`; the checked
specializations can avoid those paths. A driver/control-flow failure is a hypothesis, not a
confirmed diagnosis.

The startup log now records the **effective** portable/reconvergence policy, including explicit
overrides; the older device-policy line alone describes defaults. `Spec::hash` includes
per-lane, lane-vote, no-lockstep, subgroup-wave and append modes. Compute and graphics pipeline
loads validate translated-code fingerprints, including the maximal-reconvergence salt.
Preparation previously dropped the player's overrides; it now passes them to its children and
includes them and PC graphics settings in its checkpoint identity. This prevents a preparation
run from claiming completion for different player settings, without deleting compatible caches.

For a controlled comparison, use `DS_PLAIN_MODES=0 DS_MAX_RECONV=0` for both preparation and
play. If that fixes the scene, comparing `1/0` and `0/1` separately distinguishes the translation
path from the reconvergence declaration. Keep the Windows policy unchanged until there is GPU
evidence; shader caches need not be deleted for this comparison. The comparison confirmed correct rendering with subgroup modes and rainbow artifacts with
portable modes on Linux Blackwell; Linux NVIDIA now defaults to subgroup modes with maximal
reconvergence off. The exact translator/driver failure within the portable path is still unknown. The arithmetic NaN tests remain required in every mode.
