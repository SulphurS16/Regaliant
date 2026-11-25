# Cube texel loads and compiler quarantine

NVIDIA 615.78 on Blackwell aborted in its compiler on `llvm.nvvm.tex.load`.
The saved SPIR-V identifies an invalid input, not an integer-format limitation:
`OpImageFetch %v4float` used `OpTypeImage ... Cube ... Arrayed=1`. `spirv-val`
rejects this with `Image 'Dim' cannot be Cube`. All four observed pipeline
variants had this error. The backend's `<4 x s32>` diagnostic is its internal
representation; it does not establish the SPIR-V numeric class.

The affected programs were `e890e8e2b31bf06f` and `ffafd65f8762677f`. A learned
resource specialization described their load inputs as cube arrays. The load
emitter selected `OpImageFetch` without checking the resource dimension; this
predates the Linux port. Earlier cached recipes used ordinary 2D/array inputs
and did not exercise that invalid combination. The first local store with the
cube recipes came from a gameplay run that also reported invalid descriptors
and a huge allocation request. Why it learned those descriptors remains
unresolved; making that store safe to precompile does not resolve that separate
runtime-state issue.

`image_load` and `image_load_mip` now bind the same cube image through an
additional 2D-array view (`BindingKind::CubeFaces`). The format, swizzle and mip
range stay the same. Cube sampling retains its original cube view, including
when a program both samples and loads the resource. Coordinates address exact
texels; no float round-trip, filtering, or replacement data is involved.
For gfx10 cube integer loads the third coordinate is `slice * 6 + face`, as
specified in the [AMD RDNA ISA guide](https://gpuopen.com/wp-content/uploads/2019/08/RDNA_Shader_ISA_7July2019.pdf).
A 2D load uses face zero; an array load retains its layer. Explicit LOD stays
explicit. Cube storage/table forms that cannot use this sampled alias return a
translation error instead of emitting an invalid cube fetch.

This is a shared SPIR-V correction, not a Windows translation-policy change.
The translator source hash changes, so translations are regenerated. Pipeline
binaries are still checked against actual SPIR-V and creation-info fingerprints:
unaffected binaries remain reusable; changed modules/layouts are rebuilt.

## Offline regression checks

`crates/rdna/tests/fixtures/cube-fetch-invalid.spvasm` is a small authored negative
reproducer, with no game code. `scripts/test_linux_cube_fetch.py` assembles it,
checks that validation rejects the cube fetch, then changes only the image
dimension to a face array and checks that validation accepts it.

The Rust tests cover float/signed/unsigned data, implicit/explicit LOD, 2D/cube/
array instruction coordinates and both cube resource types. A mixed load/sample
test checks that sampling still uses the cube variable. Set `RDNA_KERNEL_DUMP`
to an existing writable directory to save these modules for `spirv-val`.
The `rdna` example `translate_fixture` translates a local binary and serialized
`Spec` for full-program validation; game fixtures must remain outside git.

## Crash containment

Linux source pipeline creation writes and fsyncs a `compile-pending/*.pending`
intent before calling the driver. Only a successful return removes it. An
existing intent prevents that source compile on later launches, including after
SIGKILL or a power loss. The identity includes the driver, translator version,
pipeline key and creation fingerprint so corrected code can be tried. Binary
replay does not incur this journal I/O. Windows retains its existing mechanism.

The fatal note contains the pipeline key, compute program hash and, when
`DS_PIPELINE_CRUMBS=1`, saved SPIR-V filenames. It is copied into preallocated
signal-report storage before entering the driver. Driver faults still add the
attributed key to `crashed.txt`, scoped to the driver/translator on Linux.

This is record-and-skip, not process isolation: a first unknown driver abort can
still terminate the game. A legitimate compile interrupted by shutdown can also
leave an intent, and a skipped pipeline can omit rendering. Keep reports and
markers for diagnosis; do not erase them to make a startup test pass. Explicit
read-only cache probes do not write intents.

The startup relay accepts repeatable `--expect-pipeline c<hex-key>` arguments.
They require successful source creation or binary replay somewhere in the suite;
merely reaching the menu by skipping every bad pipeline is insufficient.
