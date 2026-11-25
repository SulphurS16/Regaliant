# Experimental Linux frame generation

Native FSR3 interpolation is optional and off by default. Context creation and generated
frames have been exercised, but visual quality and pacing remain experimental. It does not
enable an upscaler or change the game's render resolution. Whole-run recording
(`DS_PAR_WHOLE_RUN=1`) is incompatible and remains off because it draws incorrectly.

## Backend and build

AMD's raw interpolation and optical-flow APIs produce an intermediate image without owning
presentation. Regaliant retains its Vulkan swapchain and Wayland integration.
[Interpolation API](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/blob/v1.1.4/docs/techniques/frame-interpolation.md),
[optical flow](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/blob/v1.1.4/docs/techniques/optical-flow.md).

The MIT backend is ychwch's unofficial SDK 1.1.4 Linux port, pinned to
`2c90b7f4b877b97e5779aa4d09d338c1e29cb75c`. It supplies native Vulkan host code and
GLSL-to-SPIR-V compilation with glslang, without Wine or a Windows swapchain proxy.
This is not an AMD Linux-support claim.
[Backend source](https://github.com/ychwch/FidelityFX-SDK-Linux/tree/2c90b7f4b877b97e5779aa4d09d338c1e29cb75c).
Regaliant does not integrate native Linux DLSS frame generation; Linux optical-flow support
alone does not supply that integration.
[NVIDIA integration requirements](https://github.com/NVIDIA-RTX/Streamline/blob/main/docs/ProgrammingGuideDLSS_G.md).

Build the ordinary release runtime first. Install a C++20 compiler, CMake, Python3, Vulkan
headers/loader and `glslangValidator`, then:

```sh
python3 scripts/linux-build-framegen.py
DS_FRAME_GENERATION=fsr3 target/release/dsport --app /path/to/PPSA01341-app0
```

The build downloads about 200 MB of pinned source, verifies SHA-256, excludes prebuilt SDK
binaries and builds a C ABI bridge. Allow roughly 10–20 minutes on two compile CPUs, longer
with thermal pauses. It uses the shared CPU-work lock, nice 10 and limited workers. Source
and intermediates stay under `local-data/linux-port/fsr-framegen/`; libraries, license and
provenance go beside the runtime in `target/release/upscalers/`. Preserve that directory
when copying the runtime. `--bridge-only` updates just the bridge/input shader against an
existing SDK build. The current bridge ABI is 2.

Without an environment override, **PC settings → Graphics → Frame Generation → FSR3 2x**
saves the choice. This replaces the unsupported Linux HDR row. `DS_FRAME_GENERATION=off`
forces it off without changing the saved choice. `DS_FSR_FG_LIB` can name a compatible bridge
with its backend beside it. Missing libraries or unsupported inputs leave real frames running;
a library/SDK initialization failure requires a restart. This is SDR only.

FSR builds separate compute pipelines at first activation, using the private NVIDIA/Mesa
driver cache rather than the game's pipeline-binary store. First activation can stall even
with a prepared game store; the ordinary preparation tour does not exercise this presenter.

## Inputs and ordering

The main-view TAA hook supplies depth and motion after program `0x100893aa00`, verified by
code hash `fe9965945c763809`. Resource 2 is reversed depth; resource 4 is RG16F motion in
current-minus-previous UV. The bridge scales it by `[-render_width, -render_height]` for
FSR's previous-minus-current pixels. Camera near/far and FOV come from validated inverse
projection constants. Screen-axis jitter is globals `+0x1c0.xy` times half the render extent;
the title's history-enable bit controls resets. Unreadable/unsupported constants reject that
frame rather than substituting a camera.

Render and output dimensions are independent. Depth/MV must agree and determine render size;
HUDless/final colour determine display size. The wider-than-16:9 cap of 1512 render lines
therefore does not shrink the output. Context keys include both extents and the depth
convention; a change recreates the context and resets history. The bridge passes render size
to preparation and display size to interpolation and optical flow.

The two supported SDR UI-composite programs are recognized by decoded code hash, not their
allocation address. With FG enabled, the title's composite runs once with transparent UI
and copies that result to a private HUDless image. The ordinary composite then overwrites
it with the original scene and UI. Both use the title's filtering and SDR transfer function,
including its A2B10G10R10 output. The presenter copies final colour separately. This costs
one fullscreen draw and a copy; translucent and world-anchored UI still need visual review.

Snapshots follow guest command order and pair with the actual displayed image. Deferred
callbacks use the command buffer passed at playback, not a speculative captured buffer.
Depth copies go through a device-local buffer to preserve bits without an invalid
depth-to-colour image copy. UI composite draws bypass delta/repeat batching; held speculative
recording stops at capture hooks. Snapshots are retained through the interpolation fence.
The SDK dispatches on the existing graphics queue and waits before presenting its output.
Guest flips, completion labels and simulation ticks are unchanged.

The captured motion can contain undefined previous positions. Before SDK dilation, a separate
adapter preserves every valid vector and repairs only non-finite or greater-than-one-UV
components. It chooses a valid 3×3 neighbour within 2% relative device depth, or reconstructs
camera motion from depth and matrices; an invalid previous projection falls back to zero.
This approximates animated-object motion and does not fix the unidentified source draw.
The guest target remains untouched. Raw motion and a recovery-status image are retained for
diagnostics; status-zero pixels must remain bit-identical.

History also resets on skipped/repeated TAA IDs, unmatched real frames, toggles and excessive
input age. SDK scene-change detection remains active. Each TAA snapshot is consumed once.
Pools are bounded to eight depth/MV and eight HUDless snapshots; exhaustion disables FG
instead of recycling live resources. Repeated resizing can require a restart. Resources
remain allocated until exit when disabled; device/SDK failures disable further generation.

## Pacing and acceptance

On 60 Hz FIFO, the presenter schedules 30 real plus 30 generated frames with display
backpressure. It cannot show 48 real plus 48 generated frames on a 60 Hz display. At
120/144 Hz it schedules at most twice the incoming real rate, capped at refresh; cadence can
still be uneven when rates do not divide evenly. Interpolation needs the next real image,
adds GPU work and buffering latency, and does not speed up simulation. The fence-serialized
implementation prioritizes correctness over overlap. No fixed FPS or latency improvement is
guaranteed.

Export `DS_GAME_DIR`; seed directories are copied read-only into private relay outputs:

```sh
# Visible validation/capture run: load a save, 5 seconds stationary, 5 walking.
python3 scripts/linux-framegen-test.py --save-seed /path/to/save-seed \
  --shader-seed /path/to/shader-seed
# Also warm the route and compare FG off/on, 30 seconds stationary + 60 walking.
python3 scripts/linux-framegen-test.py --steady --save-seed /path/to/save-seed \
  --shader-seed /path/to/shader-seed
```

Loading and cooling add to those durations. Runs are visible fullscreen native 4K, scripted,
silent and timeout-bounded. Select the desired desktop refresh first; the relay does not
change it. `--dry-run` prints the plan. The relay updates its runtime/bridge as needed under
the CPU-work lock. A failed short validation prevents long runs.

Outputs are under `local-data/linux-port/framegen/`. Read `summary.txt`, `comparison.json`,
`validation.txt`, `framegen-stages.txt` and the presentation CSVs. Require a created context,
generated completions, active validation with no errors, and visual review. A loaded bridge
alone is not success. `dsport.log` and `framegen-state.json` explain input gates, pairing and
creation failures. `HUDless draw skipped` counts rejected draw probes, not missed display
flips: empty attachments/scissors or asynchronous compilation cannot produce a snapshot.

Compare `generated-*.png` with matching `window-*.png`, and watch moving gameplay for HUD
smear, disocclusion, halos and flicker. `fg-inputs-*/` holds native raw texels, previews and
`inputs.json`: current/previous final and HUDless colour, depth, raw/prepared motion, generated
output, camera constants, formats, extents, history IDs and reset state. Display previews are
opaque: SDK output alpha is an inpainting weight, not display transparency. Raw dumps retain
that alpha and signed/floating-point inputs. These captures contain game data; do not commit them.

```sh
python3 scripts/linux_framegen_compare.py /path/to/fg-inputs-N \
  --output /path/to/separate-report --previews
```

NumPy is required; previews also need Pillow. The report compares RGB against the arithmetic
mean of retained real frames in stored SDR code values, with MAE/RMSE, percentiles and
non-finite counts. This is not motion-compensated ground truth. `--baseline` and
`--max-mae-increase` support same-route numerical regression checks. Invalid prepared motion,
truncated inputs and changes to valid source vectors fail independently of MAE.

Presentation IDs distinguish real/generated KHR completions from guest flips. Snapshot age
and interpolation/completion cost are reported, not physical input-to-photon latency.
Validation, capture, cold pipelines and thermal pauses invalidate performance comparisons;
physical latency requires an external measurement.
