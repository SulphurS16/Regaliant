# Whole-run recording experiment

The renderer already splits synchronization-free PM4 runs between gpu-gfx and its
`gpu-gfx-par` workers. Workers own their command processor, register snapshot, Vulkan
command pool and deferred executor. They record primary batches concurrently, then submit
in guest order. This experiment extends that path; it does not introduce secondary command
buffers or reorder GPU execution.

`DS_PAR_WHOLE_RUN=1` also hands the first chunk to a worker. gpu-gfx immediately enters the
existing collect/look-ahead path instead of recording that chunk itself. With `past_runs`
enabled it can plan the following range while the workers record. Otherwise it waits for
the chunks. The number of chunks, minimum size and worker count are unchanged: the first
chunk's work is redistributed over the existing worker queue, not an extra thread.

The first implementation only changes non-held runs on the main graphics command processor
which the existing planner accepts and which have **no indirect arguments, label waits,
CP writes or flush/cache actions**. Ordinary direct draws and direct dispatches may be
included. Register writes are walked serially to generate exact start/end snapshots.
The existing packet scanner still defines every range boundary. Live/fetched copies and
capture sessions retain their serial fallback. Compute queues and held runs are unchanged.

Ordering and failure rules remain the existing ones:

* Resolve pending fast clears, watched resources and prior unsubmitted writes before workers
  start. Whole runs also submit prior label/stamp-only batches before handing off, since
  there is no local first chunk to own those effects. Grant submission of the first worker only after the preceding main batch is submitted.
* Each worker records an independent primary batch; its resource barriers, uploads, FE2 view
  checks and reads of earlier writes still run. Recording independence does not imply absence
  of GPU memory dependencies.
* Grant turns and submit batches in command order. A stopped chunk is resumed on gpu-gfx at
  the exact stopped packet. Later labels/effects depend on the collected batch sequence.
* Look-ahead work remains deferred. Validate its end state, CPU reads against run writes,
  pending clears/reloads and stopped chunks before commit; discard/replay on failure.
* Never execute the handed-off first packet twice. Return to collect immediately after
  planning a run whose first worker starts at the current packet.

The switch defaults off on all platforms. Unit tests check command coverage and ordering
hazard exclusion; they cannot establish GPU correctness or a speedup. Logs count whole runs
and commands. Use the same warmed seed for a cooled A/B/A/B measurement:

```sh
python3 scripts/linux-steady.py --record-candidate whole-run \
  --save-seed /path/to/save-seed --shader-seed /path/to/shader-seed
```

Set `DS_GAME_DIR` first. The relay pins the switch separately in each child, keeps mprotect,
validation/captures off during timed phases, and emits adjacent CPU profiles and boundary
PNGs. Compare `recording-comparison.json`, temperatures, gpu-gfx/worker CPU, parallel wait
and fallback counts, and images. A candidate with zero whole runs did not exercise the path.
Do not enable it in normal play until both performance and image comparisons pass.

This could move useful work off gpu-gfx, but can also lose performance: four workers must
absorb what five recorders previously did. Saved main-thread time can be replaced by waits,
resource-lock contention or chunk setup. There is no projected FPS gain before the relay.
