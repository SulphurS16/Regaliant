# Nexus pipeline reuse and adaptive presentation

Build the runtime and the FSR bridge as described in [linux.md](linux.md). Export
`DS_GAME_DIR`, then run the three-case test with read-only shader and save seeds.

```sh
python3 scripts/linux-nexus-test.py \
  --save-seed /path/to/immutable/nexus-save \
  --shader-seed /path/to/immutable/prepared-shaders
```

1. **Seed, FG off.** Load the prepared store and immediately replay freshly captured
   binaries to check serialization. This extra work affects timing.
2. **Warm, adaptive FG.** Copy the first session’s application and driver caches.
   Require actual binary hits and at most two source creations per stationary 30-second
   window. Both sessions use preparation’s Vulkan device features, with present timing
   extensions disabled. The single end-of-measurement screenshot avoids periodic
   readback stalls.
3. **Validation and vertex census.** Load the Nexus with FG off. Check validation
   output, early after-load screenshots and the path to the nonempty vertex census.
   Instrumented shaders have different binaries. This run has no performance gate.

The suite prints its output directory under `local-data/linux-port/nexus/`. Read
`summary.txt` there and the linked child summaries. The adaptive CSV counts fresh images,
repeated slots and source age, including periods when the compositor holds its previous
buffer. These are application-side measurements. FG defaults to legacy pacing.
