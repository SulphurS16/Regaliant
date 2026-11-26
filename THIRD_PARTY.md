# Third-party components

The port's own code is licensed under the GNU General Public License, version 3 or (at your
option) any later version ([LICENSE](LICENSE)). The components below keep their own licenses,
all of which allow their use in a program under the GPL version 3 (MIT, Apache-2.0, ISC, BSD,
Zlib, 0BSD, CC0).

## Rust crates (fetched by Cargo, not included; versions pinned in `Cargo.lock`)

| Crate | Use | License |
| --- | --- | --- |
| windows-sys / windows | Win32 API bindings (memory, threads, WASAPI and COM, the windows, the launcher's controls and folder picker) | MIT OR Apache-2.0 |
| iced-x86 | x86-64 decoding and encoding for load-time patching (thread-local storage, the SSE4A rewrite, the title fixes) | MIT |
| parking_lot | synchronization primitives | MIT OR Apache-2.0 |
| ash | Vulkan bindings (loads `vulkan-1.dll` at run time) | MIT OR Apache-2.0 |
| rspirv / spirv | SPIR-V module construction in the shader translator (`crates/rdna`) | Apache-2.0 |
| miniz_oxide / adler2 | zlib for screenshots and the launcher's pictures | MIT OR Zlib OR Apache-2.0 / 0BSD OR MIT OR Apache-2.0 |
| serde_json | reading the game's `param.json` | MIT OR Apache-2.0 |
| mimalloc / libmimalloc-sys | the memory allocator (libmimalloc-sys builds Microsoft's mimalloc C sources bundled in it) | MIT |
| zstd / zstd-safe / zstd-sys | the game read from a `.zar` archive and the launcher's Compress (zstd-sys builds Meta's zstd C sources bundled in it) | BSD-3-Clause (zstd-sys: BSD-3-Clause AND BSD-2-Clause; the zstd library: BSD-3-Clause) |
| cc (build dependency) | compiles the vendored C sources below and mimalloc | MIT OR Apache-2.0 |

Their dependencies are MIT and/or Apache-2.0, except libloading (ISC, used by ash to load
`vulkan-1.dll`).

## In this repository

| Component | Location | Use | License |
| --- | --- | --- | --- |
| LibAtrac9 (Alex Barney), C implementation | `third_party/libatrac9` | ATRAC9 audio decoding | MIT (`third_party/libatrac9/LICENSE`) |
| minimp3 (lieff) | `third_party/minimp3/minimp3.h` | MP3 audio decoding | CC0-1.0 (`third_party/minimp3/LICENSE`) |
| Mesa addrlib's gfx10 swizzle patterns and layout logic (Advanced Micro Devices, Inc.) | `crates/runtime/src/gpu/swizzle_tables.rs` (tables generated from `gfx10SwizzlePattern.h`), `crates/runtime/src/gpu/tiling.rs` (a re-implementation) | the GPU's texture and surface layouts | MIT |
| The earlier native port by the same author | much of `crates/runtime`, `crates/sce-elf` and `crates/rdna` derives from it | the loader, system library layers and shader translator | MIT ("Copyright (c) 2026 SulphurS16 and the Bloodborne native port contributors") |
| License texts of crates that ship none | `third_party/licenses` | read by `scripts/notices.py` | iced-x86: MIT (its repository's text), rspirv and spirv: Apache-2.0 (the license text) |

## In the programs built from this repository

`regaliant.exe` and `regaliant-launcher.exe` contain the Rust standard library (MIT OR
Apache-2.0), the crates and components above and the Microsoft C runtime (linked statically).
Their notices are in `THIRD_PARTY_NOTICES.txt` (made by `scripts/notices.py`), and the player
package's `README.txt` carries them after the player instructions, followed by the text of the
GPL.

## Shipped beside the programs (the player package's `upscalers` folder)

Loaded by `regaliant.exe` at run time for the Anti-Aliasing setting, unmodified, from the vendors'
release archives (`scripts/upscalers.ps1`). They are not under the GPL: the port's license adds a
section 7 permission to link with and ship them under their own terms (README, License).

| Component | Files | License (shipped beside them) |
| --- | --- | --- |
| Intel XeSS 3.0.2 | libxess.dll | Intel Simplified Software License (libxess.license.txt, with its third-party notices in libxess.third-party-programs.txt) |
| NVIDIA Streamline 2.14.16 | sl.interposer.dll, sl.common.dll, sl.dlss.dll, sl.dlss_g.dll, sl.reflex.dll, sl.pcl.dll | MIT (streamline.license.txt, with its third-party notices in streamline.3rd-party-licenses.md) |
| NVIDIA Reflex low-latency library (from Streamline 2.14.16) | NvLowLatencyVk.dll | NVIDIA Reflex SDK license (reflex.license.txt) |
| AMD FSR SDK 2.4.0 | amd_fidelityfx_loader_vk.dll, amd_fidelityfx_upscaler_vk.dll | MIT (fidelityfx.license.md, which lists the files under each license) |

Not shipped: NVIDIA DLSS itself (`nvngx_dlss.dll`) and DLSS Frame Generation (`nvngx_dlssg.dll`),
both under NVIDIA's RTX SDKs license. Players who want them put their own copies in the folder. `regaliant.exe` uses it through Streamline after
checking that it carries NVIDIA's signature. The section 7 permission covers linking with it.

NVIDIA DLSS, NVIDIA Reflex, G-SYNC and NVIDIA Streamline are trademarks and/or registered trademarks of NVIDIA
Corporation. Intel XeSS is a trademark of Intel Corporation. AMD FSR and FidelityFX are
trademarks of Advanced Micro Devices, Inc.

## Used from Windows at run time (not redistributed)

Vulkan (`vulkan-1.dll`, installed with the graphics driver), WASAPI, XInput, Raw Input, the
common controls and the Universal C Runtime.

## The game

Demon's Souls is © Sony Interactive Entertainment Inc. The port contains none of its data or
code. Each player provides their own copy.
