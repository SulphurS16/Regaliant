# 6. Reading the project's vocabulary

| Term | Meaning in this repository |
| --- | --- |
| Guest / host | The guest is the game's console program and the environment it expects. The host is the PC process, OS, driver and hardware providing that environment. |
| ELF / SELF / PRX | ELF describes executable segments and linking information. SELF wraps a module. The parser accepts supported decrypted, uncompressed segments. PRX is the extension used by the two supplied game libraries. |
| HLE / NID | High-level emulation implements a console service at its callable interface. A NID identifies an exported symbol within the library lookup. |
| TLS / fiber | Thread-local storage gives each thread its own instance of selected data. A fiber is an execution context the program can switch within a thread. |
| VMA / direct memory | A virtual memory area records what a guest address range maps and its permissions. Direct memory models separately allocated physical ranges that can be mapped at different virtual addresses. |
| Section / view / alias slot | A section is shared backing storage. A view maps part of it at a virtual address. A second, unprotected alias view lets the GPU import the same bytes. A slot allows an old alias to survive guest remapping until references retire. |
| Import window | A range of host memory exposed to Vulkan. It can supply bytes for a GPU copy without the CPU staging a second copy first. |
| Agc / PM4 / CP | Agc is the guest graphics interface. PM4 is the packet language it submits. The command processor (CP) interprets packets, maintains register state and issues work. The graphics CP is a host CPU thread. |
| RDNA / wave / SGPR / VGPR | RDNA is the guest GPU instruction architecture. A wave executes lanes together. Scalar registers hold wave-wide values, vector registers hold per-lane values, and the execution mask controls participating lanes. |
| Dispatch / draw / batch | A dispatch launches compute work. A draw issues graphics work. A batch groups recorded host GPU commands for submission and ordering. |
| Label / promise / timeline | A label is a value written to guest memory as a synchronization signal. A promise records a value that queued work will publish. A Vulkan timeline semaphore orders the actual device work using increasing completion values. |
| FE2 / mirror / residency | Front end 2 keeps guest-addressed buffers in one sparse Vulkan buffer, the mirror. Residency supplies physical device backing to the portions used. Validity describes whether those backed bytes are current. |
| Placement | The executor's work to make a guest binding usable for the current batch, including mapping lookup, range and freshness checks and any required upload. A memo caches a previously established result subject to its validity rules. |
| Take / epoch / view | A take collects dirty-page information and resets tracking for the queried pages. An epoch identifies a synchronization interval used to stamp clean results. A queue's epoch view determines which stamps it may trust. Here “view” refers to visibility in time, distinct from a mapped section or Vulkan image view. |
| Write watch / uffd | Write watch reports CPU-written pages. Linux's default backend uses userfaultfd asynchronous write protection and the pagemap scan ioctl. Host writes that can bypass guest protection also need explicit accounting. |
| Arming | Protecting a sufficiently quiet guest page so its next CPU write faults. The surviving armed bit allows later checks to skip dirty queries. A write, protection change or remap ends that proof. |
| Prologue / scatter / dword | A prologue orders uploads before a batch uses them. A scatter write-back copies individually marked GPU-written words to guest memory. A dword is 32 bits. |
| C4 / held page / C5 | C4 holds selected GPU-written pages in the mirror, deferring transfer until CPU access. C5 specializes shader stores to skip write marks on held pages. Linux disables C4 even when optional CPU-write arming is enabled. |
| Stretch / held recording / forwarding | A stretch spans commands the recorder tries to process together, sometimes past a label wait. Held recording can be committed once its assumptions hold. Forwarding supplies a result of earlier stream work to later preparation before a normal round trip. A held recording is distinct from a C4 held memory page. |
| Specialization / recipe / pipeline binary | Specialization describes translation choices made for a particular program and resources. A learned recipe lets preparation request the pipeline again. The driver-produced binary can avoid compiling it again on a compatible device and driver. |
| Flip / vblank / present | A flip is the guest's request to display a buffer. Vblank is a display timing event. A present hands an image to the host presentation system. Generated images can add presents without adding guest flips. |
| TAA / HUDless / FG | Temporal anti-aliasing uses information from prior frames. HUDless colour is scene colour before the user interface is composited. Frame generation uses real frames and matching motion, depth and camera history to synthesize an intermediate display image. |
| Tour / relay / warm run | The tour uses the game's automation to visit scenes. A relay runs a controlled session and collects evidence. A warm run reuses preparation or previous visits. Compatibility and binary-hit counters establish what was actually reused. |

For the precise code contracts see [FE2](../../crates/runtime/src/gpu/fe2.rs), [shader store marking](../../crates/runtime/src/gpu/shaders.rs), [sections](../../crates/runtime/src/sections.rs) and [frame generation](../linux-frame-generation.md).

## Terms used in the worked examples and measurement records

| Term | How to read it in this chronicle |
| --- | --- |
| Relocation / addend / jump slot | A relocation tells the loader where to write a value once the module base or symbol address is known. The addend is the constant used by that relocation. An imported jump slot receives the address to which an indirect call should go. |
| LLE export | A definition provided by one of the loaded guest modules. The loader checks these exports before consulting HLE. |
| Trampoline | A small block of executable code that a patched instruction can branch through to carry out an operation the original encoding cannot express on this host. |
| GS binding / red zone | On Linux each entering host thread receives a GS-based area carrying guest state. The red zone is scratch space below the current stack pointer that a signal frame must preserve. These facts matter to the in-process armer. |
| V# / T# / S# | The gfx10 buffer, image and sampler descriptors. V# carries an address, stride, format and extent. T# describes image storage, levels and layout. S# controls how sampling uses coordinates, filtering, address modes and LOD. |
| Mip / LOD / MIN_LOD | A mip is one resolution level of a texture. Level of detail selects which levels to sample. The image descriptor's minimum LOD can limit that selection for a particular view. Samplers also have their own limits. |
| MIMG / TFE / LWE | MIMG denotes the RDNA image instruction encoding. TFE/LWE flags on a sample can request additional status behavior. At `0ee206e` the decoder recognizes the bits and translation warns that it does not produce their residency status. |
| LDS / EXEC / lane election | Local data share is GPU memory visible within a workgroup. EXEC masks the participating lanes of a wave. A lane election chooses one lane to act for several, for example when reserving a range with one atomic add. |
| HLE queue / CP thread / device queue | The guest submits command buffers to an emulated queue, a host CPU thread walks them, and Vulkan receives the recorded result on a device queue. Several guest compute queues can feed the same main device queue. |
| Slot / release / deferred action | A slot owns reusable recording resources and a fence. After device completion those resources can be released while guest-visible actions still await their required publication order. |
| Sparse bind / home / shadow | A sparse bind gives part of a large Vulkan buffer physical backing. Earlier front ends used a persistent device copy called a home or a temporary writable shadow whose changed contents had to be merged back. |
| Scout / dispatcher | Experiments with preparing or recording upcoming work on additional threads. They appear in historical measurements even though their unsuccessful implementations were removed. |
| Fast-clear metadata | A compact representation that allows a surface to be treated as uniformly cleared before individual pixels are stored. Reusing the metadata address for a different image requires tracking which target owns the clear. |
| Detiling / subresource | Detiling rearranges the console's tiled image storage into a layout the upload path can use. A subresource is a particular aspect, mip or layer of an image. An upload report can count the same subresource on several occasions. |
| Remap guard / mm lock | The first is the runtime's coordination between address users and changes of mapping. The second is the kernel's lock for a process's memory map. A thread can hold the first while waiting to acquire the second. |
| Fingerprint / variant / refusal | A fingerprint identifies creation inputs more precisely than a lookup key may do. A variant file retains another compatible binary under that key. A refusal is the driver's decision that a particular binary cannot be replayed. |
| Cold / warm / p99 / critical path | Cold work encounters uncached inputs. Warm measurements reuse a defined store or route. The 99th percentile is a tail measure of frame duration. The critical path is the dependency sequence that actually determines completion time. Aggregate milliseconds on four workers can exceed elapsed frame time. |
