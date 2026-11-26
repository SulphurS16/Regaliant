//! Patches of the title's own code (PPSA01341 01.007.000): stand-ins for
//! behavior this port does not reproduce yet, fixes of title bugs the port's
//! runs reach, adaptations to a shared PC (`jobidle`) and PC controls
//! (`camhook`). Each fix checks the original bytes first (another
//! title version is left alone and the mismatch logged), every applied fix is
//! logged, and `DS_TITLEFIX=0` turns them all off (to investigate the
//! underlying problem).

use std::sync::atomic::{AtomicU64, Ordering};

enum Change {
    /// Instructions that run before the ones `at` bytes into `from` (in a
    /// trampoline, `patch::Patcher::insert`), built at load time (they may
    /// name the fix's run counter).
    Insert { at: usize, code: fn(counter: u64) -> Vec<u8> },
    /// A host function called with the registers before the instructions
    /// `at` bytes into `from` (`patch::Patcher::hook`).
    Hook { at: usize, handler: extern "sysv64" fn(*const u64, u32) },
}

struct Fix {
    name: &'static str,
    /// Module-relative address of the instructions the fix expects in the main executable.
    offset: usize,
    /// Their length and FNV-1a hash (the game's own bytes are not kept in the port's source).
    from: (usize, u64),
    change: Change,
    kind: &'static str,
    why: &'static str,
    /// Applied only when one of these environment variables (`|`-separated) is set
    /// (diagnosis); `!NAME`: only when NAME is not set.
    only_with: Option<&'static str>,
}

/// How many times each fix's inserted code ran (`status`; hooks keep their
/// own statistics).
static RUNS: [AtomicU64; FIXES.len()] = [const { AtomicU64::new(0) }; FIXES.len()];
/// Which fixes were applied.
static APPLIED: [std::sync::atomic::AtomicBool; FIXES.len()] = [const { std::sync::atomic::AtomicBool::new(false) }; FIXES.len()];

/// The stand-in used until 2026-10-11 (a byte patch) kept static meshes off
/// the GPU-driven batches (eboot+0xa21e77 `jne` -> `nop; jmp`), which drew
/// nothing here: the title's batch culling stores compare ballots, which the
/// shader translation now builds completely (STATUS: architecture).
const FIXES: &[Fix] = &[
    Fix {
        name: "collision mesh unload keeps a slot index",
        // BPCollisionMeshManager's unload (eboot+0xd51f0) moves components whose
        // collision body it releases back to the pending list
        // (`mov rcx,[rcx+28h]; cmp r12,[rbx+270h]`, rcx = the component) but
        // leaves their slot index (+0x33C) set. CollisionMeshBodyDestroy
        // (eboot+0xd4780) trusts that index: once the slot belongs to another
        // component, destroying this one releases the other's body and leaves
        // this one in the pending list, freed, for the next level's
        // loading-completed job to read (eboot+0xd4e16: World 8's levels
        // toured with three locations each, then the Nexus or 1-1; FINDINGS.md
        // 2026-10-19 night). The inserted code clears the index there, as the
        // destroy's own slot path does (eboot+0xd48f4).
        offset: 0xd5372,
        from: (11, 0xa66232c848f2b6bb),
        change: Change::Insert { at: 4, code: cmesh_unload_code },
        kind: "title bug",
        why: "components the collision mesh unload returns to the pending list lose their slot index",
        only_with: None,
    },
    // BPE's wait for work (eboot+0x83b640), where its scan of this worker's
    // job counters found nothing: `pause; mov r15,[rbx+r14*8+0E10h]`.
    Fix {
        name: "idle job workers sleep (wait)",
        offset: 0x83b6fd,
        from: (10, 0x6404c4c949f860c6),
        change: Change::Hook { at: 0, handler: crate::jobidle::wait_hook },
        kind: "shared-PC adaptation, `tune job_idle`",
        why: "the title's 13 job workers poll for work on 13 logical CPUs at all times",
        only_with: None,
    },
    // BPE's take (eboot+0x83b880), at the end of its pass over one queue
    // (`add r15,9220h`; the last queue's end is a take that found nothing).
    Fix {
        name: "idle job workers sleep (take)",
        offset: 0x83bf17,
        from: (7, 0x71e3819aac9c152e),
        change: Change::Hook { at: 0, handler: crate::jobidle::take_hook },
        kind: "shared-PC adaptation, `tune job_idle`",
        why: "jobs whose conditions do not hold yet are taken and put back by every worker without pause",
        only_with: None,
    },
    // The worker loop (eboot+0x83b520) running a job it took: `mov rdi,rbx;
    // mov rsi,r15` before the call.
    Fix {
        name: "idle job workers sleep (run)",
        offset: 0x83b570,
        from: (6, 0xe2b7f7c3f95da47a),
        change: Change::Hook { at: 0, handler: crate::jobidle::run_hook },
        kind: "shared-PC adaptation, `tune job_idle`",
        why: "a worker that runs a job is busy again",
        only_with: None,
    },
    // Raw mouse look (`camhook`, FINDINGS.md 2026-10-21 evening): the camera
    // stick reader (eboot+0x18bd60) after it loaded the stick into rdx: `mov
    // edi,[rbx+74h]; mov rsi,[rbx]` (rbx the camera's work context).
    Fix {
        name: "raw mouse look (stick)",
        offset: 0x18bd99,
        from: (6, 0x6b3b326505e11815),
        change: Change::Hook { at: 0, handler: crate::camhook::stick_hook },
        kind: "PC controls, `mouse_look_raw`",
        why: "mouse look through the right stick inherits the stick's response curve and top speed",
        only_with: None,
    },
    // ... and the end of the stage that adds the stick to the camera's pitch
    // and yaw (eboot+0x18cf00), which all of its paths reach: `mov
    // rax,[r15]; cmp rax,[rbp-20h]` (its stack guard check; rbx the context).
    Fix {
        name: "raw mouse look (angles)",
        offset: 0x18d1c4,
        from: (7, 0xd82de55227e30b5e),
        change: Change::Hook { at: 0, handler: crate::camhook::apply_hook },
        kind: "PC controls, `mouse_look_raw`",
        why: "the mouse turns the camera by its counts, not through the stick",
        only_with: None,
    },
    // ... and the bow's precision aim (`camhook::aim_hook`): after its stick stage stored the
    // aim's pitch and yaw (eboot+0x17692e), `vmovss xmm1,[rbx+9Ch]` (rbx the aim controller).
    Fix {
        name: "raw mouse look (bow aim)",
        offset: 0x176936,
        from: (8, 0xd74d122ae79883ef),
        change: Change::Hook { at: 0, handler: crate::camhook::aim_hook },
        kind: "PC controls, `mouse_look_raw`",
        why: "the mouse aims the bow by its counts, not through the stick",
        only_with: None,
    },
    // Diagnosis (`DS_CAMDEBUG=1`): the camera rig update's entry
    // (eboot+0x181b00: `push rbp; mov rbp,rsp; push r15`), rdi the rig: the
    // rigs seen, for the `cam` control command.
    Fix {
        name: "camera rigs (diagnosis)",
        offset: 0x181b00,
        from: (6, 0x82d75a35aabe38aa),
        change: Change::Hook { at: 0, handler: crate::camhook::rig_hook },
        kind: "diagnosis",
        why: "the `cam` command lists the camera rigs and their controllers",
        only_with: Some("DS_CAMDEBUG"),
    },
    // The virtual geometry scene (`vgdiag`): the per-frame update before the active entity
    // list is hashed into CreateScene's generation (`mov rdi,[rbp-0E8h]; mov r8d,811C9DC5h`,
    // the FNV-1a loop's start), where a held spawn enable goes back (and with `DS_VG_DIAG` the
    // list is logged) ...
    // The job threads' reads of the culled index counts (`cmp dword ptr [r10+rbx*4], 0`,
    // `cullread`): a read waits for the GPU work in flight that writes the counts, and a record
    // read non-zero in the last two flips is not dropped on a 0 read between the GPU's clear of
    // the counts and their refill (the intro's sky flashes, walls missing for a frame) ...
    Fix {
        name: "culling results read after the GPU's writes, held over their clear",
        offset: 0xad91de,
        from: (5, 0x0731273bf58b8b71),
        change: Change::Insert { at: 0, code: crate::cullread::hold_code },
        kind: "title timing",
        why: "the CPU read the counts while the GPU's culling was still landing; a visible object then missed the depth pre-pass",
        only_with: Some("!DS_CULLREAD_DIAG"),
    },
    // ... or, with `DS_CULLREAD_DIAG`, a hook with the same hold and a log of the reads.
    Fix {
        name: "culling results read (diagnosis)",
        offset: 0xad91de,
        from: (5, 0x0731273bf58b8b71),
        change: Change::Hook { at: 0, handler: crate::cullread::read_hook },
        kind: "diagnosis",
        why: "logs what the title's CPU reads of its GPU culling results",
        only_with: Some("DS_CULLREAD_DIAG"),
    },
    // The title's CPU walks of its GPU frustum culling masks (`cullread`, the view masks): the
    // index culls per object set (eboot+0xa1f560, `cmp qword ptr [rdi+38h],0`), the culling reads
    // per set (eboot+0xa1f020, `push rbp; mov rbp,rsp; push r15`) and the draws per set
    // (eboot+0xa1f290, `push r14; push rbx; mov rax,rdi`) first wait for the GPU's writes of the
    // masks they walk ...
    Fix {
        name: "view masks walked after the GPU's writes (culls)",
        offset: 0xa1f560,
        from: (5, 0xa4d780dd637582e9),
        change: Change::Hook { at: 0, handler: crate::cullread::mask_wait_cull },
        kind: "title timing",
        why: "the culls walked the GPU's frustum culling masks while our GPU's writes were still landing; culls and draws then numbered objects differently",
        only_with: None,
    },
    Fix {
        name: "view masks walked after the GPU's writes (reads)",
        offset: 0xa1f020,
        from: (6, 0x82d75a35aabe38aa),
        change: Change::Hook { at: 0, handler: crate::cullread::mask_wait_read },
        kind: "title timing",
        why: "the culling reads walked the GPU's frustum culling masks before our GPU's writes landed",
        only_with: None,
    },
    Fix {
        name: "view masks walked after the GPU's writes (draws)",
        offset: 0xa1f290,
        from: (6, 0x7658a7ca3bc5a22c),
        change: Change::Hook { at: 0, handler: crate::cullread::mask_wait_draw },
        kind: "title timing",
        why: "the draws walked the GPU's frustum culling masks before our GPU's writes landed",
        only_with: None,
    },
    // ... and the static sections' decal draws (eboot+0xa20cae, `mov eax,[rdi+198h]`, past the
    // walk's early out) wait for the GPU's writes of the section's decal mask (`cullread::
    // decal_mask_wait`, 2026-11-20: the equipment menu's wall flicker).
    Fix {
        name: "decal masks walked after the GPU's writes",
        offset: 0xa20cae,
        from: (6, 0xa7d2d964b481f8b4),
        change: Change::Hook { at: 0, handler: crate::cullread::decal_mask_wait },
        kind: "title timing",
        why: "the static decals' draws walked the GPU's decal visibility mask before our GPU's writes landed; walls lost their decals for a frame",
        only_with: None,
    },
    // The shader preparation's self-play (`prepplay`): the levels' tour opens with a 120 s wait
    // of the title's automation; once the self-play sees the game in play that wait ends
    // (`prepplay::automation_wait_hook` at the automation's wait step, eboot+0xaf7bf0: `vmovss
    // xmm0,[rdi]; vsubss xmm0,xmm0,[rsi+8]`). Inert in any other session.
    Fix {
        name: "the self-play's opening wait ended once in play",
        offset: 0xaf7bf0,
        from: (9, 0xc4fca86b5b174336),
        change: Change::Hook { at: 0, handler: crate::prepplay::automation_wait_hook },
        kind: "self-play",
        why: "the levels' tour waited 120 s of the title's time from boot, ~148 s after the self-play had reached the game",
        only_with: None,
    },
    Fix {
        name: "VG spawning held after scene resets (per frame)",
        offset: 0xa553c7,
        from: (13, 0x9cb16a09a6dccd4c),
        change: Change::Hook { at: 0, handler: crate::vgdiag::list_hook },
        kind: "title timing",
        why: "a held spawn enable goes back once the scene has kept its numbering for a few frames",
        only_with: None,
    },
    // ... the reset of the scene's model numbering a new generation makes
    // (`mov eax,[rbp-131Ch]; mov byte ptr [r14+1],0`) ...
    Fix {
        name: "VG spawning held after scene resets (reset)",
        offset: 0xa656dc,
        from: (11, 0xeb41e0551ea737db),
        change: Change::Hook { at: 0, handler: crate::vgdiag::reset_hook },
        kind: "title timing",
        why: "plants spawned with a scene numbering a reset replaces a frame later keep the old model numbers (weeds drawn as tall grass out of the tutorial tunnel's walls)",
        only_with: None,
    },
    // ... and the spawn enable CreateScene stores (`mov [r15+1],al; mov rsi,[r15+0B0h]`),
    // held off for the frames after a reset.
    Fix {
        name: "VG spawning held after scene resets (enable)",
        offset: 0xa66b93,
        from: (11, 0xda3d353d8164bd25),
        change: Change::Hook { at: 0, handler: crate::vgdiag::enable_hook },
        kind: "title timing",
        why: "spawning waits until the scene's model numbering has held for a few frames",
        only_with: None,
    },
    // Diagnosis: the VG spawn grid's new centre stored (`vmovss [rdi+20h],xmm2; vmovss
    // [rdi+24h],xmm1`), the rebuild of the frame's spawn data marked after it.
    Fix {
        name: "VG spawn grid's centre changes logged",
        offset: 0xa5c37d,
        from: (10, 0xc67eedbe37e67377),
        change: Change::Hook { at: 0, handler: crate::vgdiag::cell_hook },
        kind: "diagnosis",
        why: "the spawn grid's centre cell changes, with the camera's position (DS_VG_DIAG)",
        only_with: Some("DS_VG_DIAG"),
    },
    // The VirtualGeometryRendering constructor after its budget stores (`mov r13,[r12+28h]`,
    // after `mov dword [r12+18h],24000h` and a call that reads only its locals).
    Fix {
        name: "VG instance pool raised",
        offset: 0xa54223,
        from: (5, 0x0b5567a3b0018a0b),
        change: Change::Hook { at: 0, handler: crate::vgdiag::pool_hook },
        kind: "title budget",
        why: "the views' plant capacities overran the 147,456-instance pool near a level view (the tutorial courtyard's leaves vanished in view as the camera tilted)",
        only_with: None,
    },
    // The TAA's per-view jitter (`jitter`): its function's entry, `mov rax,[rdi+68h]; cmp
    // dword ptr [rax+0AACh],0` (the forced index's test).
    Fix {
        name: "TAA jitter noted",
        offset: 0x9ed8d0,
        from: (11, 0x93065f1746a483d6),
        change: Change::Hook { at: 0, handler: crate::jitter::hook },
        kind: "PC settings: DLSS, FSR, XeSS",
        why: "the upscalers need each frame's jitter, which the title keeps in its projection",
        only_with: None,
    },
    // The render resolution (`dispmode`, `DS_DISPLAY_MODE`): the per-frame
    // display mode choice (eboot+0x9fe6a3..), `test ecx,ecx; cmovne esi,ecx`
    // before `test eax,eax; cmovne esi,eax` (eax the options' mode).
    Fix {
        name: "display mode held (render resolution)",
        offset: 0x9fe6a8,
        from: (5, 0x64be9b4d9841022e),
        change: Change::Hook { at: 0, handler: crate::dispmode::hook_display },
        kind: "PC settings: Render Scale, `DS_DISPLAY_MODE`",
        why: "the title's render resolution follows its options (2560x1440 at the gate) and no PC setting",
        only_with: None,
    },
    // ... the same choice kept as the effective mode (eboot+0xb26030, +0xD28),
    // the same instructions ...
    Fix {
        name: "display mode held (effective mode)",
        offset: 0xb2606e,
        from: (5, 0x64be9b4d9841022e),
        change: Change::Hook { at: 0, handler: crate::dispmode::hook_rax },
        kind: "PC settings, `DS_DISPLAY_MODE`",
        why: "the effective mode follows the options",
        only_with: None,
    },
    // ... and the UI's 4K output test (eboot+0xde88c0): `test r15d,r15d; mov
    // rcx,[rdx]` before `cmovne eax,r15d` (r15d the options' mode).
    Fix {
        name: "display mode held (UI scale)",
        offset: 0xde88c0,
        from: (6, 0x009edbe6ece0abb9),
        change: Change::Hook { at: 0, handler: crate::dispmode::hook_r15 },
        kind: "PC settings, `DS_DISPLAY_MODE`",
        why: "the UI's scale follows the options' mode (4K for modes 3-5)",
        only_with: None,
    },
    // ... and the frame's copy of the settings (eboot+0xa4f2dd: memcpy of
    // 0xD8A bytes to r14+0x390), after it: `mov edx,1`.
    Fix {
        name: "display mode held (frame settings)",
        offset: 0xa4f2e9,
        from: (5, 0x5019e61086cedfec),
        change: Change::Hook { at: 0, handler: crate::dispmode::hook_snapshot },
        kind: "PC settings, `DS_DISPLAY_MODE`",
        why: "the frame's copy of the settings keeps the options' mode",
        only_with: None,
    },
    // The render size within a mode (`dispmode`, `DS_RENDER_SIZE`; and the
    // output size, `DS_OUTPUT_SIZE`): set display mode (eboot+0x90a600)
    // after it built the sizes for the render resolution manager: `lea
    // rdi,[rbx+0A8h]` before its call.
    Fix {
        name: "render size",
        offset: 0x90a6fc,
        from: (7, 0x1ff36c6035ccf317),
        change: Change::Hook { at: 0, handler: crate::dispmode::hook_render_size },
        kind: "PC settings, `DS_RENDER_SIZE`",
        why: "the title renders at its mode's size (2560x1440 at the gate) and no PC setting",
        only_with: None,
    },
    // The output size (`dispmode`: the Aspect Ratio setting, `DS_OUTPUT_SIZE`;
    // the hooks do nothing at the title's own size): the video-out
    // constructor's size block before its use, `mov rsi,r15; mov rdi,[rax]` ...
    Fix {
        name: "output size (constructor)",
        offset: 0x90b278,
        from: (6, 0x2247b521bcdead6b),
        change: Change::Hook { at: 0, handler: crate::dispmode::hook_ctor_sizes },
        kind: "PC settings (Aspect Ratio), `DS_OUTPUT_SIZE`",
        why: "the title outputs 3840x2160 whatever the display",
        only_with: None,
    },
    // ... its two display targets, after `mov edi,0F00h; mov esi,870h`:
    // `mov edx,1` ...
    Fix {
        name: "output size (display target 0)",
        offset: 0x90af7e,
        from: (5, 0x5019e61086cedfec),
        change: Change::Hook { at: 0, handler: crate::dispmode::hook_targets },
        kind: "PC settings (Aspect Ratio), `DS_OUTPUT_SIZE`",
        why: "the display targets are made at 3840x2160",
        only_with: None,
    },
    Fix {
        name: "output size (display target 1)",
        offset: 0x90b06b,
        from: (5, 0x5019e61086cedfec),
        change: Change::Hook { at: 0, handler: crate::dispmode::hook_targets },
        kind: "PC settings (Aspect Ratio), `DS_OUTPUT_SIZE`",
        why: "the display targets are made at 3840x2160",
        only_with: None,
    },
    // ... the current mode's size getter (eboot+0x90ab90) before it stores
    // the output size: `mov [rsi+0Ch],rdx; mov [rsi+14h],r9d` ...
    Fix {
        name: "output size (size getter)",
        offset: 0x90ac57,
        from: (8, 0xe366030876dbf04d),
        change: Change::Hook { at: 0, handler: crate::dispmode::hook_size_getter },
        kind: "PC settings (Aspect Ratio), `DS_OUTPUT_SIZE`",
        why: "the mode's sizes come from per-mode tables (the UI's target among their users)",
        only_with: None,
    },
    // ... and the UI's canvas before its layout (eboot+0xde8680): `lea
    // rsi,[rbp-30h]; mov rdi,rbx`.
    Fix {
        name: "output size (UI canvas)",
        offset: 0xde85db,
        from: (7, 0xa7e4cfea432e97d5),
        change: Change::Hook { at: 0, handler: crate::dispmode::hook_ui_canvas },
        kind: "PC settings (Aspect Ratio), `DS_OUTPUT_SIZE`",
        why: "the UI is laid out on a 3840x2160 canvas in modes 3-5",
        only_with: None,
    },
    // The title's handler of a changed setting (eboot+0xb4700, an observer of
    // its settings registry; rsi the setting): `push rbp; mov rbp,rsp; push
    // rbx; push rax`. The PC settings learn every change there (`pcsettings`).
    Fix {
        name: "setting changes (PC settings)",
        offset: 0xb4700,
        from: (6, 0x83070935aae6d2c9),
        change: Change::Hook { at: 0, handler: crate::pcsettings::changed_hook },
        kind: "PC settings",
        why: "the port applies its settings when the title's SETTINGS screen changes them",
        only_with: None,
    },
    // The PC settings' values at the title's own settings paths (`pcsettings::store`): the
    // menu bindings' scans of the store (`UISetting_<type>::Evaluate`) for the PC keys ...
    // The button prompts (`pcsettings::prompts`): the text renderer's inline-image draw
    // (eboot+0x150e380) after its map from a text's button hash to a glyph entity's name hash
    // (r15d), before the entity's lookup: `lea rcx,[rbp-90h]`.
    // The menu pointer (`pcsettings::pointer`): the menu list's selection change
    // (UIBehaviorMenuList, eboot+0xe2a8d0: xmm0 the move's direction, rdi the list, esi an
    // item index used when the direction is zero, as the list's own focus handler calls it),
    // at its entry: `push rbp; mov rbp,rsp; push r15`.
    Fix {
        name: "menu pointer selects directly",
        offset: 0xe2a8d0,
        from: (6, 0x82d75a35aabe38aa),
        change: Change::Hook { at: 0, handler: crate::pcsettings::pointer::select_hook },
        kind: "PC controls",
        why: "the mouse pointer selects the item under it at once instead of stepping toward it",
        only_with: None,
    },
    // ... and the input grid's cell selection (UIBehaviorInputGrid, eboot+0xf648b0: rdi the
    // grid, esi the column, edx the row), at its entry: `push rbp; mov rbp,rsp; push r15`.
    Fix {
        name: "menu pointer selects grid cells directly",
        offset: 0xf648b0,
        from: (6, 0x82d75a35aabe38aa),
        change: Change::Hook { at: 0, handler: crate::pcsettings::pointer::grid_hook },
        kind: "PC controls",
        why: "the mouse pointer selects the grid cell under it at once instead of stepping toward it",
        only_with: None,
    },
    Fix {
        name: "button prompts (glyphs)",
        offset: 0x150e645,
        from: (7, 0x5852fe0d91453bd2),
        change: Change::Hook { at: 0, handler: crate::pcsettings::prompts::glyph_hook },
        kind: "PC controls",
        why: "the title's text shows PlayStation buttons only: keyboard, mouse and Xbox glyphs stand in for them",
        only_with: None,
    },
    Fix {
        name: "PC settings (bool lookups)",
        offset: 0xc3a8e5,
        from: (8, 0xc752abb0241b5555),
        change: Change::Hook { at: 0, handler: crate::pcsettings::store::eval_bool_hook },
        kind: "PC settings",
        why: "the achievement pop-up is shown by a setting the title's store does not hold",
        only_with: None,
    },
    Fix {
        name: "PC settings (int lookups)",
        offset: 0xc24245,
        from: (8, 0xd03f73fd5c6e9315),
        change: Change::Hook { at: 0, handler: crate::pcsettings::store::eval_int_hook },
        kind: "PC settings",
        why: "the title's store holds only the settings it registers",
        only_with: None,
    },
    Fix {
        name: "PC settings (float lookups)",
        offset: 0xc24d15,
        from: (7, 0x75b77c455c297c78),
        change: Change::Hook { at: 0, handler: crate::pcsettings::store::eval_float_hook },
        kind: "PC settings",
        why: "the title's store holds only the settings it registers",
        only_with: None,
    },
    // ... and the menu's setters (`mov qword ptr [rbp-48h],0` after each loads the store).
    Fix {
        name: "PC settings (int changes)",
        offset: 0xfa5cba,
        from: (8, 0x484e6b386f04fc65),
        change: Change::Hook { at: 0, handler: crate::pcsettings::store::set_int_hook },
        kind: "PC settings",
        why: "the port keeps and applies the PC settings the menu changes",
        only_with: None,
    },
    Fix {
        name: "PC settings (float changes)",
        offset: 0xfa5ee5,
        from: (8, 0x484e6b386f04fc65),
        change: Change::Hook { at: 0, handler: crate::pcsettings::store::set_float_hook },
        kind: "PC settings",
        why: "the port keeps and applies the PC settings the menu changes",
        only_with: None,
    },
    // The engine's frame allocator init (eboot+0x8114a0, `TaggedFrameAllocator`
    // with its arena: rdi, rsi): `mov byte ptr [rdi+8],1; lea rcx,[rdi+0C8h]`.
    Fix {
        name: "engine map (frame allocators)",
        offset: 0x8114a0,
        from: (11, 0x2e28f44da0de2f7b),
        change: Change::Hook { at: 0, handler: crate::engine::frame_alloc_hook },
        kind: "engine map (read only)",
        why: "the port learns the frame allocators and the GPU arena (front end 4)",
        only_with: None,
    },
    // The frame allocator's placement of an allocation (eboot+0x8121b2,
    // `lea rcx,[rax+r12+7]`: rax the start, r12 the size, r15 the allocator).
    Fix {
        name: "engine map (arena allocations)",
        offset: 0x8121b2,
        from: (5, 0x041cfc8b570b0201),
        change: Change::Insert { at: 0, code: crate::engine::alloc_insert_code },
        kind: "engine map (read only)",
        why: "the port records the GPU arena's allocations per MiB (front end 4's allocation table)",
        only_with: None,
    },
    // The arena's tag free with its lock held (eboot+0x7fe5d4,
    // `mov rax,[r8+728h]`: r8 the arena, r12 the tag).
    Fix {
        name: "engine map (arena tag frees)",
        offset: 0x7fe5d4,
        from: (7, 0x9ea495a12cbad53e),
        change: Change::Hook { at: 0, handler: crate::engine::arena_free_hook },
        kind: "engine map (read only)",
        why: "the port marks the GPU arena's freed units (front end 4's allocation table)",
        only_with: None,
    },
    // After the frame allocator placed an allocation (eboot+0x8121b7, `xor
    // edx,edx; and rcx,-8`: rax the start, r12 the size, r15 the allocator,
    // rbp its frame).
    Fix {
        name: "engine map (arena allocation census)",
        offset: 0x8121b7,
        from: (6, 0x9f059237e9db26e6),
        change: Change::Hook { at: 0, handler: crate::engine::alloc_census_hook },
        kind: "engine map (read only)",
        why: "the port counts the GPU arena's allocations by their call site (census)",
        only_with: Some("DS_ALLOC_CENSUS"),
    },
    // The title's texture streaming: CTexture::SetCurrentStreamingChunk after
    // its memcpy into the texture's memory (eboot+0x901465, `mov
    // [r14+0B8h],ebx`: r14 the texture, ebx the chunk index; the `je` at
    // eboot+0x9013c5 goes past it to eboot+0x90146c).
    Fix {
        name: "engine map (texture streaming)",
        offset: 0x901465,
        from: (7, 0xe317909174e15e3b),
        change: Change::Hook { at: 0, handler: crate::engine::tex_chunk_hook },
        kind: "engine map (read only)",
        why: "the port learns when a streamed texture chunk's data is complete (census)",
        only_with: Some("DS_TEX_STREAM"),
    },
    // RAD's sound system: a voice's speaker-gain setter (eboot+0x1e4b340, `cmp edx,8; mov
    // ecx,8`: rdi the voice, rsi a speaker-by-channel gain matrix, edx the speaker count).
    // The title's movie player routes Bink track k to speaker k alone (eboot+0x981dec: a 5.1
    // movie as six mono tracks), so the one movie with sound, the PlayStation Studios logo
    // (one stereo track), played its left channel from the front left and lost its right.
    Fix {
        name: "stereo movie sound on both front speakers",
        offset: 0x1e4b340,
        from: (8, 0x3933a16e8590f353),
        change: Change::Hook { at: 0, handler: crate::hle::libs::devices::rad_speaker_hook },
        kind: "title bug",
        why: "a stereo movie track routed like a 5.1 movie's mono tracks keeps only its left channel",
        only_with: None,
    },
    // The heaps' allocate's one epilogue (eboot+0x802c2a, `mov rax,rbx; add
    // rsp,28h`, a branch target: rbx the block or 0, r12 the heap or the
    // heap + 0x20, [rbp-0x40] the size asked for, rbp its frame).
    Fix {
        name: "engine map (heap census)",
        offset: 0x802c2a,
        from: (7, 0xe655972d0f24a80f),
        change: Change::Hook { at: 0, handler: crate::engine::heap_census_hook },
        kind: "engine map (read only)",
        why: "the port counts the engine heaps' blocks by their call site (census)",
        only_with: Some("DS_HEAP_CENSUS"),
    },
    // The engine's readback helper's entry (eboot+0x909970, `xor eax,eax;
    // cmp dword ptr [rdi+1Ch],1Fh`: rdi the V#, esi the size, rdx the
    // destination, ecx the offset; the `je` after reads the cmp's flags).
    Fix {
        name: "engine map (readback census)",
        offset: 0x909970,
        from: (6, 0x46dff7a8af79963b),
        change: Change::Hook { at: 0, handler: crate::engine::rb_census_hook },
        kind: "engine map (read only)",
        why: "the port counts the engine's readbacks of GPU memory by caller and allocator (census)",
        only_with: Some("DS_RB_CENSUS"),
    },
    // A static model section's hand-off of a mesh item to the GPU-driven batches
    // (eboot+0xa21e10) at its program check (`mov rax,[rbx+20h]; cmp dword ptr [rax+3D8h],4`,
    // then `jne` past the hand-off): `DS_CPU_DRAW_PATH=1` keeps every item on the title's CPU
    // draw path (the stand-in of 2026-10-10; it loses geometry: diagnosis only); with
    // `DS_CPUDRAW_CENSUS=1` the census of both paths (`cpudraw`; FINDINGS.md 2026-11-16).
    Fix {
        name: "static meshes on the CPU draw path",
        offset: 0xa21e6c,
        from: (11, 0xd65451942f7c6b6b),
        change: Change::Hook { at: 0, handler: crate::cpudraw::handoff_hook },
        kind: "rendering path (diagnosis)",
        why: "the census of the title's static mesh hand-off, or its items on the CPU draw path (DS_CPU_DRAW_PATH=1)",
        only_with: Some("DS_CPU_DRAW_PATH|DS_CPUDRAW_CENSUS"),
    },
    // The same function's epilogue (`add rsp,68h; pop rbx`): `DS_CPU_DRAW_PATH` 3 and 4 give
    // an item it handed off to the CPU draw path too (`cpudraw::epilogue_hook`; 4 loses
    // geometry: diagnosis only).
    Fix {
        name: "static meshes on both draw paths",
        offset: 0xa222a8,
        from: (5, 0xf107f2dab9ac2fc1),
        change: Change::Hook { at: 0, handler: crate::cpudraw::epilogue_hook },
        kind: "rendering path (diagnosis)",
        why: "static mesh items the title hands to its GPU-driven batches are given to its CPU draw path too",
        only_with: Some("DS_CPU_DRAW_PATH"),
    },
    // The end of the Agc library's command-buffer init (eboot+0x1d0a4fb,
    // `mov dword ptr [rax+30h],0`: rax the buffer object, rsi its begin, rdx
    // its end; a leaf function without branches).
    Fix {
        name: "engine map (command buffers)",
        offset: 0x1d0a4fb,
        from: (7, 0xea3ec3f71d77b206),
        change: Change::Hook { at: 0, handler: crate::engine::cb_init_hook },
        kind: "engine map (read only)",
        why: "the port learns the command buffers' extents (the GPU arena's coherence census)",
        only_with: Some("DS_KCO"),
    },
    // The CPU draw path's per-model adders (eboot+0xab4210 a chosen LOD's items, eboot+0xab4040
    // a model whole; both `push rbp; mov rbp,rsp; push r15`) and its loop over a pass's model
    // list (eboot+0xab4e7d, `mov rax,[r15+68h]; mov rdi,[r13]`): the census of `cpudraw`
    // (diagnosis; the hand-off's side is the hook above).
    Fix {
        name: "cpu draw census (LOD items)",
        offset: 0xab4210,
        from: (6, 0x82d75a35aabe38aa),
        change: Change::Hook { at: 0, handler: crate::cpudraw::lod_hook },
        kind: "diagnosis (read only)",
        why: "the port lists the models the CPU draw path walks by LOD",
        only_with: Some("DS_CPUDRAW_CENSUS"),
    },
    Fix {
        name: "cpu draw census (items added)",
        offset: 0xabac10,
        from: (6, 0x82d75b35aabe3a5d),
        change: Change::Hook { at: 0, handler: crate::cpudraw::added_hook },
        kind: "diagnosis (read only)",
        why: "the port lists the items the CPU draw path puts in its draw lists",
        only_with: Some("DS_CPUDRAW_CENSUS"),
    },
    Fix {
        name: "cpu draw census (listed models)",
        offset: 0xab4e7d,
        from: (8, 0xf726edfbb3111c5b),
        change: Change::Hook { at: 0, handler: crate::cpudraw::listed_hook },
        kind: "diagnosis (read only)",
        why: "the port lists the models of the CPU draw path's per-pass lists",
        only_with: Some("DS_CPUDRAW_CENSUS"),
    },
    Fix {
        name: "cpu draw census (whole models)",
        offset: 0xab4040,
        from: (6, 0x82d75a35aabe38aa),
        change: Change::Hook { at: 0, handler: crate::cpudraw::whole_hook },
        kind: "diagnosis (read only)",
        why: "the port lists the models the CPU draw path takes whole",
        only_with: Some("DS_CPUDRAW_CENSUS"),
    },
    // A per-frame simulation stepped by 1/the display's refresh rate whatever the frame took
    // (eboot+0xe6f180: `vdivss xmm0,xmm0,[rax+28h]` at +0xe6f1f4, then `mov rax,[rbx+0F8h]`):
    // its step from the port (`gamestep::fixed_step_code`: the game's step, at most the
    // title's 1/59.94 s), so that it keeps its speed above 60 frames a second.
    Fix {
        name: "a per-frame simulation keeps its speed above 60 fps",
        offset: 0xe6f1f9,
        from: (7, 0x108842cc7285899d),
        change: Change::Insert { at: 0, code: crate::gamestep::fixed_step_code },
        kind: "PC frame rates, `tune pace_step`",
        why: "it steps by 1/59.94 s every frame, so twice as fast at 120 frames a second",
        only_with: None,
    },
    // The render thread's submit of a frame with its flip (CRenderContext::SubmitAndFlip,
    // eboot+0x911d60: `push rbp; mov rbp,rsp; push r14`): which flip shows the frame each step
    // makes (`gamestep::submit_hook`), for Reflex's markers (`gpu::latency`); with
    // `DS_PACE_PAIR=1` logged beside the frame steps (diagnosis).
    Fix {
        name: "frame submits",
        offset: 0x911d60,
        from: (6, 0x82d75b35aabe3a5d),
        change: Change::Hook { at: 0, handler: crate::gamestep::submit_hook },
        kind: "PC latency (Reflex), diagnosis",
        why: "Reflex names each frame's moments by the flip that shows it",
        only_with: None,
    },
    // The frame step (`gamestep`): after the application's per-frame call of it returned
    // (eboot+0xb27f77 calls eboot+0xb280e0; `mov rax,[rbx+28E8h]`, rbx the application
    // object, its timer at +0x28F0).
    Fix {
        name: "the frame step follows the frames shown",
        offset: 0xb27f7c,
        from: (7, 0x6aae3b529a211d65),
        change: Change::Hook { at: 0, handler: crate::gamestep::after_step },
        kind: "PC frame rates, `tune pace_step`",
        why: "the title steps the game by 1/59.94 s at least and in halves of it, whatever the frames' real spacing",
        only_with: None,
    },
];

/// Whether the fix of this name was applied.
/// FNV-1a (64-bit) of `b`: how a fix's expected instructions are recognized.
fn fnv1a(b: &[u8]) -> u64 {
    b.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &x| (h ^ x as u64).wrapping_mul(0x0100_0000_01b3))
}

pub fn applied(name: &str) -> bool {
    FIXES.iter().position(|f| f.name == name).is_some_and(|i| APPLIED[i].load(Ordering::Relaxed))
}

/// `mov dword ptr [rcx+33Ch], -1` after counting the run (rax kept; the
/// System V red zone below rsp skipped; the flags the counter changes are set
/// again by the displaced `cmp`).
fn cmesh_unload_code(counter: u64) -> Vec<u8> {
    let mut c = vec![0x48, 0x8d, 0x64, 0x24, 0x80, 0x50, 0x48, 0xb8]; // lea rsp,[rsp-80h]; push rax; mov rax, imm64
    c.extend_from_slice(&counter.to_le_bytes());
    c.extend_from_slice(&[0xf0, 0x48, 0xff, 0x00]); // lock inc qword ptr [rax]
    c.extend_from_slice(&[0x58, 0x48, 0x8d, 0xa4, 0x24, 0x80, 0x00, 0x00, 0x00]); // pop rax; lea rsp,[rsp+80h]
    c.extend_from_slice(&[0xc7, 0x81, 0x3c, 0x03, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff]); // mov dword ptr [rcx+33Ch], -1
    c
}

/// Applies the fixes to the main executable (`ranges`: its functions, for
/// inserted code).
pub fn apply(p: &mut crate::patch::Patcher, ranges: &[(u64, u64)]) {
    if std::env::var("DS_TITLEFIX").ok().as_deref() == Some("0") {
        crate::info!("title fixes off (DS_TITLEFIX=0)");
        return;
    }
    crate::dispmode::init();
    crate::pcsettings::convar::init(p.base, p.image);
    crate::gamestep::init(p.base);
    for (i, f) in FIXES.iter().enumerate() {
        let unmet = |n: &str| match n.strip_prefix('!') {
            Some(x) => std::env::var_os(x).is_some(),
            None => std::env::var_os(n).is_none(),
        };
        if f.only_with.is_some_and(|v| v.split('|').all(unmet)) {
            continue;
        }
        if p.image.get(f.offset..f.offset + f.from.0).map(fnv1a) != Some(f.from.1) {
            crate::warn!("title fix '{}': unexpected bytes at eboot+{:#x} (another title version?); not applied", f.name, f.offset);
            continue;
        }
        let done = match &f.change {
            Change::Insert { at, code } => p.insert((f.offset + at) as u64, ranges, &code(RUNS[i].as_ptr() as u64)).map(|_| ()),
            Change::Hook { at, handler } => p.hook((f.offset + at) as u64, ranges, *handler as usize as u64, i as u32).map(|_| ()),
        };
        match done {
            Ok(()) => {
                APPLIED[i].store(true, Ordering::Relaxed);
                crate::info!("title fix '{}' at eboot+{:#x} ({}): {}", f.name, f.offset, f.kind, f.why);
            }
            Err(e) => crate::warn!("title fix '{}' at eboot+{:#x}: {e}; not applied", f.name, f.offset),
        }
    }
    // The idle job workers' sleep needs all three of its hooks.
    if (1..=3).all(|i| APPLIED[i].load(Ordering::Relaxed)) {
        crate::jobidle::enable();
    }
    // Raw mouse look needs both of its hooks.
    if (4..=5).all(|i| APPLIED[i].load(Ordering::Relaxed)) {
        crate::camhook::enable();
    }
}

/// For the status report: how often the applied fixes' code ran.
pub fn status() -> String {
    let runs: Vec<String> = FIXES
        .iter()
        .enumerate()
        // (The engine map's inserted code keeps no count: `engine` reports it.)
        .filter(|(i, f)| APPLIED[*i].load(Ordering::Relaxed) && matches!(f.change, Change::Insert { .. }) && !f.kind.starts_with("engine map"))
        .map(|(i, f)| format!("'{}' ran {} times", f.name, RUNS[i].load(Ordering::Relaxed)))
        .collect();
    let fixes = if runs.is_empty() { String::new() } else { format!("    title fixes: {}\n", runs.join(", ")) };
    fixes + &crate::cullread::status() + &crate::cpudraw::status() + &crate::jobidle::status() + &crate::gamestep::status() + &crate::camhook::status() + &crate::vgdiag::status() + &crate::dispmode::status() + &crate::pcsettings::store::status()
}
