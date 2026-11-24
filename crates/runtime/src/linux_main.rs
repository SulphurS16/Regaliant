// Linux startup for the shared loader, HLE and renderer. SDL stays on the main thread.
static PATHS: std::sync::OnceLock<dsconfig::paths::Paths> = std::sync::OnceLock::new();
pub fn paths() -> &'static dsconfig::paths::Paths { PATHS.get().expect("paths initialized") }

static RUN_DIR: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();
pub fn run_dir() -> Option<&'static std::path::Path> { RUN_DIR.get().map(|p| p.as_path()) }

fn utc_stamp() -> String { dsconfig::utc_stamp() }

const EXIT_CPU: u32 = 8;
fn player_error(message: &str) { eprintln!("{message}"); }
fn shutdown(code: i32) -> ! {
    hle::libs::videoout::wait_shots();
    platform::stalls::flush();
    info!("{}", platform::sparse::status());
    hle::libs::savedata::flush_on_exit();
    gpu::shutdown();
    gpu::pcache::clear_crash_suspects();
    gpu::precompile::exact_flush();
    gpu::trcache::flush();
    if let Some(dir) = run_dir() { let _ = std::fs::write(dir.join("pipeline-reuse.json"), gpu::pcache::reuse_report().to_string()); }
    info!("Linux shutdown complete: status {code}; process-lifetime workers end together");
    log::flush();
    // Guest, Vulkan and SDL workers have process lifetime. libc exit runs DSO finalizers
    // concurrently with those workers (the relay faulted on a worker during that teardown).
    // Do not unload their code/data underneath them. Persist explicitly above, then exit_group.
    unsafe { libc::_exit(code) }
}
fn fatal_exit(code: u32, why: &str) -> ! {
    error!("{why}; exit {code:#x}");
    log::flush();
    unsafe { libc::_exit((code & 255) as i32) };
}

fn main() {
    if let Err(e) = run() {
        eprintln!("Linux bootstrap: {e}");
        log::flush();
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    use std::os::fd::AsRawFd;
    use std::path::PathBuf;
    let diagnostics = apply_diagnostics();
    let mut args = std::env::args_os().skip(1);
    let mut app = std::env::var_os("DS_GAME_DIR").map(PathBuf::from);
    let mut scatter_check = None;
    let mut load_only = false;
    let mut headless = false;
    let mut offscreen = false;
    let mut run_seconds = None;
    let mut platform_check = false;
    let mut audio_check = false;
    let mut write_watch_check = false;
    let mut relay = false;
    let mut relay_continue = false;
    let mut precompile = None;
    let mut prepare = None;
    let mut prepare_seconds = None;
    let mut prepare_stage = None;
    let mut prepare_root = None;
    let mut prepare_threads = 4usize;
    let mut prepare_chunk = None;
    let mut progress = None;
    let mut pb_check = None;
    while let Some(arg) = args.next() {
        match arg.to_str() {
            Some("--scatter-check") => scatter_check = Some((PathBuf::from(args.next().ok_or("--scatter-check needs a fixture directory")?), PathBuf::from(args.next().ok_or("--scatter-check needs an output directory")?))),
            Some("--app") => app = Some(PathBuf::from(args.next().ok_or("--app needs a game folder")?)),
            Some("--precompile-all") => precompile = Some(args.next().and_then(|s| s.to_str().and_then(|s| s.parse::<u32>().ok())).filter(|&n| n > 0 && n <= 64).ok_or("--precompile-all needs 1..64 compiler threads")?),
            Some("--prepare-shaders") => prepare = Some(args.next().and_then(|s| s.to_str().and_then(|s| s.parse::<u32>().ok())).filter(|&n| n > 0 && n <= 64).ok_or("--prepare-shaders needs 1..64 compiler threads")?),
            Some("--prepare-seconds") => prepare_seconds = Some(args.next().and_then(|s| s.to_str().and_then(|s| s.parse::<u64>().ok())).filter(|&n| n > 0 && n <= 86400).ok_or("--prepare-seconds needs 1..86400")?),
            Some("--prepare-play") => prepare_stage = Some(args.next().and_then(|s| s.to_str().and_then(prepplay::Stage::parse)).ok_or("--prepare-play needs newgame or tour")?),
            Some("--prepare-root") => prepare_root = Some(PathBuf::from(args.next().ok_or("--prepare-root needs a directory")?)),
            Some("--prepare-threads") => prepare_threads = args.next().and_then(|s| s.to_str().and_then(|s| s.parse::<usize>().ok())).filter(|&n| n > 0 && n <= 64).ok_or("--prepare-threads needs 1..64")?,
            Some("--prepare-chunk") => {
                let t = args.next().ok_or("--prepare-chunk needs i/n")?.to_string_lossy().into_owned();
                let (i, n) = t.split_once('/').ok_or("--prepare-chunk needs i/n")?;
                let (i, n) = (i.parse::<usize>().map_err(|e| e.to_string())?, n.parse::<usize>().map_err(|e| e.to_string())?);
                if i == 0 || i > n || n > 64 { return Err("invalid preparation chunk".into()); }
                prepare_chunk = Some((i, n));
            }
            Some("--progress") => progress = Some(PathBuf::from(args.next().ok_or("--progress needs a file")?)),
            Some("--pb-check") => {
                let p = args.next().ok_or("--pb-check needs write/read/probe")?.to_string_lossy().into_owned();
                if !["write", "read", "probe"].contains(&p.as_str()) { return Err("--pb-check needs write/read/probe".into()); }
                pb_check = Some(p);
            }
            Some("--relay-newgame") => relay = true,
            Some("--relay-continue") => { relay = true; relay_continue = true; },
            Some("--headless") => headless = true,
            Some("--offscreen") => offscreen = true,
            Some("--run-seconds") => {
                let value = args.next().and_then(|s| s.to_str().and_then(|s| s.parse::<u64>().ok())).filter(|&n| n > 0 && n <= 3600).ok_or("--run-seconds needs 1..3600")?;
                run_seconds = Some(value);
            }
            Some("--platform-check") => platform_check = true,
            Some("--audio-check") => audio_check = true,
            Some("--write-watch-check") => write_watch_check = true,
            Some("--load-only") => load_only = true,
            Some("--help" | "-h") => {
                println!("dsport --app <game-folder> [--offscreen | --headless] [--load-only]\n--offscreen renders and captures game flips without a visible window, swapchain or audio.\n--headless disables GPU execution and audio. Set DSPORT_HOME outside the game tree.\n--run-seconds N bounds execution; --relay-newgame enables isolated self-play; --relay-continue loads its existing saves.\n--prepare-shaders N resumes isolated new-game, compute preparation and the full shared level tour.\n--prepare-seconds N bounds that invocation (default14400); rerun to resume interrupted work.\n--pb-check write|read|probe checks cross-process pipeline store reuse.\n--precompile-all N compiles known shader specs without self-play.\n--platform-check tests a hidden SDL/Vulkan surface; --audio-check queues silence.\n--write-watch-check exercises exact CPU dirty tracking without a GPU.\n--scatter-check <local-fixture> <output-dir> runs only the isolated fog shader diagnostic.");
                return Ok(());
            }
            _ => return Err(format!("unknown argument {}", arg.to_string_lossy())),
        }
    }
    // Hitch diagnostics can run during mapping/unmapping even in CPU-only modes.
    // Publish the clock before any mapping, signal service or GPU worker exists.
    hle::kernel::time::init();
    if let Some(threads) = prepare {
        if headless || load_only || relay || precompile.is_some() || platform_check || audio_check || write_watch_check || scatter_check.is_some() || run_seconds.is_some() || prepare_stage.is_some() || pb_check.is_some() {
            return Err("--prepare-shaders accepts --app and optional --offscreen only; its children have built-in deadlines".into());
        }
        let code = platform::prepare::run(&app.ok_or("--app <game-folder> is required")?, threads, prepare_seconds.unwrap_or(14400))?;
        std::process::exit(code);
    }
    if prepare_seconds.is_some() { return Err("--prepare-seconds requires --prepare-shaders".into()); }
    if prepare_stage.is_none() && (prepare_root.is_some() || prepare_chunk.is_some() || progress.is_some()) { return Err("internal preparation arguments require --prepare-play".into()); }
    if prepare_stage.is_some() && (headless || load_only || relay || precompile.is_some() || pb_check.is_some() || !offscreen) { return Err("--prepare-play needs its isolated --offscreen child".into()); }
    if let Some((input, output)) = scatter_check {
        if let Some(app) = &app {
            let game = dsconfig::game::check(app).map_err(|e| e.to_string())?;
            platform::files::protect_game(&game.app).map_err(|e| e.to_string())?;
        }
        gpu::scatter_check::run(&input, &output)?;
        return Ok(());
    }
    if platform_check {
        let mut frontend = platform::frontend::Frontend::new(true)?;
        println!("SDL hidden window and Vulkan surface created");
        println!("SDL/Vulkan surface supports: {:?}", frontend.present_devices()?);
        frontend.poll();
        return Ok(());
    }
    if write_watch_check { platform::write_watch::check()?; return Ok(()); }
    if audio_check { audio::smoke_test()?; println!("SDL audio silence queue consumed"); return Ok(()); }
    if precompile.is_some() && (headless || load_only || relay) { return Err("--precompile-all requires the GPU and cannot be combined with --headless, --load-only or --relay-newgame".into()); }
    if precompile.is_some() || pb_check.is_some() { offscreen = true; }
    if offscreen && headless { return Err("--offscreen and --headless are mutually exclusive".into()); }
    if offscreen { unsafe { std::env::set_var("DS_LINUX_HIDDEN", "1"); } }
    if relay {
        if run_seconds.is_none() { return Err("--relay-newgame/--relay-continue requires --run-seconds".into()); }
        // Before any worker starts. Reuse the Windows self-play's title text and name hook.
        unsafe { std::env::set_var("DS_TEXT_HOOK", "86d2ee:r13+8*:r13+10*"); std::env::set_var("DS_IME_TEXT", "Regaliant"); }
        input::SYSTEM_OFF.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    let app = app.ok_or("--app <game-folder> is required")?;
    let game = dsconfig::game::check(&app).map_err(|e| e.to_string())?;
    platform::files::protect_game(&game.app).map_err(|e| e.to_string())?;
    game::set(game.source.clone());
    if let Some(n) = std::env::var("DS_ZAR_THREADS").ok().and_then(|v| v.parse().ok()) { dsconfig::zar::set_threads(n); }
    let paths = PATHS.get_or_init(dsconfig::paths::Paths::current);
    let cache = platform::files::output_path(&paths.cache, &[&game.app]).map_err(|e| e.to_string())?;
    let user = platform::files::output_path(&paths.user, &[&game.app]).map_err(|e| e.to_string())?;
    let shader_path = std::env::var_os("DS_LINUX_SHADER_CACHE").map(PathBuf::from).unwrap_or_else(|| paths.shaders());
    let shaders = platform::files::output_path(&shader_path, &[&game.app]).map_err(|e| e.to_string())?;
    for output in [&cache, &user, &shaders] { platform::files::validate_output_tree(output).map_err(|e| e.to_string())?; }
    platform::files::configure_driver_cache(&cache)?;
    for key in ["DS_SPV_DUMP", "DS_DUMP_DIR"] {
        if let Some(value) = std::env::var_os(key) {
            let path = std::path::absolute(value).map_err(|e| e.to_string())?;
            platform::files::validate_output_tree(&path).map_err(|e| format!("{key}: {e}"))?;
        }
    }
    if prepare_stage.is_some() {
        platform::prepare::validate_child(prepare_root.as_deref().ok_or("missing --prepare-root")?, paths)?;
    }
    let run_dir = cache.join("linux-runs").join(format!("{}-{}", dsconfig::utc_stamp(), std::process::id()));
    let run_dir = platform::files::output_path(&run_dir, &[&game.app]).map_err(|e| e.to_string())?;
    std::fs::create_dir_all(&run_dir).map_err(|e| e.to_string())?;
    // Environment-backed setup must precede log::init, which starts a flush thread.
    if std::env::var_os("DS_SECTIONS").is_none() { unsafe { std::env::set_var("DS_SECTIONS", "1"); } }
    if headless || load_only { unsafe { std::env::set_var("DS_GPU", "0"); } }
    else { gpu::exec::default_on(); }
    audio::SPATIAL.store(0, std::sync::atomic::Ordering::Relaxed);
    audio::SILENT.store(headless || offscreen || load_only, std::sync::atomic::Ordering::Relaxed);
    if let Some(stage) = prepare_stage {
        let root = prepare_root.as_ref().unwrap();
        let name = if stage == prepplay::Stage::NewGame { "newgame".into() } else { format!("tour-{}", prepare_chunk.ok_or("tour needs --prepare-chunk")?.0) };
        let expected = root.join(format!("{name}.progress"));
        if progress.as_ref() != Some(&expected) { return Err("preparation progress must stay in its owned directory".into()); }
        prepplay::setup_linux(stage, paths, &shaders, expected, prepare_threads, prepare_chunk)?;
    }
    log::init(&run_dir.join("dsport.log")).map_err(|e| e.to_string())?;
    RUN_DIR.set(run_dir.clone()).ok();
    std::fs::write(cache.join("active-pid"), format!("{}\n", std::process::id())).map_err(|e| e.to_string())?;
    log::set_thread_name("linux-bootstrap");
    info!("Regaliant Linux native bootstrap; game {} {}; read-only {}", game.id, game.version, game.app.display());
    info!("output: {}; user data: {}", run_dir.display(), user.display());
    info!("Linux shared renderer: offscreen={offscreen} headless={headless} load_only={load_only}");
    std::panic::set_hook(Box::new(|p| {
        eprintln!("FATAL Linux runtime panic: {p}\n{}", std::backtrace::Backtrace::force_capture());
        // A panic in the fault service must not strand a faulting guest forever.
        unsafe { libc::_exit(101) };
    }));
    if !diagnostics.is_empty() { info!("diagnostics.txt: {}", diagnostics.join(", ")); }
    let hash = dsconfig::game::executable_hash(&game).map_err(|e| e.to_string())?;
    info!("decrypted eboot SHA-256 {hash}; tested fingerprint match {}", hash == dsconfig::game::TESTED_EBOOT);
    cpucheck::check();
    mem::reserve_guest_range()?;
    gpu::resident::init_linux_faults();
    tls::init_slot()?;
    let crash = std::fs::OpenOptions::new().create(true).append(true).open(run_dir.join("crash.log")).map_err(|e| e.to_string())?;
    let _signal_stack = platform::signals::SignalStack::new().map_err(|e| e.to_string())?;
    platform::signals::install(crash.as_raw_fd()).map_err(|e| e.to_string())?;
    let runtime = paths.runtime();
    for name in ["temp0", "data", "download0", "devlog", "devlog/logs"] {
        platform::files::output_path(&runtime.join(name), &[&game.app]).map_err(|e| e.to_string())?;
    }
    boot::mount_filesystems(&game.source, &runtime);
    let trophies = platform::files::output_path(&user.join("achievements.json"), &[&game.app]).map_err(|e| e.to_string())?;
    achievements::init(&game.source, trophies, "en-GB");
    input::init(&paths.controls_file());
    display::init(paths.display_file());
    if !headless && !offscreen && !load_only {
        if let Err(e) = audio::initialize() { warn!("audio initialization: {e}; paced null output will be used"); }
    }

    pcsettings::init(paths.settings_file(), false);
    hle::libs::savedata::init(pcsettings::saves::start(paths, &game.id));

    if pb_check.as_deref() == Some("probe") { gpu::pcache::set_read_only(); }
    gpu::pcache::set_dir(shaders.join("pipelines"));
    gpu::trcache::set_dir(shaders.join("translations"));
    gpu::precompile::set_store(shaders.join("shader-specs.txt"));
    gpu::capture::init_from_env();
    if offscreen { gpu::tune::set("headless_present", 0); }
    if !headless && !load_only {
        // SDL instance extensions and surfaces must be obtained on this thread, before HLE
        // workers can initialize Vulkan. Offscreen flips use the same GPU image readback as
        // Windows captures; they never acquire a hidden Wayland swapchain image.
        gpu::vk::init()?;
        if !offscreen { gpu::vk::present::init("Regaliant - Demon's Souls")?; }
        else {
            let c = gpu::vk::ctx();
            // Exercise the real SDL surface without creating/acquiring a swapchain. Hidden
            // Wayland windows may never receive the configure event required for acquisition.
            unsafe {
                let surface = gpu::vk::window::surface(c, 0, 0)?;
                let supported = c.surface_i.get_physical_device_surface_support(c.pdev, c.queue_family, surface);
                c.surface_i.destroy_surface(surface, None);
                info!("offscreen SDL surface: selected queue presentation support {supported:?}");
            }
            info!("offscreen rendering: swapchain disabled; DS_SCREENSHOT captures guest flips");
        }
        gpu::particles::start(game.source.clone());
        gpu::start_profiler();
    }
    if let Some(phase) = pb_check { shutdown(gpu::pcache::pb_check(&phase)); }
    if let Some(threads) = precompile {
        gpu::pcache::background_process();
        let code = gpu::compbench::precompile_all(&game.source, threads as usize, None, Some(&shaders.join("prepared.txt")));
        shutdown(code);
    }
    let load = |name: &str, main: bool| {
        let rel = game::join(&game.code_dir, name);
        let bytes = game.source.read(&rel).map_err(|e| e.to_string())?;
        loader::load_bytes(bytes, &game.app.join(&rel), main)
    };
    let eboot = load("eboot.bin", true)?;
    let mut libs = Vec::new();
    for name in dsconfig::game::MODULES {
        libs.push(load(&format!("sce_module/{name}"), false)?);
    }
    platform::signals::modules(loader::MODULES.read().iter().map(|m| (m.base, m.base + m.size, m.name.clone())).collect());
    for m in std::iter::once(&eboot).chain(libs.iter()) {
        let stats = loader::relocate(m)?;
        info!("{}: linked {stats:?}", m.name);
        let patched = loader::finalize(m)?;
        loader::report_patch(m, &patched);
        if !patched.fs_unknown.is_empty() {
            return Err(format!("{} has {} unpatched FS instructions; refusing execution with host FS active", m.name, patched.fs_unknown.len()));
        }
    }
    let unresolved = hle::UNRESOLVED.read();
    let mut imports: Vec<_> = unresolved.iter().map(|u| format!("{} [{}] imported by {}", hle::describe_nid(&u.nid), u.lib, u.importer)).collect();
    imports.sort();
    std::fs::write(run_dir.join("unresolved.txt"), imports.join("\n") + "\n").map_err(|e| e.to_string())?;
    info!("load/link/patch complete; {} distinct unresolved function imports", imports.len());
    drop(unresolved);
    log::flush();
    if load_only { log::flush(); return Ok(()); }
    if prepare_stage.is_some() { prepplay::start(); }
    if relay { prepplay::start_relay(relay_continue)?; }
    if !headless {
        std::thread::Builder::new().name("linux-gpu-diagnostics".into()).spawn(|| {
            log::set_thread_name("linux-gpu-diagnostics");
            loop {
                std::thread::sleep(std::time::Duration::from_secs(15));
                info!("Linux GPU snapshot:\n{}\n{}\n{}\n{}\n{}", gpu::status_ex(false), gpu::track::status(), gpu::resident::status(), gpu::fe2::status(), gpu::vk::present::stats_line());
                info!("{}", gpu::resident::linux_fault_timing());
                info!("{}", platform::write_watch::status());
                info!("{}", platform::sparse::status());
                info!("{}", audio::status());
                info!("attachment metadata rebindings: {}", gpu::images::METADATA_REBINDS.load(std::sync::atomic::Ordering::Relaxed));
            }
        }).map_err(|e| e.to_string())?;
    }
    let (stack, name, priority) = boot::read_proc_param(&eboot);
    let stack = stack.max(16 << 20);
    let guest = thread::new_thread(&name, eboot.entry, 0, stack, priority);
    // Rust supplies a guarded host stack; guest fibers use their own guarded alternate stacks.
    let thread = std::thread::Builder::new().name(name).stack_size(stack as usize).spawn(move || {
        let _signal_stack = platform::signals::SignalStack::new().expect("guest signal stack");
        thread::enter(guest);
        log::set_thread_name("guest-main");
        for m in &libs {
            if let Some(init) = m.init {
                info!("entering module {} at {}", m.name, loader::describe_addr(init));
                let result = thread::run_entry(guest, init, 0, 0);
                info!("{} init returned {result:#x}", m.name);
            }
        }
        let params = mem::MEM.lock().alloc_internal(mem::PAGE, mem::PROT_CPU_READ | mem::PROT_CPU_WRITE, "entry parameters").expect("entry parameters");
        unsafe {
            *(params as *mut u32) = 1;
            *((params + 8) as *mut u64) = params + 64;
            std::ptr::copy_nonoverlapping(c"eboot.bin".as_ptr(), (params + 64) as *mut _, 10);
        }
        info!("entering eboot at {}", loader::describe_addr(eboot.entry));
        let result = thread::run_entry(guest, eboot.entry, params, hle::kernel::misc::rtld_exit_fn());
        info!("eboot returned {result:#x}");
    }).map_err(|e| e.to_string())?;
    let started = std::time::Instant::now();
    let mut reported = 0;
    let close_via_window = relay && std::env::var("DS_RELAY_CLOSE_WINDOW").ok().as_deref() == Some("1");
    let mut close_requested = false;
    let relay_stop = std::env::var_os("DS_RELAY_STOP_FILE").map(PathBuf::from);
    while !thread.is_finished() {
        let seconds = started.elapsed().as_secs();
        if seconds / 5 > reported {
            reported = seconds / 5;
            use std::sync::atomic::Ordering::Relaxed;
            let [upload_bytes, writeback_bytes, upload_runs] = gpu::fe2::traffic_counters();
            let [uffd_takes, uffd_scans, uffd_pages, uffd_ns, _, uffd_errors, _] = platform::write_watch::counters();
            let [dirty_pages, unarmable_pages] = gpu::resident::dirty_counters();
            info!("Linux heartbeat: {seconds}s flips={} submits={} draws={} dispatches={} recovered_faults={} direct_faults={} kernel_io_retries={} upload_bytes={upload_bytes} writeback_bytes={writeback_bytes} upload_runs={upload_runs} dirty_pages={dirty_pages} unarmable_pages={unarmable_pages} uffd_takes={uffd_takes} uffd_scans={uffd_scans} uffd_pages={uffd_pages} uffd_ns={uffd_ns} uffd_errors={uffd_errors}",
                gpu::FRAMES.load(Relaxed), gpu::SUBMITS.load(Relaxed), gpu::cp::DRAWS.load(Relaxed), gpu::cp::DISPATCHES.load(Relaxed),
                platform::signals::RECOVERED.load(Relaxed), platform::signals::DIRECT.load(Relaxed), mem::KERNEL_IO[1].load(Relaxed));
            if !headless && reported % 3 == 0 && std::env::var("DS_LINUX_INTERVAL_SHOTS").ok().as_deref() != Some("0") {
                let mut shot = hle::libs::videoout::SHOT_REQUEST.lock();
                if shot.is_none() { *shot = Some(format!("elapsed-{seconds:04}s")); }
            }
        }
        if relay {
            let result = prepplay::RELAY_RESULT.load(std::sync::atomic::Ordering::Acquire);
            if result >= 0 {
                if result == 0 && close_via_window {
                    if !close_requested {
                        info!("relay requesting SDL window-close event after successful gameplay");
                        gpu::vk::window::request_close()?;
                        close_requested = true;
                    }
                } else { shutdown(result); }
            }
        }
        if relay_stop.as_ref().is_some_and(|p| p.is_file()) {
            info!("relay thermal/cancellation stop requested; flushing completed saves");
            shutdown(75);
        }
        if run_seconds.is_some_and(|limit| seconds >= limit) {
            info!("bounded Linux run finished after {seconds}s; waiting for frame PNG writers");
            hle::libs::videoout::wait_shots();
            shutdown(0);
        }
        if !headless && !gpu::vk::window::poll() { info!("SDL window-close received; flushing completed saves"); shutdown(0); }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    thread.join().map_err(|_| "guest thread panicked")?;
    shutdown(0)
}


pub const EXIT_DEVICE_LOST:u32=5;
pub const EXIT_NO_GPU:u32=11;
pub const EXIT_MEMORY:u32=7;
pub fn keep_run(reason:&str){warn!("retaining Linux run diagnostics: {reason}");}
pub fn no_gpu_exit(reason:&str)->!{fatal_exit(EXIT_NO_GPU,reason)}
pub fn hung(why:&str,code:i32)->!{fatal_exit(code as u32,why)}

#[cfg(test)]
mod linux_shutdown_tests {
    #[test]
    fn process_lifetime_workers_skip_dso_teardown() {
        if std::env::var_os("REGALIANT_SHUTDOWN_TEST").is_some() {
            extern "C" fn unsafe_teardown() { unsafe { libc::_exit(99) } }
            assert_eq!(unsafe { libc::atexit(unsafe_teardown) }, 0);
            std::thread::spawn(|| loop { std::thread::park(); });
            super::shutdown(0);
        }
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "linux_shutdown_tests::process_lifetime_workers_skip_dso_teardown", "--nocapture"])
            .env("REGALIANT_SHUTDOWN_TEST", "1").output().unwrap();
        assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    }
}
