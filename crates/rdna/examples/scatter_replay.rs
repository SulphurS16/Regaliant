//! Offline replay of a local NaN-probe capture. No title bytes are embedded in this tool.
//! cargo run -p rdna --example scatter_replay -- shader.bin event/before shader-specs.txt output-dir
use rdna::{
    analysis::Ptr,
    emu::{Env, Group, Launch},
    translate::{self, Spec},
};
use std::{collections::BTreeMap, path::Path};
fn words(p: &Path) -> Vec<u32> {
    std::fs::read(p)
        .expect("read words")
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
#[derive(Default)]
struct Snapshot {
    mem: BTreeMap<u64, u8>,
    images: BTreeMap<u64, (Vec<u8>, usize, usize, usize)>,
    stores: Vec<(u32, u32, u32, [u32; 4])>,
    missing: Vec<String>,
}
impl Snapshot {
    fn put(&mut self, addr: u64, bytes: &[u8]) {
        for (i, &b) in bytes.iter().enumerate() {
            self.mem.insert(addr + i as u64, b);
        }
    }
    fn ptr(&self, p: &Ptr, user: &[u32]) -> Option<u64> {
        match p {
            Ptr::User(k) => Some(user[*k as usize] as u64 | (user[*k as usize + 1] as u64) << 32),
            Ptr::Mem(p, o) => {
                let a = self.ptr(p, user)? + *o as u64;
                let mut b = [0; 8];
                for (i, v) in b.iter_mut().enumerate() {
                    *v = *self.mem.get(&(a + i as u64))?;
                }
                Some(u64::from_le_bytes(b))
            }
            Ptr::Pc(_) => None,
        }
    }
}
impl Env for Snapshot {
    fn read(&mut self, a: u64, out: &mut [u8]) -> bool {
        let mut ok = true;
        for (i, v) in out.iter_mut().enumerate() {
            if let Some(b) = self.mem.get(&(a + i as u64)) {
                *v = *b;
            } else {
                *v = 0;
                ok = false;
            }
        }
        if !ok && self.missing.len() < 16 {
            self.missing.push(format!("memory {a:#x} +{}", out.len()));
        }
        ok
    }
    fn write(&mut self, a: u64, b: &[u8]) {
        self.put(a, b);
    }
    fn texel(&mut self, t: &[u32; 8], x: u32, y: u32, z: u32, level: u32) -> Option<[u32; 4]> {
        let base = (t[0] as u64 | ((t[1] & 255) as u64) << 32) << 8;
        let Some((b, w, h, n)) = self.images.get(&base) else {
            if self.missing.len() < 16 {
                self.missing.push(format!("image {base:#x}"));
            }
            return None;
        };
        let at = ((z as usize * h + y as usize) * w + x as usize) * n;
        if level != 0 || x as usize >= *w || y as usize >= *h || at + n > b.len() {
            return None;
        }
        let mut raw = [0u8; 16];
        raw[..*n].copy_from_slice(&b[at..at + n]);
        Some(std::array::from_fn(|i| u32::from_le_bytes(raw[i * 4..i * 4 + 4].try_into().unwrap())))
    }
    fn store_texel(&mut self, _: &[u32; 8], x: u32, y: u32, z: u32, _: u32, raw: [u32; 4]) {
        self.stores.push((x, y, z, raw));
    }
}
fn main() {
    let a: Vec<_> = std::env::args().skip(1).collect();
    assert_eq!(a.len(), 4, "shader.bin before-dir shader-specs.txt output-dir");
    let bin = words(Path::new(&a[0]));
    let code = rdna::decode_program(&bin).unwrap();
    let id = bin[..bin.iter().position(|&w| w == 0xbf810000).map_or(bin.len(), |i| i + 1)]
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .fold(0xcbf29ce484222325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100000001b3));
    println!("code {} instructions; prefix FNV {id:#x}", code.len());
    assert_eq!(id, 0xcb83739fea041817, "this fixture expects the reviewed fogscatter program");
    let before = Path::new(&a[1]);
    let out = Path::new(&a[3]);
    std::fs::create_dir_all(out).unwrap();
    let listing = std::fs::read_to_string(before.join("inputs.txt")).unwrap();
    let line = listing.lines().find(|l| l.starts_with("program ")).unwrap();
    let addr = line.split_whitespace().nth(1).unwrap();
    let hash = line.split_whitespace().nth(3).unwrap().trim_end_matches(',');
    let key = addr.trim_start_matches("0x");
    let user = words(&before.join(format!("{key}-user.bin")));
    let sgpr = words(&before.join(format!("{key}-sgpr-before-binding.bin")));
    let specs = std::fs::read_to_string(&a[2]).unwrap();
    let l = specs.lines().find(|l| l.starts_with(&format!("K {} ", hash.trim_start_matches("0x")))).expect("exact spec");
    let serialized = &l[l.find("s1 ").unwrap()..l.rfind(' ').unwrap()];
    let mut spec = Spec::from_words(serialized).unwrap();
    spec.code_base = u64::from_str_radix(key, 16).unwrap();
    println!(
        "captured spec per_lane={} lane_votes={} wave64={} local={:?} gpu_loads={:#x}",
        spec.per_lane, spec.lane_votes, spec.wave64, spec.cs_local_size, spec.gpu_loads
    );
    let plan = rdna::analysis::analyze_opts(&code, spec.user_sgprs, spec.user_base, spec.gpu_loads != 0);
    let translated = translate::translate(&code, &plan, &spec).expect("translate actual shader");
    println!("SPIR-V store guards: {} bit-tested conditional ImageWrites", check_store_guards(&translated.spirv));
    std::fs::write(out.join("translated.spv"), translated.spirv.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<_>>()).unwrap();
    std::fs::write(out.join("shader.bin"), bin.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<_>>()).unwrap();
    std::fs::write(out.join("spec.txt"), spec.to_words()).unwrap();
    std::fs::write(out.join("sgpr.bin"), sgpr.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<_>>()).unwrap();
    println!(
        "SPIR-V {} words; lockstep={} wave_masks={} warnings={:?}",
        translated.spirv.len(),
        translated.lockstep,
        translated.wave_masks,
        translated.warnings
    );
    // The same manifest drives the GPU check, so every requested probe is translated
    // and validated offline too. Probes save values after the PC; they do NOT stop
    // execution. An unexecuted probe keeps its default zero (consult EXEC probes).
    let uint_s = translate::PROBE_SGPR | translate::PROBE_UINT;
    let mut variants: Vec<(String, Option<(u32, u32)>)> = vec![
        ("normal".into(), None),
        ("normal-half".into(), None),
        ("no-lane-votes".into(), None),
        ("no-lane-guards".into(), None),
        ("depth-limit".into(), Some((0x7c, translate::PROBE_UINT))),
        ("media-alpha".into(), Some((0x170, 4))),
        ("history-alpha".into(), Some((0x284, 8))),
        ("history-accept-mask".into(), Some((0x294, uint_s | 60))),
        ("history-combined-mask".into(), Some((0x4448, uint_s | 106))),
        ("before-history".into(), Some((0x4430, 3))),
    ];
    let mut manifest = String::new();
    let mut instructions = String::from("Each probe saves ONE register after the instruction and replaces both stores with it. Not an execution cutoff. Unreached PCs save zero. Scalar integers are converted to f32. EXEC bit0 is the local lane.\n");
    for i in code.iter().filter(|i| (0x43ec..0x4528).contains(&i.pc)) {
        let op = i.base_name().unwrap_or("");
        if i.branch_target().is_some() {
            instructions.push_str(&format!("{} ; branch uses the preceding EXEC probe; branches are not instrumented by Spec::probe\n", rdna::disasm(i)));
            continue;
        }
        let reg = if op == "s_and_saveexec_b64" {
            uint_s | 126 // new EXEC, not the saved old EXEC in s0
        } else if op.starts_with("v_cmp") {
            uint_s | i.sdwa_sdst().unwrap_or(if i.enc == rdna::Enc::Vop3 { i.vdst() } else { rdna::OP_VCC_LO }) as u32
        } else if op.starts_with("v_") {
            i.vdst() as u32
        } else if op == "ds_read_addtid_b32" {
            rdna::bits(i.w1, 24, 8)
        } else if op.starts_with("s_load_") || op.starts_with("s_mov_") || op.starts_with("s_and_") {
            uint_s | i.sdst() as u32
        } else {
            uint_s | 126 // branch/wait: EXEC, to distinguish rejected history
        };
        let name = format!("history-pc-{:04x}", i.pc);
        instructions.push_str(&format!("{name}: {} ; probe register {reg:#x}\n", rdna::disasm(i)));
        variants.push((name, Some((i.pc, reg))));
    }
    for (name, probe) in variants {
        manifest.push_str(&match probe {
            Some((pc, reg)) => format!("{name} {pc} {reg}\n"),
            None => format!("{name} - -\n"),
        });
        let mut s = spec.clone();
        s.addresses = false;
        s.sgpr_pc = None;
        s.sgpr_ubo = 0;
        s.batched = false;
        s.write_marks = false;
        s.gpu_loads = 0;
        s.gpu_res = 0;
        s.lod_stats = false;
        s.probe = probe;
        match name.as_str() {
            "no-lane-votes" => {
                s.lane_votes = false;
                s.known_active = false;
            }
            "no-lane-guards" => s.lane_guards = false,
            _ => {}
        }
        let tr = translate::translate(&code, &plan, &s).expect("standalone translate");
        std::fs::write(
            out.join(format!("standalone-{name}.spv")),
            tr.spirv.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<_>>(),
        )
        .unwrap();
        println!("standalone {name}: {} SPIR-V words, bindings {:?}", tr.spirv.len(), tr.bindings);
    }
    std::fs::write(out.join("probes.txt"), manifest).unwrap();
    std::fs::write(out.join("history-instructions.txt"), instructions).unwrap();
    let mut env = Snapshot::default();
    let offs = translate::load_offsets(&plan);
    // Pointer loads precede their consumers in the capture's plan. Resolve in passes,
    // refusing missing inputs rather than silently treating uncaptured data as zero.
    let mut pending: Vec<_> = (0..plan.loads.len()).collect();
    while !pending.is_empty() {
        let old = pending.len();
        pending.retain(|&i| {
            let l = &plan.loads[i];
            let Some(base) = env.ptr(&l.ptr, &user) else { return true };
            let o = offs[i] as usize;
            env.put(
                base + l.offset as u64,
                &sgpr[o..o + l.dwords as usize].iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<_>>(),
            );
            false
        });
        assert!(pending.len() < old, "unresolved captured pointer chain {pending:?}");
    }
    for e in std::fs::read_dir(before).unwrap().flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        if name.starts_with("shader-") && name.ends_with(".bin") {
            let base = u64::from_str_radix(name.trim_end_matches(".bin").rsplit('-').next().unwrap(), 16).unwrap();
            env.put(base, &std::fs::read(e.path()).unwrap());
        }
        if name.starts_with("input-") && name.ends_with("-m0-slice0.raw") {
            let parts: Vec<_> = name.split('-').collect();
            let base = u64::from_str_radix(parts[1], 16).unwrap();
            let wh: Vec<usize> = parts[2].split('x').map(|s| s.parse().unwrap()).collect();
            let n = match parts[3] {
                "R16_SFLOAT" | "R16_UNORM" => 2,
                "R32_UINT" | "R32_SFLOAT" => 4,
                "R16G16B16A16_SFLOAT" => 8,
                _ => continue,
            };
            env.images.insert(base, (std::fs::read(e.path()).unwrap(), wh[0], wh[1], n));
        }
    }
    let history = &env.images[&0x10f5f60000].0;
    assert!(history.chunks_exact(2).all(|b| { let h = u16::from_le_bytes([b[0], b[1]]); h & 0x7c00 == 0x7c00 && h & 0x3ff != 0 }), "fixture must contain NaN history");
    println!("CPU replay history: {} half components, ALL NaN (not zero history)", history.len() / 2);
    for z in [0, 1, 35, 71] {
        env.stores.clear();
        env.missing.clear();
        let launch = Launch {
            wave64: spec.wave64,
            local_size: spec.cs_local_size,
            rsrc2: spec.cs_rsrc2,
            user: user.clone(),
            group: [0, 0, z],
            lds_bytes: 32768,
            code_base: spec.code_base,
            probe: None,
        };
        let mut g = Group::new(&code, &mut env, launch);
        let trace = std::rc::Rc::new(std::cell::RefCell::new(String::new()));
        let t = trace.clone();
        g.trace = Some(Box::new(move |_, i, s, v| {
            if matches!(i.pc, 0x7c | 0x80 | 0xac | 0x294 | 0x43ec | 0x444c | 0x4518 | 0x4528) {
                use std::fmt::Write;
                let _ = writeln!(
                    t.borrow_mut(),
                    "pc {:#x} exec {:08x}:{:08x} s60 {:08x} v0..8 {:?}",
                    i.pc,
                    s[127],
                    s[126],
                    s[60],
                    v[..9].iter().map(|v| format!("{:08x}", v[0])).collect::<Vec<_>>()
                );
            }
        }));
        let result = g.run(100_000);
        drop(g);
        println!(
            "group z={z}: {result:?}; stores {} first {:?}; missing {:?}",
            env.stores.len(),
            env.stores.first(),
            env.missing
        );
        std::fs::write(out.join(format!("group-{z}.txt")), trace.borrow().as_bytes()).unwrap();
        if result.is_err() || !env.missing.is_empty() {
            std::process::exit(1);
        }
        assert_eq!(env.stores.len(), 64);
        assert!(env.stores.iter().all(|s| s.3 == [0; 4]), "captured zero-media groups must reject NaN history");
    }
    // Counterfactual: reproduce the GPU's observed s60=1 by accepting every
    // active lane at the class test. This is only a CPU diagnostic, never a
    // production patch or a shader sent to Vulkan. Locate the first poisoned
    // output component independently of the translator's floating arithmetic.
    let mut accept_nan = code.clone();
    let cmp = accept_nan.iter_mut().find(|i| i.pc == 0x294).unwrap();
    assert_eq!(cmp.base_name(), Some("v_cmp_class_f32"));
    assert_eq!(cmp.enc, rdna::Enc::Vopc);
    cmp.op = 15; // v_cmp_tru_f32, preserving the SDWA s60 destination
    cmp.w0 = (cmp.w0 & !(255 << 17)) | (15 << 17);
    env.stores.clear();
    env.missing.clear();
    let launch = Launch { wave64: true, local_size: [8, 8, 1], rsrc2: spec.cs_rsrc2, user: user.clone(), group: [0, 0, 0], lds_bytes: 32768, code_base: spec.code_base, probe: None };
    let trace = std::rc::Rc::new(std::cell::RefCell::new(String::new()));
    let t = trace.clone();
    let mut g = Group::new(&accept_nan, &mut env, launch);
    g.trace = Some(Box::new(move |_, i, s, v| {
        if i.pc == 0x294 || i.pc >= 0x43ec {
            use std::fmt::Write;
            let _ = writeln!(t.borrow_mut(), "pc {:#x} {} EXEC {:08x}:{:08x} v0..8 {:?}", i.pc, i.base_name().unwrap_or("?"), s[127], s[126], v[..9].iter().map(|v| format!("{:08x}", v[0])).collect::<Vec<_>>());
        }
    }));
    g.run(100_000).unwrap();
    drop(g);
    assert!(env.missing.is_empty(), "{:?}", env.missing);
    let poisoned = env.stores.iter().filter(|s| (0..4).all(|i| { let h = (s.3[i / 2] >> ((i % 2) * 16)) as u16; h & 0x7c00 == 0x7c00 && h & 0x3ff != 0 })).count();
    assert!(poisoned > 0, "accepting NaN history must reproduce poisoned valid voxels");
    println!("Counterfactual accepting NaN history: {poisoned}/{} valid voxels all NaN; see accept-nan-counterfactual.txt", env.stores.len());
    std::fs::write(out.join("accept-nan-counterfactual.txt"), trace.borrow().as_bytes()).unwrap();
    // Same controlled-depth cases as --scatter-check, verified with hardware EXEC
    // semantics on the host before the relay asks a GPU to run them.
    let coeff = &plan.loads[*plan.inst_load.get(&0x28).unwrap()];
    let coeff_addr = env.ptr(&coeff.ptr, &user).unwrap() + coeff.offset as u64;
    let slice = &plan.loads[*plan.inst_load.get(&0x30).unwrap()];
    let slice_addr = env.ptr(&slice.ptr, &user).unwrap() + slice.offset as u64;
    env.put(slice_addr, &0u32.to_le_bytes());
    let depth = env.images.get_mut(&0x10d3200000).expect("captured fog depth");
    for b in depth.0.chunks_exact_mut(2) {
        b.copy_from_slice(&0x3c00u16.to_le_bytes());
    }
    for limit in [0u32, 4, 71] {
        env.put(coeff_addr, &[1f32, 0., 1., limit as f32].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
        let mut finite = [0usize; 64];
        for z in 0..72 {
            env.stores.clear();
            env.missing.clear();
            let launch = Launch {
                wave64: true,
                local_size: [8, 8, 1],
                rsrc2: spec.cs_rsrc2,
                user: user.clone(),
                group: [0, 0, z],
                lds_bytes: 32768,
                code_base: spec.code_base,
                probe: None,
            };
            Group::new(&code, &mut env, launch).run(100_000).unwrap();
            assert!(env.missing.is_empty(), "{:?}", env.missing);
            assert_eq!(env.stores.len(), 64);
            for &(x, y, zz, raw) in &env.stores {
                assert_eq!(zz, z);
                let bits: [u16; 4] = std::array::from_fn(|i| (raw[i / 2] >> ((i % 2) * 16)) as u16);
                if z <= limit {
                    assert_eq!(bits, [0; 4]);
                    finite[(y * 8 + x) as usize] += 1;
                } else {
                    assert!(bits.iter().all(|b| b & 0x7c00 == 0x7c00 && b & 0x3ff != 0));
                }
            }
        }
        assert_eq!(finite, [limit as usize + 1; 64]);
        println!(
            "controlled depth limit {limit}: all 64 columns have {} finite zero slices; remaining slices NaN sentinels",
            limit + 1
        );
    }
}

// Structural regression: both scatter ImageWrites must sit on the true edge of a
// bit-tested guard. This checks emitted control flow, not NVIDIA's execution of it.
fn check_store_guards(words: &[u32]) -> usize {
    use rspirv::{dr::Operand, spirv::Op};
    let m = rspirv::dr::load_words(words).unwrap();
    let defs: BTreeMap<_, _> = m.all_inst_iter().filter_map(|i| i.result_id.map(|id| (id, i))).collect();
    fn bit_guard(id: u32, defs: &BTreeMap<u32, &rspirv::dr::Instruction>, depth: u8) -> bool {
        if depth == 0 {
            return false;
        }
        let Some(i) = defs.get(&id) else { return false };
        if i.class.opcode == Op::BitwiseAnd {
            return i
                .operands
                .iter()
                .any(|o| matches!(o,Operand::IdRef(k) if defs.get(k).is_some_and(|v|v.class.opcode==Op::Constant&&v.operands==[Operand::LiteralBit32(1)])));
        }
        if !matches!(i.class.opcode, Op::INotEqual | Op::LogicalAnd) {
            return false;
        }
        i.operands.iter().any(|o| matches!(o,Operand::IdRef(k) if bit_guard(*k,defs,depth-1)))
    }
    let mut n = 0;
    for f in &m.functions {
        for b in &f.blocks {
            if !b.instructions.iter().any(|i| i.class.opcode == Op::ImageWrite) {
                continue;
            }
            let label = b.label.as_ref().unwrap().result_id.unwrap();
            let guards: Vec<_> = f
                .blocks
                .iter()
                .filter_map(|p| p.instructions.last())
                .filter(|i| i.class.opcode == Op::BranchConditional && i.operands.get(1) == Some(&Operand::IdRef(label)))
                .collect();
            assert!(
                guards.iter().any(|i| matches!(i.operands[0],Operand::IdRef(k) if bit_guard(k,&defs,4))),
                "ImageWrite in block {label} lacks the per-lane bit guard"
            );
            n += 1;
        }
    }
    assert_eq!(n, 2, "expected sentinel and valid scatter stores");
    n
}
