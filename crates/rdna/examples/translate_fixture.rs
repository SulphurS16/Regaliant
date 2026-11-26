//! Offline translation of local fixtures. Never embeds or writes to title data.
//! cargo run -p rdna --example translate_fixture -- shader.bin spec.txt output.spv
use rdna::translate::{self, Spec};
use std::hash::{Hash, Hasher};
fn hash(v: &impl Hash) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    v.hash(&mut h);
    h.finish()
}
fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    assert_eq!(args.len(),3,"shader.bin spec.txt output.spv");
    let bytes = std::fs::read(&args[0]).unwrap();
    assert_eq!(bytes.len()%4,0);
    let words: Vec<_> = bytes.chunks_exact(4).map(|b|u32::from_le_bytes(b.try_into().unwrap())).collect();
    let code = rdna::decode_program(&words).unwrap();
    let spec = Spec::from_words(std::fs::read_to_string(&args[1]).unwrap().trim()).unwrap();
    println!("program={:016x} spec={:016x} translator={}", hash(&words), hash(&spec), rdna::SOURCE_HASH);
    let plan = rdna::analysis::analyze_opts(&code,spec.user_sgprs,spec.user_base,spec.stage==translate::Stage::Vertex);
    let result = translate::translate(&code,&plan,&spec).unwrap();
    for w in &result.warnings { eprintln!("{w}"); }
    std::fs::write(&args[2],result.spirv.iter().flat_map(|w|w.to_le_bytes()).collect::<Vec<_>>()).unwrap();
    println!("bindings: {:?}",result.bindings);
}
