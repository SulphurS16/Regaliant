//! Linux reservation failures already name occupied mappings from /proc/self/maps.
//! DLL-injection simulation and Windows foreign-module classification do not apply here.
pub fn sim_module() {}
pub fn foreign_modules() -> Vec<String> { Vec::new() }
