//! Guest images on the GPU: one VkImage per guest surface (address,
//! format, size, layers, levels), shared by every use of that memory as a
//! texture, storage image or render target. Images first used as textures
//! are uploaded from guest memory (detiled with `tiling`); images the GPU
//! writes are owned by the GPU from then on. Samplers come from S#.
//!
//! All images stay in VK_IMAGE_LAYOUT_GENERAL; work is ordered by the
//! executor's global memory barriers.

use super::desc::Tsharp;
use super::tiling;
use super::vk::ctx;
use ash::vk;
use parking_lot::Mutex;
use super::fx::Map as HashMap;

/// Changes whenever what `Images::lookup` finds for a key may change (images
/// added or removed, an image turning GPU-written, stencil links): the
/// per-thread lookup memos (`tls`) and the global write notes depend on it.
pub static LOOKUP_GEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
/// Changes when the GPU-written images (render targets, storage images,
/// their aliases at the same addresses) change: what `Images::depth_view`
/// answers for a texture depends only on them (texture keys have color
/// formats; uploads, which stream all the time, do not move it).
pub static GPU_IMAGES_GEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn gpu_images_changed() {
    GPU_IMAGES_GEN.fetch_add(1, std::sync::atomic::Ordering::Release);
}

/// Generations of the images by address (4,096 buckets): a bucket moves when an image at an
/// address in it is added or removed (`map_insert`, `map_remove`). With GPU_IMAGES_GEN they hold
/// what a T#'s lookup finds (`Images::lookup` looks at its own address, and at another only
/// through the GPU-written images' stencil aliases).
static BASE_GENS: [std::sync::atomic::AtomicU32; 4096] = [const { std::sync::atomic::AtomicU32::new(0) }; 4096];

fn base_gen(base: u64) -> &'static std::sync::atomic::AtomicU32 {
    &BASE_GENS[((base >> 8).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 52) as usize]
}

/// A sampled view's memo (`view_hit`, `Images::view`): it holds while the images at the T#'s
/// address (`base_gen`), the GPU-written images (`gpu_gen`) and the image's own tracked region
/// (`region`, None for images without one: GPU-written) keep their generations, and in its frame
/// unless textures are scrubbed. `stale`: made when the image's region was not known before its
/// lookup (a new image): good for that call only.
#[derive(Clone, Copy)]
struct ViewMemo {
    frame: u64,
    gpu_gen: u64,
    base_gen: u32,
    region: Option<&'static std::sync::atomic::AtomicU32>,
    region_gen: u32,
    stale: bool,
    view: vk::ImageView,
    image: vk::Image,
    used: UseStamp,
}

/// `DS_CMP_ALWAYS=1` (diagnosis): comparison samplers always pass (`ImageCache::sampler`).
fn cmp_always() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_CMP_ALWAYS").ok().as_deref() == Some("1"))
}

impl ViewMemo {
    /// 0 when it holds for `t` in `frame`, else why not (`memo_note`'s kinds: 2 another frame,
    /// 3 the images changed, 4 its region changed).
    fn miss(&self, t: &Tsharp, frame: u64, any_frame: bool) -> usize {
        use std::sync::atomic::Ordering::Acquire;
        if !(any_frame || self.frame == frame) {
            2
        } else if self.stale || self.gpu_gen != GPU_IMAGES_GEN.load(Acquire) || self.base_gen != base_gen(t.base).load(Acquire) {
            3
        } else if self.region.is_some_and(|g| g.load(Acquire) != self.region_gen) {
            4
        } else {
            0
        }
    }
}

/// Fast clears and overwritten images pending (published at the end of
/// every `with`): draws check them without the lock while both are 0.
static PENDING: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// `DS_TRACE_CLEARS=1` (diagnosis): fast clears made pending and applied, with the flip and
/// thread (`trace clears` lines).
pub(crate) static CLEAR_PROBE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub fn trace_clears() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    CLEAR_PROBE.load(std::sync::atomic::Ordering::Relaxed) || *ON.get_or_init(|| std::env::var("DS_TRACE_CLEARS").is_ok_and(|v| v == "1"))
}

/// A pending fast clear: its metadata address, the value a fill gave it (None: a write of
/// unknown data), and the value of a fill that such a write replaced, kept aside for the use
/// (`merge_clear`). The block is read at the image's use: one code throughout is a later clear
/// (a fill kernel's other codes and CP DMA fills reach us only as unknown writes), anything else
/// is other data over the block, and the kept fill's clear stands (2026-11-18 night: the title
/// reuses metadata blocks an image no longer uses). The kept value lives in the entry, so it
/// moves and ends with it (2026-11-19: a table of its own outlived entries dropped elsewhere).
/// The fourth field: the layers whose metadata the writes covered, a bit a layer, for a layered
/// depth surface whose HTILE is laid out layer by layer (`meta_layer_bytes`); 0: the whole image.
pub type PendingClear = (u64, Option<u32>, Option<u32>, u128);

/// Two pending clears' layers together (0, the whole image, wins).
fn union_layers(a: u128, b: u128) -> u128 {
    if a == 0 || b == 0 { 0 } else { a | b }
}

/// Notes a pending fast clear of `key` (its metadata at `meta`; `v` the value a fill gave,
/// None: a write of unknown data; `layers` the layers it covers, 0 all), replacing the one
/// pending but for its layers, which add up; a fill's value an unknown write replaces at the
/// same metadata is kept aside in the entry for its use.
pub fn merge_clear<S: std::hash::BuildHasher>(map: &mut std::collections::HashMap<ImageKey, PendingClear, S>, key: ImageKey, meta: u64, v: Option<u32>, layers: u128) {
    let prev = map.get(&key).copied();
    // (Layers that meet: either clear is the whole image's (0), or their masks overlap. A fill
    // kept aside answers only for the layers it filled: one layer's value must not clear another
    // layer an unknown write reached, the release review of 2026-11-19.)
    let meets = |pl: u128| pl == 0 || layers == 0 || pl & layers != 0;
    let kept = match (v, prev) {
        (None, Some((m, Some(old), _, pl))) if m == meta && meets(pl) => Some(old),
        (None, Some((m, None, kept, pl))) if m == meta && meets(pl) => kept,
        _ => None,
    };
    // A fill of another value than the pending one's over fewer layers than it pends: no one
    // value holds for them all, so none is taken (each run reads its own back when recorded).
    let covers = |pl: u128| layers == 0 || (pl != 0 && pl & !layers == 0);
    let v = match (v, prev) {
        (Some(x), Some((m, pv, _, pl))) if m == meta && pv != Some(x) && !covers(pl) => None,
        _ => v,
    };
    let layers = prev.map_or(layers, |e| union_layers(e.3, layers));
    map.insert(key, (meta, v, kept, layers));
}

/// Merges a pending clear `e` noted later (a held stretch's, a chunk's) into `map`: its own
/// fill or kept fill wins; with neither, it is an unknown write over what `map` holds. The
/// layers add up.
pub fn merge_pending<S: std::hash::BuildHasher>(map: &mut std::collections::HashMap<ImageKey, PendingClear, S>, key: ImageKey, e: PendingClear) {
    match e {
        (meta, None, None, layers) => merge_clear(map, key, meta, None, layers),
        // (The stretch's fill: with `merge_clear`'s layer rules, so that one value never answers
        // for layers another value filled: the release review of 2026-11-20.)
        (meta, Some(v), _, layers) => merge_clear(map, key, meta, Some(v), layers),
        (meta, None, kept, layers) => {
            let prev = map.get(&key).copied();
            // (The stretch's kept value answers only for its own layers: the map's layers pending
            // at that metadata must lie inside them.)
            let kept = match prev {
                Some((m, _, _, pl)) if m == meta && !(layers == 0 || (pl != 0 && pl & !layers == 0)) => None,
                _ => kept,
            };
            let layers = prev.map_or(layers, |p| union_layers(p.3, layers));
            map.insert(key, (meta, None, kept, layers));
        }
    }
}

/// The HTILE bytes a layer of a layered depth image takes (4 bytes an 8x8 tile), when its
/// layers' metadata is laid out one after another from the surface's metadata address: the
/// hardware addresses a slice's HTILE past the ones before it. The title binds every layer of
/// its shadow atlas (1024x1024, 80 layers) with the atlas's one address and clears a layer by
/// filling that layer's 64 KiB (2026-11-19: a fill of layer 0's cleared all 80 here, the
/// cached shadows of the others with it, and fills of the others cleared nothing: lights lost
/// their shadows for a frame, the menu's preview view making it happen often). Sizes whose
/// layers are whole 64 KiB blocks only (the ones measured); others keep whole-image clears.
pub fn meta_layer_bytes(k: &ImageKey) -> Option<u64> {
    // (`tune layer_clears 0` / `DS_LAYER_CLEARS=0`: whole-image clears, as before.)
    // (One mip level only: a mip-mapped array lays its levels' metadata out otherwise.)
    if !is_depth(k.format) || k.layers <= 1 || k.layers > 128 || k.levels != 1 || k.ty3d || !super::tune::layer_clears() {
        return None;
    }
    let s = k.width.div_ceil(8) as u64 * k.height.div_ceil(8) as u64 * 4;
    (s > 0 && s % (64 << 10) == 0).then_some(s)
}

/// The layers of a layered surface (metadata at `meta`, `per` bytes a layer, `layers`) whose
/// metadata starts inside [b, e) (`fill`: with its first dword inside), a bit a layer; 0: none.
fn layers_reached(meta: u64, per: u64, layers: u32, b: u64, e: u64, fill: bool) -> u128 {
    let mut mask = 0u128;
    for j in 0..layers.min(128) {
        let at = meta + j as u64 * per;
        if at >= b && at < e && (!fill || at + 4 <= e) {
            mask |= 1u128 << j;
        }
    }
    mask
}

/// The word of a metadata block whose first dwords `words` hold one byte value throughout, as
/// a fast clear leaves it (DCC codes 00, 40, 80, C0, 10, 20 and FF; HTILE 0 and all ones);
/// None: other data (`Exec::record_fast_clears`, CP DMA fills).
pub fn uniform_word(words: &[u32]) -> Option<u32> {
    let first = *words.first()?;
    (first.to_le_bytes().iter().all(|&x| x == first as u8) && words.iter().all(|&w| w == first)).then_some(first)
}

#[cfg(test)]
mod clear_tests {
    use super::*;

    fn key(base: u64) -> ImageKey {
        ImageKey { base, format: vk::Format::R8_UNORM, width: 512, height: 512, layers: 1, levels: 1, ty3d: false, cube: false }
    }

    #[test]
    fn fractional_ui_clear_after_metadata_rebind() {
        // CPU offscreen model: 240x135 logical at 1.6 density, two moving UI rectangles.
        // Vulkan is deliberately not needed; exercise the real attachment memo invalidation
        // and pending-clear routing. Merely repainting the new rectangle leaves a trail.
        let mut a = key(0x1000); a.width = 384; a.height = 216;
        let mut b = a; b.width = 240; b.height = 135;
        let surface = Surface { base: a.base, format: a.format, width: a.width, height: a.height,
            layers: 1, levels: 1, ty3d: false, mip: 0, slice: 0, stencil_base: 0, meta: 0x8000, clear: [0; 2] };
        let target = Target { key: a, image: vk::Image::null(), view: vk::ImageView::null(),
            used: UseStamp::new(0), format: a.format, width: a.width, height: a.height };
        let mut metadata = std::collections::BTreeMap::new();
        assert!(bind_clear_metadata(&mut metadata, surface.meta, (a, [0; 2])));
        tls(|m| { m.targets.insert(surface_key(&surface), target); });
        let mut pixels = vec![0u8; 384 * 216];
        for x in 32..64 { pixels[32 * 384 + x] = 255; }
        // A differently sized incarnation is bound between frames.
        assert!(bind_clear_metadata(&mut metadata, surface.meta, (b, [0; 2])));
        let cached = tls(|m| m.targets.get(&surface_key(&surface)).copied());
        assert!(cached.is_none(), "stale attachment would leave the clear routed to the other image");
        assert!(bind_clear_metadata(&mut metadata, surface.meta, (a, [0; 2])));
        let mut pending = std::collections::HashMap::new();
        merge_clear(&mut pending, metadata[&surface.meta].0, surface.meta, Some(0), 0);
        if pending.remove(&a).is_some() { pixels.fill(0); }
        for x in 48..80 { pixels[48 * 384 + x] = 255; }
        assert!(pixels[32 * 384 + 32..32 * 384 + 64].iter().all(|v| *v == 0));
        assert!(pixels[48 * 384 + 48..48 * 384 + 80].iter().all(|v| *v == 255));
        let generation = LOOKUP_GEN.load(std::sync::atomic::Ordering::Acquire);
        assert!(!bind_clear_metadata(&mut metadata, surface.meta, (a, [0; 2])));
        assert_eq!(LOOKUP_GEN.load(std::sync::atomic::Ordering::Acquire), generation);
        // Same image with different clear words must invalidate too.
        assert!(bind_clear_metadata(&mut metadata, surface.meta, (a, [1; 2])));
    }

    #[test]
    fn others_room_only_when_it_fits() {
        const GIB: u64 = 1 << 30;
        // (This RTX 4090: others ~6 GB beside ours ~11 of 24: the room as before.)
        assert_eq!(others_room(6 * GIB, 11 * GIB, 24 * GIB), Some(18 * GIB));
        // (Issue #5's 12 GB card: others counted at 11.7 and 38 GB beside ours ~5.8: not taken.)
        assert_eq!(others_room(11 * GIB + 700 * (1 << 20), 5 * GIB + 800 * (1 << 20), 12 * GIB), None);
        assert_eq!(others_room(38 * GIB, 5 * GIB, 12 * GIB), None);
        // (Exactly full still counts; one byte more does not.)
        assert_eq!(others_room(4 * GIB, 8 * GIB, 12 * GIB), Some(8 * GIB));
        assert_eq!(others_room(4 * GIB + 1, 8 * GIB, 12 * GIB), None);
        assert_eq!(others_room(u64::MAX, 1, 12 * GIB), None);
    }

    #[test]
    fn others_taken_holds_against_our_overshoot() {
        const GIB: u64 = 1 << 30;
        const MIB: u64 = 1 << 20;
        let (margin, heap24, heap12) = (512 * MIB, 24 * GIB + 142 * MIB, 12 * GIB + 115 * MIB);
        // (2026-11-02's case: others 7.8 GB really resident, ours past their limit at 20 GB:
        // still taken, the room kept.)
        let floor24 = 11 * GIB + 700 * MIB;
        assert_eq!(others_taken(7 * GIB + 800 * MIB, 20 * GIB, heap24, margin, floor24), Some(heap24 - 7 * GIB - 800 * MIB));
        // (This PC tonight: others 6.1 GB beside ours 11: taken as before.)
        assert!(others_taken(6 * GIB + 100 * MIB, 11 * GIB, heap24, margin, floor24).is_some());
        // (dwm's leak, others counted at 22 GB: not taken; the floor's eviction cost hitches.)
        assert_eq!(others_taken(22 * GIB, 11 * GIB, heap24, margin, floor24), None);
        // (Issue #5's 12 GB card: 11.7 and 38 GB counted beside ours 5.8: not taken.)
        let floor12 = 5 * GIB + 600 * MIB;
        assert_eq!(others_taken(11 * GIB + 700 * MIB, 5 * GIB + 800 * MIB, heap12, margin, floor12), None);
        assert_eq!(others_taken(38 * GIB, 5 * GIB + 800 * MIB, heap12, margin, floor12), None);
        // (A small real count on that card: taken.)
        assert!(others_taken(GIB, 6 * GIB, heap12, margin, floor12).is_some());
    }

    #[test]
    fn uniform_words() {
        assert_eq!(uniform_word(&[0x4040_4040; 16]), Some(0x4040_4040));
        assert_eq!(uniform_word(&[0; 16]), Some(0));
        assert_eq!(uniform_word(&[u32::MAX; 1]), Some(u32::MAX));
        assert_eq!(uniform_word(&[0x3F80_0000; 16]), None);
        let mut w = [0x8080_8080u32; 16];
        w[9] = 0x8080_8000;
        assert_eq!(uniform_word(&w), None);
        assert_eq!(uniform_word(&[]), None);
    }

    #[test]
    fn fill_kept_aside() {
        let mut m: HashMap<ImageKey, PendingClear> = HashMap::default();
        let k = key(0x7_0000_0000);
        // A fill, then other data over the block: the write replaces it, the fill's value kept.
        merge_clear(&mut m, k, 0x2000, Some(0), 0);
        merge_clear(&mut m, k, 0x2000, None, 0);
        assert_eq!(m.get(&k), Some(&(0x2000, None, Some(0), 0)));
        // More unknown writes keep it.
        merge_clear(&mut m, k, 0x2000, None, 0);
        assert_eq!(m.get(&k), Some(&(0x2000, None, Some(0), 0)));
        // A later fill takes the place of the one kept aside.
        merge_clear(&mut m, k, 0x2000, Some(u32::MAX), 0);
        assert_eq!(m.get(&k), Some(&(0x2000, Some(u32::MAX), None, 0)));
        // An unknown write with no fill before keeps nothing aside; nor does one at other
        // metadata (the image rendered with another block).
        let k2 = key(0x7_0010_0000);
        merge_clear(&mut m, k2, 0x3000, None, 0);
        assert_eq!(m.get(&k2), Some(&(0x3000, None, None, 0)));
        merge_clear(&mut m, k, 0x2800, None, 0);
        assert_eq!(m.get(&k), Some(&(0x2800, None, None, 0)));
        // Taken (used or dropped), nothing stays behind for a later unknown write.
        m.remove(&k);
        merge_clear(&mut m, k, 0x2000, None, 0);
        assert_eq!(m.get(&k), Some(&(0x2000, None, None, 0)));
    }

    #[test]
    fn held_entries_merge() {
        let k = key(0x7_0020_0000);
        // A stretch that filled v1 and then wrote other data, over a committed fill v0: v1 kept.
        let mut c: HashMap<ImageKey, PendingClear> = HashMap::default();
        merge_clear(&mut c, k, 0x4000, Some(0x40), 0);
        let mut h: HashMap<ImageKey, PendingClear> = HashMap::default();
        merge_clear(&mut h, k, 0x4000, Some(0x80), 0);
        merge_clear(&mut h, k, 0x4000, None, 0);
        merge_pending(&mut c, k, h[&k]);
        assert_eq!(c.get(&k), Some(&(0x4000, None, Some(0x80), 0)));
        // A stretch that only wrote other data over a committed fill: the committed one kept.
        let mut c: HashMap<ImageKey, PendingClear> = HashMap::default();
        merge_clear(&mut c, k, 0x4000, Some(0x40), 0);
        merge_pending(&mut c, k, (0x4000, None, None, 0));
        assert_eq!(c.get(&k), Some(&(0x4000, None, Some(0x40), 0)));
        // A stretch's own fill wins.
        merge_pending(&mut c, k, (0x4000, Some(0xC0), None, 0));
        assert_eq!(c.get(&k), Some(&(0x4000, Some(0xC0), None, 0)));
    }

    #[test]
    fn layer_clears() {
        // The shadow atlas: 1024x1024 D32, 80 layers, 64 KiB of HTILE a layer.
        let atlas = ImageKey { base: 0x10_ab90_0000, format: vk::Format::D32_SFLOAT, width: 1024, height: 1024, layers: 80, levels: 1, ty3d: false, cube: false };
        let per = meta_layer_bytes(&atlas).unwrap();
        assert_eq!(per, 0x10000);
        let m = 0x10_0820_0000u64;
        // Layer 0's fill reaches layer 0 alone, layer 5's layer 5, the 4 layers of a cascade map
        // all 4.
        assert_eq!(layers_reached(m, per, 80, m, m + per, true), 1);
        assert_eq!(layers_reached(m, per, 80, m + 5 * per, m + 6 * per, true), 1 << 5);
        assert_eq!(layers_reached(m, per, 4, m, m + 4 * per, true), 0xF);
        // A fill ending inside a layer's first dword does not reach it.
        assert_eq!(layers_reached(m, per, 80, m + per - 8, m + per + 2, true), 0);
        // One layer image and colour images: whole-image clears.
        assert_eq!(meta_layer_bytes(&ImageKey { layers: 1, ..atlas }), None);
        assert_eq!(meta_layer_bytes(&ImageKey { format: vk::Format::R16G16B16A16_SFLOAT, ..atlas }), None);
        // Pending layers add up; a whole-image clear wins.
        let mut c: HashMap<ImageKey, PendingClear> = HashMap::default();
        merge_clear(&mut c, atlas, m, Some(0), 1);
        merge_clear(&mut c, atlas, m, Some(0), 1 << 5);
        assert_eq!(c[&atlas].3, 1 | 1 << 5);
        merge_pending(&mut c, atlas, (m, Some(0), None, 0));
        assert_eq!(c[&atlas].3, 0);
        // A mip-mapped depth array: whole-image clears (its levels' metadata lies otherwise).
        assert_eq!(meta_layer_bytes(&ImageKey { levels: 2, ..atlas }), None);
    }

    /// Fills of layers keep their values to their layers: a value kept aside answers only for
    /// the layers it filled, and fills of other values over other layers take none.
    #[test]
    fn layer_merges() {
        let atlas = ImageKey { base: 0x10_ab90_0000, format: vk::Format::D32_SFLOAT, width: 1024, height: 1024, layers: 80, levels: 1, ty3d: false, cube: false };
        let m = 0x10_0820_0000u64;
        let mut c: HashMap<ImageKey, PendingClear> = HashMap::default();
        // Layer 3 filled, then an unknown write reaching layer 5 only: layer 3's value is not
        // kept for it.
        merge_clear(&mut c, atlas, m, Some(0), 1 << 3);
        merge_clear(&mut c, atlas, m, None, 1 << 5);
        assert_eq!(c[&atlas], (m, None, None, 1 << 3 | 1 << 5));
        // An unknown write over the filled layer keeps the fill aside.
        c.clear();
        merge_clear(&mut c, atlas, m, Some(0), 1 << 3);
        merge_clear(&mut c, atlas, m, None, 1 << 3);
        assert_eq!(c[&atlas], (m, None, Some(0), 1 << 3));
        // Two layers filled with other values: no one value for both.
        c.clear();
        merge_clear(&mut c, atlas, m, Some(0), 1 << 3);
        merge_clear(&mut c, atlas, m, Some(u32::MAX), 1 << 5);
        assert_eq!(c[&atlas], (m, None, None, 1 << 3 | 1 << 5));
        // The same value, or a later fill over every pending layer: that value.
        c.clear();
        merge_clear(&mut c, atlas, m, Some(0), 1 << 3);
        merge_clear(&mut c, atlas, m, Some(0), 1 << 5);
        assert_eq!(c[&atlas], (m, Some(0), None, 1 << 3 | 1 << 5));
        merge_clear(&mut c, atlas, m, Some(u32::MAX), 1 << 3 | 1 << 5 | 1 << 7);
        assert_eq!(c[&atlas], (m, Some(u32::MAX), None, 1 << 3 | 1 << 5 | 1 << 7));
        // A layer's unknown write, then its fill, then another layer's: no one value (each run
        // reads its own back when recorded: a flush at the use, never another layer's value).
        // (Since 2026-11-21 callers note a fill's own range as the fill alone,
        // `note_meta_writes_but_fill`: per-layer fills of one value merge as just above.)
        c.clear();
        merge_clear(&mut c, atlas, m, None, 1 << 3);
        merge_clear(&mut c, atlas, m, Some(0), 1 << 3);
        merge_clear(&mut c, atlas, m, None, 1 << 5);
        merge_clear(&mut c, atlas, m, Some(0), 1 << 5);
        assert_eq!(c[&atlas], (m, None, None, 1 << 3 | 1 << 5));
        // A held stretch's fill of another value over another layer: no one value either.
        c.clear();
        merge_clear(&mut c, atlas, m, Some(0), 1 << 3);
        merge_pending(&mut c, atlas, (m, Some(u32::MAX), None, 1 << 5));
        assert_eq!(c[&atlas], (m, None, None, 1 << 3 | 1 << 5));
        // A stretch's kept value answers only where it covers the map's pending layers.
        c.clear();
        merge_clear(&mut c, atlas, m, Some(0), 1 << 3);
        merge_pending(&mut c, atlas, (m, None, Some(u32::MAX), 1 << 5));
        assert_eq!(c[&atlas], (m, None, None, 1 << 3 | 1 << 5));
        c.clear();
        merge_clear(&mut c, atlas, m, Some(0), 1 << 3);
        merge_pending(&mut c, atlas, (m, None, Some(u32::MAX), 1 << 3 | 1 << 5));
        assert_eq!(c[&atlas], (m, None, Some(u32::MAX), 1 << 3 | 1 << 5));
    }
}

/// Whether any fast clear or overwrite reload is pending.
pub fn any_pending() -> bool {
    PENDING.load(std::sync::atomic::Ordering::Acquire) != 0
}

/// The images with a fast clear or an overwrite reload pending (published
/// by `with`), and its generation.
static PENDING_SET: Mutex<Option<std::sync::Arc<super::fx::Set<ImageKey>>>> = Mutex::new(None);
static PENDING_GEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

thread_local! {
    static PENDING_TLS: std::cell::RefCell<(u64, std::sync::Arc<super::fx::Set<ImageKey>>)> = std::cell::RefCell::new((u64::MAX, std::sync::Arc::new(Default::default())));
}

/// Whether any of `keys` has a fast clear or an overwrite reload pending
/// (`clears_pending || overwrites_pending` without the cache's lock).
pub fn pending_hit(keys: &[ImageKey]) -> bool {
    if !any_pending() {
        return false;
    }
    let g = PENDING_GEN.load(std::sync::atomic::Ordering::Acquire);
    PENDING_TLS.with(|t| {
        let mut t = t.borrow_mut();
        if t.0 != g {
            *t = (g, PENDING_SET.lock().clone().unwrap_or_default());
        }
        keys.iter().any(|k| t.1.contains(k))
    })
}

type SurfaceKey = (u64, i32, u32, u32, u32, u32, bool, u32, u32, u64, u64, [u32; 2]);

fn surface_key(s: &Surface) -> SurfaceKey {
    (s.base, s.format.as_raw(), s.width, s.height, s.layers, s.levels, s.ty3d, s.mip, s.slice, s.stencil_base, s.meta, s.clear)
}

/// Register a target's clear-metadata binding. Attachment and fixed-draw memos must not
/// hide a later rebind after another layout/image used the same metadata allocation.
/// NOTE_GEN alone invalidates write notes, not those attachment lookups.
pub(crate) static METADATA_REBINDS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
fn bind_clear_metadata(map: &mut std::collections::BTreeMap<u64, (ImageKey, [u32; 2])>, meta: u64, binding: (ImageKey, [u32; 2])) -> bool {
    let previous = map.insert(meta, binding);
    if previous == Some(binding) { return false; }
    if previous.is_some() {
        METADATA_REBINDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        LOOKUP_GEN.fetch_add(1, std::sync::atomic::Ordering::Release);
        if trace_clears() {
            crate::info!("trace clears: metadata {meta:#x} rebound from {:?} to {:?}; attachment memos invalidated", previous, binding);
        }
    }
    true
}

/// Entries the memos of T# lookups hold before they are emptied (the
/// per-thread `Tls::keys` and `Tls::views`, the cache's `key_memo` and
/// `view_memo`); they are made with room for that many, so an insert never
/// grows them. Growing one rehashed it under the cache's lock: 1-6 ms on
/// one recording thread while every other one waited for that lock (the
/// vista's stalled graphics runs; FINDINGS.md 2026-11-09). Their entries
/// are T#s, whose words change as the title streams (the memos filled to
/// this cap within minutes).
const MEMO_CAP: usize = 1 << 16;

/// A T#'s words as the memos of T# lookups key it: without the hardware LOD
/// statistics' fields (word 5 bit 25, LOD_HDW_CNT_EN; word 6 bits 0..7, the
/// counter bank: `lodstats`), which the title's streamer changes every frame
/// for the textures it lends banks to and which neither the image nor its
/// view depends on (each preparation reads them from the T# itself). Keyed
/// with them, 98% of the T#s new to a thread's memo at the vista differed
/// from one it held only there (~190 a flip; in-run x1.013, FINDINGS.md
/// 2026-11-09).
#[inline]
fn memo_raw(t: &Tsharp) -> [u32; 8] {
    let mut r = t.raw;
    r[5] &= !(1 << 25);
    r[6] &= !0xFF;
    r
}

/// A memo of T# lookups made with room for MEMO_CAP entries.
fn memo_map<K, V>() -> HashMap<K, V> {
    memo_map_of(MEMO_CAP)
}

fn memo_map_of<K, V>(cap: usize) -> HashMap<K, V> {
    HashMap::with_capacity_and_hasher(cap, Default::default())
}

/// The per-thread memos' cap (`Tls::keys`, `Tls::views`): with the LOD
/// statistics' fields out of their keys they hold a thread's live T#s;
/// 4,096 and 65,536 measured level at the vista (FINDINGS.md 2026-11-09),
/// this one a quarter of the latter's memory a thread.
const TLS_MEMO_CAP: usize = 1 << 14;

fn tls_cap() -> usize {
    TLS_MEMO_CAP
}

/// Per-thread memos of the lookups every draw repeats (the recording
/// threads of a parallel run all wanted the cache's one lock, which made
/// every draw step 1.5-2x slower with three workers). Keys, targets, depth
/// growth and GPU-written bases hold while LOOKUP_GEN does (`gen`); views
/// also need the frame and the tracker's generation (`view`'s refresh);
/// samplers are never dropped.
#[derive(Default)]
struct Tls {
    lookup_gen: u64,
    /// The cap `keys` and `views` were made with (`tls_cap`).
    cap: usize,
    /// Resolved keys with the generations they hold for (GPU_IMAGES_GEN, the T#'s `base_gen`).
    keys: HashMap<([u32; 8], bool), (Option<ImageKey>, u64, u32)>,
    targets: HashMap<SurfaceKey, Target>,
    grows: HashMap<(u64, u32, u32, i32, u32), Option<ImageKey>>,
    gpu_base: HashMap<u64, bool>,
    depth_feed: HashMap<u64, bool>,
    views: HashMap<[u32; 8], ViewMemo>,
    samplers: HashMap<([u32; 4], bool, u8), vk::Sampler>,
}

/// The PC settings' texture filtering: 0 the title's own, otherwise the anisotropy (8, 16) its
/// mipmapped surface samplers get (new samplers from the next bind; `pcsettings::apply`).
pub static ANISO: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

thread_local! {
    static TLS: std::cell::RefCell<Tls> = std::cell::RefCell::new(Tls { cap: tls_cap(), keys: memo_map_of(tls_cap()), views: memo_map_of(tls_cap()), ..Tls::default() });
}

/// `f` on this thread's memos, those valid for one LOOKUP_GEN emptied if
/// it moved.
fn tls<R>(f: impl FnOnce(&mut Tls) -> R) -> R {
    TLS.with(|t| {
        let mut t = t.borrow_mut();
        let g = LOOKUP_GEN.load(std::sync::atomic::Ordering::Acquire);
        if t.lookup_gen != g {
            t.lookup_gen = g;
            t.targets.clear();
            t.grows.clear();
            t.gpu_base.clear();
            t.depth_feed.clear();
        }
        f(&mut t)
    })
}

/// Memo statistics (DS_PROFILE_GPU; `profile report`): view lookups [hit,
/// not memoized, other frame, image-cache generation moved, write-tracking
/// generation moved], key lookups [hit, miss], and the generations at the
/// last `profile reset` [lookup, tracking].
static MEMO_STATS: [std::sync::atomic::AtomicU64; 7] = [const { std::sync::atomic::AtomicU64::new(0) }; 7];
static MEMO_GENS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];
static TRACK_WHY_AT_RESET: [std::sync::atomic::AtomicU64; 6] = [const { std::sync::atomic::AtomicU64::new(0) }; 6];

fn memo_note(k: usize) {
    if super::exec::profiling_on() {
        MEMO_STATS[k].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Clears the memo statistics (`profile reset`).
pub fn reset_memo_stats() {
    for c in &MEMO_STATS {
        c.store(0, std::sync::atomic::Ordering::Relaxed);
    }
    MEMO_GENS[0].store(LOOKUP_GEN.load(std::sync::atomic::Ordering::Relaxed), std::sync::atomic::Ordering::Relaxed);
    MEMO_GENS[1].store(super::track::GEN.load(std::sync::atomic::Ordering::Relaxed), std::sync::atomic::Ordering::Relaxed);
    for (r, c) in TRACK_WHY_AT_RESET.iter().zip(super::track::GEN_WHY.iter()) {
        r.store(c.load(std::sync::atomic::Ordering::Relaxed), std::sync::atomic::Ordering::Relaxed);
    }
}

/// One profile line on the per-thread memos.
pub fn memo_stats_line() -> String {
    let s: Vec<u64> = MEMO_STATS.iter().map(|c| c.load(std::sync::atomic::Ordering::Relaxed)).collect();
    format!(
        "    image memos: views hit {}, missed {} (not memoized {}, other frame {}, image cache changed {}, write tracking changed {}); keys hit {}, missed {}; generations since reset: image cache {}, write tracking {} (forced dirty, page not writable, CPU write fault, host write, GPU write, remap: {:?})\n",
        s[0],
        s[1] + s[2] + s[3] + s[4],
        s[1],
        s[2],
        s[3],
        s[4],
        s[5],
        s[6],
        LOOKUP_GEN.load(std::sync::atomic::Ordering::Relaxed).saturating_sub(MEMO_GENS[0].load(std::sync::atomic::Ordering::Relaxed)),
        super::track::GEN.load(std::sync::atomic::Ordering::Relaxed).saturating_sub(MEMO_GENS[1].load(std::sync::atomic::Ordering::Relaxed)),
        super::track::GEN_WHY.iter().zip(TRACK_WHY_AT_RESET.iter()).map(|(c, r)| c.load(std::sync::atomic::Ordering::Relaxed).saturating_sub(r.load(std::sync::atomic::Ordering::Relaxed))).collect::<Vec<u64>>()
    )
}

/// `Images::view` from this thread's memo (sampled views only).
pub fn view_hit(t: &Tsharp, frame: u64) -> Option<(vk::ImageView, vk::Image)> {
    let any_frame = scrubbing();
    let e = tls(|m| m.views.get(&memo_raw(t)).copied());
    let k = e.map_or(1, |e| e.miss(t, frame, any_frame));
    memo_note(k);
    if k == 1 && tsharp_diff_on() {
        tsharp_diff(t);
    }
    if k == 0 {
        e.map(|e| {
            e.used.stamp(frame);
            (e.view, e.image)
        })
    } else {
        None
    }
}

/// `DS_TSHARP_DIFF=1` (diagnosis): a T# this thread's view memo did not
/// hold is compared with the last one it saw at the same base address; the
/// dwords that differ are counted (`tsharp_diff_line`) and the first 24
/// pairs logged: which T# words the title changes for one texture.
fn tsharp_diff_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_TSHARP_DIFF").ok().as_deref() == Some("1"))
}

/// [misses, misses with an earlier T# at the base, then per dword the
/// misses where it differed].
static TSHARP_DIFF: [std::sync::atomic::AtomicU64; 10] = [const { std::sync::atomic::AtomicU64::new(0) }; 10];
/// The bits that differed, per dword (OR of the pairs' XORs).
static TSHARP_XOR: [std::sync::atomic::AtomicU32; 8] = [const { std::sync::atomic::AtomicU32::new(0) }; 8];

thread_local! {
    static LAST_AT_BASE: std::cell::RefCell<HashMap<u64, [u32; 8]>> = std::cell::RefCell::new(HashMap::default());
}

fn tsharp_diff(t: &Tsharp) {
    use std::sync::atomic::Ordering::Relaxed;
    TSHARP_DIFF[0].fetch_add(1, Relaxed);
    let before = LAST_AT_BASE.with(|m| m.borrow_mut().insert(t.base, t.raw));
    let Some(b) = before.filter(|b| *b != t.raw) else { return };
    TSHARP_DIFF[1].fetch_add(1, Relaxed);
    for i in 0..8 {
        if b[i] != t.raw[i] {
            TSHARP_DIFF[2 + i].fetch_add(1, Relaxed);
            TSHARP_XOR[i].fetch_or(b[i] ^ t.raw[i], Relaxed);
        }
    }
    // Pairs that differ only in dwords 5-7 (the first 24 of them too).
    static LOGGED_HI: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    if b[..5] == t.raw[..5] && LOGGED_HI.fetch_add(1, Relaxed) < 24 {
        let w = |r: &[u32; 8]| r.iter().map(|x| format!("{x:08x}")).collect::<Vec<_>>().join(" ");
        crate::info!("images: T# at {:#x} changed in dwords 5-7: [{}] -> [{}]", t.base, w(&b), w(&t.raw));
    }
    static LOGGED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    if LOGGED.fetch_add(1, Relaxed) < 24 {
        let w = |r: &[u32; 8]| r.iter().map(|x| format!("{x:08x}")).collect::<Vec<_>>().join(" ");
        crate::info!("images: T# at {:#x} changed: [{}] -> [{}]", t.base, w(&b), w(&t.raw));
    }
}

/// The T# change census (`DS_TSHARP_DIFF`; "" when off).
pub fn tsharp_diff_line() -> String {
    if !tsharp_diff_on() {
        return String::new();
    }
    use std::sync::atomic::Ordering::Relaxed;
    let v: Vec<u64> = TSHARP_DIFF.iter().map(|c| c.load(Relaxed)).collect();
    let x: Vec<String> = TSHARP_XOR.iter().map(|c| format!("{:08x}", c.load(Relaxed))).collect();
    format!("    T#s new to a thread's view memo (`DS_TSHARP_DIFF`): {}, {} of them another T# at a base seen before; differing dwords 0-7: {:?}; bits that differed: [{}]\n", v[0], v[1], &v[2..], x.join(" "))
}

/// `Images::sampler` from this thread's memo.
pub fn sampler_hit(s: [u32; 4], compare: bool) -> Option<vk::Sampler> {
    let aniso = ANISO.load(std::sync::atomic::Ordering::Relaxed);
    tls(|m| m.samplers.get(&(s, compare, aniso)).copied())
}

/// `Images::resolved_key` from this thread's memo (None: not memoized).
pub fn key_hit(t: &Tsharp, storage: bool) -> Option<Option<ImageKey>> {
    use std::sync::atomic::Ordering::Acquire;
    let r = tls(|m| m.keys.get(&(memo_raw(t), storage)).copied()).filter(|e| e.1 == GPU_IMAGES_GEN.load(Acquire) && e.2 == base_gen(t.base).load(Acquire)).map(|e| e.0);
    memo_note(if r.is_some() { 5 } else { 6 });
    r
}

/// `Images::target`, from this thread's memo when it holds.
pub fn target(s: &Surface) -> Result<Target, String> {
        let _cpu = crate::gpu::cpu_profile::scope(crate::gpu::cpu_profile::Part::ImageLookup);
    let k = surface_key(s);
    if let Some(t) = tls(|m| m.targets.get(&k).copied()) {
        t.used.stamp(super::FRAMES.load(std::sync::atomic::Ordering::Relaxed));
        return Ok(t);
    }
    let g = LOOKUP_GEN.load(std::sync::atomic::Ordering::Acquire);
    let t = with(|im| im.target(s))?;
    tls(|m| {
        if m.lookup_gen == g {
            m.targets.insert(k, t);
        }
    });
    Ok(t)
}

/// `Images::depth_to_grow`, from this thread's memo when it holds.
pub fn depth_to_grow(base: u64, width: u32, height: u32, format: vk::Format, layers: u32) -> Option<ImageKey> {
    let k = (base, width, height, format.as_raw(), layers);
    if let Some(r) = tls(|m| m.grows.get(&k).copied()) {
        return r;
    }
    let g = LOOKUP_GEN.load(std::sync::atomic::Ordering::Acquire);
    let r = with(|im| im.depth_to_grow(base, width, height, Images::depth_family(format), layers));
    tls(|m| {
        if m.lookup_gen == g {
            m.grows.insert(k, r);
        }
    });
    r
}

/// Whether block copies may be pending for `keys` (`block_copies_possible`
/// from this thread's memo of GPU-written bases).
pub fn block_copies_possible(keys: &[ImageKey]) -> bool {
    let cands: Vec<u64> = keys.iter().filter(|k| is_compressed(k.format) && !k.ty3d).map(|k| k.base).collect();
    if cands.is_empty() {
        return false;
    }
    let known: Option<bool> = tls(|m| {
        let mut any = false;
        for b in &cands {
            match m.gpu_base.get(b) {
                Some(&x) => any |= x,
                None => return None,
            }
        }
        Some(any)
    });
    if let Some(x) = known {
        return x;
    }
    let g = LOOKUP_GEN.load(std::sync::atomic::Ordering::Acquire);
    let flags: Vec<bool> = with(|im| cands.iter().map(|b| im.gpu_bases.contains(b)).collect());
    tls(|m| {
        if m.lookup_gen == g {
            for (b, f) in cands.iter().zip(&flags) {
                m.gpu_base.insert(*b, *f);
            }
        }
    });
    flags.iter().any(|&f| f)
}

/// The color format GPU work writes a depth surface's depth plane through
/// (the title's compute shaders store depth as R32 float texels under D32,
/// R16 unorm under D16).
fn depth_feed_format(depth: vk::Format) -> Option<vk::Format> {
    use vk::Format as F;
    match depth {
        F::D32_SFLOAT | F::D32_SFLOAT_S8_UINT => Some(F::R32_SFLOAT),
        F::D16_UNORM | F::D16_UNORM_S8_UINT => Some(F::R16_UNORM),
        _ => None,
    }
}

/// Whether depth copies may be pending for `keys` (`depth_feed_possible`
/// from this thread's memo of depth-feed bases).
pub fn depth_copies_possible(keys: &[ImageKey]) -> bool {
    let cands: Vec<u64> = keys.iter().filter(|k| depth_feed_format(k.format).is_some()).map(|k| k.base).collect();
    if cands.is_empty() {
        return false;
    }
    let known: Option<bool> = tls(|m| {
        let mut any = false;
        for b in &cands {
            match m.depth_feed.get(b) {
                Some(&x) => any |= x,
                None => return None,
            }
        }
        Some(any)
    });
    if let Some(x) = known {
        return x;
    }
    let g = LOOKUP_GEN.load(std::sync::atomic::Ordering::Acquire);
    let flags: Vec<bool> = with(|im| cands.iter().map(|b| im.depth_feeds.contains(b)).collect());
    tls(|m| {
        if m.lookup_gen == g {
            for (b, f) in cands.iter().zip(&flags) {
                m.depth_feed.insert(*b, *f);
            }
        }
    });
    flags.iter().any(|&f| f)
}

/// A depth image's depth plane to refresh from the color image GPU work
/// stored it through (`Images::take_depth_copies`).
pub struct DepthCopy {
    pub key: ImageKey,
    pub dst: vk::Image,
    pub src: vk::Image,
    pub width: u32,
    pub height: u32,
    pub layers: u32,
    /// Bytes per texel of the source (and of the buffer the copy passes through).
    pub texel: u32,
    pub seq: u64,
}

/// Vulkan format of a gfx10 image FORMAT.
pub fn vk_format(fmt: u16) -> Option<vk::Format> {
    use vk::Format as F;
    Some(match fmt {
        1 => F::R8_UNORM,
        2 => F::R8_SNORM,
        5 => F::R8_UINT,
        6 => F::R8_SINT,
        7 => F::R16_UNORM,
        8 => F::R16_SNORM,
        11 => F::R16_UINT,
        12 => F::R16_SINT,
        13 => F::R16_SFLOAT,
        14 => F::R8G8_UNORM,
        15 => F::R8G8_SNORM,
        18 => F::R8G8_UINT,
        19 => F::R8G8_SINT,
        20 => F::R32_UINT,
        21 => F::R32_SINT,
        22 => F::R32_SFLOAT,
        23 => F::R16G16_UNORM,
        24 => F::R16G16_SNORM,
        27 => F::R16G16_UINT,
        28 => F::R16G16_SINT,
        29 => F::R16G16_SFLOAT,
        36 => F::B10G11R11_UFLOAT_PACK32,
        50 => F::A2B10G10R10_UNORM_PACK32,
        54 => F::A2B10G10R10_UINT_PACK32,
        56 => F::R8G8B8A8_UNORM,
        57 => F::R8G8B8A8_SNORM,
        60 => F::R8G8B8A8_UINT,
        61 => F::R8G8B8A8_SINT,
        62 => F::R32G32_UINT,
        63 => F::R32G32_SINT,
        64 => F::R32G32_SFLOAT,
        65 => F::R16G16B16A16_UNORM,
        66 => F::R16G16B16A16_SNORM,
        69 => F::R16G16B16A16_UINT,
        70 => F::R16G16B16A16_SINT,
        71 => F::R16G16B16A16_SFLOAT,
        75 => F::R32G32B32A32_UINT,
        76 => F::R32G32B32A32_SINT,
        77 => F::R32G32B32A32_SFLOAT,
        128 => F::R8_SRGB,
        129 => F::R8G8_SRGB,
        130 => F::R8G8B8A8_SRGB,
        132 => F::E5B9G9R9_UFLOAT_PACK32,
        133 => F::B5G6R5_UNORM_PACK16,
        169 => F::BC1_RGBA_UNORM_BLOCK,
        170 => F::BC1_RGBA_SRGB_BLOCK,
        171 => F::BC2_UNORM_BLOCK,
        172 => F::BC2_SRGB_BLOCK,
        173 => F::BC3_UNORM_BLOCK,
        174 => F::BC3_SRGB_BLOCK,
        175 => F::BC4_UNORM_BLOCK,
        176 => F::BC4_SNORM_BLOCK,
        177 => F::BC5_UNORM_BLOCK,
        178 => F::BC5_SNORM_BLOCK,
        179 => F::BC6H_UFLOAT_BLOCK,
        180 => F::BC6H_SFLOAT_BLOCK,
        181 => F::BC7_UNORM_BLOCK,
        182 => F::BC7_SRGB_BLOCK,
        _ => return None,
    })
}

/// Bytes per element: texel, or 4x4 block of compressed formats (log2).
/// Bytes of guest memory a T#'s resource spans (all mips and layers), if
/// its format and swizzle mode are known.
pub fn guest_size(t: &Tsharp) -> Option<u64> {
    let key = Images::key_of(t)?;
    let el = elem_log2(t.format)?;
    if key.ty3d {
        return tiling::layout_3d(t.sw_mode, el, key.width, key.height, key.layers).map(|l| l.size());
    }
    tiling::layout(t.sw_mode, el, key.width, key.height, key.levels, is_block(t.format)).map(|l| l.size(key.layers))
}

pub fn elem_log2(fmt: u16) -> Option<u32> {
    let f = rdna::format::gfx10(fmt)?;
    Some(f.bytes().trailing_zeros())
}

fn is_block(fmt: u16) -> bool {
    (169..=182).contains(&fmt)
}

/// Bytes per texel (per 4x4 block for compressed formats); 0 if unknown.
pub fn texel_bytes(f: vk::Format) -> u32 {
    use vk::Format as F;
    match f {
        F::R8_UNORM | F::R8_SNORM | F::R8_UINT | F::R8_SINT | F::R8_SRGB | F::S8_UINT => 1,
        F::R16_UNORM | F::R16_SNORM | F::R16_UINT | F::R16_SINT | F::R16_SFLOAT | F::R8G8_UNORM | F::R8G8_SNORM | F::R8G8_UINT | F::R8G8_SINT | F::R8G8_SRGB | F::B5G6R5_UNORM_PACK16 | F::D16_UNORM => 2,
        F::R32_UINT | F::R32_SINT | F::R32_SFLOAT | F::R16G16_UNORM | F::R16G16_SNORM | F::R16G16_UINT | F::R16G16_SINT | F::R16G16_SFLOAT | F::B10G11R11_UFLOAT_PACK32 | F::A2B10G10R10_UNORM_PACK32 | F::A2B10G10R10_UINT_PACK32
        | F::A2R10G10B10_UNORM_PACK32 | F::R8G8B8A8_UNORM | F::R8G8B8A8_SNORM | F::R8G8B8A8_UINT | F::R8G8B8A8_SINT | F::R8G8B8A8_SRGB | F::B8G8R8A8_UNORM | F::B8G8R8A8_SRGB | F::E5B9G9R9_UFLOAT_PACK32 | F::D32_SFLOAT => 4,
        F::R32G32_UINT | F::R32G32_SINT | F::R32G32_SFLOAT | F::R16G16B16A16_UNORM | F::R16G16B16A16_SNORM | F::R16G16B16A16_UINT | F::R16G16B16A16_SINT | F::R16G16B16A16_SFLOAT | F::D32_SFLOAT_S8_UINT => 8,
        F::R32G32B32A32_UINT | F::R32G32B32A32_SINT | F::R32G32B32A32_SFLOAT => 16,
        F::BC1_RGBA_UNORM_BLOCK | F::BC1_RGBA_SRGB_BLOCK | F::BC4_UNORM_BLOCK | F::BC4_SNORM_BLOCK => 8,
        F::BC2_UNORM_BLOCK | F::BC2_SRGB_BLOCK | F::BC3_UNORM_BLOCK | F::BC3_SRGB_BLOCK | F::BC5_UNORM_BLOCK | F::BC5_SNORM_BLOCK | F::BC6H_UFLOAT_BLOCK | F::BC6H_SFLOAT_BLOCK | F::BC7_UNORM_BLOCK | F::BC7_SRGB_BLOCK => 16,
        _ => 0,
    }
}

/// GPU-owned images cleared right after a zero fill of their memory
/// (`Images::zero_filled`).
pub static ZERO_FILL_CLEARS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// `Images::keyed_views`.
static KEYED_VIEWS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

pub fn is_depth(f: vk::Format) -> bool {
    matches!(f, vk::Format::D16_UNORM | vk::Format::D32_SFLOAT | vk::Format::D32_SFLOAT_S8_UINT | vk::Format::D16_UNORM_S8_UINT | vk::Format::D24_UNORM_S8_UINT | vk::Format::S8_UINT)
}

pub fn has_stencil(f: vk::Format) -> bool {
    matches!(f, vk::Format::D32_SFLOAT_S8_UINT | vk::Format::D16_UNORM_S8_UINT | vk::Format::D24_UNORM_S8_UINT | vk::Format::S8_UINT)
}

/// All aspects of a format.
pub fn aspects(f: vk::Format) -> vk::ImageAspectFlags {
    match f {
        vk::Format::D16_UNORM | vk::Format::D32_SFLOAT => vk::ImageAspectFlags::DEPTH,
        vk::Format::S8_UINT => vk::ImageAspectFlags::STENCIL,
        f if is_depth(f) => vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
        _ => vk::ImageAspectFlags::COLOR,
    }
}

fn is_compressed(f: vk::Format) -> bool {
    (vk::Format::BC1_RGB_UNORM_BLOCK.as_raw()..=vk::Format::BC7_SRGB_BLOCK.as_raw()).contains(&f.as_raw())
}

/// Two formats can view the same image (MUTABLE_FORMAT): same texel size,
/// both color, both compressed or both not.
fn view_compatible(a: vk::Format, b: vk::Format) -> bool {
    a == b || (texel_bytes(a) != 0 && texel_bytes(a) == texel_bytes(b) && !is_depth(a) && !is_depth(b) && is_compressed(a) == is_compressed(b))
}

/// CB color format (CB_COLORn_INFO FORMAT, NUMBER_TYPE, COMP_SWAP) -> Vulkan format.
pub fn cb_format(fmt: u32, ntype: u32, swap: u32) -> Option<vk::Format> {
    use vk::Format as F;
    // NUMBER_TYPE: 0 unorm, 1 snorm, 4 uint, 5 sint, 6 srgb, 7 float.
    let f = match (fmt, ntype) {
        (1, 0) => F::R8_UNORM,
        (1, 1) => F::R8_SNORM,
        (1, 4) => F::R8_UINT,
        (1, 5) => F::R8_SINT,
        (1, 6) => F::R8_SRGB,
        (2, 0) => F::R16_UNORM,
        (2, 1) => F::R16_SNORM,
        (2, 4) => F::R16_UINT,
        (2, 5) => F::R16_SINT,
        (2, 7) => F::R16_SFLOAT,
        (3, 0) => F::R8G8_UNORM,
        (3, 1) => F::R8G8_SNORM,
        (3, 4) => F::R8G8_UINT,
        (3, 5) => F::R8G8_SINT,
        (3, 6) => F::R8G8_SRGB,
        (4, 4) => F::R32_UINT,
        (4, 5) => F::R32_SINT,
        (4, 7) => F::R32_SFLOAT,
        (5, 0) => F::R16G16_UNORM,
        (5, 1) => F::R16G16_SNORM,
        (5, 4) => F::R16G16_UINT,
        (5, 5) => F::R16G16_SINT,
        (5, 7) => F::R16G16_SFLOAT,
        (6, 7) => F::B10G11R11_UFLOAT_PACK32,
        (9, 0) => F::A2B10G10R10_UNORM_PACK32,
        (9, 4) => F::A2B10G10R10_UINT_PACK32,
        (10, 0) => F::R8G8B8A8_UNORM,
        (10, 1) => F::R8G8B8A8_SNORM,
        (10, 4) => F::R8G8B8A8_UINT,
        (10, 5) => F::R8G8B8A8_SINT,
        (10, 6) => F::R8G8B8A8_SRGB,
        (11, 4) => F::R32G32_UINT,
        (11, 5) => F::R32G32_SINT,
        (11, 7) => F::R32G32_SFLOAT,
        (12, 0) => F::R16G16B16A16_UNORM,
        (12, 1) => F::R16G16B16A16_SNORM,
        (12, 4) => F::R16G16B16A16_UINT,
        (12, 5) => F::R16G16B16A16_SINT,
        (12, 7) => F::R16G16B16A16_SFLOAT,
        (14, 4) => F::R32G32B32A32_UINT,
        (14, 5) => F::R32G32B32A32_SINT,
        (14, 7) => F::R32G32B32A32_SFLOAT,
        (16, 0) => F::B5G6R5_UNORM_PACK16,
        _ => return None,
    };
    // COMP_SWAP 1 (ALT) on 4-component formats stores BGRA.
    Some(match (f, swap) {
        (_, 0) => f,
        (F::R8G8B8A8_UNORM, 1) => F::B8G8R8A8_UNORM,
        (F::R8G8B8A8_SRGB, 1) => F::B8G8R8A8_SRGB,
        (F::A2B10G10R10_UNORM_PACK32, 1) => F::A2R10G10B10_UNORM_PACK32,
        _ => return None,
    })
}

pub fn half_to_f32(h: u16) -> f32 {
    let s = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = ((h >> 10) & 0x1F) as i32;
    let m = (h & 0x3FF) as f32;
    s * match e {
        0 => m * 2f32.powi(-24),
        31 if m == 0.0 => f32::INFINITY,
        31 => f32::NAN,
        _ => (1.0 + m / 1024.0) * 2f32.powi(e - 15),
    }
}

/// Converts tightly packed texels to RGBA8 (screenshots, dumps).
pub fn to_rgba8(format: vk::Format, raw: &[u8]) -> Option<Vec<u8>> {
    use vk::Format as F;
    let unit = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    Some(match format {
        F::R8G8B8A8_UNORM | F::R8G8B8A8_SRGB => raw.to_vec(),
        F::B8G8R8A8_UNORM | F::B8G8R8A8_SRGB => raw.chunks_exact(4).flat_map(|p| [p[2], p[1], p[0], p[3]]).collect(),
        F::A2B10G10R10_UNORM_PACK32 | F::A2R10G10B10_UNORM_PACK32 => raw
            .chunks_exact(4)
            .flat_map(|p| {
                let v = u32::from_le_bytes([p[0], p[1], p[2], p[3]]);
                let lo = ((v & 0x3FF) >> 2) as u8;
                let g = (((v >> 10) & 0x3FF) >> 2) as u8;
                let hi = (((v >> 20) & 0x3FF) >> 2) as u8;
                let a = ((v >> 30) * 85) as u8;
                if format == F::A2B10G10R10_UNORM_PACK32 { [lo, g, hi, a] } else { [hi, g, lo, a] }
            })
            .collect(),
        F::R16G16B16A16_SFLOAT => raw
            .chunks_exact(8)
            .flat_map(|p| {
                let h = |i: usize| half_to_f32(u16::from_le_bytes([p[2 * i], p[2 * i + 1]]));
                [unit(h(0)), unit(h(1)), unit(h(2)), unit(h(3))]
            })
            .collect(),
        F::R8_UNORM => raw.iter().flat_map(|&v| [v, v, v, 255]).collect(),
        F::R32_SFLOAT => raw
            .chunks_exact(4)
            .flat_map(|p| {
                let v = unit(f32::from_le_bytes([p[0], p[1], p[2], p[3]]));
                [v, v, v, 255]
            })
            .collect(),
        F::R16G16_SFLOAT => raw
            .chunks_exact(4)
            .flat_map(|p| [unit(half_to_f32(u16::from_le_bytes([p[0], p[1]]))), unit(half_to_f32(u16::from_le_bytes([p[2], p[3]]))), 0, 255])
            .collect(),
        F::R32_UINT => raw.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2], 255]).collect(),
        F::R8_UINT => raw.iter().flat_map(|&v| [v, v, v, 255]).collect(),
        F::R8G8_UINT | F::R8G8_UNORM => raw.chunks_exact(2).flat_map(|p| [p[0], p[1], 0, 255]).collect(),
        F::R16_SFLOAT => raw
            .chunks_exact(2)
            .flat_map(|p| {
                let v = unit(half_to_f32(u16::from_le_bytes([p[0], p[1]])));
                [v, v, v, 255]
            })
            .collect(),
        _ => return None,
    })
}

/// Per-channel statistics of a float image (finite min / max / mean, NaN
/// and infinity counts): HDR targets clamp to white or black as PNG.
pub fn float_stats(format: vk::Format, raw: &[u8]) -> Option<String> {
    use vk::Format as F;
    let (channels, half) = match format {
        F::R16G16B16A16_SFLOAT => (4, true),
        F::R16G16_SFLOAT => (2, true),
        F::R16_SFLOAT => (1, true),
        F::R32G32B32A32_SFLOAT => (4, false),
        F::R32G32_SFLOAT => (2, false),
        // Integer targets also as f32 bits (probed values, packed data).
        F::R32_SFLOAT | F::R32_UINT | F::R32_SINT => (1, false),
        _ => return None,
    };
    let size = if half { 2 } else { 4 };
    let mut st = vec![(f32::INFINITY, f32::NEG_INFINITY, 0f64, 0u64, 0u64, 0u64); channels];
    for texel in raw.chunks_exact(size * channels) {
        for (c, s) in st.iter_mut().enumerate() {
            let p = &texel[c * size..];
            let v = if half { half_to_f32(u16::from_le_bytes([p[0], p[1]])) } else { f32::from_le_bytes([p[0], p[1], p[2], p[3]]) };
            if v.is_nan() {
                s.4 += 1;
            } else if v.is_infinite() {
                s.5 += 1;
            } else {
                s.0 = s.0.min(v);
                s.1 = s.1.max(v);
                s.2 += v as f64;
                s.3 += 1;
            }
        }
    }
    let parts: Vec<String> = st
        .iter()
        .zip("rgba".chars())
        .map(|(s, n)| {
            let mean = if s.3 > 0 { s.2 / s.3 as f64 } else { 0.0 };
            format!("{n} [{:.4}, {:.4}] mean {mean:.4} nan {} inf {}", s.0, s.1, s.4, s.5)
        })
        .collect();
    let prefix = if matches!(format, F::R32_UINT | F::R32_SINT) { "integer target interpreted as f32 bits: " } else { "" };
    Some(format!("{prefix}{}", parts.join("; ")))
}

/// A render target or depth/stencil attachment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Target {
    pub key: ImageKey,
    pub image: vk::Image,
    pub view: vk::ImageView,
    /// The image's use stamp (memoized targets stamp it too).
    pub used: UseStamp,
    /// Attachment (view) format.
    pub format: vk::Format,
    /// Size of the attached mip level.
    pub width: u32,
    pub height: u32,
}

/// Color or depth surface described by CB/DB registers.
#[derive(Clone, Copy, Debug)]
pub struct Surface {
    pub base: u64,
    pub format: vk::Format,
    pub width: u32,
    pub height: u32,
    /// Array layers of the resource (3D: depth).
    pub layers: u32,
    pub levels: u32,
    pub ty3d: bool,
    pub mip: u32,
    pub slice: u32,
    /// Address of the stencil plane (depth/stencil surfaces with stencil).
    pub stencil_base: u64,
    /// Metadata a fast clear writes (DCC of color, HTILE of depth; 0: none).
    pub meta: u64,
    /// Clear values: CB_COLOR_CLEAR_WORD0/1, or DB_DEPTH_CLEAR and DB_STENCIL_CLEAR.
    pub clear: [u32; 2],
}

/// A fast clear found pending for an image: its metadata and clear values.
#[derive(Clone, Copy)]
pub struct FastClear {
    pub key: ImageKey,
    pub image: vk::Image,
    pub meta: u64,
    pub clear: [u32; 2],
    /// The metadata's first word, known from the fill that wrote it (no read
    /// back after the GPU work).
    pub known: Option<u32>,
    /// A fill's value a later write of unknown data replaced (`PendingClear`).
    pub kept: Option<u32>,
    /// The layers to clear, a bit a layer (`PendingClear`'s fourth field); 0: the whole image.
    pub layers: u128,
}

/// Blocks of a block-compressed texture that GPU work stored through an
/// uncompressed view of its memory (one texel per block), to be copied in.
pub struct BlockCopy {
    pub key: ImageKey,
    /// The view of the whole chain (null when only the tail was written):
    /// its mips `mips`.
    pub src: vk::Image,
    pub dst: vk::Image,
    /// Block grid of mip 0.
    pub width: u32,
    pub height: u32,
    /// The mips copied from `src` (each the same level of both).
    pub mips: Vec<u32>,
    pub layers: u32,
    /// The mip tail stored through a view of the tail block alone (one
    /// swizzle block at the surface's base, a single level the size of the
    /// block in elements: the title's transcoders write the small mips
    /// there): the view and, per tail mip, (mip, x, y, width, height) in
    /// blocks (where the mip sits inside the tail block).
    pub tail: Option<(vk::Image, Vec<(u32, u32, u32, u32, u32)>)>,
    seq: u64,
}

/// UNORM equivalent of sRGB formats (storage views cannot be sRGB).
fn storage_format(f: vk::Format) -> vk::Format {
    match f {
        vk::Format::R8G8B8A8_SRGB => vk::Format::R8G8B8A8_UNORM,
        vk::Format::R8G8_SRGB => vk::Format::R8G8_UNORM,
        vk::Format::R8_SRGB => vk::Format::R8_UNORM,
        x => x,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ImageKey {
    pub base: u64,
    pub format: vk::Format,
    pub width: u32,
    pub height: u32,
    /// Array layers (cube faces), or depth of 3D images.
    pub layers: u32,
    pub levels: u32,
    pub ty3d: bool,
    pub cube: bool,
}

pub struct Image {
    pub image: vk::Image,
    pub key: ImageKey,
    /// Views by (format, type, base mip, mips, base layer, layers, swizzle,
    /// usage: 0 sampled, 1 storage, 2 attachment). Usage is part of the key:
    /// a view created for one usage is not valid for another.
    views: HashMap<(vk::Format, vk::ImageViewType, u32, u32, u32, u32, [u8; 4], u8), vk::ImageView>,
    /// Written by the GPU: guest memory is no longer its source.
    pub gpu_written: bool,
    /// Write-tracking region of the guest bytes the upload came from.
    source: Option<u32>,
    /// Sampled hashes of those bytes, one per page slice (see slice_hash).
    hashes: [u64; SLICES],
    /// Frame of the last slice check.
    checked_frame: u64,
    /// Generation of its last binding as a storage image (see `Images::write_seq`).
    write_gen: u64,
    /// Block-compressed textures: generation of the block view last copied
    /// in (see `take_block_copies`).
    block_gen: u64,
    /// The (mip, layer)s whose blocks GPU work wrote through block views
    /// (block copies), with the hash of their own guest bytes then
    /// (`own_part_hash`): guest memory never holds those blocks, so a
    /// re-upload leaves such a part as it is while its guest bytes stay
    /// the same (`choose_2d`).
    gpu_parts: Vec<(u32, u32, u64)>,
    /// GPU-written images: the write generation of each mip level a
    /// storage view last had as its base (a storage view writes its base
    /// level only), 0 for levels never bound so (`block_copies`).
    mip_gens: Vec<u64>,
    /// Guest bytes of its layout and the T# that described it exactly (0 /
    /// None: not seen through a T#): GPU buffer writes over GPU-owned images.
    span: u64,
    layout: Option<Tsharp>,
    /// GPU buffer writes or CP DMA covered all of its guest bytes since its
    /// last use: guest memory holds its contents (`reload_overwritten`).
    overwritten: bool,
    /// The shader that last bound it as a storage image (diagnosis:
    /// `fillimg cs=`).
    writer: u64,
    /// Its tracked region's generation (`track::gen_of`; with `source`).
    source_gen: Option<&'static std::sync::atomic::AtomicU32>,
    /// Its device memory and bytes, and the frame of its last use.
    mem: super::slab::Mem,
    bytes: u64,
    used: UseStamp,
}

/// An image's use stamp: the frame of its last lookup, memoized lookups
/// included, so that the least recently used images can go when video
/// memory runs short (`Images::evict`). Leaked (8 bytes per image ever
/// made): the memos read it without the cache's lock.
#[derive(Clone, Copy)]
pub struct UseStamp(&'static Stamps);

/// An image's use stamp and the frames draws wrote it (`UseStamp::drawn`:
/// the last such frame shifted left by 16, a bit per frame before it).
pub struct Stamps {
    used: std::sync::atomic::AtomicU64,
    drawn: std::sync::atomic::AtomicU64,
}

impl UseStamp {
    fn new(frame: u64) -> Self {
        UseStamp(Box::leak(Box::new(Stamps { used: std::sync::atomic::AtomicU64::new(frame), drawn: std::sync::atomic::AtomicU64::new(0) })))
    }

    /// Notes a use in `frame` (one store per image and frame).
    #[inline]
    pub fn stamp(&self, frame: u64) {
        if self.0.used.load(std::sync::atomic::Ordering::Relaxed) < frame {
            self.0.used.store(frame, std::sync::atomic::Ordering::Relaxed);
        }
    }

    fn get(&self) -> u64 {
        self.0.used.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Notes that a draw writes the image as an attachment in `frame` (one
    /// store per image and frame; racing threads may lose a frame's bit).
    #[inline]
    pub fn stamp_drawn(&self, frame: u64) {
        let v = self.0.drawn.load(std::sync::atomic::Ordering::Relaxed);
        let last = v >> 16;
        if last == frame && v & 1 != 0 {
            return;
        }
        let mask = match frame.checked_sub(last) {
            Some(d) if d < 16 => ((v & 0xFFFF) << d) | 1,
            _ => 1,
        };
        self.0.drawn.store(frame << 16 | (mask & 0xFFFF), std::sync::atomic::Ordering::Relaxed);
    }

    /// In how many of the 4 frames before `frame` draws wrote the image.
    pub fn drawn_before(&self, frame: u64) -> u32 {
        let v = self.0.drawn.load(std::sync::atomic::Ordering::Relaxed);
        let last = v >> 16;
        let Some(d) = frame.checked_sub(last) else { return 0 };
        if d >= 5 {
            return 0;
        }
        // Bit j of `rel`: drawn in frame `frame - j`.
        let rel = (v & 0xFFFF) << d;
        (rel & 0b11110).count_ones()
    }
}

impl PartialEq for UseStamp {
    fn eq(&self, o: &Self) -> bool {
        std::ptr::eq(self.0, o.0)
    }
}

impl Eq for UseStamp {}

impl std::fmt::Debug for UseStamp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "used at {}", self.get())
    }
}

/// An image the cache dropped (`Images::evict`, `grow_depth`), destroyed
/// once no batch can use it any more (`exec::bury_images`).
pub struct Evicted {
    pub image: vk::Image,
    pub mem: super::slab::Mem,
    pub views: Vec<vk::ImageView>,
}

/// Video memory of the cached images, bytes.
static IMAGE_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Images evicted: [count, bytes, eviction rounds].
static EVICTED: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];
/// An uploaded image is evicted under pressure only when unused for this
/// many frames. GPU-written ones, whose contents guest memory does not
/// hold, are never evicted: evicting those unused for 600 frames (data a
/// level load wrote and drew much later) lost 4-2's architecture in a
/// stress tour, and taking images over memory the title unmapped for dead
/// broke every level (the title maps its direct memory again with its
/// contents; 2026-10-19).
const EVICT_AGE: u64 = 120;
/// The age for a texture in video memory while video memory is under pressure (`under_pressure`,
/// with the system-memory fallback on): it comes back in system memory (`create_texture`), so
/// the video memory it held goes to the mirror. Tried without the fallback on 2026-11-22 small
/// hours, when evicted textures came back in video memory: no effect above the tour's spread
/// (TRIED.md).
const EVICT_AGE_PRESSURE: u64 = 30;
/// Video-memory budget checks at most this often.
const EVICT_CHECK_MS: u64 = 500;
/// Textures moved back from system memory a check at most (`promote`), bytes: each comes back at
/// its next use as an upload.
const PROMOTE_MAX: u64 = 128 << 20;

/// On a card with less than `SMALL_CARD` of video memory (`vk::card_size`), an image of
/// `BIG_IMAGE` bytes or more is made in system memory from the start (render targets, grown depth
/// images, storage images and textures; `create_in`): the shadow atlas's depth (320 MiB), the
/// lighting volumes (~280 MiB), texture arrays (342 and 86 MiB) and the title's 3840x2160 RGBA16F
/// targets (63 MiB each) among them. The mirror then has the room it needs to draw right
/// (NVIDIA's sparse buffers take video memory only). In a 4 GB room the mirror got ~1.8 of its
/// ~2.5 GB: 82-116 million placements went without it over a tour and one of three tours drew
/// 8-2 with most of its geometry missing (2026-11-22); at 64 MiB the census still found 1.11 GB
/// of GPU-written images and 0.51 GB of textures in video memory. The cost: the GPU reads and
/// writes those images across the bus on such a card.
/// Under `TINY_CARD` (a 4 GB card leaves a game ~3.5 GB) the line is `BIG_IMAGE_TINY`: at 3.5 GB
/// the 32 MiB line left 0.82 GB of render targets in video memory, the mirror got 1.73 GB and 8-2
/// drew wrong (19.2), 85 million placements without the mirror (-115433); nearly every render
/// target then goes to system memory, read and written across the bus on such a card.
const SMALL_CARD: u64 = 5632 << 20;
const BIG_IMAGE: u64 = 32 << 20;
const TINY_CARD: u64 = 4608 << 20;
const BIG_IMAGE_TINY: u64 = 8 << 20;

/// Whether `key` is such an image on such a card (`SMALL_CARD`, with the fallback on).
fn big_on_small_card(key: &ImageKey) -> bool {
    if !super::vk::system_fallback() {
        return false;
    }
    let card = super::vk::card_size();
    let line = if card < TINY_CARD { BIG_IMAGE_TINY } else if card < SMALL_CARD { BIG_IMAGE } else { return false };
    // (Block-compressed formats: `texel_bytes` is a 4x4 block's; until the pre-release review of
    // 2026-11-22 afternoon a texel was counted as a block, every BC texture 16 times its size.)
    let (w, h) = if is_compressed(key.format) { (key.width.div_ceil(4), key.height.div_ceil(4)) } else { (key.width, key.height) };
    w as u64 * h as u64 * key.layers.max(1) as u64 * texel_bytes(key.format).max(1) as u64 >= line
}
/// A texture used within this many flips stays where it is (`promote`).
const PROMOTE_AGE: u64 = 2;
/// This process over its eviction mark (`tune vram_high` percent of its budget) at the last
/// check (`maybe_evict`): textures then go to system memory first (`create_texture`), the mark's
/// last 15% left to the mirror. With pressure alone (an allocation failed in the last 10 s) the
/// 7 GB tour of 2026-11-22 morning still bound 1.2 million mirror placements as zeros: each spell
/// ended, new textures filled video memory again and the mirror's next block failed.
static OVER_HIGH: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// The share of the budget other processes' memory never pushes eviction
/// below (percent). With others at ~24.5 GB the 1-1 walk's Castlewall and
/// Front ran level without that eviction (44.5 / 54.1 against 44.2 / 55.3
/// flips/s), its p99 frames lower (34-42 against 50-63 ms at Castlewall)
/// and its detiling about halved (FINDINGS.md 2026-11-15 late night).
const OTHERS_FLOOR_PCT: u64 = 50;

/// Images not made for want of memory (or another failure of their creation) and the last such key;
/// front end 2's placements whose mirror pages were bound as zeros (`exec`). Counted for the census
/// beside the over-budget warning, whose own messages stop at their cap early in a session (the 7 GB
/// tours of 2026-11-22: a session went wrong for good at World 8 with no line saying why).
pub static IMAGE_FAILS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static ZERO_PLACEMENTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static LAST_IMAGE_FAIL: parking_lot::Mutex<Option<ImageKey>> = parking_lot::Mutex::new(None);

/// Video memory under pressure until this time (`pressure_clock_ms`): set when evicting textures
/// left this process over the high mark (`maybe_evict`), read by the mirror's cold-page thread
/// (`resident::maybe_unback`), which then gives cold pages back sooner. And how often it was set.
static PRESSURE_UNTIL_MS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static PRESSURE_SPELLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// How long a pressure reading holds (the eviction checks come every `EVICT_CHECK_MS`).
const PRESSURE_HOLD_MS: u64 = 10_000;

fn pressure_clock_ms() -> u64 {
    static T0: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    T0.get_or_init(std::time::Instant::now).elapsed().as_millis() as u64
}

/// A device-memory allocation failed (the mirror, shadows, batches): video memory under pressure
/// from now on for `PRESSURE_HOLD_MS`. A spell begun is counted in `PRESSURE_SPELLS` (until
/// 2026-11-21 night every call was: 5 million in one session, each a failure that the refusal
/// window, `vk::refuse_alloc`, now spares the driver; the failures are `vk::GATE_STATS`).
pub fn note_pressure() {
    use std::sync::atomic::Ordering::Relaxed;
    let now = pressure_clock_ms();
    if PRESSURE_UNTIL_MS.swap(now + PRESSURE_HOLD_MS, Relaxed) <= now {
        PRESSURE_SPELLS.fetch_add(1, Relaxed);
    }
}
/// The highest share of its video-memory budget this process used, per mille, and that budget in
/// MiB (`maybe_evict`'s readings, every `EVICT_CHECK_MS`; the preparation's health line).
pub static VRAM_PEAK: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// The status line of video-memory pressure (`PRESSURE_SPELLS`).
pub fn pressure_line() -> String {
    let n = PRESSURE_SPELLS.load(std::sync::atomic::Ordering::Relaxed);
    if n == 0 {
        return String::new();
    }
    format!(
        "    video memory under pressure {n} times (spells of 10 s or more: textures alone could not bring this process under its mark, or an allocation failed; the mirror's cold pages given back sooner); in all {} mirror placements bound as zeros, {} images not made\n",
        ZERO_PLACEMENTS.load(std::sync::atomic::Ordering::Relaxed),
        IMAGE_FAILS.load(std::sync::atomic::Ordering::Relaxed)
    )
}

/// Whether video memory is under pressure now (`PRESSURE_UNTIL_MS`).
pub fn under_pressure() -> bool {
    PRESSURE_UNTIL_MS.load(std::sync::atomic::Ordering::Relaxed) > pressure_clock_ms()
}

/// What holds this process's video memory when it is over its budget: images (slabs and their
/// own), the mirror's blocks, the batch slots, and the rest (the driver's own memory for
/// pipelines, descriptors and the like, and anything not counted here). Added 2026-11-21 night: a
/// test with 7 GB of room ended a session at 12.6 GB used, every counted allocation refused past
/// the room, and nothing said what had grown.
pub fn over_budget_line(used: u64, budget: u64) -> String {
    let (img, mirror, slots) = (super::slab::held_bytes(), super::resident::mirror_bytes(), super::exec::slot_device_bytes());
    let mib = |b: Option<u64>| b.map_or("? (busy)".to_string(), |b| format!("{}", b >> 20));
    let rest = match (img, mirror) {
        (Some(i), Some(m)) => Some(used.saturating_sub(i + m + slots)),
        _ => None,
    };
    format!(
        "video memory over this process's budget: {} of {} MiB at flip {}: images {} MiB, the mirror {} MiB, batch slots {} MiB, the rest {} MiB; device allocations failed {}, refused with no driver call {}",
        used >> 20,
        budget >> 20,
        super::FRAMES.load(std::sync::atomic::Ordering::Relaxed),
        mib(img),
        mib(mirror),
        slots >> 20,
        mib(rest),
        super::vk::GATE_STATS[1].load(std::sync::atomic::Ordering::Relaxed),
        super::vk::GATE_STATS[0].load(std::sync::atomic::Ordering::Relaxed),
    )
}

/// The heap's room beside other processes' dedicated video memory, as the image cache's limit
/// takes it (`maybe_evict`), or None when Windows' count of it does not fit on the card beside
/// ours: then it cannot all be resident (WDDM counts allocations it has paged out to system
/// memory too), and yielding to it only evicts the scene's own textures. A player's 12 GB card
/// counted others at 11.7 GB and, in a later session, 38 GB (issue #5, 2026-11-24): the cache
/// stayed at its floor, half the budget, evicting and uploading all session, and the pressure
/// it signalled split the shader preparation 10 times. Counts that fit (this RTX 4090: others
/// ~6 GB beside ours ~11 of 24) give the room as before.
fn others_room(others: u64, ours: u64, heap: u64) -> Option<u64> {
    (others.saturating_add(ours) <= heap).then(|| heap - others)
}

/// `others_room` as `maybe_evict` asks it: ours counted at most at the limit the count itself
/// sets (the heap less others and `margin`, never under `floor`), so that our own use passing
/// that limit (a load's overshoot within a check, idle slabs) cannot switch the limit off; a
/// count that leaves less than `floor` beside it is not taken (the review of 2026-11-24: with
/// ours counted whole, others really resident at 7.8 GB on the RTX 4090 would have lost their
/// room for good once ours passed 15.8 GB).
fn others_taken(others: u64, used: u64, heap: u64, margin: u64, floor: u64) -> Option<u64> {
    let mark = heap.saturating_sub(others).saturating_sub(margin).max(floor);
    others_room(others, used.min(mark), heap)
}

/// When this process uses more video memory than `tune vram_high` percent of
/// the device-local budget, evicts the least recently used images until
/// about `vram_low` percent (completion thread, at most every
/// EVICT_CHECK_MS; skipped while the cache is busy). Other processes'
/// dedicated video memory counts too (`tune vram_others`): the budget the
/// driver reports here leaves it out (23.4 of the RTX 4090's 24.1 GB while
/// others held 7.8 GB: textures of four levels, 11.8 GB, were kept and the
/// GPU's time per frame rose as ours passed 16 GB, FINDINGS.md 2026-11-02).
pub fn maybe_evict() {
    use std::sync::atomic::Ordering::Relaxed;
    if !super::tune::image_evict() {
        return;
    }
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    static LAST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let ms = START.get_or_init(std::time::Instant::now).elapsed().as_millis() as u64 + EVICT_CHECK_MS;
    if ms < LAST.load(Relaxed) + EVICT_CHECK_MS {
        return;
    }
    LAST.store(ms, Relaxed);
    let frame = super::FRAMES.load(Relaxed);
    let Some((raw_used, budget)) = ctx().device_budget() else { return };
    // (Slab memory no image uses is free for new images.)
    let used = raw_used.saturating_sub(super::slab::idle_bytes());
    if budget > 0 {
        let pm = used.saturating_mul(1000) / budget;
        if pm > VRAM_PEAK[0].load(Relaxed) {
            VRAM_PEAK[0].store(pm, Relaxed);
            VRAM_PEAK[1].store(budget >> 20, Relaxed);
        }
        // (Over the budget: what holds the memory, at most every 10 s.)
        static LAST_OVER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        if pm > 1000 && ms >= LAST_OVER.load(Relaxed) + 10_000 {
            LAST_OVER.store(ms, Relaxed);
            crate::warn!("{}", over_budget_line(raw_used, budget));
        }
    }
    let (mut high, mut low) = (budget / 100 * super::tune::vram_high(), budget / 100 * super::tune::vram_low());
    // What the heap leaves beside other processes' dedicated memory, less
    // `vram_others` MiB (high) and a GiB more (low), but never under
    // OTHERS_FLOOR_PCT of the budget: others' memory that grows over a day
    // (dwm.exe and the NVIDIA overlay held 22 GB on 2026-11-15) otherwise
    // evicts the scene's own textures every half second.
    let margin = super::tune::vram_others() << 20;
    if margin != 0 {
        if let Some(others) = crate::hostload::others_vram() {
            // (The card as the choices made by its size see it: `DS_SIM_VRAM_MIB`'s smaller one.)
            let heap = super::vk::card_size();
            let floor = budget / 100 * OTHERS_FLOOR_PCT;
            let taken = others_taken(others, used, heap, margin, floor);
            // (Each change of whether the count is taken, at most every 10 s.)
            static LAST: [std::sync::atomic::AtomicU64; 2] = [std::sync::atomic::AtomicU64::new(2), std::sync::atomic::AtomicU64::new(0)];
            let state = taken.is_some() as u64;
            if LAST[0].load(Relaxed) != state && ms >= LAST[1].load(Relaxed) + 10_000 {
                LAST[0].store(state, Relaxed);
                LAST[1].store(ms, Relaxed);
                crate::info!(
                    "images: other processes' video memory as Windows counts it, {} MiB, {} (the card {} MiB, ours {} MiB, {} with idle slabs; issue #5)",
                    others >> 20,
                    if taken.is_some() { "taken: the image cache leaves it room" } else { "not taken: it does not fit on the card beside ours, so not all of it is resident; the image cache keeps to its budget" },
                    heap >> 20,
                    used >> 20,
                    raw_used >> 20
                );
            }
            if let Some(room) = taken {
                high = high.min(room.saturating_sub(margin).max(floor));
                low = low.min(room.saturating_sub(margin + (1 << 30)).max(floor.saturating_sub(1 << 30)));
            }
        }
    }
    OVER_HIGH.store(budget > 0 && used > high, Relaxed);
    // (Room again under the low mark, no pressure: textures in system memory move back, as many
    // as fit under it, `PROMOTE_MAX` a check at most: evicted, they come back to video memory at
    // their next use.)
    if budget > 0 && used < low && !under_pressure() && super::vk::SYSTEM_STATS[0].load(Relaxed) > 0 {
        if let Some(mut g) = IMAGES.try_lock() {
            if let Some(im) = g.as_mut() {
                let out = im.promote(frame, (low - used).min(PROMOTE_MAX));
                drop(g);
                if !out.is_empty() {
                    super::vk::SYSTEM_STATS[3].fetch_add(out.len() as u64, Relaxed);
                    super::exec::bury_images(out);
                }
            }
        }
    }
    if budget == 0 || used <= high {
        return;
    }
    let want = used.saturating_sub(low);
    let Some(mut g) = IMAGES.try_lock() else { return };
    let Some(im) = g.as_mut() else { return };
    let out = im.evict(frame, want);
    // (Over budget: the GPU-written images' census beside the warning, at most every 10 s.)
    static LAST_CENSUS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    if raw_used > budget && ms >= LAST_CENSUS.load(Relaxed) + 10_000 {
        LAST_CENSUS.store(ms, Relaxed);
        let (all, cold) = im.gpu_written_census(frame, 600);
        let (vgw, vtex, vtop) = im.video_census(4);
        let (tex, top) = im.texture_census();
        // (Since the last census: placements drawn from zeros, images not made, the last of them.)
        static SEEN: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];
        let (z, f) = (ZERO_PLACEMENTS.load(Relaxed), IMAGE_FAILS.load(Relaxed));
        let (dz, df) = (z - SEEN[0].swap(z, Relaxed), f - SEEN[1].swap(f, Relaxed));
        let last = if df > 0 { LAST_IMAGE_FAIL.lock().map(|k| format!(" (the last {}x{}x{} {:?} at {:#x})", k.width, k.height, k.layers, k.format, k.base)).unwrap_or_default() } else { String::new() };
        crate::info!(
            "images: GPU-written {} MiB, {} MiB of it unused for 600 flips or more; textures {} MiB, ~{} MiB of it the top mips of large ones; since the last: {dz} placements bound as zeros, {df} images not made{last}; images in system memory now {} MiB; in video memory GPU-written {} MiB, textures {} MiB, the largest {vtop} (at flip {frame})",
            all >> 20,
            cold >> 20,
            tex >> 20,
            top >> 20,
            super::vk::SYSTEM_STATS[0].load(Relaxed) >> 20,
            vgw >> 20,
            vtex >> 20
        );
    }
    // (Textures alone could not bring it under the high mark: the rest is memory images the GPU
    // wrote, the mirror and the like. The mirror then gives its cold pages back sooner for a while,
    // `under_pressure`: the other PC's 12 GB card ran 2 GB over its budget with 45 MiB of textures
    // left to evict, and its allocations failed, 2026-11-21.)
    if used.saturating_sub(out.1) > high {
        note_pressure();
    }
    // What holds the memory (the first reports).
    let largest = if EVICTED[2].load(Relaxed) < 5 { format!("; largest: {}", im.largest(6)) } else { String::new() };
    drop(g);
    if out.0.is_empty() {
        return;
    }
    let rounds = EVICTED[2].fetch_add(1, Relaxed) + 1;
    if rounds <= 20 || rounds % 100 == 0 {
        crate::info!(
            "images: video memory {} of budget {} MiB (evicting above {} MiB, others' {} MiB) at flip {frame}: evicted {} textures ({} MiB), {} images cached ({} MiB){largest}",
            used >> 20,
            budget >> 20,
            high >> 20,
            crate::hostload::others_vram().map_or(0, |o| o >> 20),
            out.0.len(),
            out.1 >> 20,
            out.2,
            IMAGE_BYTES.load(Relaxed) >> 20
        );
    }
    super::exec::bury_images(out.0);
}

/// Two GPU timestamps for an upload's command buffer: when the GPU reached
/// it and when it finished (`exec::batch_stamps`; null without them).
fn upload_stamps() -> vk::QueryPool {
    if !super::exec::batch_stamps() {
        return vk::QueryPool::null();
    }
    unsafe { ctx().device.create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(2), None) }.unwrap_or(vk::QueryPool::null())
}

/// Records an upload command buffer's start (k = 0, after a reset) or end
/// (k = 1) timestamp.
unsafe fn stamp(cmd: vk::CommandBuffer, qp: vk::QueryPool, k: u32) {
    if qp == vk::QueryPool::null() {
        return;
    }
    let d = &ctx().device;
    unsafe {
        if k == 0 {
            d.cmd_reset_query_pool(cmd, qp, 0, 2);
        }
        d.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, qp, k);
    }
}

/// Uploads whose GPU span went to the hitch reports.
static UPLOAD_SPANS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// A completed upload's span on the GPU, for the hitch reports' GPU busy
/// time (`hitch::gpu_batch`, queue "uploads").
fn note_upload_span(qp: vk::QueryPool, what: impl FnOnce() -> String) {
    if qp == vk::QueryPool::null() {
        return;
    }
    let mut t = [0u64; 2];
    if unsafe { ctx().device.get_query_pool_results(qp, 0, &mut t, vk::QueryResultFlags::TYPE_64) }.is_ok() {
        let n = UPLOAD_SPANS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        super::hitch::gpu_batch("uploads", n, t[0], t[1], 1, 0, what);
    }
}

/// A host-visible buffer texture data is uploaded from (`staging_take`):
/// its bytes, which the batched uploads' in-flight limit counts.
struct Staging {
    buf: vk::Buffer,
    mem: vk::DeviceMemory,
    ptr: *mut u8,
    cap: u64,
}

/// What a GPU-detiled upload in flight holds until the GPU is done with it:
/// the video-memory buffers it used and the copy queue's command buffer
/// that filled them.
struct Held {
    dev: Vec<(vk::Buffer, vk::DeviceMemory, u64)>,
    copy_cmd: Option<vk::CommandBuffer>,
}

impl Held {
    fn bytes(&self) -> u64 {
        self.dev.iter().map(|b| b.2).sum::<u64>()
    }
}

/// GPU-detiled uploads in flight before the next one waits for the oldest
/// (their bytes are bounded by DETILE_INFLIGHT; an arrival's hundreds of
/// small textures waited for the oldest at 32).
const MAX_INFLIGHT: usize = 256;
/// Uploads that waited for an earlier one (`MAX_INFLIGHT`).
static UPLOAD_WAITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// `tune upload_batch`: uploads detiled on the CPU that would wait for their
/// transfer on the main queue are recorded into one command buffer they all
/// share, submitted ahead of the next batch (`flush_uploads`, from
/// `exec::submit_batch`) or of the cache's own next submission, and not
/// waited for; their staging buffers are freed once it completes. A batch
/// that binds such a texture runs after its upload: on the main queue in
/// its order (the upload's last barrier holds all later work), on the
/// compute family's queue after all the main queue's work submitted before
/// it (ordered async compute); so does any batch an image buried after it
/// waits for.
struct Batched {
    pool: vk::CommandPool,
    /// The open command buffer, its fence, the staging it reads, its bytes.
    open: Option<(vk::CommandBuffer, vk::Fence, Vec<(vk::Buffer, vk::DeviceMemory)>, u64)>,
    inflight: std::collections::VecDeque<(vk::CommandBuffer, vk::Fence, Vec<(vk::Buffer, vk::DeviceMemory)>, u64)>,
    spare: Vec<(vk::CommandBuffer, vk::Fence)>,
}

static BATCHED: Mutex<Option<Batched>> = Mutex::new(None);
/// The open command buffer and those in flight (`flush_uploads` returns at
/// once at 0).
static BATCHED_LIVE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Uploads recorded batched, command buffers submitted, waits for one in
/// flight (`BATCHED_MAX_BYTES`).
static BATCHED_STATS: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];
/// Staging bytes recorded or in flight at most before an upload waits for
/// the oldest command buffer.
const BATCHED_MAX_BYTES: u64 = 256 << 20;
/// Staging bytes the open command buffer reads at most before it is
/// submitted at once (no batch submitted meanwhile: level loads).
const BATCHED_OPEN_BYTES: u64 = 64 << 20;

fn free_batched(held: Vec<(vk::Buffer, vk::DeviceMemory)>) {
    for (buf, mem) in held {
        staging_free(buf, mem);
    }
}

/// Staging buffers of CPU-detiled uploads kept mapped for reuse, by size
/// class (64 KiB << class, up to 64 MiB): each upload had made a buffer,
/// allocated, bound and mapped its memory, written fresh pages and freed it
/// all again once the GPU was done (in play 0.24 ms a MiB uploaded against
/// 0.05 from the pool, FINDINGS.md 2026-11-02 night).
struct StagingPool {
    /// Free buffers per class: buffer, memory, mapped pointer.
    free: [Vec<(vk::Buffer, vk::DeviceMemory, usize)>; STAGING_CLASSES],
    /// Bytes on the free lists (at most `STAGING_POOL_MAX`).
    bytes: u64,
    /// The pool's allocations, free or in use, by memory: class, pointer.
    owned: HashMap<vk::DeviceMemory, (usize, usize)>,
}

const STAGING_CLASSES: usize = 11;
const STAGING_POOL_MAX: u64 = 128 << 20;
static STAGING_POOL: Mutex<Option<StagingPool>> = Mutex::new(None);
/// Staging buffers taken from the pool, made for it, made outside it
/// (diagnostics).
static STAGING_STATS: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

fn staging_class(size: u64) -> Option<usize> {
    let k = size.max(64 << 10).next_power_of_two().trailing_zeros().checked_sub(16)? as usize;
    (k < STAGING_CLASSES).then_some(k)
}

/// A mapped staging buffer of at least `size` bytes for an upload's copy
/// source: a free one of its class from the pool, or made (`staging_free`
/// gives it back).
fn staging_take(size: u64) -> Result<Staging, String> {
    use std::sync::atomic::Ordering::Relaxed;
    let Some(k) = staging_class(size) else {
        STAGING_STATS[2].fetch_add(1, Relaxed);
        let (buf, mem, ptr) = ctx().host_buffer(size.max(4), vk::BufferUsageFlags::TRANSFER_SRC)?;
        return Ok(Staging { buf, mem, ptr, cap: size.max(4) });
    };
    let hit = {
        let mut g = STAGING_POOL.lock();
        let p = g.get_or_insert_with(|| StagingPool { free: Default::default(), bytes: 0, owned: HashMap::default() });
        let e = p.free[k].pop();
        if e.is_some() {
            p.bytes -= 64 << 10 << k;
        }
        e
    };
    if let Some((buf, mem, ptr)) = hit {
        STAGING_STATS[0].fetch_add(1, Relaxed);
        return Ok(Staging { buf, mem, ptr: ptr as *mut u8, cap: 64 << 10 << k });
    }
    STAGING_STATS[1].fetch_add(1, Relaxed);
    let (buf, mem, ptr) = ctx().host_buffer(64 << 10 << k, vk::BufferUsageFlags::TRANSFER_SRC)?;
    if let Some(p) = STAGING_POOL.lock().as_mut() {
        p.owned.insert(mem, (k, ptr as usize));
    }
    Ok(Staging { buf, mem, ptr, cap: 64 << 10 << k })
}

/// Gives a staging buffer back once the GPU is done with it: to the pool's
/// free list if it is the pool's and the list has room, else frees it.
fn staging_free(buf: vk::Buffer, mem: vk::DeviceMemory) {
    {
        let mut g = STAGING_POOL.lock();
        if let Some(p) = g.as_mut() {
            if let Some(&(k, ptr)) = p.owned.get(&mem) {
                let n = 64u64 << 10 << k;
                if p.bytes + n <= STAGING_POOL_MAX {
                    p.free[k].push((buf, mem, ptr));
                    p.bytes += n;
                    return;
                }
                p.owned.remove(&mem);
            }
        }
    }
    let c = ctx();
    unsafe {
        c.device.destroy_buffer(buf, None);
        c.device.unmap_memory(mem);
        c.device.free_memory(mem, None);
    }
}

impl Batched {
    /// Submits the open command buffer, if any, to the main queue.
    fn submit_open(&mut self) {
        let Some((cmd, fence, held, bytes)) = self.open.take() else { return };
        let c = ctx();
        unsafe { c.device.end_command_buffer(cmd).ok() };
        match c.submit(cmd, &[], &[], fence) {
            Ok(()) => {
                self.inflight.push_back((cmd, fence, held, bytes));
                BATCHED_STATS[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            Err(e) => {
                if super::exec::lose_device() {
                    crate::error!("images: batched uploads' submit failed: {e}; GPU execution stops; checkpoints: {}", super::vk::checkpoint_report());
                }
                free_batched(held);
                unsafe { c.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).ok() };
                self.spare.push((cmd, fence));
                BATCHED_LIVE.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }

    /// Frees what completed command buffers held, waiting for the oldest
    /// while more than `max_bytes` are recorded or in flight.
    fn reap(&mut self, max_bytes: u64) {
        let c = ctx();
        while let Some(&(_, fence, _, _)) = self.inflight.front() {
            let done = unsafe { c.device.get_fence_status(fence) }.unwrap_or(true);
            if !done {
                let bytes: u64 = self.inflight.iter().map(|x| x.3).sum::<u64>() + self.open.as_ref().map_or(0, |o| o.3);
                if bytes <= max_bytes {
                    break;
                }
                BATCHED_STATS[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let t0 = std::time::Instant::now();
                unsafe { c.device.wait_for_fences(&[fence], true, 10_000_000_000).ok() };
                super::hitch::note_since(super::hitch::Kind::GpuWait, t0, || format!("batched texture uploads in flight ({} MiB)", bytes >> 20));
            }
            let Some((cmd, fence, held, _)) = self.inflight.pop_front() else { break };
            free_batched(held);
            unsafe {
                c.device.reset_fences(&[fence]).ok();
                c.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).ok();
            }
            self.spare.push((cmd, fence));
            BATCHED_LIVE.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

/// Records an upload (`f`) into the shared open command buffer; its staging
/// (buffer, memory) is freed once that completes.
fn record_batched(staging: (vk::Buffer, vk::DeviceMemory), bytes: u64, f: &dyn Fn(vk::CommandBuffer)) -> Result<(), String> {
    let c = ctx();
    let mut g = BATCHED.lock();
    if g.is_none() {
        let pool = unsafe { c.device.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(c.queue_family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None) }.map_err(|e| format!("batched uploads' pool: {e}"))?;
        *g = Some(Batched { pool, open: None, inflight: Default::default(), spare: Vec::new() });
    }
    let Some(b) = g.as_mut() else { return Err("no batched uploads".into()) };
    b.reap(BATCHED_MAX_BYTES.saturating_sub(bytes));
    if b.open.is_none() {
        let (cmd, fence) = match b.spare.pop() {
            Some(x) => x,
            None => unsafe {
                let cmd = c.device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(b.pool).command_buffer_count(1)).map_err(|e| format!("batched uploads' command buffer: {e}"))?[0];
                (cmd, c.device.create_fence(&vk::FenceCreateInfo::default(), None).map_err(|e| format!("batched uploads' fence: {e}"))?)
            },
        };
        unsafe { c.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).ok() };
        b.open = Some((cmd, fence, Vec::new(), 0));
        BATCHED_LIVE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    let Some(o) = b.open.as_mut() else { return Err("no open command buffer".into()) };
    f(o.0);
    o.2.push(staging);
    o.3 += bytes;
    BATCHED_STATS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if o.3 >= BATCHED_OPEN_BYTES {
        b.submit_open();
    }
    Ok(())
}

/// Submits the shared upload command buffer to the main queue if one is
/// open (`tune upload_batch`: ahead of every batch, `exec::submit_batch`,
/// and of the cache's own submissions), and frees what completed ones held.
pub fn flush_uploads() {
    let _trace = super::timeline::section("submit uploads");
    if BATCHED_LIVE.load(std::sync::atomic::Ordering::Relaxed) == 0 {
        return;
    }
    let mut g = BATCHED.lock();
    let Some(b) = g.as_mut() else { return };
    b.submit_open();
    b.reap(u64::MAX);
}

/// Pages of a texture are checked for CPU writes in SLICES interleaved
/// slices, one slice per frame (a change is seen within SLICES frames).
const SLICES: usize = 8;

/// Re-uploads after mapping changes (diagnostics), by kind of change
/// (`track::REMAP_*`): [count, with the sampled hashes unchanged, covering
/// part of the texture only, upload ns].
static REMAP_UPLOADS: [[std::sync::atomic::AtomicU64; 4]; 3] = [const { [const { std::sync::atomic::AtomicU64::new(0) }; 4] }; 3];
/// Partial re-uploads (`tune upload_parts`): [uploads, (mip, layer)s
/// uploaded, (mip, layer)s left out, bytes detiled].
static PART_UPLOADS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];
/// Bytes detiled by 2D uploads (all of them).
static DETILED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// DS_UPLOAD_CHECK=1: every upload hashes all guest bytes of each mapped
/// (mip, layer); a partial re-upload checks that the ones it leaves out are
/// unchanged since the last upload ([checked, changed]): what the written
/// ranges and write-watch marks miss.
fn upload_check() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_UPLOAD_CHECK").ok().as_deref() == Some("1"))
}
static SUB_HASHES: Mutex<Option<HashMap<ImageKey, (std::time::Instant, Vec<u64>)>>> = Mutex::new(None);
/// Hashes of each (mip, layer)'s guest bytes at its last upload, by texture
/// (`tune upload_parts`; SUB_UNMAPPED: not mapped then, SUB_NONE: unknown).
static SUB_SUMS: Mutex<Option<HashMap<ImageKey, Vec<u64>>>> = Mutex::new(None);
const SUB_NONE: u64 = 0xAAAA_AAAA_AAAA_AAAA;
/// Partial re-uploads' candidates left out as unchanged (their hash).
static SUB_SAME: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static SUB_CHECKS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];
const SUB_UNMAPPED: u64 = 0x5555_5555_5555_5555;

/// 2D uploads' time (diagnostics), ns: detiling, the staging buffer (made,
/// filled, freed), the transfer (submitted and waited for: the GPU's queue
/// drains first).
static UPLOAD_SPLIT: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];
/// The detiling phase of CPU-detiled 2D uploads (diagnostics), ns: the
/// staging buffer's allocation (made and mapped), the detiling into it, a
/// first upload's hashes.
static UPLOAD_PHASES: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];
/// Uploads whose pages were checked in the mapping table and held (no
/// mapping change over them before the reader lock), and those checked again
/// with the operating system (`mem::table_view_held`); of the first, those
/// held only for their own range while other mappings changed
/// (`mem::table_view_held_for`, since 2026-11-21).
static UPLOAD_TABLE: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// 2D uploads detiled on the GPU (`tune gpu_detile`): [uploads, (mip,
/// layer)s, bytes copied from guest memory, bytes detiled, uploads left to
/// the CPU (no import window), ns recording and waiting, (mip, layer)s
/// compared with the CPU's detiling (`DS_GPU_DETILE_CHECK=1`), of them
/// differing, of them whose guest bytes changed during the upload, video-
/// memory buffers made, uploads whose transfer ran on the copy queue, 3D volumes (their chunks
/// counted as uploads)].
static GPU_DETILE: [std::sync::atomic::AtomicU64; 12] = [const { std::sync::atomic::AtomicU64::new(0) }; 12];

/// `DS_ZERO_CENSUS=1` (diagnosis): re-uploaded (mip, layer)s by cause (0 none,
/// 1 a commit, 2 another mapping change, 3 a file read, 4 a CPU write, 5 a GPU
/// write, 6 other): [parts, parts all zero in guest memory, bytes, zero
/// bytes, upload ns].
static ZERO_CENSUS: [[std::sync::atomic::AtomicU64; 5]; 7] = [const { [const { std::sync::atomic::AtomicU64::new(0) }; 5] }; 7];

fn zero_census() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_ZERO_CENSUS").ok().as_deref() == Some("1"))
}

thread_local! {
    /// The cause of the re-upload in progress (`ZERO_CENSUS`).
    static REUPLOAD_CAUSE: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
}

/// Status part: `ZERO_CENSUS` ("" when off).
pub fn zero_census_line() -> String {
    if !zero_census() {
        return String::new();
    }
    use std::sync::atomic::Ordering::Relaxed;
    const NAMES: [&str; 7] = ["first", "commit", "other mapping change", "file read", "CPU write", "GPU write", "other"];
    let parts: Vec<String> = (1..7)
        .filter(|&k| ZERO_CENSUS[k][0].load(Relaxed) > 0)
        .map(|k| {
            let v: Vec<u64> = ZERO_CENSUS[k].iter().map(|c| c.load(Relaxed)).collect();
            format!("{} {} parts ({} all zero; {} MiB, {} MiB zero; {} ms)", NAMES[k], v[0], v[1], v[2] >> 20, v[3] >> 20, v[4] / 1_000_000)
        })
        .collect();
    format!("    re-uploaded parts by cause (`DS_ZERO_CENSUS`): {}\n", parts.join("; "))
}

/// `DS_GPU_DETILE_CHECK=1`: every upload detiled on the GPU reads back the
/// tiled bytes it copied and the rows it made; the CPU's detiling of that
/// copy must give the same rows, (mip, layer) by (mip, layer), and guest
/// bytes that differ from the copy changed during the upload.
fn gpu_detile_check() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_GPU_DETILE_CHECK").ok().as_deref() == Some("1"))
}

/// Video-memory buffers kept for GPU detiling (tiled bytes, packed rows) up
/// to this size each and DETILE_POOL in all; larger uploads make and free
/// their own. Uploads detiled on the GPU are in flight (not waited for) with
/// up to DETILE_INFLIGHT bytes of such buffers.
const DETILE_KEEP: u64 = 128 << 20;
const DETILE_POOL: u64 = 256 << 20;
const DETILE_INFLIGHT: u64 = 512 << 20;

struct DetilePipe {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
}

/// The detiling kernel's pipeline (`rdna::kernels::detile`; push
/// descriptors: binding 0 the tiled bytes, 1 the packed rows).
fn detile_pipe() -> Result<&'static DetilePipe, String> {
    static PIPE: std::sync::OnceLock<Result<DetilePipe, String>> = std::sync::OnceLock::new();
    PIPE.get_or_init(|| unsafe {
        let c = ctx();
        let d = &c.device;
        let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..2)
            .map(|b| vk::DescriptorSetLayoutBinding::default().binding(b).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE))
            .collect();
        let set_layout = d
            .create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR).bindings(&bindings), None)
            .map_err(|e| format!("detile set layout: {e}"))?;
        let ranges = [vk::PushConstantRange { stage_flags: vk::ShaderStageFlags::COMPUTE, offset: 0, size: 4 * rdna::kernels::DETILE_PC_WORDS as u32 }];
        let layout = d.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[set_layout]).push_constant_ranges(&ranges), None).map_err(|e| format!("detile layout: {e}"))?;
        let code = rdna::kernels::detile();
        let module = d.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None).map_err(|e| format!("detile module: {e}"))?;
        let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main");
        let ci = vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout);
        let pipeline = d.create_compute_pipelines(vk::PipelineCache::null(), &[ci], None).map_err(|(_, e)| format!("detile pipeline: {e}"));
        d.destroy_shader_module(module, None);
        Ok(DetilePipe { pipeline: pipeline?[0], layout })
    })
    .as_ref()
    .map_err(|e| e.clone())
}

/// Hashes of (index, guest address, bytes) parts, `threads` groups at a
/// time on the helper pool (`pool`).
fn hash_parts(todo: &[(usize, u64, u64)], threads: usize) -> Vec<(usize, u64)> {
    let per = todo.len().div_ceil(threads.max(1)).max(1);
    let out: Vec<std::sync::atomic::AtomicU64> = todo.iter().map(|_| std::sync::atomic::AtomicU64::new(0)).collect();
    super::pool::for_each(todo.len().div_ceil(per), &|c| {
        for k in c * per..((c + 1) * per).min(todo.len()) {
            let (_, a, n) = todo[k];
            out[k].store(hash_bytes(a, n as usize), std::sync::atomic::Ordering::Relaxed);
        }
    });
    todo.iter().zip(out).map(|(t, h)| (t.0, h.into_inner())).collect()
}

/// Hash of `n` bytes at `a` (four independent lanes over 64-bit words).
pub(super) fn hash_bytes(a: u64, n: usize) -> u64 {
    let mut h = [0x9E37_79B9_7F4A_7C15u64, 0xC2B2_AE3D_27D4_EB4F, 0x1656_67B1_9E37_79F9, 0x27D4_EB2F_1656_67C5];
    let words = n / 8;
    let p = a as *const u64;
    let mut i = 0;
    while i + 4 <= words {
        for (l, hl) in h.iter_mut().enumerate() {
            let x = unsafe { p.add(i + l).read_unaligned() };
            *hl = (*hl ^ x).wrapping_mul(0x0000_0100_0000_01B3).rotate_left(29);
        }
        i += 4;
    }
    let mut tail = 0u64;
    for k in i * 8..n {
        tail = tail.rotate_left(8) ^ unsafe { *((a + k as u64) as *const u8) } as u64;
    }
    h[0] ^ h[1].rotate_left(17) ^ h[2].rotate_left(31) ^ h[3].rotate_left(47) ^ tail.wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

/// Sampled hash of one slice of a guest range: the first 256 bytes of every
/// SLICES-th 4 KiB page starting at page `slice`. Unmapped pages count as
/// a marker (partially resident textures map only some of their mips).
fn slice_hash(base: u64, size: u64, slice: usize) -> u64 {
    let mut h = 0u64;
    let mut a = base + 4096 * slice as u64;
    while a < base + size {
        let n = 256.min(base + size - a) as usize;
        let v = if super::cp::guest_readable(a, n as u64) { hash_bytes(a, n) } else { 0x5555_5555_5555_5555 };
        h = h.rotate_left(5) ^ v;
        a += 4096 * SLICES as u64;
    }
    h
}

/// Sampled hash of a guest range (diagnostics): all of it up to 4 KiB, else
/// 256 bytes at 16 evenly spaced points and its last 256 bytes.
pub(crate) fn sampled_hash(base: u64, size: u64) -> u64 {
    let piece = |a: u64, n: u64| if super::cp::guest_readable(a, n) { hash_bytes(a, n as usize) } else { 0x5555_5555_5555_5555 };
    if size <= 4096 {
        return piece(base, size);
    }
    let mut h = 0u64;
    for k in 0..16 {
        h = h.rotate_left(5) ^ piece(base + (size - 256) * k / 16, 256);
    }
    h.rotate_left(5) ^ piece(base + size - 256, 256)
}

fn all_hashes(base: u64, size: u64) -> [u64; SLICES] {
    std::array::from_fn(|s| slice_hash(base, size, s))
}

pub struct Images {
    map: HashMap<ImageKey, Image>,
    /// The keys of `map` by base address (`alias` lookups).
    by_base: HashMap<u64, Vec<ImageKey>>,
    /// Addresses with a GPU-written image (render targets, storage images,
    /// stencil planes).
    gpu_bases: super::fx::Set<u64>,
    /// Depth/stencil image of each rendered stencil plane address.
    stencil_of: HashMap<u64, ImageKey>,
    /// Surface of each metadata address (DCC, HTILE) and its clear values.
    /// Fast-clear metadata address -> (image, clear value); ordered for the
    /// range lookups of `note_meta_writes`.
    meta_of: std::collections::BTreeMap<u64, (ImageKey, [u32; 2])>,
    /// Each surface's metadata address at its last bind as a target (0: none) and that
    /// frame (`metamap`).
    meta_now: HashMap<ImageKey, (u64, u64)>,
    /// The writes `note_writes` last applied, with LOOKUP_GEN and note_gen
    /// then: the per-item culls report the same two ranges thousands of
    /// times per frame, and noting them again changes nothing until a
    /// pending clear or overwrite is taken or the images change.
    last_note: Option<(Vec<(u64, u64)>, Vec<(u64, u64)>, u64, u64)>,
    note_gen: u64,
    /// Sampled views by T#: (frame, LOOKUP_GEN, track::GEN, view, image).
    /// Within a frame, while no image was created, removed or re-aliased and
    /// no tracked texture source was written, `view` would do nothing but
    /// return the same view (draws bind the same textures thousands of times
    /// per frame; each full lookup took the tracker's lock three times).
    view_memo: HashMap<[u32; 8], ViewMemo>,
    /// `resolved_key` by (T#, storage): (LOOKUP_GEN, key).
    key_memo: HashMap<([u32; 8], bool), (Option<ImageKey>, u64, u32)>,
    /// Images whose metadata a GPU or CP write covered since (fast clears).
    /// Images whose fast-clear metadata was written: (metadata address, its
    /// first word when the writer's value is known: `note_meta_fill`).
    pending_clears: HashMap<ImageKey, PendingClear>,
    samplers: HashMap<([u32; 4], bool, u8), vk::Sampler>,
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    /// The one-shot's GPU timestamps (`upload_stamps`).
    qp: vk::QueryPool,
    /// Staging buffers not in use (`staging`), uploads submitted without
    /// waiting (command buffer, fence, its timestamps, the staging buffer it
    /// reads; oldest first) and the command buffers, fences and timestamps
    /// of completed ones.
    inflight: std::collections::VecDeque<(vk::CommandBuffer, vk::Fence, vk::QueryPool, Held)>,
    spare_cmds: Vec<(vk::CommandBuffer, vk::Fence, vk::QueryPool)>,
    /// GPU detiling's video-memory buffers not in use: (buffer, memory,
    /// capacity).
    detile_free: Vec<(vk::Buffer, vk::DeviceMemory, u64)>,
    /// GPU detiling's transfers on the copy queue: its command pool (made
    /// on first use), spare command buffers, and the timeline its copies
    /// signal with `copy_value` (the main queue's part waits on it).
    copy_pool: vk::CommandPool,
    copy_spare: Vec<vk::CommandBuffer>,
    copy_timeline: vk::Semaphore,
    copy_value: u64,
    reported: HashMap<String, u64>,
    pub uploads: u64,
    pub reuploads: u64,
    pub upload_bytes: u64,
    pub upload_ns: u64,
    /// Counts storage-image bindings (write generations).
    write_seq: u64,
    /// GPU-owned images with a known guest span, by address (see
    /// `note_overwrites`), the largest span, and how many are marked.
    owned: std::collections::BTreeMap<u64, Vec<ImageKey>>,
    max_span: u64,
    pending_overwrites: usize,
    /// Images marked overwritten (with `pending_clears`' keys: the set
    /// `pending_hit` reads), and a count of changes to both (published at
    /// the end of `with` when it moved).
    overwritten_keys: super::fx::Set<ImageKey>,
    pending_change: u64,
    pending_published: u64,
    /// Textures the scrubber visits in turn (`scrub_batch`), rebuilt when
    /// LOOKUP_GEN moves, and its position.
    scrub_keys: Vec<ImageKey>,
    scrub_gen: u64,
    scrub_at: usize,
    /// GPU-owned images reloaded from guest memory after buffer writes
    /// covered them: cleared (all zero) / uploaded.
    pub overwrite_clears: u64,
    pub overwrite_uploads: u64,
    /// Images reported for buffer writes covering only part of them.
    partial_seen: super::fx::Set<u64>,
    /// Zero images for unused image table entries, by view type.
    dummies: HashMap<vk::ImageViewType, vk::ImageView>,
    /// Depth images grown (status line).
    pub depth_grows: u64,
    /// Bases of R32 float / R16 unorm images bound as storage: compute
    /// views of a depth plane (`depth_copies`).
    depth_feeds: super::fx::Set<u64>,
    /// Images whose re-upload a thread left for after the lock
    /// (`RefreshJob`).
    refreshing: HashMap<ImageKey, std::sync::Arc<RefreshWait>>,
}

unsafe impl Send for Images {}

pub static IMAGES: Mutex<Option<Images>> = Mutex::new(None);

thread_local! {
    /// The shader whose bindings are being resolved on this thread
    /// (diagnostics: which pass first writes an uploaded image).
    pub static BINDING_SHADER: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Runs `f` with the image cache.
/// Texture scrubbing (`tune tex_scrub`): a background thread samples the
/// guest bytes of uploaded textures for CPU writes (`scrub_batch`) instead
/// of the first draw that binds each in a frame (`refresh`); texture views
/// then stay valid across frames (`view_hit`).
pub fn scrubbing() -> bool {
    super::tune::tex_scrub()
}

static SCRUB_CHECKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Write-watch scrubbing (`tex_scrub` 2): takes and the pages they covered.
static SCRUB_TAKES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static SCRUB_TAKE_PAGES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static SCRUB_EDGE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static SCRUB_DIRTY: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn scrubber() {
    super::profile_this_thread("scrub");
    loop {
        let marks = super::tune::tex_scrub_mode() == 2;
        std::thread::sleep(std::time::Duration::from_millis(if marks { super::tune::scrub_ms() } else { 2 }));
        if !scrubbing() || super::shutting_down() {
            continue;
        }
        if marks {
            // Write-watch marks of the next textures' pages: taking them
            // tells the write tracker which pages the CPU wrote (the
            // texture's own region included: re-uploaded at its next use).
            // In pieces of at most 16 MiB (the copy cache's lock is held
            // during a take).
            let work = with(|im| im.scrub_regions(64));
            let mut pages = Vec::new();
            for (b, n) in work {
                let mut a = b;
                while a < b + n {
                    let len = (b + n - a).min(16 << 20);
                    pages.clear();
                    SCRUB_TAKES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    SCRUB_TAKE_PAGES.fetch_add(len / 4096, std::sync::atomic::Ordering::Relaxed);
                    let _ = super::precache::take_written_for(a, len, &mut pages, None, 3);
                    SCRUB_DIRTY.fetch_add((!pages.is_empty()) as u64, std::sync::atomic::Ordering::Relaxed);
                    // Written pages only where the texture shares a page with
                    // other data (its first and last page).
                    let (first, last) = (b & !4095, (b + n - 1) & !4095);
                    if !pages.is_empty() && pages.iter().all(|&p| p == first || p == last) {
                        SCRUB_EDGE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    a += len;
                }
            }
            continue;
        }
        let work = with(|im| im.scrub_batch(32));
        for (region, b, n, s, stored) in work {
            SCRUB_CHECKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if slice_hash(b, n, s) != stored {
                SCRUB_DIRTY.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                super::track::sampled_dirty(region);
            }
        }
    }
}

/// One status line ("" before the scrubber checked anything).
pub fn scrub_status() -> String {
    let takes = SCRUB_TAKES.load(std::sync::atomic::Ordering::Relaxed);
    if takes > 0 {
        return format!(
            "    texture scrubbing by write-watch marks: {takes} takes over {} pages, {} found CPU writes ({} only on a first or last page)\n",
            SCRUB_TAKE_PAGES.load(std::sync::atomic::Ordering::Relaxed),
            SCRUB_DIRTY.load(std::sync::atomic::Ordering::Relaxed),
            SCRUB_EDGE.load(std::sync::atomic::Ordering::Relaxed)
        );
    }
    let n = SCRUB_CHECKS.load(std::sync::atomic::Ordering::Relaxed);
    if n == 0 {
        return String::new();
    }
    format!("    texture scrubbing: {n} slices checked, {} changed (re-uploaded at their next use)\n", SCRUB_DIRTY.load(std::sync::atomic::Ordering::Relaxed))
}

/// Follows `Images::note_gen` (read without the lock: `Exec`'s write-note
/// memo).
pub static NOTE_GEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Pages (64 KiB, hashed) holding a GPU-owned image's bytes or a fast-clear
/// metadata address (`Images::owned`, `meta_of`), marked when indexed and
/// never cleared (both only grow): buffer writes on no marked page have
/// nothing for `note_writes` and skip the image cache's lock (`may_note`;
/// every writing dispatch took it, the most waited-for lock after the copy
/// cache's, 2026-10-15).
const NOTE_BUCKETS: usize = 1 << 16;
static NOTE_PAGES: [std::sync::atomic::AtomicBool; NOTE_BUCKETS] = [const { std::sync::atomic::AtomicBool::new(false) }; NOTE_BUCKETS];

fn note_bucket(page: u64) -> usize {
    (page.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 48) as usize
}

/// The same marks inside the guest span, one bit a page (exact; the buckets
/// above serve pages outside it): once ~1.7 GiB of GPU-owned images were
/// indexed the buckets marked about a third of their pages, so most writes
/// passed `may_note` and took the lock (exact: +1.35% at the vista,
/// FINDINGS.md 2026-11-01 morning).
const SPAN_PAGES: usize = ((super::resident::MIRROR_END - super::resident::MIRROR_BASE) >> 16) as usize;
static NOTE_SPAN: [std::sync::atomic::AtomicU64; SPAN_PAGES / 64] = [const { std::sync::atomic::AtomicU64::new(0) }; SPAN_PAGES / 64];

fn span_page(page: u64) -> Option<usize> {
    let i = page.wrapping_sub(super::resident::MIRROR_BASE >> 16);
    (i < SPAN_PAGES as u64).then_some(i as usize)
}

fn mark_note_pages(base: u64, len: u64) {
    if len == 0 {
        return;
    }
    let (first, last) = (base >> 16, (base + len - 1) >> 16);
    for page in first..=last.min(first + SPAN_PAGES as u64) {
        if let Some(i) = span_page(page) {
            NOTE_SPAN[i / 64].fetch_or(1 << (i % 64), std::sync::atomic::Ordering::Release);
        }
    }
    if last - first >= NOTE_BUCKETS as u64 {
        for b in NOTE_PAGES.iter() {
            b.store(true, std::sync::atomic::Ordering::Release);
        }
        return;
    }
    for page in first..=last {
        NOTE_PAGES[note_bucket(page)].store(true, std::sync::atomic::Ordering::Release);
    }
}

/// Whether `note_writes` may have something to do for these writes (false:
/// certainly not; when a mark races a write, that write was not ordered
/// after the image's indexing under the lock either).
pub fn may_note(writes: &[(u64, u64)]) -> bool {
    if !super::tune::note_filter() {
        return true;
    }
    writes.iter().any(|&(b, n)| {
        if n == 0 {
            return false;
        }
        let (first, last) = (b >> 16, (b + n - 1) >> 16);
        last - first >= 1024
            || (first..=last).any(|p| match span_page(p) {
                Some(i) => NOTE_SPAN[i / 64].load(std::sync::atomic::Ordering::Acquire) & (1 << (i % 64)) != 0,
                None => NOTE_PAGES[note_bucket(p)].load(std::sync::atomic::Ordering::Acquire),
            })
    })
}

/// `DS_IMAGE_HOLDS=1` (diagnosis): the image cache's holds of 200 us or more
/// (`with`): count, total and longest ns, by the holder's thread class
/// (graphics CP, its workers, compute queues' CPs and workers, others); and
/// its waits of 1 us or more by the waiter's class: count, ns.
static HOLDS: [[std::sync::atomic::AtomicU64; 3]; 4] = [const { [const { std::sync::atomic::AtomicU64::new(0) }; 3] }; 4];
static WAITS: [[std::sync::atomic::AtomicU64; 2]; 4] = [const { [const { std::sync::atomic::AtomicU64::new(0) }; 2] }; 4];

/// `image_for`'s steps under the lock (`DS_IMAGE_HOLDS`): refresh of an
/// image held, create, upload (prepared), upload (the cache's own), write
/// watch, hashes; each [count, ns, count of 200 us or more].
static STEPS: [[std::sync::atomic::AtomicU64; 3]; 6] = [const { [const { std::sync::atomic::AtomicU64::new(0) }; 3] }; 6];

fn note_step(k: usize, t: Option<std::time::Instant>) {
    use std::sync::atomic::Ordering::Relaxed;
    if let Some(t) = t {
        let ns = t.elapsed().as_nanos() as u64;
        STEPS[k][0].fetch_add(1, Relaxed);
        STEPS[k][1].fetch_add(ns, Relaxed);
        if ns >= 200_000 {
            STEPS[k][2].fetch_add(1, Relaxed);
        }
    }
}

/// The long holds (HOLDS) by the call site of `with`: count, ns.
static HOLD_SITES: Mutex<Option<HashMap<(&'static str, u32), (u64, u64)>>> = Mutex::new(None);
/// The waits (WAITS) by the call site of `with`: count, ns.
static WAIT_SITES: Mutex<Option<HashMap<(&'static str, u32), (u64, u64)>>> = Mutex::new(None);

thread_local! {
    /// This thread's class for HOLDS and WAITS (255: not looked up yet).
    static HOLD_CLASS: std::cell::Cell<u8> = const { std::cell::Cell::new(255) };
}

fn holds_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| matches!(std::env::var("DS_IMAGE_HOLDS").ok().as_deref(), Some("1" | "2")))
}

/// `DS_IMAGE_HOLDS=2` (diagnosis): as 1, and a thread that waited 1 ms for
/// the image cache's lock samples the holder's stack, logged with the
/// holder's thread and call site of `with` (the first 200, from flip
/// `DS_HOLDS_FROM` on).
fn holder_trace() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_IMAGE_HOLDS").ok().as_deref() == Some("2"))
}

/// The image cache's holder (`holder_trace`): its host thread id, and its
/// thread name and call site.
static HOLDER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
static HOLDER_AT: Mutex<(String, &'static str, u32)> = Mutex::new((String::new(), "", 0));

/// Logs the image cache's holder's stack (`holder_trace`).
fn sample_holder(waiter: &std::panic::Location<'static>) {
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    // (`DS_HOLDS_FROM=<flip>`: from that flip on.)
    static FROM: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let from = *FROM.get_or_init(|| std::env::var("DS_HOLDS_FROM").ok().and_then(|v| v.parse().ok()).unwrap_or(0));
    let tid = HOLDER.load(std::sync::atomic::Ordering::Acquire);
    if tid == 0 || super::FRAMES.load(std::sync::atomic::Ordering::Relaxed) < from || N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) >= 200 {
        return;
    }
    let (name, file, line) = HOLDER_AT.lock().clone();
    let Some((frames, _)) = crate::exception::sample_frames(tid, 0) else { return };
    let labels: Vec<String> = frames
        .iter()
        .enumerate()
        .take(18)
        .map(|(i, &a)| {
            let f = crate::exception::frame_label(a);
            match crate::exception::host_line(a.saturating_sub((i > 0) as u64)) {
                Some(l) if f.starts_with("dsport::") => format!("{} ({l})", f.rsplit("::").next().unwrap_or(&f)),
                _ => f,
            }
        })
        .collect();
    crate::info!(
        "images: the cache's lock held over 1 ms at flip {} by {name} (from {file}:{line}; {} waiting at {}:{}): {}",
        super::FRAMES.load(std::sync::atomic::Ordering::Relaxed),
        crate::log::thread_name(),
        waiter.file(),
        waiter.line(),
        labels.join(" <- ")
    );
}

fn hold_class() -> usize {
    HOLD_CLASS.with(|c| {
        if c.get() == 255 {
            let n = crate::log::thread_name();
            c.set(match n.as_str() {
                "gpu-gfx" => 0,
                n if n.starts_with("gpu-gfx-par") => 1,
                n if n.starts_with("gpu-acb") => 2,
                _ => 3,
            });
        }
        c.get() as usize
    })
}

/// Clears the hold census (`profile reset`, through `lockstat`).
pub fn reset_holds() {
    for c in HOLDS.iter().flatten().chain(WAITS.iter().flatten()).chain(STEPS.iter().flatten()) {
        c.store(0, std::sync::atomic::Ordering::Relaxed);
    }
    *HOLD_SITES.lock() = None;
    *WAIT_SITES.lock() = None;
}

/// The hold census for the contended-locks status line ("" when off).
pub fn holds_part() -> String {
    if !holds_on() {
        return String::new();
    }
    use std::sync::atomic::Ordering::Relaxed;
    let names = ["graphics CP", "its workers", "compute queues", "others"];
    let parts: Vec<String> = names
        .iter()
        .zip(HOLDS.iter())
        .map(|(n, h)| format!("{n} {} ({} ms, longest {:.1} ms)", h[0].load(Relaxed), h[1].load(Relaxed) / 1_000_000, h[2].load(Relaxed) as f64 / 1e6))
        .collect();
    let waits: Vec<String> = names.iter().zip(WAITS.iter()).map(|(n, w)| format!("{n} {} ({} ms)", w[0].load(Relaxed), w[1].load(Relaxed) / 1_000_000)).collect();
    let mut sites: Vec<((&'static str, u32), (u64, u64))> = HOLD_SITES.lock().as_ref().map(|m| m.iter().map(|(k, v)| (*k, *v)).collect()).unwrap_or_default();
    sites.sort_by_key(|s| std::cmp::Reverse(s.1 .1));
    let sites: Vec<String> = sites.iter().take(8).map(|((f, l), (n, ns))| format!("{f}:{l} {n} ({} ms)", ns / 1_000_000)).collect();
    let mut wsites: Vec<((&'static str, u32), (u64, u64))> = WAIT_SITES.lock().as_ref().map(|m| m.iter().map(|(k, v)| (*k, *v)).collect()).unwrap_or_default();
    wsites.sort_by_key(|s| std::cmp::Reverse(s.1 .1));
    let wsites: Vec<String> = wsites.iter().take(10).map(|((f, l), (n, ns))| format!("{f}:{l} {n} ({} ms)", ns / 1_000_000)).collect();
    let step_names = ["refresh", "create", "prepared upload", "own upload", "write watch", "hashes"];
    let steps: Vec<String> = step_names.iter().zip(STEPS.iter()).map(|(n, s)| format!("{n} {} ({} ms, {} of 200 us or more)", s[0].load(Relaxed), s[1].load(Relaxed) / 1_000_000, s[2].load(Relaxed))).collect();
    format!("; image cache held 200 us or more (`DS_IMAGE_HOLDS`): {}; waited for by: {}; waits by site: {}; long holds by site: {}; image_for's steps: {}", parts.join(", "), waits.join(", "), wsites.join(", "), sites.join(", "), steps.join(", "))
}

/// A 2D texture upload's (mip, layer)s to upload, read from guest memory
/// under the guest-reader lock (`readers`, held until they are detiled), and
/// where each goes in the staging buffer (`choose_2d`).
struct Chosen {
    lay: tiling::Layout,
    layers: u32,
    elem: u64,
    block: bool,
    size: u64,
    precise: bool,
    held: bool,
    gpu: bool,
    windows: Vec<(u64, u64, vk::Buffer)>,
    readers: parking_lot::RwLockReadGuard<'static, ()>,
    sums_on: bool,
    sums: Vec<u64>,
    end: u64,
    regions: Vec<vk::BufferImageCopy>,
    zeroed: Vec<(usize, u32, usize)>,
    parts: Vec<(usize, u32, usize, usize)>,
    total: usize,
    zero_len: usize,
    zero_off: usize,
    detiled: usize,
    skipped: usize,
    parts_done: u64,
    parts_left: u64,
    t_detile: std::time::Instant,
    t_table: std::time::Instant,
    t_locked: std::time::Instant,
    t_chosen: std::time::Instant,
    take_ms: f64,
    /// When the choice began, and its wait for front end 4's held pages (`c4_release_sync`).
    t_start: std::time::Instant,
    c4_ms: f64,
}

enum Chose {
    /// Nothing to upload: (base, size).
    Done((u64, u64)),
    Parts(Chosen),
}

/// The guest range of a (mip, layer) of a 2D texture's layout, relative to
/// its base (the mip tail: its first block).
fn part_range(lay: &tiling::Layout, elem: u64, mip: usize, layer: u32) -> (u64, u64) {
    let m = &lay.mips[mip];
    let start = layer as u64 * lay.slice_size + if m.tail.is_some() { 0 } else { m.offset };
    (start, if m.tail.is_some() { 1u64 << lay.block_log2 } else { m.pitch as u64 * m.padded_height as u64 * elem })
}

/// Hash of one (mip, layer)'s own elements in guest memory: its range, or
/// for a mip in the mip tail only its elements inside the tail block (the
/// tail's mips share that block: a write to one leaves the others' hashes).
/// The caller holds the guest-reader lock with the part mapped.
fn own_part_hash(lay: &tiling::Layout, elem: u64, base: u64, mip: usize, layer: u32) -> u64 {
    let m = &lay.mips[mip];
    if m.tail.is_none() {
        let (s, n) = part_range(lay, elem, mip, layer);
        return hash_bytes(base + s, n as usize);
    }
    let mut h = 0x9E37_79B9_7F4A_7C15u64;
    for y in 0..m.height {
        for x in 0..m.width {
            let v = hash_bytes(base + lay.offset(mip, layer, x, y), elem as usize);
            h = (h ^ v).wrapping_mul(0x0000_0100_0000_01B3).rotate_left(29);
        }
    }
    h
}

/// Re-uploads' parts that GPU work wrote through block views (`Image::gpu_parts`):
/// [checked, left as the GPU wrote them (guest bytes unchanged), uploaded (changed)].
pub static GPU_PARTS_KEPT: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// Chooses what an upload of a 2D texture reads (`Images::upload_parts`'
/// first half, without the cache: under its lock for its own uploads, after
/// it for re-uploads left by `refresh`): `first` (no image yet), `region`
/// (its write tracking), GPU detiling allowed (`allow_gpu`); the cache's
/// messages go to `notes`.
#[allow(clippy::too_many_arguments)]
fn choose_2d(key: ImageKey, t: &Tsharp, only: Option<&[(u64, u64)]>, all: bool, first: bool, region: Option<u32>, allow_gpu: bool, keep: &[(u32, u32, u64)], notes: &mut Vec<String>) -> Result<Chose, String> {
    let t_start = std::time::Instant::now();
    let precise = super::tune::upload_parts() && !first && !all && only.is_some();
    let el = elem_log2(t.format).ok_or("unknown element size")?;
    let block = is_block(t.format);
    let layers = if key.ty3d { 1 } else { key.layers };
    let lay = tiling::layout(t.sw_mode, el, key.width, key.height, key.levels, block).ok_or_else(|| format!("swizzle mode {} not supported", t.sw_mode))?;
    if tiling::is_xor(t.sw_mode) {
        notes.push(format!("texture with XOR swizzle mode {} uploaded with an assumed pipe count (2^{})", t.sw_mode, tiling::pipes_log2()));
    }
    let size = lay.size(layers);
    // (Front end 4's held pages: their newest words put back in guest
    // memory before the texture reads it.)
    let t_c4 = std::time::Instant::now();
    super::fe2::c4_release_sync(t.base, size, false);
    let c4_ms = t_c4.elapsed().as_secs_f64() * 1e3;
    // GPU detiling for textures of at least `tune gpu_detile` MiB.
    let gpu = allow_gpu
        && match super::tune::gpu_detile() {
            0 => false,
            n => size >= n << 20,
        };
    // Guest memory where a video-memory home holds newer GPU-written data
    // (`homes`): this upload reads the older bytes (diagnostics).
    if super::homes::any() && super::homes::any_stale(&[(t.base, size)]) {
        let n = super::homes::IMAGE_STALE[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if n < 20 {
            crate::warn!(
                "images: {} {:#x} {}x{}x{} {:?} uploaded from guest memory where home {} holds GPU-written data not written back",
                if first { "new image" } else { "image" },
                t.base,
                key.width,
                key.height,
                layers,
                t.format,
                super::homes::describe(t.base, size)
            );
        }
    }
    // Streamed textures are partially resident: the title maps single
    // layers of texture arrays (light cookies), and within a layer the
    // mip tail and the smaller mips (at the start of the chain). Every
    // mapped (mip, layer) is uploaded; on the first upload the others
    // are zero-filled, never left as undefined image memory (which holds
    // whatever the device memory held before). Later mappings or writes
    // re-upload (write tracking).
    let elem = 1u64 << el;
    let range = |mip: usize, layer: u32| part_range(&lay, elem, mip, layer);
    // A partial re-upload (`precise`): the (mip, layer)s the written
    // ranges (mappings, file reads) or the CPU's writes since the last
    // take overlap. The title maps a layer of a streamed array at a time
    // and fills mapped layers and small mips with CPU writes later. The
    // marks are taken before the bytes are read (later writes are seen
    // next time); taking resets them, so the copy cache and other
    // textures on those pages are told (`precache::take_written_for`).
    let mut written: Vec<(u64, u64)> = Vec::new();
    let t_take = std::time::Instant::now();
    let precise = precise && {
        let mut pages = Vec::new();
        let ok = super::precache::take_written_for(t.base, size, &mut pages, region, 4).is_some();
        for p in pages {
            match written.last_mut() {
                Some(r) if r.0 + r.1 == p => r.1 += 4096,
                _ => written.push((p, 4096)),
            }
        }
        ok
    };
    let take_ms = t_take.elapsed().as_secs_f64() * 1e3;
    // Pages checked and read under the guest-reader lock: the title may
    // unmap layers of a streamed array meanwhile (`mem::GUEST_READERS`).
    // Checked in the mapping table first, before the lock (which must
    // not take the table's), and again with the operating system only
    // when a mapping changed in between (`mem::table_view_held`).
    let t_detile = std::time::Instant::now();
    let g0 = crate::mem::MAP_GEN.load(std::sync::atomic::Ordering::SeqCst);
    let table_whole = super::cp::valid(t.base, size) && crate::mem::host_readable(t.base, size);
    let n_parts = key.levels as usize * layers as usize;
    let table_parts: Vec<bool> = if table_whole {
        Vec::new()
    } else {
        (0..n_parts)
            .map(|i| {
                let (s, n) = range(i / layers as usize, (i % layers as usize) as u32);
                super::cp::valid(t.base + s, n) && crate::mem::host_readable(t.base + s, n)
            })
            .collect()
    };
    // GPU detiling: the import windows over the mapped parts, made now
    // (making one takes the mapping table's lock, which the guest-reader
    // lock's holders must not: decommits hold it while they wait for
    // them).
    let mut windows: Vec<(u64, u64, vk::Buffer)> = Vec::new();
    if gpu {
        for i in 0..n_parts {
            if !(table_whole || table_parts[i]) {
                continue;
            }
            let (s, n) = range(i / layers as usize, (i % layers as usize) as u32);
            let (mut x, end) = (t.base + s, t.base + s + n);
            while x < end {
                if let Some(&(_, we, _)) = windows.iter().find(|&&(ws, we, _)| ws <= x && x < we) {
                    x = we.min(end);
                    continue;
                }
                let e = ((x & !(super::exec::window_bytes() - 1)) + super::exec::window_bytes()).min(end);
                match super::exec::import_window(x, e - x) {
                    Some((b, ws, we)) if ws <= x && e <= we => {
                        windows.push((ws, we, b));
                        x = we.min(end);
                    }
                    _ => break,
                }
            }
        }
    }
    let t_table = std::time::Instant::now();
    let readers = crate::mem::GUEST_READERS.read();
    let t_locked = std::time::Instant::now();
    let held_all = crate::mem::table_view_held(g0);
    let held = held_all || crate::mem::table_view_held_for(g0, t.base, t.base + size);
    UPLOAD_TABLE[usize::from(!held)].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if held && !held_all {
        UPLOAD_TABLE[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    let whole = if held { table_whole } else { super::cp::valid(t.base, size) && crate::mem::os_readable(t.base, size) };
    let mapped = |mip: usize, layer: u32| -> bool {
        if whole {
            return true;
        }
        if held {
            return table_parts[mip * layers as usize + layer as usize];
        }
        let (s, n) = range(mip, layer);
        super::cp::valid(t.base + s, n) && crate::mem::os_readable(t.base + s, n)
    };
    let candidate = |mip: usize, layer: u32| -> bool {
        if !precise {
            return true;
        }
        let (s, n) = range(mip, layer);
        let (a, e) = (t.base + s, t.base + s + n);
        // `written` is ascending and disjoint.
        let i = written.partition_point(|&(pb, pl)| pb + pl <= a);
        written.get(i).is_some_and(|&(pb, _)| pb < e) || only.is_some_and(|parts| parts.iter().any(|&(pb, pl)| pb < e && a < pb.saturating_add(pl)))
    };
    // Of the candidates, those whose bytes changed since their last
    // upload (hashes kept per (mip, layer)): the title rewrites texture
    // memory with the same data (16,143 of 16,146 parts re-uploaded after
    // CPU writes were unchanged, run 20261013-082344). Mapping is checked
    // for candidates only (one memory query each; a streamed array has
    // thousands of (mip, layer)s, most of them not candidates).
    let sums_on = super::tune::upload_parts();
    let n_sub = key.levels as usize * layers as usize;
    let mut sums: Vec<u64> = if sums_on {
        SUB_SUMS.lock().as_ref().and_then(|m| m.get(&key).cloned()).filter(|v| v.len() == n_sub).unwrap_or_else(|| vec![SUB_NONE; n_sub])
    } else {
        Vec::new()
    };
    let t_query = std::time::Instant::now();
    // A re-upload's candidates are hashed first, on several threads when
    // they are large (one by one, a 1,200-part array's re-upload spent
    // 50 ms here).
    let mut pre_hash: Vec<u64> = Vec::new();
    if sums_on && !first {
        let todo: Vec<(usize, u64, u64)> = (0..n_sub)
            .filter_map(|i| {
                let (mip, layer) = (i / layers as usize, (i % layers as usize) as u32);
                if !candidate(mip, layer) || !mapped(mip, layer) {
                    return None;
                }
                let (s, n) = range(mip, layer);
                Some((i, t.base + s, n))
            })
            .collect();
        let bytes: u64 = todo.iter().map(|x| x.2).sum();
        pre_hash = vec![SUB_NONE; n_sub];
        let threads = if bytes >= 1 << 20 { super::pool::threads() } else { 1 };
        for (i, h) in hash_parts(&todo, threads) {
            pre_hash[i] = h;
        }
    }
    let mut want = vec![false; n_sub];
    let mut use_mapped = vec![false; n_sub];
    let mut end = 0u64;
    for mip in 0..key.levels as usize {
        for layer in 0..layers {
            let i = mip * layers as usize + layer as usize;
            if !candidate(mip, layer) {
                continue;
            }
            if !mapped(mip, layer) {
                want[i] = true;
                if sums_on {
                    sums[i] = SUB_UNMAPPED;
                }
                continue;
            }
            let (s, n) = range(mip, layer);
            // Blocks GPU work wrote through a block view (`gpu_parts`): left
            // as they are while their own guest bytes stay as they were then
            // (guest memory never held them; a write elsewhere in the shared
            // mip-tail block re-uploaded the tail with stale bytes and the
            // title's encoded character textures showed noise, 2026-11-11).
            if let Some(&(_, _, h0)) = keep.iter().find(|k| k.0 as usize == mip && k.1 == layer) {
                use std::sync::atomic::Ordering::Relaxed;
                GPU_PARTS_KEPT[0].fetch_add(1, Relaxed);
                if own_part_hash(&lay, elem, t.base, mip, layer) == h0 {
                    GPU_PARTS_KEPT[1].fetch_add(1, Relaxed);
                    if sums_on && !first && pre_hash.get(i).is_some_and(|&h| h != SUB_NONE) {
                        sums[i] = pre_hash[i];
                    }
                    continue;
                }
                GPU_PARTS_KEPT[2].fetch_add(1, Relaxed);
            }
            // (A first upload hashes its parts after the detiling, on
            // several threads: one by one here it cost as long as
            // detiling a large array.)
            if sums_on && !first {
                let h = pre_hash[i];
                if precise && sums[i] == h {
                    SUB_SAME.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    continue;
                }
                sums[i] = h;
            }
            want[i] = true;
            use_mapped[i] = true;
            end = end.max(s + n);
        }
    }
    UPLOAD_SPLIT[3].fetch_add(t_query.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
    let t_chosen = std::time::Instant::now();
    let wanted = |mip: usize, layer: u32| want[mip * layers as usize + layer as usize];
    if upload_check() {
        let now: Vec<u64> = (0..key.levels as usize)
            .flat_map(|mip| (0..layers).map(move |layer| (mip, layer)))
            .map(|(mip, layer)| {
                let (s, n) = range(mip, layer);
                if mapped(mip, layer) { hash_bytes(t.base + s, n as usize) } else { SUB_UNMAPPED }
            })
            .collect();
        let mut g = SUB_HASHES.lock();
        let map = g.get_or_insert_with(HashMap::default);
        if let (true, Some((at, prev))) = (precise, map.get(&key)) {
            for (i, (&a, &b)) in prev.iter().zip(&now).enumerate() {
                let (mip, layer) = (i / layers as usize, (i % layers as usize) as u32);
                if wanted(mip, layer) && a != SUB_UNMAPPED && b != SUB_UNMAPPED {
                    // Uploaded again: did its bytes change?
                    SUB_CHECKS[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if a == b {
                        SUB_CHECKS[3].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                }
                if wanted(mip, layer) || b == SUB_UNMAPPED {
                    continue;
                }
                // Written since this upload's take: uploaded next time.
                let (s, n) = range(mip, layer);
                if region.is_some_and(|r| super::track::pending_overlaps(r, t.base + s, n)) {
                    continue;
                }
                SUB_CHECKS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if a != b && SUB_CHECKS[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 20 {
                    let ev = super::track::events_since(t.base + s, n, *at);
                    let takes = region.map(|r| super::track::takes_since(r, *at)).unwrap_or_default();
                    crate::warn!(
                        "upload check: texture {:#x} {}x{}x{} mip {mip} layer {layer} ({:#x}+{n:#x}): changed although no written range overlaps it (ranges {only:x?}); last upload {} ms ago; writes noted on it since: {}; region {region:?} taken since: {}",
                        t.base, key.width, key.height, layers, t.base + s, at.elapsed().as_millis(), if ev.is_empty() { "none".to_string() } else { ev.join("; ") }, takes.join(" | ")
                    );
                }
            }
        }
        map.insert(key, (std::time::Instant::now(), now));
    }
    if end == 0 && precise {
        // Nothing mapped changed.
        if sums_on {
            SUB_SUMS.lock().get_or_insert_with(HashMap::default).insert(key, sums);
        }
        PART_UPLOADS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        PART_UPLOADS[2].fetch_add(key.levels as u64 * layers as u64, std::sync::atomic::Ordering::Relaxed);
        return Ok(Chose::Done((t.base, size)));
    }
    if end == 0 && !first {
        return Err(format!("texture {:#x}+{size:#x} not mapped", t.base));
    }
    // Detiling reads only mapped regions (holes between them are not read).
    let src = unsafe { std::slice::from_raw_parts(t.base as *const u8, end as usize) };
    let mut skipped = 0;
    // Every wanted, mapped (mip, layer) is detiled straight into the
    // staging buffer at its offset (sizes first: no intermediate copies,
    // which took as long as the detiling, FINDINGS.md 2026-10-23);
    // unmapped ones of a first upload copy from one shared zero area at
    // the end.
    let mut regions = Vec::new();
    let mut zeroed: Vec<(usize, u32, usize)> = Vec::new();
    let mut parts: Vec<(usize, u32, usize, usize)> = Vec::new();
    let mut total = 0usize;
    let (mut parts_done, mut parts_left) = (0u64, 0u64);
    for mip in 0..key.levels as usize {
        for layer in 0..layers {
            if !wanted(mip, layer) {
                parts_left += 1;
                continue;
            }
            parts_done += 1;
            if !use_mapped[mip * layers as usize + layer as usize] {
                skipped += 1;
                if first {
                    let m = &lay.mips[mip];
                    let bytes = m.width as usize * m.height as usize * elem as usize;
                    zeroed.push((mip, layer, bytes));
                }
                continue;
            }
            let m = &lay.mips[mip];
            let len = m.width as usize * m.height as usize * elem as usize;
            let off = total as u64;
            parts.push((mip, layer, total, len));
            total += len;
            let (w, h) = (m.texel_w, m.texel_h);
            regions.push(
                vk::BufferImageCopy::default()
                    .buffer_offset(off)
                    .buffer_row_length(if block { m.width * 4 } else { m.width })
                    .buffer_image_height(if block { m.height * 4 } else { m.height })
                    .image_subresource(vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: mip as u32, base_array_layer: layer, layer_count: 1 })
                    .image_extent(vk::Extent3D { width: w, height: h, depth: 1 }),
            );
        }
    }
    if regions.is_empty() && zeroed.is_empty() {
        return Err(format!("texture {:#x}+{size:#x}: no mip mapped", t.base));
    }
    if zero_census() && !first {
        use std::sync::atomic::Ordering::Relaxed;
        let cause = (REUPLOAD_CAUSE.with(|c| c.get()) as usize).min(6);
        for &(mip, layer, _, len) in &parts {
            let (s, n) = range(mip, layer);
            let zero = ((s + n) as usize) <= src.len() && src[s as usize..(s + n) as usize].chunks(8).all(|c| c.iter().all(|&b| b == 0));
            ZERO_CENSUS[cause][0].fetch_add(1, Relaxed);
            ZERO_CENSUS[cause][2].fetch_add(len as u64, Relaxed);
            if zero {
                ZERO_CENSUS[cause][1].fetch_add(1, Relaxed);
                ZERO_CENSUS[cause][3].fetch_add(len as u64, Relaxed);
            }
        }
    }
    let zero_len = zeroed.iter().map(|z| z.2).max().unwrap_or(0);
    let (detiled, zero_off) = (total, total);
    total += zero_len;
    Ok(Chose::Parts(Chosen { lay, layers, elem, block, size, precise, held, gpu, windows, readers, sums_on, sums, end, regions, zeroed, parts, total, zero_len, zero_off, detiled, skipped, parts_done, parts_left, t_detile, t_table, t_locked, t_chosen, take_ms, t_start, c4_ms }))
}

/// Detiles the chosen (mip, layer)s of `src` straight into `dst` at their
/// offsets (`Chosen::parts`).
fn detile_parts(lay: &tiling::Layout, src: &[u8], parts: &[(usize, u32, usize, usize)], dst: &mut [u8]) {
    // Parts of under 1 MiB (array layers, small mips) are detiled on
    // several threads at once (`detile_into` splits larger ones itself):
    // a 360-layer array of 288 KiB layers took 91 ms one part at a time.
    let small: Vec<(usize, u32, usize, usize)> = parts.iter().copied().filter(|p| p.3 < 1 << 20).collect();
    let small_bytes: usize = small.iter().map(|p| p.3).sum();
    for &(mip, layer, off, len) in parts {
        if len >= 1 << 20 || small_bytes < 1 << 20 {
            lay.detile_into(src, mip, layer, &mut dst[off..off + len]);
        }
    }
    if small_bytes >= 1 << 20 {
        let threads = super::pool::threads();
        // Disjoint destination slices, handed out in order of offset.
        let mut rest: &mut [u8] = &mut dst[..];
        let mut base = 0usize;
        let mut jobs: Vec<Vec<(usize, u32, &mut [u8])>> = (0..threads).map(|_| Vec::new()).collect();
        let mut sorted = small;
        sorted.sort_by_key(|p| p.2);
        for (k, &(mip, layer, off, len)) in sorted.iter().enumerate() {
            let (_, tail) = std::mem::take(&mut rest).split_at_mut(off - base);
            let (mine, after) = tail.split_at_mut(len);
            rest = after;
            base = off + len;
            jobs[k % threads].push((mip, layer, mine));
        }
        let jobs: Vec<parking_lot::Mutex<Vec<(usize, u32, &mut [u8])>>> = jobs.into_iter().map(parking_lot::Mutex::new).collect();
        super::pool::for_each(jobs.len(), &|k| {
            for (mip, layer, out) in jobs[k].lock().iter_mut() {
                lay.detile_into(src, *mip, *layer, out);
            }
        });
    }
}

/// Records an upload's copy of `buf` into `image` (`regions`), with the
/// barriers around it.
fn record_upload(cmd: vk::CommandBuffer, image: vk::Image, first: bool, buf: vk::Buffer, regions: &[vk::BufferImageCopy], base: u64) {
    let c = ctx();
    unsafe {
        super::vk::checkpoint(cmd, super::vk::mark::UPLOAD, base);
        // A first upload starts from UNDEFINED; a re-upload keeps the
        // layout and waits for earlier work using the image.
        let b = vk::ImageMemoryBarrier::default()
            .old_layout(if first { vk::ImageLayout::UNDEFINED } else { vk::ImageLayout::GENERAL })
            .new_layout(vk::ImageLayout::GENERAL)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: vk::REMAINING_MIP_LEVELS, base_array_layer: 0, layer_count: vk::REMAINING_ARRAY_LAYERS })
            .src_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
        c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[b]);
        c.device.cmd_copy_buffer_to_image(cmd, buf, image, vk::ImageLayout::GENERAL, regions);
        let a = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
        c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[a], &[], &[]);
    }
}

fn partial_residency_note(key: ImageKey, skipped: usize, first: bool) -> String {
    format!("partially resident texture: {skipped} (mip, layer) not mapped, {}; image {:#x} {}x{} layers {} levels {} {:?}",
        if first { "zero-filled" } else { "left as they were" }, key.base, key.width, key.height, key.layers, key.levels, key.format)
}

/// A texture's re-upload left by `refresh` for after the image cache's lock
/// (since 2026-11-01): its write marks taken and re-armed under the lock,
/// its parts chosen and detiled outside it by the thread that took them, its
/// copy recorded and its hashes stored under the lock again
/// (`finish_refreshes`); since 2026-11-02 also a new texture's first upload
/// (`image_for`).
struct RefreshJob {
    key: ImageKey,
    image: vk::Image,
    t: Tsharp,
    region: u32,
    /// The tracked guest range (the hashes').
    range: (u64, u64),
    parts: Option<Vec<(u64, u64)>>,
    all: bool,
    cause: u8,
    remap: Option<(u64, u64, u8)>,
    wait: std::sync::Arc<RefreshWait>,
    /// A new texture's first upload (`image_for`): unmapped (mip, layer)s
    /// zero-filled, the image's first layout, its parts' sums.
    first: bool,
    /// The image's `gpu_parts` when the job was made.
    keep: Vec<(u32, u32, u64)>,
}

/// Set once a deferred re-upload is recorded or dropped: threads that bound
/// its texture meanwhile wait for it after their lock, before their work
/// can be submitted.
#[derive(Default)]
struct RefreshWait {
    done: Mutex<bool>,
    cv: parking_lot::Condvar,
}

/// A deferred re-upload detiled into a staging buffer of its own: the copy
/// its commit records (None: nothing mapped changed), the texture's size,
/// the cache's messages.
struct Prepared {
    copy: Option<(Staging, Vec<vk::BufferImageCopy>, usize)>,
    size: u64,
    notes: Vec<String>,
}

thread_local! {
    /// This thread's refreshes may leave their re-uploads to
    /// `finish_refreshes` (`Exec::bind_stage`).
    static REFRESH_DEFER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Re-uploads this thread left for after the lock.
    static REFRESH_JOBS: std::cell::RefCell<Vec<RefreshJob>> = const { std::cell::RefCell::new(Vec::new()) };
    /// Deferred re-uploads (another thread's, or its own) of textures this
    /// thread bound meanwhile.
    static REFRESH_WAITS: std::cell::RefCell<Vec<std::sync::Arc<RefreshWait>>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Deferred re-uploads (`RefreshJob`): [left for after the lock,
/// recorded, dropped (the image gone or GPU-written meanwhile), bindings
/// that waited for one, ns outside the lock, ns of those waits].
static REFRESH_OUT: [std::sync::atomic::AtomicU64; 6] = [const { std::sync::atomic::AtomicU64::new(0) }; 6];

/// First uploads left for after the lock (`image_for`):
/// [deferred, recorded, dropped (the image replaced or evicted meanwhile),
/// failed, ns outside the lock].
static FIRST_OUT: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];

/// Whether a re-upload of `key` detiles on the CPU into the upload command
/// buffer the uploads share, as one left for after the lock does: 2D,
/// smaller than `tune gpu_detile` MiB, `tune upload_batch`.
fn reupload_on_cpu(key: &ImageKey, t: &Tsharp) -> bool {
    if key.ty3d || !super::tune::upload_batch() {
        return false;
    }
    let Some(el) = elem_log2(t.format) else { return false };
    let Some(lay) = tiling::layout(t.sw_mode, el, key.width, key.height, key.levels, is_block(t.format)) else { return false };
    match super::tune::gpu_detile() {
        0 => true,
        n => lay.size(key.layers) < n << 20,
    }
}

/// Lets this thread's texture refreshes leave their re-uploads to
/// `finish_refreshes` (`Exec::bind_stage`, around its lock for a stage's
/// images).
pub fn defer_refreshes() {
    REFRESH_DEFER.with(|d| d.set(true));
}

/// After the lock: this thread's deferred re-uploads (chosen and detiled
/// outside it, recorded under it), then its waits for those of other
/// threads whose textures it bound meanwhile; its own come first, so no two
/// threads wait for each other. The first error, as `refresh` returned it.
pub fn finish_refreshes() -> Result<(), String> {
    use std::sync::atomic::Ordering::Relaxed;
    REFRESH_DEFER.with(|d| d.set(false));
    let jobs = REFRESH_JOBS.with(|j| std::mem::take(&mut *j.borrow_mut()));
    let mut err = None;
    for job in jobs {
        if let Err(e) = run_refresh(job) {
            err.get_or_insert(e);
        }
    }
    let waits = REFRESH_WAITS.with(|w| std::mem::take(&mut *w.borrow_mut()));
    for w in waits {
        let t0 = std::time::Instant::now();
        let mut g = w.done.lock();
        while !*g {
            if w.cv.wait_for(&mut g, std::time::Duration::from_secs(5)).timed_out() && !*g {
                crate::warn!("images: waited 5 s for another thread's re-upload of a texture this one bound; going on");
                break;
            }
        }
        REFRESH_OUT[5].fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
    }
    match err {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// One deferred re-upload: prepared outside the lock, committed under it.
/// The title's streamed texture chunks completed (`engine::tex_chunk_hook`,
/// `DS_TEX_STREAM=1`): their guest ranges and when, the last STREAM_KEEP.
static STREAMED: Mutex<std::collections::VecDeque<(u64, u64, std::time::Instant)>> = Mutex::new(std::collections::VecDeque::new());
const STREAM_KEEP: usize = 8192;
/// [chunks, bytes]; the uploads made where a binding needs them by how long
/// before them a chunk over their range completed: [first uploads, re-uploads]
/// x [< 1 ms, 1-5, 5-20, 20-100, 100-1000, 1 s or more, none], with their
/// ns of upload time.
static STREAM_STATS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];
static STREAM_UPLOADS: [[[std::sync::atomic::AtomicU64; 2]; 7]; 2] = [const { [const { [const { std::sync::atomic::AtomicU64::new(0) }; 2] }; 7] }; 2];

/// A streamed texture chunk's guest range holds its final data now.
pub fn stream_chunk_done(a: u64, n: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    STREAM_STATS[0].fetch_add(1, Relaxed);
    STREAM_STATS[1].fetch_add(n, Relaxed);
    let mut g = STREAMED.lock();
    if g.len() >= STREAM_KEEP {
        g.pop_front();
    }
    g.push_back((a, a + n, std::time::Instant::now()));
}

/// The census of an upload of guest [a, a+n) taking `ns` (`run_refresh`).
fn stream_census(first: bool, a: u64, n: u64, ns: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    if STREAM_STATS[0].load(Relaxed) == 0 {
        return;
    }
    let age = {
        let g = STREAMED.lock();
        g.iter().rev().find(|e| e.0 < a + n && a < e.1).map(|e| e.2.elapsed())
    };
    let k = match age.map(|d| d.as_micros()) {
        Some(us) if us < 1_000 => 0,
        Some(us) if us < 5_000 => 1,
        Some(us) if us < 20_000 => 2,
        Some(us) if us < 100_000 => 3,
        Some(us) if us < 1_000_000 => 4,
        Some(_) => 5,
        None => 6,
    };
    let s = &STREAM_UPLOADS[!first as usize][k];
    s[0].fetch_add(1, Relaxed);
    s[1].fetch_add(ns, Relaxed);
}

/// The texture streaming census's line ("" before a chunk).
pub fn stream_status() -> String {
    use std::sync::atomic::Ordering::Relaxed;
    let chunks = STREAM_STATS[0].load(Relaxed);
    if chunks == 0 {
        return String::new();
    }
    let names = ["< 1 ms", "1-5 ms", "5-20 ms", "20-100 ms", "0.1-1 s", "1 s or more", "none"];
    let part = |w: usize| -> String {
        names.iter().enumerate().map(|(k, n)| {
            let s = &STREAM_UPLOADS[w][k];
            format!("{n} {} ({:.0} ms)", s[0].load(Relaxed), s[1].load(Relaxed) as f64 / 1e6)
        }).collect::<Vec<_>>().join(", ")
    };
    format!(
        "    texture streaming (`DS_TEX_STREAM`): {chunks} chunks completed ({} MiB); uploads where a binding needed them by how long before a chunk over their range completed: first uploads {}; re-uploads {}
",
        STREAM_STATS[1].load(Relaxed) >> 20,
        part(0),
        part(1)
    )
}

fn run_refresh(job: RefreshJob) -> Result<(), String> {
    use std::sync::atomic::Ordering::Relaxed;
    let t0 = std::time::Instant::now();
    REUPLOAD_CAUSE.with(|c| c.set(job.cause));
    let prepared = prepare_reupload(&job);
    REUPLOAD_CAUSE.with(|c| c.set(0));
    // Hashed once the bytes were read, outside the guest-reader lock (their
    // page checks take the mapping table's).
    let hashes = prepared.is_ok().then(|| all_hashes(job.range.0, job.range.1));
    if let Ok(p) = &prepared {
        let (key, n) = (job.key, p.copy.as_ref().map_or(0, |c| c.2));
        super::hitch::note_since(super::hitch::Kind::Upload, t0, || format!("{}x{}x{}{} {:?} {} KiB", key.width, key.height, key.layers, if key.ty3d { " 3D" } else { "" }, key.format, n >> 10));
    }
    if zero_census() {
        ZERO_CENSUS[(job.cause as usize).min(6)][4].fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
    }
    if job.first {
        FIRST_OUT[4].fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
    } else {
        REFRESH_OUT[4].fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
    }
    stream_census(job.first, job.range.0, job.range.1, t0.elapsed().as_nanos() as u64);
    let r = with(|im| im.commit_refresh(&job, prepared, hashes, t0));
    *job.wait.done.lock() = true;
    job.wait.cv.notify_all();
    r
}

/// A deferred upload's parts chosen and detiled, without the cache (a
/// re-upload's, or a first upload's with its parts' sums).
fn prepare_reupload(job: &RefreshJob) -> Result<Prepared, String> {
    use std::sync::atomic::Ordering::Relaxed;
    let mut notes = Vec::new();
    let Chosen { lay, layers, elem, size, precise, sums_on, mut sums, end, mut regions, zeroed, parts, total, zero_off, detiled, skipped, parts_done, parts_left, t_detile, readers, block, .. } = match choose_2d(job.key, &job.t, job.parts.as_deref(), job.all, job.first, Some(job.region), false, &job.keep, &mut notes)? {
        Chose::Done((_, size)) => return Ok(Prepared { copy: None, size, notes }),
        Chose::Parts(c) => c,
    };
    let t_alloc0 = std::time::Instant::now();
    let st = staging_take(total as u64)?;
    let src = unsafe { std::slice::from_raw_parts(job.t.base as *const u8, end as usize) };
    let dst = unsafe { std::slice::from_raw_parts_mut(st.ptr, total) };
    let t_alloc = std::time::Instant::now();
    detile_parts(&lay, src, &parts, dst);
    dst[zero_off..].fill(0);
    let t_detiled = std::time::Instant::now();
    // A first upload's sums (for later partial re-uploads), as `upload_inner`
    // makes them.
    if sums_on && job.first {
        let todo: Vec<(usize, u64, u64)> = parts
            .iter()
            .map(|&(mip, layer, _, _)| {
                let (s, n) = part_range(&lay, elem, mip, layer);
                (mip * layers as usize + layer as usize, job.t.base + s, n)
            })
            .collect();
        let threads = if detiled >= 1 << 20 { super::pool::threads() } else { 1 };
        for (i, h) in hash_parts(&todo, threads) {
            sums[i] = h;
        }
    }
    UPLOAD_PHASES[0].fetch_add((t_alloc - t_alloc0).as_nanos() as u64, Relaxed);
    UPLOAD_PHASES[1].fetch_add((t_detiled - t_alloc).as_nanos() as u64, Relaxed);
    UPLOAD_PHASES[2].fetch_add(t_detiled.elapsed().as_nanos() as u64, Relaxed);
    // Guest memory is no longer read.
    drop(readers);
    UPLOAD_SPLIT[0].fetch_add(t_detile.elapsed().as_nanos() as u64, Relaxed);
    DETILED.fetch_add(detiled as u64, Relaxed);
    if sums_on {
        SUB_SUMS.lock().get_or_insert_with(HashMap::default).insert(job.key, sums);
    }
    if precise {
        PART_UPLOADS[0].fetch_add(1, Relaxed);
        PART_UPLOADS[1].fetch_add(parts_done, Relaxed);
        PART_UPLOADS[2].fetch_add(parts_left, Relaxed);
        PART_UPLOADS[3].fetch_add(detiled as u64, Relaxed);
    }
    push_zero_regions(&lay, block, zero_off as u64, &zeroed, &mut regions);
    if skipped > 0 {
        notes.push(partial_residency_note(job.key, skipped, job.first));
    }
    Ok(Prepared { copy: Some((st, regions, total)), size, notes })
}

/// The copies of a first upload's unmapped (mip, layer)s from the zero area
/// at `off` in its staging buffer.
fn push_zero_regions(lay: &tiling::Layout, block: bool, off: u64, zeroed: &[(usize, u32, usize)], regions: &mut Vec<vk::BufferImageCopy>) {
    for &(mip, layer, _) in zeroed {
        let m = &lay.mips[mip];
        regions.push(
            vk::BufferImageCopy::default()
                .buffer_offset(off)
                .buffer_row_length(if block { m.width * 4 } else { m.width })
                .buffer_image_height(if block { m.height * 4 } else { m.height })
                .image_subresource(vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: mip as u32, base_array_layer: layer, layer_count: 1 })
                .image_extent(vk::Extent3D { width: m.texel_w, height: m.texel_h, depth: 1 }),
        );
    }
}

#[track_caller]
pub fn with<R>(f: impl FnOnce(&mut Images) -> R) -> R {
    static SCRUBBER: std::sync::Once = std::sync::Once::new();
    SCRUBBER.call_once(|| {
        if let Err(e) = crate::thread::spawn_host("gpu-texscrub", 1 << 20, Box::new(scrubber)) {
            crate::warn!("images: no texture scrubber thread ({e})");
        }
    });
    let asked = holds_on().then(std::time::Instant::now);
    let mut g = if holder_trace() {
        match IMAGES.try_lock_for(std::time::Duration::from_millis(1)) {
            Some(g) => g,
            None => {
                sample_holder(std::panic::Location::caller());
                super::lockstat::IMAGE_CACHE.lock(&IMAGES)
            }
        }
    } else {
        super::lockstat::IMAGE_CACHE.lock(&IMAGES)
    };
    if holder_trace() {
        let l = std::panic::Location::caller();
        *HOLDER_AT.lock() = (crate::log::thread_name(), l.file(), l.line());
        HOLDER.store(unsafe { crate::platform::os::Win32::System::Threading::GetCurrentThreadId() }, std::sync::atomic::Ordering::Release);
    }
    let held_from = holds_on().then(std::time::Instant::now);
    if let (Some(a), Some(b)) = (asked, held_from) {
        let ns = (b - a).as_nanos() as u64;
        if ns >= 1000 {
            let w = &WAITS[hold_class()];
            w[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            w[1].fetch_add(ns, std::sync::atomic::Ordering::Relaxed);
            let l = std::panic::Location::caller();
            let mut s = WAIT_SITES.lock();
            let e = s.get_or_insert_with(HashMap::default).entry((l.file(), l.line())).or_insert((0, 0));
            e.0 += 1;
            e.1 += ns;
        }
    }
    if g.is_none() {
        let c = ctx();
        unsafe {
            let pool = c
                .device
                .create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(c.queue_family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None)
                .expect("image pool");
            let cmd = c.device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(1)).expect("image cmd")[0];
            let fence = c.device.create_fence(&vk::FenceCreateInfo::default(), None).expect("image fence");
            let qp = upload_stamps();
            *g = Some(Images { qp, map: HashMap::default(), by_base: HashMap::default(), gpu_bases: Default::default(), stencil_of: HashMap::default(), meta_of: Default::default(), meta_now: HashMap::default(), last_note: None, note_gen: 0, view_memo: memo_map(), key_memo: memo_map(), pending_clears: HashMap::default(), samplers: HashMap::default(), pool, cmd, fence, inflight: Default::default(), spare_cmds: Vec::new(), detile_free: Vec::new(), copy_pool: vk::CommandPool::null(), copy_spare: Vec::new(), copy_timeline: vk::Semaphore::null(), copy_value: 0, reported: HashMap::default(), uploads: 0, reuploads: 0, upload_bytes: 0, upload_ns: 0, write_seq: 0, owned: Default::default(), max_span: 0, pending_overwrites: 0, overwritten_keys: Default::default(), pending_change: 0, pending_published: 0, scrub_keys: Vec::new(), scrub_gen: 0, scrub_at: 0, overwrite_clears: 0, overwrite_uploads: 0, partial_seen: Default::default(), dummies: HashMap::default(), depth_grows: 0, depth_feeds: Default::default(), refreshing: HashMap::default() });
        }
    }
    let im = g.as_mut().unwrap();
    let r = f(&mut *im);
    PENDING.store((im.pending_clears.len() + im.pending_overwrites) as u64, std::sync::atomic::Ordering::Release);
    if im.pending_change != im.pending_published {
        im.pending_published = im.pending_change;
        let set: super::fx::Set<ImageKey> = im.pending_clears.keys().copied().chain(im.overwritten_keys.iter().copied()).collect();
        *PENDING_SET.lock() = Some(std::sync::Arc::new(set));
        PENDING_GEN.fetch_add(1, std::sync::atomic::Ordering::Release);
    }
    if let Some(t) = held_from {
        let ns = t.elapsed().as_nanos() as u64;
        if ns >= 200_000 {
            use std::sync::atomic::Ordering::Relaxed;
            let h = &HOLDS[hold_class()];
            h[0].fetch_add(1, Relaxed);
            h[1].fetch_add(ns, Relaxed);
            h[2].fetch_max(ns, Relaxed);
            let l = std::panic::Location::caller();
            let mut s = HOLD_SITES.lock();
            let e = s.get_or_insert_with(HashMap::default).entry((l.file(), l.line())).or_insert((0, 0));
            e.0 += 1;
            e.1 += ns;
        }
    }
    if holder_trace() {
        HOLDER.store(0, std::sync::atomic::Ordering::Release);
    }
    r
}

impl Images {
    fn report(&mut self, msg: String) {
        let n = self.reported.entry(msg.clone()).or_insert(0);
        *n += 1;
        if *n == 1 {
            crate::warn!("images: {msg}");
        }
    }

    /// Creates the VkImage for `key` (GENERAL layout, contents undefined); a failure counted with
    /// its key (`IMAGE_FAILS`, `LAST_IMAGE_FAIL`: the census beside the over-budget warning).
    fn create(&mut self, key: ImageKey) -> Result<(vk::Image, super::slab::Mem, u64), String> {
        self.create_in(key, false)
    }

    /// `create` for a texture: while video memory is under pressure (`under_pressure`) or this
    /// process is over its eviction mark (`OVER_HIGH`) its memory is system memory first
    /// (`slab::alloc`), video memory left to what cannot go there (the mirror: NVIDIA's sparse
    /// buffers take video memory only) or costs more there (the images the GPU writes). It comes
    /// back once video memory has room (`maybe_evict`). Not for a storage binding (`storage`):
    /// the GPU writes such an image, which then never leaves its memory (the review of
    /// 2026-11-22 morning).
    fn create_texture(&mut self, key: ImageKey, storage: bool) -> Result<(vk::Image, super::slab::Mem, u64), String> {
        self.create_in(key, big_on_small_card(&key) || (!storage && (under_pressure() || OVER_HIGH.load(std::sync::atomic::Ordering::Relaxed))))
    }

    fn create_in(&mut self, key: ImageKey, system_first: bool) -> Result<(vk::Image, super::slab::Mem, u64), String> {
        let r = self.create_inner(key, system_first);
        if r.is_err() {
            IMAGE_FAILS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            *LAST_IMAGE_FAIL.lock() = Some(key);
        }
        r
    }

    fn create_inner(&mut self, key: ImageKey, system_first: bool) -> Result<(vk::Image, super::slab::Mem, u64), String> {
        let c = ctx();
        let props = unsafe { c.instance.get_physical_device_format_properties(c.pdev, key.format) };
        let f = props.optimal_tiling_features;
        let mut usage = vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC;
        if f.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE) {
            usage |= vk::ImageUsageFlags::SAMPLED;
        }
        if !is_depth(key.format) {
            let sprops = unsafe { c.instance.get_physical_device_format_properties(c.pdev, storage_format(key.format)) };
            if sprops.optimal_tiling_features.contains(vk::FormatFeatureFlags::STORAGE_IMAGE) {
                usage |= vk::ImageUsageFlags::STORAGE;
            }
        }
        if f.contains(vk::FormatFeatureFlags::COLOR_ATTACHMENT) {
            usage |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
        }
        if f.contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT) {
            usage |= vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
        }
        let mut flags = if is_depth(key.format) { vk::ImageCreateFlags::empty() } else { vk::ImageCreateFlags::MUTABLE_FORMAT | vk::ImageCreateFlags::EXTENDED_USAGE };
        if key.cube && key.layers % 6 == 0 {
            flags |= vk::ImageCreateFlags::CUBE_COMPATIBLE;
        }
        let (ty, extent, layers) = if key.ty3d {
            (vk::ImageType::TYPE_3D, vk::Extent3D { width: key.width, height: key.height, depth: key.layers }, 1)
        } else {
            (vk::ImageType::TYPE_2D, vk::Extent3D { width: key.width, height: key.height, depth: 1 }, key.layers)
        };
        let ci = vk::ImageCreateInfo::default()
            .flags(flags)
            .image_type(ty)
            .format(key.format)
            .extent(extent)
            .mip_levels(key.levels)
            .array_layers(layers)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        // Shared with the compute family's queue when it exists (ordered
        // async compute: its batches sample and store images in GENERAL
        // layout as the main queue's do).
        let fams = [c.queue_family, c.compute.as_ref().map_or(c.queue_family, |q| q.family)];
        let ci = if c.compute.is_some() { ci.sharing_mode(vk::SharingMode::CONCURRENT).queue_family_indices(&fams) } else { ci };
        // (Inside the refusal window after a failed allocation, `vk::gate_refuses`, an image whose
        // smallest possible size (half a byte a texel, one level) would be refused is not made:
        // while memory was out every draw sampling a texture not made yet tried again.)
        // (With the system-memory fallback such an image is made there, unless the fallback
        // refused moments ago too: `vk::system_refusing`.)
        let least = (key.width as u64 * key.height as u64 * key.layers.max(1) as u64) / 2;
        if least >= 64 << 10 && (super::vk::system_refusing(least) || super::vk::images_lack_system_type()) && super::vk::gate_refuses(least) {
            return Err(format!("image {}x{}x{} {:?}: video memory ran out moments ago (refused with no driver call)", key.width, key.height, key.layers, key.format));
        }
        unsafe {
            let t0 = std::time::Instant::now();
            let image = c.device.create_image(&ci, None).map_err(|e| format!("vkCreateImage {key:?}: {e}"))?;
            // (Whether the driver requires or prefers the image in an allocation of its own.)
            let mut ded = vk::MemoryDedicatedRequirements::default();
            let mut req2 = vk::MemoryRequirements2::default().push_next(&mut ded);
            c.device.get_image_memory_requirements2(&vk::ImageMemoryRequirementsInfo2::default().image(image), &mut req2);
            let req = req2.memory_requirements;
            c.note_image_bits(req.memory_type_bits);
            let (requires, prefers) = (ded.requires_dedicated_allocation != 0, ded.prefers_dedicated_allocation != 0);
            let t1 = std::time::Instant::now();
            // (The image destroyed when its memory fails: until 2026-11-21 night each failure left
            // its VkImage behind, and while memory was out every draw sampling a texture not made
            // yet failed again.)
            let mem = match super::slab::alloc(req, requires, prefers, system_first) {
                Ok(m) => m,
                Err(e) => {
                    c.device.destroy_image(image, None);
                    return Err(e);
                }
            };
            let t2 = std::time::Instant::now();
            if let Err(e) = c.device.bind_image_memory(image, mem.memory, mem.offset) {
                c.device.destroy_image(image, None);
                super::slab::free(mem);
                return Err(format!("bind image: {e}"));
            }
            // (Its parts: the image, the memory's allocation, the bind.)
            super::hitch::note_since(super::hitch::Kind::Upload, t0, || {
                let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
                format!("image creation {}x{} ({} MiB; create {:.1}, allocate {:.1}, bind {:.1} ms)", key.width, key.height, req.size >> 20, ms(t1 - t0), ms(t2 - t1), ms(t2.elapsed()))
            });
            Ok((image, mem, req.size))
        }
    }

    /// Recycles the uploads the GPU completed (in submission order); with
    /// more than `max` in flight or more than `max_bytes` of their staging
    /// buffers, waits for the oldest.
    fn reap(&mut self, max: usize, max_bytes: u64) {
        let c = ctx();
        while let Some(&(_, fence, _, _)) = self.inflight.front() {
            let done = unsafe { c.device.get_fence_status(fence) }.unwrap_or(true);
            if !done {
                let bytes: u64 = self.inflight.iter().map(|x| x.3.bytes()).sum();
                if self.inflight.len() <= max && bytes <= max_bytes {
                    break;
                }
                UPLOAD_WAITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let t0 = std::time::Instant::now();
                let r = unsafe { c.device.wait_for_fences(&[fence], true, 10_000_000_000) };
                super::hitch::note_since(super::hitch::Kind::GpuWait, t0, || format!("earlier texture uploads ({} in flight, {} MiB)", self.inflight.len(), bytes >> 20));
                if let Err(e) = r {
                    if super::exec::lose_device() {
                        crate::error!("images: upload failed: {e}; GPU execution stops; checkpoints: {}", super::vk::checkpoint_report());
                    }
                }
            }
            let (cmd, fence, qp, held) = self.inflight.pop_front().unwrap();
            note_upload_span(qp, || format!(": texture upload detiled on the GPU ({} KiB of buffers)", held.dev.iter().map(|b| b.2).sum::<u64>() >> 10));
            unsafe {
                c.device.reset_fences(&[fence]).ok();
                c.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).ok();
            }
            self.spare_cmds.push((cmd, fence, qp));
            self.release_held(held);
        }
    }

    /// What an upload the GPU is done with (or that was not submitted) held.
    fn release_held(&mut self, held: Held) {
        self.release_detile(held.dev);
        if let Some(x) = held.copy_cmd {
            unsafe { ctx().device.reset_command_buffer(x, vk::CommandBufferResetFlags::empty()).ok() };
            self.copy_spare.push(x);
        }
    }

    /// The copy queue's part of an upload: `record` on one of its command
    /// buffers, submitted to signal the timeline's next value. Returns the
    /// command buffer and that value (None: nothing was submitted).
    fn submit_copy_part(&mut self, pool: vk::CommandPool, sem: vk::Semaphore, record: impl FnOnce(vk::CommandBuffer)) -> Option<(vk::CommandBuffer, u64)> {
        let c = ctx();
        let x = match self.copy_spare.pop() {
            Some(x) => x,
            None => unsafe { c.device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(1)).ok()?[0] },
        };
        let ok = unsafe {
            c.device.begin_command_buffer(x, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).ok();
            record(x);
            c.device.end_command_buffer(x).ok();
            c.submit_timeline(true, &[x], &[], &[(sem, self.copy_value + 1)], vk::Fence::null()).is_ok()
        };
        if !ok {
            unsafe { c.device.reset_command_buffer(x, vk::CommandBufferResetFlags::empty()).ok() };
            self.copy_spare.push(x);
            return None;
        }
        self.copy_value += 1;
        Some((x, self.copy_value))
    }

    /// The copy queue's part of GPU detiling: its command pool and timeline,
    /// made on first use (None: no copy queue, or `tune copy_queue` 0).
    fn copy_parts(&mut self) -> Option<(vk::CommandPool, vk::Semaphore)> {
        let c = ctx();
        let q = c.copy.as_ref().filter(|_| super::tune::copy_queue())?;
        if self.copy_pool == vk::CommandPool::null() {
            unsafe {
                let pool = c.device.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(q.family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None).ok()?;
                let mut tci = vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE).initial_value(0);
                let Ok(sem) = c.device.create_semaphore(&vk::SemaphoreCreateInfo::default().push_next(&mut tci), None) else {
                    c.device.destroy_command_pool(pool, None);
                    return None;
                };
                self.copy_pool = pool;
                self.copy_timeline = sem;
            }
        }
        Some((self.copy_pool, self.copy_timeline))
    }

    /// Records and submits a transfer reading what `s` holds without waiting
    /// for it (on the one queue it runs after the work submitted before it
    /// and before batches submitted later, and its barriers order it against
    /// the image's other uses; `reap` recycles it), with up to `max_bytes` of
    /// such buffers in flight, after the copy queue's timeline reaches `wait`
    /// (semaphore, value) when given.
    fn submit_held(&mut self, s: Held, max_bytes: u64, wait: Option<(vk::Semaphore, u64)>, f: impl FnOnce(vk::CommandBuffer)) {
        // (After the batched uploads recorded before it, `tune upload_batch`.)
        flush_uploads();
        self.reap(MAX_INFLIGHT - 1, max_bytes.saturating_sub(s.bytes()));
        let c = ctx();
        let (cmd, fence, qp) = match self.spare_cmds.pop() {
            Some(x) => x,
            None => unsafe {
                let cmd = c.device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(self.pool).command_buffer_count(1)).expect("upload cmd")[0];
                (cmd, c.device.create_fence(&vk::FenceCreateInfo::default(), None).expect("upload fence"), upload_stamps())
            },
        };
        unsafe {
            c.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).ok();
            stamp(cmd, qp, 0);
            f(cmd);
            stamp(cmd, qp, 1);
            c.device.end_command_buffer(cmd).ok();
            let r = match wait {
                Some((sem, v)) => c.submit_timeline(false, &[cmd], &[(sem, v, vk::PipelineStageFlags2::ALL_COMMANDS)], &[], fence),
                None => c.submit(cmd, &[], &[], fence),
            };
            match r {
                Ok(()) => self.inflight.push_back((cmd, fence, qp, s)),
                Err(e) => {
                    if super::exec::lose_device() {
                        crate::error!("images: submit failed: {e}; GPU execution stops; checkpoints: {}", super::vk::checkpoint_report());
                    }
                    c.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).ok();
                    self.spare_cmds.push((cmd, fence, qp));
                    self.release_held(s);
                }
            }
        }
    }

    fn one_shot(&mut self, f: impl FnOnce(vk::CommandBuffer)) {
        self.one_shot_after(None, f)
    }

    /// `one_shot` after the copy queue's timeline reaches `wait` (semaphore,
    /// value) when given.
    fn one_shot_after(&mut self, wait: Option<(vk::Semaphore, u64)>, f: impl FnOnce(vk::CommandBuffer)) {
        // (Its timestamps: `stamp`, `note_upload_span`. After the batched
        // uploads recorded before it, `tune upload_batch`.)
        flush_uploads();
        let c = ctx();
        unsafe {
            c.device.begin_command_buffer(self.cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).ok();
            stamp(self.cmd, self.qp, 0);
            f(self.cmd);
            stamp(self.cmd, self.qp, 1);
            c.device.end_command_buffer(self.cmd).ok();
            let r = match wait {
                Some((sem, v)) => c.submit_timeline(false, &[self.cmd], &[(sem, v, vk::PipelineStageFlags2::ALL_COMMANDS)], &[], self.fence),
                None => c.submit(self.cmd, &[], &[], self.fence),
            };
            match r {
                Ok(()) => {
                    let t0 = std::time::Instant::now();
                    let r = c.device.wait_for_fences(&[self.fence], true, 10_000_000_000);
                    super::hitch::note_since(super::hitch::Kind::GpuWait, t0, || "a texture cache one-shot (upload or transition)".to_string());
                    if r.is_ok() {
                        note_upload_span(self.qp, || ": texture cache one-shot (upload or transition)".to_string());
                    }
                    if let Err(e) = r {
                        if super::exec::lose_device() {
                            crate::error!("images: upload/transition failed: {e}; GPU execution stops; checkpoints: {}", super::vk::checkpoint_report());
                        }
                    }
                }
                Err(e) => {
                    if super::exec::lose_device() {
                        crate::error!("images: submit failed: {e}; GPU execution stops; checkpoints: {}", super::vk::checkpoint_report());
                    }
                }
            }
            c.device.reset_fences(&[self.fence]).ok();
            c.device.reset_command_buffer(self.cmd, vk::CommandBufferResetFlags::empty()).ok();
        }
    }

    /// Value statistics of every cached image at guest address `base` (mip
    /// 0, all layers or slices; float formats), with the largest magnitude
    /// of the first channel per layer/slice; BC1 textures: each layer's mean
    /// endpoint color and share of all-zero blocks (diagnosis: `imgstat`).
    pub fn stats_at(&mut self, base: u64) -> Vec<String> {
        let imgs: Vec<(vk::Image, ImageKey)> = self.map.values().filter(|i| i.key.base == base).map(|i| (i.image, i.key)).collect();
        let mut out = Vec::new();
        for (image, k) in imgs {
            let bpp = texel_bytes(k.format) as u64;
            let half = matches!(k.format, vk::Format::R16G16B16A16_SFLOAT | vk::Format::R16G16_SFLOAT | vk::Format::R16_SFLOAT);
            let full = matches!(k.format, vk::Format::R32G32B32A32_SFLOAT | vk::Format::R32G32_SFLOAT | vk::Format::R32_SFLOAT);
            let bc1 = matches!(k.format, vk::Format::BC1_RGBA_SRGB_BLOCK | vk::Format::BC1_RGBA_UNORM_BLOCK);
            let unorm8 = matches!(k.format, vk::Format::R8_UNORM | vk::Format::R8G8_UNORM | vk::Format::R8G8B8A8_UNORM | vk::Format::B8G8R8A8_UNORM);
            let unorm16 = matches!(k.format, vk::Format::R16_UNORM | vk::Format::R16G16_UNORM | vk::Format::R16G16B16A16_UNORM);
            if bpp == 0 || !(half || full || bc1 || unorm8 || unorm16) {
                out.push(format!("{base:#x} {}x{}x{} {:?}: not a float format", k.width, k.height, k.layers, k.format));
                continue;
            }
            let (layers, depth) = if k.ty3d { (1, k.layers) } else { (k.layers, 1) };
            let slice = if bc1 { bpp * k.width.div_ceil(4) as u64 * k.height.div_ceil(4) as u64 } else { bpp * k.width as u64 * k.height as u64 };
            let size = slice * (layers * depth) as u64;
            let c = ctx();
            let Ok((buf, mem, ptr)) = c.host_buffer(size, vk::BufferUsageFlags::TRANSFER_DST) else {
                out.push(format!("{base:#x}: no staging buffer"));
                continue;
            };
            self.one_shot(|cmd| unsafe {
                let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ);
                c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
                let region = vk::BufferImageCopy::default()
                    .image_subresource(vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: layers })
                    .image_extent(vk::Extent3D { width: k.width, height: k.height, depth });
                c.device.cmd_copy_image_to_buffer(cmd, image, vk::ImageLayout::GENERAL, buf, &[region]);
                let hb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ);
                c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[hb], &[], &[]);
            });
            let raw = unsafe { std::slice::from_raw_parts(ptr, size as usize).to_vec() };
            unsafe {
                c.device.destroy_buffer(buf, None);
                c.device.unmap_memory(mem);
                c.device.free_memory(mem, None);
            }
            if bc1 {
                let rgb = |c: u16| [((c >> 11) & 31) as f32 / 31.0, ((c >> 5) & 63) as f32 / 63.0, (c & 31) as f32 / 31.0];
                let per: Vec<String> = raw
                    .chunks(slice as usize)
                    .enumerate()
                    .map(|(l, s)| {
                        let (mut sum, mut zero, mut n) = ([0f64; 3], 0usize, 0usize);
                        for b in s.chunks_exact(8) {
                            n += 1;
                            if b.iter().all(|&x| x == 0) {
                                zero += 1;
                            }
                            let (c0, c1) = (rgb(u16::from_le_bytes([b[0], b[1]])), rgb(u16::from_le_bytes([b[2], b[3]])));
                            for i in 0..3 {
                                sum[i] += (c0[i] + c1[i]) as f64 / 2.0;
                            }
                        }
                        let n = n.max(1) as f64;
                        format!("{l}:{:.2}/{:.2}/{:.2}{}", sum[0] / n, sum[1] / n, sum[2] / n, if zero > 0 { format!(" z{:.0}%", 100.0 * zero as f64 / n) } else { String::new() })
                    })
                    .collect();
                out.push(format!("{base:#x} {}x{}x{} {:?}: mean endpoint rgb per layer: {}", k.width, k.height, k.layers, k.format, per.join(" ")));
                continue;
            }
            if unorm8 || unorm16 {
                // Per channel mean and share of zero texels, per layer.
                let w = if unorm16 { 2 } else { 1 };
                let max = if unorm16 { 65535.0 } else { 255.0 };
                let per: Vec<String> = raw
                    .chunks(slice as usize)
                    .enumerate()
                    .map(|(l, s)| {
                        let n = s.len() / bpp as usize;
                        let ch = bpp as usize / w;
                        let mut sum = [0u64; 4];
                        let mut zero = 0usize;
                        for t in s.chunks_exact(bpp as usize) {
                            if t.iter().all(|&x| x == 0) {
                                zero += 1;
                            }
                            for (i, c) in t.chunks_exact(w).enumerate() {
                                sum[i] += if w == 2 { u16::from_le_bytes([c[0], c[1]]) as u64 } else { c[0] as u64 };
                            }
                        }
                        let means: Vec<String> = (0..ch).map(|i| format!("{:.3}", sum[i] as f64 / max / n.max(1) as f64)).collect();
                        format!("{l}:{}{}", means.join("/"), if zero > 0 { format!(" z{:.0}%", 100.0 * zero as f64 / n.max(1) as f64) } else { String::new() })
                    })
                    .collect();
                out.push(format!("{base:#x} {}x{}x{} {:?}: channel means per layer: {}", k.width, k.height, k.layers, k.format, per.join(" ")));
                continue;
            }
            let st = float_stats(k.format, &raw).unwrap_or_default();
            let maxima: Vec<String> = raw
                .chunks(slice as usize)
                .map(|s| {
                    let m = s
                        .chunks_exact(bpp as usize)
                        .map(|t| if half { half_to_f32(u16::from_le_bytes([t[0], t[1]])) } else { f32::from_le_bytes([t[0], t[1], t[2], t[3]]) })
                        .fold(0f32, |m, v| if v.is_nan() { f32::NAN } else { m.max(v.abs()) });
                    format!("{m:.3}")
                })
                .collect();
            out.push(format!("{base:#x} {}x{}x{} {:?}: {st}; first-channel max per {}: {}", k.width, k.height, k.layers, k.format, if k.ty3d { "slice" } else { "layer" }, maxima.join(" ")));
        }
        out
    }

    /// Fills every cached image at guest address `base` (all layers and
    /// mips) with one color value (diagnosis: `fillimg`; experiments with
    /// GPU-written inputs). Returns how many images.
    pub fn fill_at(&mut self, base: u64, value: f32) -> usize {
        let imgs: Vec<(vk::Image, ImageKey)> = self.map.values().filter(|i| i.key.base == base).map(|i| (i.image, i.key)).collect();
        let c = ctx();
        for (image, k) in &imgs {
            if is_depth(k.format) {
                continue;
            }
            let layers = if k.ty3d { 1 } else { k.layers };
            let range = vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: k.levels.max(1), base_array_layer: 0, layer_count: layers };
            let color = vk::ClearColorValue { float32: [value; 4] };
            self.one_shot(|cmd| unsafe {
                let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
                c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
                c.device.cmd_clear_color_image(cmd, *image, vk::ImageLayout::GENERAL, &color, &[range]);
                let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
                c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[mb], &[], &[]);
            });
        }
        imgs.len()
    }

    /// Guest addresses of the images a shader last bound as storage
    /// (diagnosis: `fillimg cs=<shader>`).
    pub fn written_by(&self, shader: u64) -> Vec<u64> {
        let mut v: Vec<u64> = self.map.values().filter(|i| i.writer == shader).map(|i| i.key.base).collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// Uploads the CPU-sourced textures at guest address `base` (all of
    /// them without one) again at their next use (diagnosis: `reupload`);
    /// returns how many.
    pub fn force_reupload(&mut self, base: Option<u64>) -> usize {
        let mut n = 0;
        for img in self.map.values() {
            if img.gpu_written || base.is_some_and(|b| b != img.key.base) {
                continue;
            }
            if let Some(region) = img.source {
                super::track::force_dirty(region);
                n += 1;
            }
        }
        n
    }

    /// Diagnosis (`texcheck`): the 2D textures uploaded from guest memory
    /// whose guest bytes no longer match their last upload, (mip, layer) by
    /// (mip, layer) against the sums `tune upload_parts` keeps, while no
    /// write is marked over the changed parts: writes the tracking missed
    /// (they stay stale until something else re-uploads them). Write-watch
    /// marks over a changed part are taken first, so CPU writes the
    /// scrubber has not reached yet count as marked. Returns report lines.
    pub fn check_sources() -> Vec<String> {
        let items: Vec<(ImageKey, Tsharp, u32, usize)> = with(|im| {
            im.map
                .values()
                .filter(|i| !i.gpu_written && !i.key.ty3d && !im.refreshing.contains_key(&i.key))
                .filter_map(|i| Some((i.key, i.layout?, i.source?, im.by_base.get(&i.key.base).map_or(0, |v| v.len()))))
                .collect()
        });
        let sums = SUB_SUMS.lock().clone().unwrap_or_default();
        let (mut checked, mut parts_checked, mut no_sums, mut pending_tex, mut missed_tex, mut missed_parts, mut reuploaded) = (0u64, 0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
        let mut lines = Vec::new();
        for (key, t, region, at_base) in items {
            let Some(prev) = sums.get(&key) else {
                no_sums += 1;
                continue;
            };
            let (Some(el), layers) = (elem_log2(t.format), key.layers) else { continue };
            let Some(lay) = tiling::layout(t.sw_mode, el, key.width, key.height, key.levels, is_block(t.format)) else { continue };
            if prev.len() != key.levels as usize * layers as usize {
                continue;
            }
            let elem = 1u64 << el;
            checked += 1;
            // Parts with a sum from a mapped upload, mapped now (the table
            // first, then held under the guest-reader lock, as uploads do).
            let g0 = crate::mem::MAP_GEN.load(std::sync::atomic::Ordering::SeqCst);
            let todo: Vec<(usize, u64, u64)> = (0..prev.len())
                .filter(|&i| prev[i] != SUB_NONE && prev[i] != SUB_UNMAPPED)
                .map(|i| {
                    let (s, n) = part_range(&lay, elem, i / layers as usize, (i % layers as usize) as u32);
                    (i, t.base + s, n)
                })
                .filter(|&(_, a, n)| super::cp::valid(a, n) && crate::mem::host_readable(a, n))
                .collect();
            let readers = crate::mem::GUEST_READERS.read();
            let held = crate::mem::table_view_held(g0);
            let now: Vec<(usize, u64, u64, u64)> = todo.iter().filter(|&&(_, a, n)| held || crate::mem::os_readable(a, n)).map(|&(i, a, n)| (i, a, n, hash_bytes(a, n as usize))).collect();
            drop(readers);
            parts_checked += now.len() as u64;
            let differ: Vec<(usize, u64, u64)> = now.iter().filter(|x| x.3 != prev[x.0]).map(|x| (x.0, x.1, x.2)).collect();
            if differ.is_empty() {
                continue;
            }
            // CPU writes not taken yet: marked on their regions by the take.
            let mut pages = Vec::new();
            for &(_, a, n) in &differ {
                pages.clear();
                let _ = super::precache::take_written_for(a, n, &mut pages, None, 3);
            }
            let missed: Vec<(usize, u64, u64)> = differ.iter().copied().filter(|&(_, a, n)| !super::track::pending_overlaps(region, a, n)).collect();
            if missed.is_empty() {
                pending_tex += 1;
                continue;
            }
            // Uploaded meanwhile (the sums moved): not counted.
            if SUB_SUMS.lock().as_ref().and_then(|m| m.get(&key)).is_none_or(|v| missed.iter().any(|&(i, _, _)| v.get(i) != prev.get(i))) {
                reuploaded += 1;
                continue;
            }
            missed_tex += 1;
            missed_parts += missed.len() as u64;
            if lines.len() < 40 {
                let parts: Vec<String> = missed.iter().take(6).map(|&(i, a, n)| format!("m{} l{} {a:#x}+{n:#x}", i / layers as usize, i % layers as usize)).collect();
                let since = SUB_HASHES.lock().as_ref().and_then(|m| m.get(&key)).map(|x| x.0);
                let events = since.map(|s| missed.iter().take(2).flat_map(|&(_, a, n)| super::track::events_since(a, n, s)).take(6).collect::<Vec<_>>().join("; "));
                lines.push(format!(
                    "texcheck: {:#x} {}x{}x{} {:?} sw {} levels {} (T# levels {}..{}, {} images at base): {} of {} checked parts changed with no write marked ({}{}); region {region} dirty {}{}{}",
                    t.base,
                    key.width,
                    key.height,
                    layers,
                    key.format,
                    t.sw_mode,
                    key.levels,
                    t.base_level,
                    t.last_level,
                    at_base,
                    missed.len(),
                    now.len(),
                    parts.join(", "),
                    if missed.len() > 6 { ", ..." } else { "" },
                    super::track::is_dirty(region),
                    match (since, events) {
                        (Some(s), Some(e)) => format!("; last upload {} ms ago, writes noted since: {}", s.elapsed().as_millis(), if e.is_empty() { "none" } else { &e }),
                        _ => String::new(),
                    },
                    // (`DS_WRITE_JOURNAL=1`: the GPU's last writes over its
                    // first changed part.)
                    missed.first().map(|&(_, a, n)| super::exec::journal_for(a, n)).unwrap_or_default()
                ));
            }
        }
        lines.push(format!(
            "texcheck: {checked} textures checked ({parts_checked} (mip, layer)s, {no_sums} without sums), {missed_tex} stale with no write marked ({missed_parts} parts), {pending_tex} changed with writes marked (uploaded at their next use), {reuploaded} uploaded during the check"
        ));
        lines
    }

    /// Zero-fills every cached image at guest address `base` (diagnosis:
    /// `clearimg` control command); returns how many.
    pub fn clear_at(&mut self, base: u64) -> usize {
        let images: Vec<(vk::Image, vk::Format)> = self.map.values().filter(|i| i.key.base == base).map(|i| (i.image, i.key.format)).collect();
        for &(image, format) in &images {
            self.one_shot(|cmd| unsafe {
                let range = vk::ImageSubresourceRange { aspect_mask: aspects(format), base_mip_level: 0, level_count: vk::REMAINING_MIP_LEVELS, base_array_layer: 0, layer_count: vk::REMAINING_ARRAY_LAYERS };
                let d = &ctx().device;
                let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
                d.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
                if is_depth(format) {
                    d.cmd_clear_depth_stencil_image(cmd, image, vk::ImageLayout::GENERAL, &vk::ClearDepthStencilValue { depth: 0.0, stencil: 0 }, &[range]);
                } else if !is_compressed(format) {
                    d.cmd_clear_color_image(cmd, image, vk::ImageLayout::GENERAL, &vk::ClearColorValue::default(), &[range]);
                }
                let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
                d.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[mb], &[], &[]);
            });
        }
        images.len()
    }

    /// Moves a new image to GENERAL and zero-fills it (depth: 0, stencil: 0).
    fn to_general(&mut self, image: vk::Image, format: vk::Format) {
        self.one_shot(|cmd| unsafe {
            let range = vk::ImageSubresourceRange { aspect_mask: aspects(format), base_mip_level: 0, level_count: vk::REMAINING_MIP_LEVELS, base_array_layer: 0, layer_count: vk::REMAINING_ARRAY_LAYERS };
            let b = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .image(image)
                .subresource_range(range)
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
            let d = &ctx().device;
            d.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[b]);
            if is_depth(format) {
                d.cmd_clear_depth_stencil_image(cmd, image, vk::ImageLayout::GENERAL, &vk::ClearDepthStencilValue { depth: 0.0, stencil: 0 }, &[range]);
            } else if !is_compressed(format) {
                d.cmd_clear_color_image(cmd, image, vk::ImageLayout::GENERAL, &vk::ClearColorValue::default(), &[range]);
            }
            let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
            d.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[mb], &[], &[]);
        });
    }

    /// Uploads all mips/layers of a guest texture (detiled) into `image`.
    /// A video-memory buffer of at least `size` bytes for GPU detiling: the
    /// smallest free one large enough, else a new one (sizes rounded up to
    /// powers of two up to DETILE_KEEP; larger ones exact and not kept).
    fn detile_take(&mut self, size: u64) -> Result<(vk::Buffer, vk::DeviceMemory, u64), String> {
        if let Some(i) = self.detile_free.iter().enumerate().filter(|(_, b)| b.2 >= size).min_by_key(|(_, b)| b.2).map(|(i, _)| i) {
            return Ok(self.detile_free.swap_remove(i));
        }
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
        let cap = if size > DETILE_KEEP { size } else { size.next_power_of_two().clamp(1 << 20, DETILE_KEEP) };
        let t0 = std::time::Instant::now();
        let (b, m) = ctx().device_buffer(cap, usage)?;
        super::hitch::note_since(super::hitch::Kind::Upload, t0, || format!("a new GPU detiling buffer of {} MiB", cap >> 20));
        GPU_DETILE[9].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok((b, m, cap))
    }

    /// GPU detiling buffers the GPU is done with: kept for reuse (up to
    /// DETILE_POOL in all) or freed.
    fn release_detile(&mut self, v: Vec<(vk::Buffer, vk::DeviceMemory, u64)>) {
        for b in v {
            let kept: u64 = self.detile_free.iter().map(|b| b.2).sum();
            if b.2 <= DETILE_KEEP && kept + b.2 <= DETILE_POOL {
                self.detile_free.push(b);
                continue;
            }
            unsafe {
                ctx().device.destroy_buffer(b.0, None);
                ctx().device.free_memory(b.1, None);
            }
        }
    }

    /// The wanted, mapped (mip, layer)s `parts` of a 2D (array) texture
    /// detiled on the GPU (`tune gpu_detile`): their tiled bytes copied from
    /// the guest-memory import windows into video memory, detiled there into
    /// packed rows (`rdna::kernels::detile`, the CPU detiler's addresses)
    /// and copied into `image` with `part_regions` (one per part, in order;
    /// their buffer offsets are replaced), the unmapped parts of a first
    /// upload (`zeroed`) from a zero-filled area, the bytes read through the
    /// import `windows` (start, end, buffer) made before the guest-reader
    /// lock, which the caller holds (nothing here may take the mapping
    /// table's lock). Submitted without waiting (`submit_held`; the GPU
    /// reads guest memory later, through imports that outlive a remap until
    /// settled, as front end 2's uploads do), or waited for with
    /// `DS_GPU_DETILE_CHECK=1`. Err: nothing was recorded (the caller
    /// detiles on the CPU). Returns the bytes detiled.
    #[allow(clippy::too_many_arguments)]
    fn upload_gpu(&mut self, image: vk::Image, first: bool, lay: &tiling::Layout, t: &Tsharp, block: bool, parts: &[(usize, u32, usize, usize)], zeroed: &[(usize, u32, usize)], part_regions: &[vk::BufferImageCopy], range: &dyn Fn(usize, u32) -> (u64, u64), windows: &[(u64, u64, vk::Buffer)]) -> Result<u64, String> {
        use std::sync::atomic::Ordering::Relaxed;
        use rdna::kernels::{DETILE_LOCAL_SIZE, DETILE_PC_WORDS};
        let t0 = std::time::Instant::now();
        let pipe = detile_pipe()?;
        let pat = tiling::pattern(lay.sw, lay.elem_log2).ok_or("no swizzle pattern")?;
        let mut masks = [0u32; 16];
        for (i, m) in masks.iter_mut().enumerate().take(lay.block_log2 as usize) {
            *m = (pat[i] & 0xFFFF) as u32 | (((pat[i] >> 16) & 0xFFFF) as u32) << 16;
        }
        // The parts' tiled bytes, packed (a layer's mip-tail mips share its
        // tail block).
        let mut tiled: Vec<(u64, u64, u64)> = Vec::new();
        let mut t_of: HashMap<u64, u64> = HashMap::default();
        let mut t_total = 0u64;
        let mut src_at = Vec::with_capacity(parts.len());
        for &(mip, layer, _, _) in parts {
            let (s, n) = range(mip, layer);
            let at = match t_of.get(&s) {
                Some(&at) => at,
                None => {
                    let at = t_total;
                    tiled.push((t.base + s, n, at));
                    t_of.insert(s, at);
                    t_total = (t_total + n).next_multiple_of(256);
                    at
                }
            };
            src_at.push(at);
        }
        if t_total > u32::MAX as u64 {
            return Err(format!("{t_total} tiled bytes"));
        }
        // Their copies from the import windows, split at window boundaries.
        let mut copies: Vec<(vk::Buffer, vk::BufferCopy)> = Vec::new();
        for &(g, n, at) in &tiled {
            let end = g + n;
            let mut x = g;
            while x < end {
                let &(start, wend, buf) = windows.iter().find(|&&(ws, we, _)| ws <= x && x < we).ok_or_else(|| format!("no import window over {x:#x}"))?;
                let e = end.min(wend);
                copies.push((buf, vk::BufferCopy { src_offset: x - start, dst_offset: at + (x - g), size: e - x }));
                x = e;
            }
        }
        // Packed rows at 16-byte offsets (whole dwords per part whatever the
        // element size), then the zero area.
        let mut l_at = Vec::with_capacity(parts.len());
        let mut l_total = 0u64;
        for &(_, _, _, len) in parts {
            l_at.push(l_total);
            l_total = (l_total + len as u64).next_multiple_of(16);
        }
        let zero_len = zeroed.iter().map(|z| z.2 as u64).max().unwrap_or(0).next_multiple_of(16);
        let zero_at = l_total;
        l_total += zero_len;
        let mut regions: Vec<vk::BufferImageCopy> = part_regions.iter().zip(&l_at).map(|(r, &o)| r.buffer_offset(o)).collect();
        for &(mip, layer, _) in zeroed {
            let m = &lay.mips[mip];
            regions.push(
                vk::BufferImageCopy::default()
                    .buffer_offset(zero_at)
                    .buffer_row_length(if block { m.width * 4 } else { m.width })
                    .buffer_image_height(if block { m.height * 4 } else { m.height })
                    .image_subresource(vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: mip as u32, base_array_layer: layer, layer_count: 1 })
                    .image_extent(vk::Extent3D { width: m.texel_w, height: m.texel_h, depth: 1 }),
            );
        }
        // Each part's dispatch: its push constants and grid.
        let (lw, lh) = (lay.block_w.trailing_zeros(), lay.block_h.trailing_zeros());
        let mut dispatches: Vec<(Vec<u8>, u32, u32)> = Vec::with_capacity(parts.len());
        let mut detiled = 0u64;
        for (i, &(mip, _, _, len)) in parts.iter().enumerate() {
            let m = &lay.mips[mip];
            let (tx, ty) = m.tail.unwrap_or((0, 0));
            let count = (len as u64).div_ceil(4);
            let groups = count.div_ceil(DETILE_LOCAL_SIZE as u64).max(1);
            let gx = groups.min(65535);
            let gy = groups.div_ceil(gx);
            let mut pc = [0u32; DETILE_PC_WORDS];
            pc[0] = src_at[i] as u32;
            pc[1] = (l_at[i] / 4) as u32;
            pc[2] = m.width;
            pc[3] = m.height;
            pc[4] = tx;
            pc[5] = ty;
            pc[6] = (m.pitch / lay.block_w).max(1);
            pc[7] = lay.block_log2 | lw << 8 | lh << 16 | lay.elem_log2 << 24;
            pc[8] = count as u32;
            pc[9] = (gx * DETILE_LOCAL_SIZE as u64) as u32;
            pc[10] = t_total as u32;
            pc[12..28].copy_from_slice(&masks);
            dispatches.push((pc.iter().flat_map(|w| w.to_le_bytes()).collect(), gx as u32, gy as u32));
            detiled += len as u64;
        }
        let c = ctx();
        let tbuf = self.detile_take(t_total.max(256))?;
        let lbuf = match self.detile_take(l_total.max(16)) {
            Ok(b) => b,
            Err(e) => {
                self.release_detile(vec![tbuf]);
                return Err(e);
            }
        };
        let (tb, lb) = (tbuf.0, lbuf.0);
        // (Rows, then the tiled bytes.)
        let readback = if gpu_detile_check() {
            match c.host_buffer(l_total.max(16) + t_total.max(256), vk::BufferUsageFlags::TRANSFER_DST) {
                Ok(r) => Some(r),
                Err(e) => {
                    self.release_detile(vec![tbuf, lbuf]);
                    return Err(e);
                }
            }
        } else {
            None
        };
        let rb = readback.as_ref().map(|r| r.0);
        let rows_bytes = l_total.max(16);
        let base = t.base;
        // The transfer from guest memory on the copy queue (beside the
        // frames' work on the main queue), the rest on the main queue once
        // it is done (checked runs wait for that part); all on the main queue
        // without a copy queue.
        let on_copy = self.copy_parts();
        let record_copies = |cmd: vk::CommandBuffer| unsafe {
            let mut k = 0;
            while k < copies.len() {
                let src = copies[k].0;
                let mut regs = Vec::new();
                while k < copies.len() && copies[k].0 == src {
                    regs.push(copies[k].1);
                    k += 1;
                }
                c.device.cmd_copy_buffer(cmd, src, tb, &regs);
            }
        };
        let record = |cmd: vk::CommandBuffer| unsafe {
            super::vk::checkpoint(cmd, super::vk::mark::UPLOAD, base);
            let b = vk::ImageMemoryBarrier::default()
                .old_layout(if first { vk::ImageLayout::UNDEFINED } else { vk::ImageLayout::GENERAL })
                .new_layout(vk::ImageLayout::GENERAL)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: vk::REMAINING_MIP_LEVELS, base_array_layer: 0, layer_count: vk::REMAINING_ARRAY_LAYERS })
                .src_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[b]);
            if on_copy.is_none() {
                record_copies(cmd);
            }
            if zero_len > 0 {
                c.device.cmd_fill_buffer(cmd, lb, zero_at, zero_len, 0);
            }
            let m = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[m], &[], &[]);
            c.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipe.pipeline);
            let ti = [vk::DescriptorBufferInfo { buffer: tb, offset: 0, range: vk::WHOLE_SIZE }];
            let li = [vk::DescriptorBufferInfo { buffer: lb, offset: 0, range: vk::WHOLE_SIZE }];
            let w = [
                vk::WriteDescriptorSet::default().dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&ti),
                vk::WriteDescriptorSet::default().dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&li),
            ];
            c.push_descriptors(cmd, vk::PipelineBindPoint::COMPUTE, pipe.layout, 0, &w);
            for (pc, gx, gy) in &dispatches {
                c.device.cmd_push_constants(cmd, pipe.layout, vk::ShaderStageFlags::COMPUTE, 0, pc);
                c.device.cmd_dispatch(cmd, *gx, *gy, 1);
            }
            let m = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[m], &[], &[]);
            if let Some(rb) = rb {
                c.device.cmd_copy_buffer(cmd, lb, rb, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: l_total.max(16) }]);
                c.device.cmd_copy_buffer(cmd, tb, rb, &[vk::BufferCopy { src_offset: 0, dst_offset: rows_bytes, size: t_total.max(256) }]);
                let h = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ);
                c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[h], &[], &[]);
            }
            c.device.cmd_copy_buffer_to_image(cmd, lb, image, vk::ImageLayout::GENERAL, &regions);
            let a = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[a], &[], &[]);
        };
        // Submitted without waiting (the one queue runs it after the work
        // submitted before and before batches submitted later; its barriers
        // order it against the image's other uses), unless checked.
        if readback.is_some() && on_copy.is_none() {
            self.one_shot(record);
            self.release_detile(vec![tbuf, lbuf]);
        } else if let Some((pool, sem)) = on_copy {
            match self.submit_copy_part(pool, sem, record_copies) {
                Some((x, v)) => {
                    GPU_DETILE[10].fetch_add(1, Relaxed);
                    let held = Held { dev: vec![tbuf, lbuf], copy_cmd: Some(x) };
                    if readback.is_some() {
                        self.one_shot_after(Some((sem, v)), record);
                        self.release_held(held);
                    } else {
                        self.submit_held(held, DETILE_INFLIGHT, Some((sem, v)), record);
                    }
                }
                None => {
                    self.release_detile(vec![tbuf, lbuf]);
                    return Err("the copy queue refused the transfer".into());
                }
            }
        } else {
            self.submit_held(Held { dev: vec![tbuf, lbuf], copy_cmd: None }, DETILE_INFLIGHT, None, record);
        }
        // DS_GPU_DETILE_CHECK=1: each part against the CPU's detiling of the
        // tiled bytes the GPU copied (laid out at their texture offsets).
        if let Some((rbuf, rmem, rptr)) = readback {
            let got = unsafe { std::slice::from_raw_parts(rptr, l_total as usize) };
            let copied = unsafe { std::slice::from_raw_parts(rptr.add(rows_bytes as usize), t_total as usize) };
            let end = tiled.iter().map(|&(g, n, _)| g + n - t.base).max().unwrap_or(0);
            let mut src = vec![0u8; end as usize];
            let mut changed = 0u64;
            for &(g, n, at) in &tiled {
                let (s, n) = ((g - t.base) as usize, n as usize);
                let mine = &copied[at as usize..at as usize + n];
                src[s..s + n].copy_from_slice(mine);
                if unsafe { std::slice::from_raw_parts(g as *const u8, n) } != mine {
                    changed += 1;
                }
            }
            GPU_DETILE[8].fetch_add(changed, Relaxed);
            let mut differ = 0u64;
            for (i, &(mip, layer, _, len)) in parts.iter().enumerate() {
                let mut want = vec![0u8; len];
                lay.detile_into(&src, mip, layer, &mut want);
                let have = &got[l_at[i] as usize..l_at[i] as usize + len];
                if want != have {
                    differ += 1;
                    if GPU_DETILE[7].load(Relaxed) + differ <= 8 {
                        let at = want.iter().zip(have).position(|(a, b)| a != b).unwrap_or(0);
                        let m = &lay.mips[mip];
                        crate::warn!(
                            "images: GPU detiling of {:#x} {:?} mip {mip} layer {layer} ({}x{} elements of {} bytes, swizzle {}, tail {:?}) differs from the CPU's at byte {at} of {len}",
                            t.base,
                            t.format,
                            m.width,
                            m.height,
                            1u32 << lay.elem_log2,
                            lay.sw,
                            m.tail
                        );
                    }
                }
            }
            GPU_DETILE[6].fetch_add(parts.len() as u64, Relaxed);
            GPU_DETILE[7].fetch_add(differ, Relaxed);
            unsafe {
                c.device.destroy_buffer(rbuf, None);
                c.device.unmap_memory(rmem);
                c.device.free_memory(rmem, None);
            }
        }
        GPU_DETILE[0].fetch_add(1, Relaxed);
        GPU_DETILE[1].fetch_add(parts.len() as u64, Relaxed);
        GPU_DETILE[2].fetch_add(tiled.iter().map(|x| x.1).sum::<u64>(), Relaxed);
        GPU_DETILE[3].fetch_add(detiled, Relaxed);
        GPU_DETILE[5].fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
        Ok(detiled)
    }

    fn upload(&mut self, image: vk::Image, key: ImageKey, t: &Tsharp) -> Result<(u64, u64), String> {
        self.upload_parts(image, key, t, None, true)
    }

    /// Uploads the mapped (mip, layer)s of a guest texture: all of them with
    /// `all` or without written ranges `only` (and on a first upload, and
    /// for 3D textures); otherwise, with `tune upload_parts`, those that
    /// `only` or the CPU's writes since the last upload (write-watch marks)
    /// overlap.
    fn upload_parts(&mut self, image: vk::Image, key: ImageKey, t: &Tsharp, only: Option<&[(u64, u64)]>, all: bool) -> Result<(u64, u64), String> {
        let t0 = std::time::Instant::now();
        let r = self.upload_inner(image, key, t, only, all);
        if zero_census() {
            let cause = REUPLOAD_CAUSE.with(|c| c.get()) as usize;
            ZERO_CENSUS[cause.min(6)][4].fetch_add(t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        }
        if let Ok((_, n)) = &r {
            self.upload_bytes += n;
            super::hitch::note_since(super::hitch::Kind::Upload, t0, || format!("{}x{}x{}{} {:?} {} KiB", key.width, key.height, key.layers, if key.ty3d { " 3D" } else { "" }, key.format, n >> 10));
        }
        self.upload_ns += t0.elapsed().as_nanos() as u64;
        r
    }

    fn upload_inner(&mut self, image: vk::Image, key: ImageKey, t: &Tsharp, only: Option<&[(u64, u64)]>, all: bool) -> Result<(u64, u64), String> {
        let first = !self.map.contains_key(&key);
        let region = self.map.get(&key).and_then(|i| i.source);
        if key.ty3d {
            return self.upload_3d(image, key, t, first);
        }
        let mut notes = Vec::new();
        let keep = self.map.get(&key).map(|i| i.gpu_parts.clone()).unwrap_or_default();
        let chose = choose_2d(key, t, only, all, first, region, true, &keep, &mut notes);
        for n in notes {
            self.report(n);
        }
        let Chosen { lay, layers, elem, block, size, precise, held, gpu, windows, readers, sums_on, mut sums, end, mut regions, zeroed, parts, total, zero_len, zero_off, detiled, skipped, parts_done, parts_left, t_detile, t_table, t_locked, t_chosen, take_ms, t_start, c4_ms } = match chose? {
            Chose::Done(r) => return Ok(r),
            Chose::Parts(c) => c,
        };
        let range = |mip: usize, layer: u32| part_range(&lay, elem, mip, layer);
        let src = unsafe { std::slice::from_raw_parts(t.base as *const u8, end as usize) };
        // (Only while no mapping changed since the windows were made.)
        if gpu && held {
            let t_gpu = std::time::Instant::now();
            match self.upload_gpu(image, first, &lay, t, block, &parts, &zeroed, &regions, &range, &windows) {
                Ok(bytes) => {
                    let t_recorded = std::time::Instant::now();
                    // A first upload's sums, as the CPU path makes them (a
                    // part rewritten with the same bytes is left out later).
                    if sums_on && first {
                        let todo: Vec<(usize, u64, u64)> = parts
                            .iter()
                            .map(|&(mip, layer, _, _)| {
                                let (s, n) = range(mip, layer);
                                (mip * layers as usize + layer as usize, t.base + s, n)
                            })
                            .collect();
                        let threads = if bytes >= 1 << 20 { super::pool::threads() } else { 1 };
                        for (i, h) in hash_parts(&todo, threads) {
                            sums[i] = h;
                        }
                    }
                    drop(readers);
                    // GPU-detiled uploads of 50 ms or more: where their time
                    // went (diagnosis, as for the CPU path's).
                    let ms = |a: std::time::Instant, b: std::time::Instant| (b - a).as_secs_f64() * 1e3;
                    let now = std::time::Instant::now();
                    if ms(t_start, now) >= 30.0 {
                        crate::info!(
                            "images: GPU-detiled upload of {:#x} {}x{}x{} {:?} ({} parts, {} KiB{}) took {:.0} ms: held pages' release {c4_ms:.0}, write marks {take_ms:.0}, the rest before the choice {:.0}, choosing {:.0} (table and import windows {:.0}, reader lock {:.0}, the rest: queries, hashes), recording and submitting {:.0}, the first upload's hashes {:.0}",
                            t.base,
                            key.width,
                            key.height,
                            layers,
                            t.format,
                            parts.len(),
                            bytes >> 10,
                            if first { ", first" } else { "" },
                            ms(t_start, now),
                            (ms(t_start, t_detile) - c4_ms - take_ms).max(0.0),
                            ms(t_detile, t_chosen),
                            ms(t_detile, t_table),
                            ms(t_table, t_locked),
                            ms(t_gpu, t_recorded),
                            ms(t_recorded, now)
                        );
                    }
                    if sums_on {
                        SUB_SUMS.lock().get_or_insert_with(HashMap::default).insert(key, sums);
                    }
                    UPLOAD_SPLIT[0].fetch_add(t_detile.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
                    DETILED.fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
                    if precise {
                        use std::sync::atomic::Ordering::Relaxed;
                        PART_UPLOADS[0].fetch_add(1, Relaxed);
                        PART_UPLOADS[1].fetch_add(parts_done, Relaxed);
                        PART_UPLOADS[2].fetch_add(parts_left, Relaxed);
                        PART_UPLOADS[3].fetch_add(bytes, Relaxed);
                    }
                    if skipped > 0 {
                        self.report(partial_residency_note(key, skipped, first));
                    }
                    self.uploads += 1;
                    return Ok((t.base, size));
                }
                Err(e) => {
                    if GPU_DETILE[4].fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8 {
                        crate::warn!("images: texture {:#x} {}x{}x{} {:?} detiled on the CPU: {e}", t.base, key.width, key.height, layers, t.format);
                    }
                }
            }
        }
        // A staging buffer of its own, freed after the upload: one kept
        // across uploads made the driver refuse to import guest memory
        // windows thousands of times a session (2026-10-24: 3,000-75,000
        // failed imports against 48, then unmapped textures and device losses
        // at cold boots; the pooled buffers of the removed `tune
        // upload_async` likewise).
        let st = staging_take(total as u64)?;
        let t_alloc = std::time::Instant::now();
        let dst = unsafe { std::slice::from_raw_parts_mut(st.ptr, total) };
        detile_parts(&lay, src, &parts, dst);
        let t_detiled = std::time::Instant::now();
        dst[zero_off..].fill(0);
        UPLOAD_PHASES[0].fetch_add((t_alloc - t_chosen).as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        UPLOAD_PHASES[1].fetch_add(t_alloc.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        // A first upload's sums (for later partial re-uploads), hashed here
        // on several threads rather than one by one before the detiling.
        if sums_on && first {
            let todo: Vec<(usize, u64, u64)> = parts
                .iter()
                .map(|&(mip, layer, _, _)| {
                    let (s, n) = range(mip, layer);
                    (mip * layers as usize + layer as usize, t.base + s, n)
                })
                .collect();
            let threads = if detiled >= 1 << 20 { super::pool::threads() } else { 1 };
            let t_hash = std::time::Instant::now();
            for (i, h) in hash_parts(&todo, threads) {
                sums[i] = h;
            }
            UPLOAD_PHASES[2].fetch_add(t_hash.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        }
        // Guest memory is no longer read.
        drop(readers);
        UPLOAD_SPLIT[0].fetch_add(t_detile.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        DETILED.fetch_add(detiled as u64, std::sync::atomic::Ordering::Relaxed);
        if sums_on {
            SUB_SUMS.lock().get_or_insert_with(HashMap::default).insert(key, sums);
        }
        if precise {
            use std::sync::atomic::Ordering::Relaxed;
            PART_UPLOADS[0].fetch_add(1, Relaxed);
            PART_UPLOADS[1].fetch_add(parts_done, Relaxed);
            PART_UPLOADS[2].fetch_add(parts_left, Relaxed);
            PART_UPLOADS[3].fetch_add(detiled as u64, Relaxed);
        }
        if zero_len > 0 {
            push_zero_regions(&lay, block, zero_off as u64, &zeroed, &mut regions);
        }
        if skipped > 0 {
            self.report(partial_residency_note(key, skipped, first));
        }
        let t_stage = std::time::Instant::now();
        let (buf, mem) = (st.buf, st.mem);
        let base = t.base;
        let t_copy = std::time::Instant::now();
        UPLOAD_SPLIT[1].fetch_add((t_copy - t_stage).as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        let record = |cmd: vk::CommandBuffer| record_upload(cmd, image, first, buf, &regions, base);
        // (`tune upload_batch`: into the command buffer the uploads share,
        // not waited for; its own staging freed once that completes.)
        if super::tune::upload_batch() {
            match record_batched((st.buf, st.mem), st.cap, &record) {
                Ok(()) => {
                    UPLOAD_SPLIT[2].fetch_add(t_copy.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
                    // Batched uploads of 50 ms or more: where their time went (diagnosis).
                    let ms = |a: std::time::Instant, b: std::time::Instant| (b - a).as_secs_f64() * 1e3;
                    let now = std::time::Instant::now();
                    if ms(t_start, now) >= 50.0 {
                        crate::info!(
                            "images: batched upload of {:#x} {}x{}x{} {:?} ({} parts, {} KiB{}, the GPU path {}) took {:.0} ms: held pages' release {c4_ms:.0}, write marks {take_ms:.0}, choosing {:.0} (table {:.0}, reader lock {:.0}, the rest: queries, hashes), staging buffer {:.0}, detiling {:.0}, the rest {:.0}",
                            t.base,
                            key.width,
                            key.height,
                            layers,
                            t.format,
                            parts.len(),
                            detiled >> 10,
                            if first { ", first" } else { "" },
                            if !gpu { "not wanted" } else if !held { "left: a mapping over it changed" } else { "failed" },
                            ms(t_start, now),
                            ms(t_detile, t_chosen),
                            ms(t_detile, t_table),
                            ms(t_table, t_locked),
                            ms(t_chosen, t_alloc),
                            ms(t_alloc, t_detiled),
                            ms(t_detiled, now)
                        );
                    }
                    self.uploads += 1;
                    return Ok((t.base, size));
                }
                Err(e) => {
                    static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                    if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                        crate::warn!("images: upload not batched: {e}");
                    }
                }
            }
        }
        self.one_shot(record);
        let t_free = std::time::Instant::now();
        UPLOAD_SPLIT[2].fetch_add((t_free - t_copy).as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        // The GPU is done with it (one_shot waited).
        staging_free(buf, mem);
        UPLOAD_SPLIT[1].fetch_add(t_free.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        // Uploads of 50 ms or more: where their time went (diagnosis).
        let ms = |a: std::time::Instant, b: std::time::Instant| (b - a).as_secs_f64() * 1e3;
        let now = std::time::Instant::now();
        if ms(t_start, now) >= 50.0 {
            crate::info!(
                "images: upload of {:#x} {}x{}x{} {:?} ({} parts, {} KiB) took {:.0} ms: held pages' release {c4_ms:.0}, write marks {take_ms:.0}, choosing {:.0} (table {:.0}, reader lock {:.0}, the rest: queries, hashes), staging buffer {:.0}, detiling {:.0}, transfer {:.0}, freeing {:.0}",
                t.base,
                key.width,
                key.height,
                layers,
                t.format,
                parts.len(),
                detiled >> 10,
                ms(t_start, now),
                ms(t_detile, t_chosen),
                ms(t_detile, t_table),
                ms(t_table, t_locked),
                ms(t_chosen, t_alloc),
                ms(t_alloc, t_detiled),
                ms(t_copy, t_free),
                ms(t_free, now)
            );
        }
        self.uploads += 1;
        Ok((t.base, size))
    }

    /// Uploads a 3D texture (one mip, thick standard swizzle).
    fn upload_3d(&mut self, image: vk::Image, key: ImageKey, t: &Tsharp, first: bool) -> Result<(u64, u64), String> {
        if !tiling::is_thick_3d(t.sw_mode) || key.levels > 1 || is_block(t.format) {
            return Err(format!("3D texture layout not implemented (swizzle mode {}, {} mips)", t.sw_mode, key.levels));
        }
        let el = elem_log2(t.format).ok_or("unknown element size")?;
        let lay = tiling::layout_3d(t.sw_mode, el, key.width, key.height, key.layers).ok_or_else(|| format!("3D swizzle mode {} not supported", t.sw_mode))?;
        let size = lay.size();
        // As for 2D textures: the mapping table before the lock, the
        // operating system after it only when a mapping changed meanwhile.
        let g0 = crate::mem::MAP_GEN.load(std::sync::atomic::Ordering::SeqCst);
        // GPU detiling for volumes of at least `tune gpu_detile` MiB whose swizzle it can do: the
        // import windows over the volume made now (before the guest-reader lock, as for 2D).
        let want_gpu = match super::tune::gpu_detile() {
            0 => false,
            n => size >= n << 20,
        };
        let masks = if want_gpu { lay.gpu_masks() } else { None };
        if want_gpu && masks.is_none() && GPU_DETILE[4].fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8 {
            crate::warn!("images: 3D texture {:#x} {}x{}x{} {:?} detiled on the CPU: its swizzle mode {} takes coordinate bits above its blocks", t.base, key.width, key.height, key.layers, t.format, t.sw_mode);
        }
        let mut windows: Vec<(u64, u64, vk::Buffer)> = Vec::new();
        if masks.is_some() {
            let (mut x, end) = (t.base, t.base + size);
            while x < end {
                let e = ((x & !(super::exec::window_bytes() - 1)) + super::exec::window_bytes()).min(end);
                match super::exec::import_window(x, e - x) {
                    Some((b, ws, we)) if ws <= x && e <= we => {
                        windows.push((ws, we, b));
                        x = we.min(end);
                    }
                    _ => {
                        windows.clear();
                        break;
                    }
                }
            }
        }
        let table = super::cp::valid(t.base, size) && crate::mem::host_readable(t.base, size);
        let readers = crate::mem::GUEST_READERS.read();
        let held_all = crate::mem::table_view_held(g0);
        let held = held_all || crate::mem::table_view_held_for(g0, t.base, t.base + size);
        UPLOAD_TABLE[usize::from(!held)].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if held && !held_all {
            UPLOAD_TABLE[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        let ok = if held { table } else { super::cp::valid(t.base, size) && crate::mem::os_readable(t.base, size) };
        if !ok {
            return Err(format!("3D texture {:#x}+{size:#x} not mapped", t.base));
        }
        // (Only while no mapping changed since the windows were made.)
        if let (Some(masks), true, false) = (masks, held, windows.is_empty()) {
            match self.upload_gpu_3d(image, first, &lay, t, &masks, &windows) {
                Ok(_) => {
                    drop(readers);
                    self.uploads += 1;
                    return Ok((t.base, size));
                }
                Err(e) => {
                    if GPU_DETILE[4].fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8 {
                        crate::warn!("images: 3D texture {:#x} detiled on the CPU: {e}", t.base);
                    }
                }
            }
        } else if masks.is_some() {
            GPU_DETILE[4].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        let src = unsafe { std::slice::from_raw_parts(t.base as *const u8, size as usize) };
        // Detiled straight into the staging buffer (no intermediate copy).
        let c = ctx();
        let len = lay.width as usize * lay.height as usize * lay.depth as usize * (1usize << el);
        let (buf, mem, ptr) = c.host_buffer(len.max(4) as u64, vk::BufferUsageFlags::TRANSFER_SRC)?;
        lay.detile_into(src, unsafe { std::slice::from_raw_parts_mut(ptr, len) });
        drop(readers);
        let region = vk::BufferImageCopy::default()
            .image_subresource(vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 })
            .image_extent(vk::Extent3D { width: key.width, height: key.height, depth: key.layers });
        let base = t.base;
        self.one_shot(|cmd| unsafe {
            super::vk::checkpoint(cmd, super::vk::mark::UPLOAD, base);
            let b = vk::ImageMemoryBarrier::default()
                .old_layout(if first { vk::ImageLayout::UNDEFINED } else { vk::ImageLayout::GENERAL })
                .new_layout(vk::ImageLayout::GENERAL)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: vk::REMAINING_MIP_LEVELS, base_array_layer: 0, layer_count: vk::REMAINING_ARRAY_LAYERS })
                .src_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[b]);
            c.device.cmd_copy_buffer_to_image(cmd, buf, image, vk::ImageLayout::GENERAL, &[region]);
            let a = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[a], &[], &[]);
        });
        unsafe {
            c.device.destroy_buffer(buf, None);
            c.device.unmap_memory(mem);
            c.device.free_memory(mem, None);
        }
        self.uploads += 1;
        Ok((t.base, size))
    }

    /// A thick 3D texture detiled on the GPU (`upload_3d`, a volume of at least `tune gpu_detile`
    /// MiB whose swizzle `Layout3d::gpu_masks` gives), in chunks of whole slabs of at most
    /// `CHUNK_3D` tiled bytes (the detiling buffers' pool keeps them): each chunk's tiled bytes
    /// copied from the import windows into video memory, a `detile` dispatch a slice (the slab's
    /// offset in its source offset, the slice's depth term XORed into its elements' in-block
    /// offsets), the packed slices copied into the image; submitted without waiting (`Held`).
    /// Until 2026-11-21 every volume was detiled on the CPU on the recording thread into a host
    /// buffer made for it, then copied and waited for: 2-19 ms each where hitch notes named them
    /// (the 1-1 walk's 4-38 MiB volumes, 128x128x128 RGBA8 the most often). Err: the CPU path
    /// uploads the volume whole (after any chunks submitted before the error).
    fn upload_gpu_3d(&mut self, image: vk::Image, first: bool, lay: &tiling::Layout3d, t: &Tsharp, masks: &[u32; 16], windows: &[(u64, u64, vk::Buffer)]) -> Result<u64, String> {
        use rdna::kernels::{DETILE_LOCAL_SIZE, DETILE_PC_WORDS};
        use std::sync::atomic::Ordering::Relaxed;
        const CHUNK_3D: u64 = 64 << 20;
        let t0 = std::time::Instant::now();
        let pipe = detile_pipe()?;
        let size = lay.size();
        let slab = lay.slab_size();
        if slab == 0 || size > u32::MAX as u64 {
            return Err(format!("{size} tiled bytes"));
        }
        let per = (CHUNK_3D / slab).max(1) as u32;
        let slabs = lay.padded_depth / lay.block_d;
        let e = 1u64 << lay.elem_log2;
        let slice = lay.width as u64 * lay.height as u64 * e;
        let slice_at = slice.next_multiple_of(16);
        let (lw, lh) = (lay.block_w.trailing_zeros(), lay.block_h.trailing_zeros());
        let count = slice.div_ceil(4);
        let groups = count.div_ceil(DETILE_LOCAL_SIZE as u64).max(1);
        let gx = groups.min(65535);
        let gy = groups.div_ceil(gx);
        let c = ctx();
        let check = gpu_detile_check();
        let mut k0 = 0u32;
        let mut first = first;
        while k0 < slabs {
            let k1 = (k0 + per).min(slabs);
            let (z0, z1) = (k0 * lay.block_d, (k1 * lay.block_d).min(lay.depth));
            if z0 >= z1 {
                break;
            }
            // The chunk's tiled bytes and their copies from the windows.
            let (lo, hi) = (k0 as u64 * slab, (k1 as u64 * slab).min(size));
            let t_total = (hi - lo).next_multiple_of(4);
            let mut copies: Vec<(vk::Buffer, vk::BufferCopy)> = Vec::new();
            let (mut x, end) = (t.base + lo, t.base + hi);
            while x < end {
                let &(start, wend, buf) = windows.iter().find(|&&(ws, we, _)| ws <= x && x < we).ok_or_else(|| format!("no import window over {x:#x}"))?;
                let e = end.min(wend);
                copies.push((buf, vk::BufferCopy { src_offset: x - start, dst_offset: x - t.base - lo, size: e - x }));
                x = e;
            }
            let n = (z1 - z0) as u64;
            let l_total = slice_at * n;
            let mut dispatches: Vec<Vec<u8>> = Vec::with_capacity(n as usize);
            for z in z0..z1 {
                let mut pc = [0u32; DETILE_PC_WORDS];
                pc[0] = (lay.slab_offset(z) - lo) as u32;
                pc[1] = ((z - z0) as u64 * slice_at / 4) as u32;
                pc[2] = lay.width;
                pc[3] = lay.height;
                pc[6] = lay.pitch_blocks();
                pc[7] = lay.block_log2 | lw << 8 | lh << 16 | lay.elem_log2 << 24;
                pc[8] = count as u32;
                pc[9] = (gx * DETILE_LOCAL_SIZE as u64) as u32;
                pc[10] = t_total as u32;
                pc[11] = lay.slice_xor(z);
                pc[12..28].copy_from_slice(masks);
                dispatches.push(pc.iter().flat_map(|w| w.to_le_bytes()).collect());
            }
            let sub = vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 };
            let regions: Vec<vk::BufferImageCopy> = if slice_at == slice {
                vec![vk::BufferImageCopy::default().image_subresource(sub).image_offset(vk::Offset3D { x: 0, y: 0, z: z0 as i32 }).image_extent(vk::Extent3D { width: lay.width, height: lay.height, depth: z1 - z0 })]
            } else {
                (z0..z1)
                    .map(|z| vk::BufferImageCopy::default().buffer_offset((z - z0) as u64 * slice_at).image_subresource(sub).image_offset(vk::Offset3D { x: 0, y: 0, z: z as i32 }).image_extent(vk::Extent3D { width: lay.width, height: lay.height, depth: 1 }))
                    .collect()
            };
            let tbuf = self.detile_take(t_total.max(256))?;
            let lbuf = match self.detile_take(l_total.max(16)) {
                Ok(b) => b,
                Err(e) => {
                    self.release_detile(vec![tbuf]);
                    return Err(e);
                }
            };
            let (tb, lb) = (tbuf.0, lbuf.0);
            // (`DS_GPU_DETILE_CHECK=1`: the rows, then the tiled bytes, read back.)
            let readback = if check {
                match c.host_buffer(l_total.max(16) + t_total.max(256), vk::BufferUsageFlags::TRANSFER_DST) {
                    Ok(r) => Some(r),
                    Err(e) => {
                        self.release_detile(vec![tbuf, lbuf]);
                        return Err(e);
                    }
                }
            } else {
                None
            };
            let rb = readback.as_ref().map(|r| r.0);
            let rows_bytes = l_total.max(16);
            let on_copy = self.copy_parts();
            let record_copies = |cmd: vk::CommandBuffer| unsafe {
                let mut k = 0;
                while k < copies.len() {
                    let src = copies[k].0;
                    let mut regs = Vec::new();
                    while k < copies.len() && copies[k].0 == src {
                        regs.push(copies[k].1);
                        k += 1;
                    }
                    c.device.cmd_copy_buffer(cmd, src, tb, &regs);
                }
            };
            let base = t.base;
            let was_first = first;
            let record = |cmd: vk::CommandBuffer| unsafe {
                super::vk::checkpoint(cmd, super::vk::mark::UPLOAD, base);
                let b = vk::ImageMemoryBarrier::default()
                    .old_layout(if was_first { vk::ImageLayout::UNDEFINED } else { vk::ImageLayout::GENERAL })
                    .new_layout(vk::ImageLayout::GENERAL)
                    .image(image)
                    .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: vk::REMAINING_MIP_LEVELS, base_array_layer: 0, layer_count: vk::REMAINING_ARRAY_LAYERS })
                    .src_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
                c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[b]);
                if on_copy.is_none() {
                    record_copies(cmd);
                }
                let m = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
                c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[m], &[], &[]);
                c.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipe.pipeline);
                let ti = [vk::DescriptorBufferInfo { buffer: tb, offset: 0, range: vk::WHOLE_SIZE }];
                let li = [vk::DescriptorBufferInfo { buffer: lb, offset: 0, range: vk::WHOLE_SIZE }];
                let w = [
                    vk::WriteDescriptorSet::default().dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&ti),
                    vk::WriteDescriptorSet::default().dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&li),
                ];
                c.push_descriptors(cmd, vk::PipelineBindPoint::COMPUTE, pipe.layout, 0, &w);
                for pc in &dispatches {
                    c.device.cmd_push_constants(cmd, pipe.layout, vk::ShaderStageFlags::COMPUTE, 0, pc);
                    c.device.cmd_dispatch(cmd, gx as u32, gy as u32, 1);
                }
                let m = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ);
                c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[m], &[], &[]);
                if let Some(rb) = rb {
                    c.device.cmd_copy_buffer(cmd, lb, rb, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: l_total.max(16) }]);
                    c.device.cmd_copy_buffer(cmd, tb, rb, &[vk::BufferCopy { src_offset: 0, dst_offset: rows_bytes, size: t_total.max(256) }]);
                    let h = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ);
                    c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[h], &[], &[]);
                }
                c.device.cmd_copy_buffer_to_image(cmd, lb, image, vk::ImageLayout::GENERAL, &regions);
                let a = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
                c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[a], &[], &[]);
            };
            // Submitted without waiting, as for 2D, unless checked.
            if readback.is_some() && on_copy.is_none() {
                self.one_shot(record);
                self.release_detile(vec![tbuf, lbuf]);
            } else if let Some((pool, sem)) = on_copy {
                match self.submit_copy_part(pool, sem, record_copies) {
                    Some((x, v)) => {
                        GPU_DETILE[10].fetch_add(1, Relaxed);
                        let held = Held { dev: vec![tbuf, lbuf], copy_cmd: Some(x) };
                        if readback.is_some() {
                            self.one_shot_after(Some((sem, v)), record);
                            self.release_held(held);
                        } else {
                            self.submit_held(held, DETILE_INFLIGHT, Some((sem, v)), record);
                        }
                    }
                    None => {
                        self.release_detile(vec![tbuf, lbuf]);
                        return Err("the copy queue refused the transfer".into());
                    }
                }
            } else {
                self.submit_held(Held { dev: vec![tbuf, lbuf], copy_cmd: None }, DETILE_INFLIGHT, None, record);
            }
            first = false;
            // DS_GPU_DETILE_CHECK=1: the chunk's slices against the CPU's detiling of the bytes copied.
            if let Some((rbuf, rmem, rptr)) = readback {
                let got = unsafe { std::slice::from_raw_parts(rptr, l_total as usize) };
                let copied = unsafe { std::slice::from_raw_parts(rptr.add(rows_bytes as usize), (hi - lo) as usize) };
                let mut src = vec![0u8; size as usize];
                src[lo as usize..hi as usize].copy_from_slice(copied);
                let mut want = vec![0u8; (slice * n) as usize];
                lay.detile_slices(&src, z0, z1, &mut want);
                let mut differ = 0u64;
                for (i, z) in (z0..z1).enumerate() {
                    let w = &want[i * slice as usize..(i + 1) * slice as usize];
                    let h = &got[i * slice_at as usize..i * slice_at as usize + slice as usize];
                    if w != h {
                        differ += 1;
                        if GPU_DETILE[7].load(Relaxed) + differ <= 8 {
                            let at = w.iter().zip(h).position(|(a, b)| a != b).unwrap_or(0);
                            crate::warn!("images: GPU detiling of 3D {:#x} {:?} slice {z} ({}x{} elements of {} bytes, swizzle {}) differs from the CPU's at byte {at} of {slice}", t.base, t.format, lay.width, lay.height, e, t.sw_mode);
                        }
                    }
                }
                GPU_DETILE[6].fetch_add(n, Relaxed);
                GPU_DETILE[7].fetch_add(differ, Relaxed);
                unsafe {
                    c.device.destroy_buffer(rbuf, None);
                    c.device.unmap_memory(rmem);
                    c.device.free_memory(rmem, None);
                }
            }
            GPU_DETILE[0].fetch_add(1, Relaxed);
            GPU_DETILE[1].fetch_add(n, Relaxed);
            GPU_DETILE[2].fetch_add(hi - lo, Relaxed);
            GPU_DETILE[3].fetch_add(slice * n, Relaxed);
            k0 = k1;
        }
        GPU_DETILE[11].fetch_add(1, Relaxed);
        GPU_DETILE[5].fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
        Ok(slice * lay.depth as u64)
    }

    /// A 1x1 zero image view of a dimension (unused image table entries).
    pub fn dummy_view(&mut self, dim: rdna::translate::ImageDim) -> Result<vk::ImageView, String> {
        use rdna::translate::ImageDim as D;
        let (vt, layers, ty3d) = match dim {
            D::D1 | D::D2 | D::D2Msaa => (vk::ImageViewType::TYPE_2D, 1, false),
            D::D1Array | D::D2Array | D::D2MsaaArray => (vk::ImageViewType::TYPE_2D_ARRAY, 1, false),
            D::Cube => (vk::ImageViewType::CUBE, 6, false),
            D::CubeArray => (vk::ImageViewType::CUBE_ARRAY, 6, false),
            D::D3 | D::D3OrD2 => (vk::ImageViewType::TYPE_3D, 1, true),
        };
        if let Some(&v) = self.dummies.get(&vt) {
            return Ok(v);
        }
        let key = ImageKey { base: 0, format: vk::Format::R8G8B8A8_UNORM, width: 1, height: 1, layers, levels: 1, ty3d, cube: layers == 6 };
        let (image, _, _) = self.create(key)?;
        self.to_general(image, key.format);
        let ci = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vt)
            .format(key.format)
            .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: if ty3d { 1 } else { layers } });
        let v = unsafe { ctx().device.create_image_view(&ci, None) }.map_err(|e| format!("dummy view {vt:?}: {e}"))?;
        self.dummies.insert(vt, v);
        Ok(v)
    }

    /// Key of the image a T# describes.
    pub fn key_of(t: &Tsharp) -> Option<ImageKey> {
        let format = vk_format(t.format)?;
        // The resource's mip count (T# MAX_MIP), not the view's range: a
        // view may name levels past it, which the hardware clamps (the
        // particle lighting atlas's writers of levels 1-3 bind levels
        // m..m+3 of its 4-level resource). Keyed by their range, each such
        // writer got an image of its own that the atlas's readers never
        // saw: smoke lit from those levels came out black (4-2's fog).
        // A zero MAX_MIP with a longer view keeps the view's count (no such
        // descriptor seen: possibly a MAX_MIP not filled in).
        let levels = if t.max_mip > 0 { t.max_mip + 1 } else { (t.last_level + 1).max(1) };
        let (ty3d, cube) = (t.ty == 10, t.ty == 11);
        let layers = match t.ty {
            10 | 11 | 12 | 13 | 15 => t.depth.max(1),
            _ => 1,
        };
        Some(ImageKey { base: t.base, format, width: t.width, height: t.height, layers, levels, ty3d, cube })
    }

    /// The depth image at `base` (size, format family, mips) with the most
    /// layers, when it has fewer than `layers` and none has enough. Depth
    /// views name only their slice range (DB_DEPTH_VIEW), so a shadow array
    /// rendered slice by slice, or sampled whole, makes its image grow
    /// (`grow_depth`) instead of splitting into one image per view.
    pub fn depth_to_grow(&self, base: u64, width: u32, height: u32, depths: &[vk::Format], layers: u32) -> Option<ImageKey> {
        let same = |k: &ImageKey| k.width == width && k.height == height && !k.ty3d && depths.contains(&k.format);
        let at = self.by_base.get(&base)?;
        if at.iter().any(|k| same(k) && k.layers >= layers) {
            return None;
        }
        at.iter().filter(|k| same(k) && k.layers < layers && self.map.get(k).is_some_and(|i| i.gpu_written)).copied().max_by_key(|k| k.layers)
    }

    /// Depth formats a sampled texture of `format` can read (R32 float over
    /// D32, R16 unorm over D16), or a depth surface's own format.
    pub fn depth_family(format: vk::Format) -> &'static [vk::Format] {
        use vk::Format as F;
        match format {
            F::R32_SFLOAT | F::D32_SFLOAT | F::D32_SFLOAT_S8_UINT => &[F::D32_SFLOAT, F::D32_SFLOAT_S8_UINT],
            F::R16_UNORM | F::D16_UNORM | F::D16_UNORM_S8_UINT => &[F::D16_UNORM, F::D16_UNORM_S8_UINT],
            _ => &[],
        }
    }

    /// Replaces depth image `old` by one with `layers` layers that holds its
    /// contents (layers above stay cleared). The caller has completed the
    /// work that used `old`.
    pub fn grow_depth(&mut self, old: ImageKey, layers: u32) -> Result<ImageKey, String> {
        let new_key = ImageKey { layers, ..old };
        let (image, mem, bytes) = self.create_in(new_key, big_on_small_card(&new_key))?;
        self.to_general(image, new_key.format);
        let Some(prev) = self.map_remove(&old) else { return Err("grow_depth: image gone".into()) };
        let aspect = aspects(old.format);
        self.one_shot(|cmd| unsafe {
            let d = &ctx().device;
            let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE);
            d.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
            let regions: Vec<vk::ImageCopy> = (0..old.levels)
                .map(|m| {
                    let sub = vk::ImageSubresourceLayers { aspect_mask: aspect, mip_level: m, base_array_layer: 0, layer_count: old.layers };
                    vk::ImageCopy { src_subresource: sub, src_offset: vk::Offset3D::default(), dst_subresource: sub, dst_offset: vk::Offset3D::default(), extent: vk::Extent3D { width: (old.width >> m).max(1), height: (old.height >> m).max(1), depth: 1 } }
                })
                .collect();
            d.cmd_copy_image(cmd, prev.image, vk::ImageLayout::GENERAL, image, vk::ImageLayout::GENERAL, &regions);
            let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
            d.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[mb], &[], &[]);
        });
        self.map_insert(new_key, Image { image, key: new_key, views: HashMap::default(), gpu_written: true, source: None, source_gen: None, hashes: [0; SLICES], checked_frame: 0, write_gen: prev.write_gen, block_gen: 0, gpu_parts: Vec::new(), mip_gens: Vec::new(), span: 0, layout: None, overwritten: false, writer: 0, mem, bytes, used: UseStamp::new(super::FRAMES.load(std::sync::atomic::Ordering::Relaxed)) });
        for v in self.stencil_of.values_mut() {
            if *v == old {
                *v = new_key;
            }
        }
        for v in self.meta_of.values_mut() {
            if v.0 == old {
                v.0 = new_key;
            }
        }
        if let Some(n) = self.pending_clears.remove(&old) {
            self.pending_clears.insert(new_key, n);
            self.pending_change += 1;
        }
        self.note_gen += 1;
        NOTE_GEN.fetch_add(1, std::sync::atomic::Ordering::Release);
        // Destroyed once the batches that may name it are done.
        super::exec::bury_images(vec![Evicted { image: prev.image, mem: prev.mem, views: prev.views.values().copied().collect() }]);
        self.depth_grows += 1;
        if self.depth_grows <= 16 {
            crate::info!("images: depth image {:#x} {}x{} {:?} grown from {} to {} layers", old.base, old.width, old.height, old.format, old.layers, layers);
        }
        Ok(new_key)
    }

    /// A rendered depth surface a texture of `key` reads (sampling only):
    /// same address and size, R32 float over D32, R16 unorm over D16, R8 uint
    /// over the stencil plane. Render targets are not written back, so guest
    /// memory under them is stale (a reversed-Z depth of 0 is the far plane,
    /// stencil 0 no material: the lighting saw only sky).
    fn depth_alias(&self, key: &ImageKey) -> Option<ImageKey> {
        use vk::Format as F;
        if key.format == F::R8_UINT {
            return self.stencil_of.get(&key.base).filter(|k| k.width == key.width && k.height == key.height && k.layers >= key.layers && self.map.contains_key(k)).copied();
        }
        let depths: &[vk::Format] = match key.format {
            F::R32_SFLOAT => &[F::D32_SFLOAT, F::D32_SFLOAT_S8_UINT],
            F::R16_UNORM => &[F::D16_UNORM, F::D16_UNORM_S8_UINT],
            _ => return None,
        };
        // The images at this address (it scanned the whole cache per lookup).
        self.by_base
            .get(&key.base)?
            .iter()
            .filter(|k| {
                self.map.get(k).is_some_and(|i| i.gpu_written) && k.width == key.width && k.height == key.height && k.layers >= key.layers && !k.ty3d && !key.ty3d && depths.contains(&k.format)
            })
            .copied()
            .max_by_key(|k| k.layers)
    }

    /// The cached image a texture of `key` uses, and how it was found. A
    /// GPU-written image of the same address and size (other format,
    /// rendered depth or stencil) holds newer data than guest memory: render
    /// targets are not written back, so an upload made before the surface
    /// was first rendered (e.g. the depth buffer read at boot) is stale.
    /// Whether binding `t` may read its guest memory now: a texture not cached
    /// yet or one uploaded from guest memory (refreshed when written there).
    /// Images the GPU owns are read from their Vulkan image; buffer writes
    /// over them are reloaded separately (`reload_overwritten`, which waits
    /// for the writers itself).
    pub fn reads_guest(&self, t: &Tsharp) -> bool {
        let Some(key) = Self::key_of(t) else { return true };
        match self.lookup(&key, false) {
            Some((k, _)) => !self.map.get(&k).is_some_and(|i| i.gpu_written),
            None => true,
        }
    }

    fn lookup(&self, key: &ImageKey, storage: bool) -> Option<(ImageKey, &'static str)> {
        let exact = self.map.get(key).map(|i| i.gpu_written);
        if exact == Some(true) {
            return Some((*key, "exact"));
        }
        if self.gpu_bases.contains(&key.base) {
            if let Some(k) = self.alias(key, true) {
                return Some((k, "alias"));
            }
            if !storage {
                if let Some(k) = self.depth_alias(key) {
                    return Some((k, if key.format == vk::Format::R8_UINT { "stencil of" } else { "depth of" }));
                }
            }
        }
        exact.map(|_| (*key, "exact"))
    }

    /// Block copies pending for the block-compressed textures among `keys`:
    /// the title transcodes streamed textures with compute shaders that store
    /// each 16- or 8-byte block as one R32G32B32A32 / R32G32 texel of a view
    /// whose size is the block grid. Images here do not share memory, so the
    /// blocks are copied into the texture (size-compatible formats) once per
    /// write generation. Mips with partial blocks are left out.
    /// Whether `take_block_copies` may find something for `keys`
    /// (block-compressed images whose guest base GPU work wrote).
    pub fn block_copies_possible(&self, keys: &[ImageKey]) -> bool {
        !self.gpu_bases.is_empty() && keys.iter().any(|k| is_compressed(k.format) && !k.ty3d && self.gpu_bases.contains(&k.base))
    }

    pub fn take_block_copies(&mut self, keys: &[ImageKey]) -> Vec<BlockCopy> {
        let out = self.block_copies(keys);
        for c in &out {
            let parts = Self::block_copy_parts(c, self.map.get(&c.key).and_then(|i| i.layout));
            if let Some(i) = self.map.get_mut(&c.key) {
                i.block_gen = c.seq;
                for (m, l, h) in parts {
                    match i.gpu_parts.iter_mut().find(|p| p.0 == m && p.1 == l) {
                        Some(p) => p.2 = h,
                        None => i.gpu_parts.push((m, l, h)),
                    }
                }
            }
        }
        out
    }

    /// The (mip, layer)s a block copy writes, with their own guest bytes'
    /// hashes now (`Image::gpu_parts`; parts not mapped now are left out).
    fn block_copy_parts(c: &BlockCopy, t: Option<Tsharp>) -> Vec<(u32, u32, u64)> {
        let Some(t) = t else { return Vec::new() };
        let Some(el) = elem_log2(t.format) else { return Vec::new() };
        let Some(lay) = tiling::layout(t.sw_mode, el, c.key.width, c.key.height, c.key.levels, is_block(t.format)) else { return Vec::new() };
        let elem = 1u64 << el;
        let mut want: Vec<(u32, u32)> = Vec::new();
        if c.src != vk::Image::null() {
            for &m in &c.mips {
                for l in 0..c.layers {
                    want.push((m, l));
                }
            }
        }
        if let Some((_, regs)) = &c.tail {
            want.extend(regs.iter().map(|r| (r.0, 0)));
        }
        let _readers = crate::mem::GUEST_READERS.read();
        want.into_iter()
            .filter(|&(m, _)| (m as usize) < lay.mips.len())
            .filter_map(|(m, l)| {
                let (s, n) = part_range(&lay, elem, m as usize, l);
                (super::cp::valid(t.base + s, n) && crate::mem::os_readable(t.base + s, n)).then(|| (m, l, own_part_hash(&lay, elem, t.base, m as usize, l)))
            })
            .collect()
    }

    /// Whether `depth_copies` may find something for `keys` (depth
    /// attachments over memory GPU work wrote through a color view).
    pub fn depth_feed_possible(&self, keys: &[ImageKey]) -> bool {
        !self.depth_feeds.is_empty() && keys.iter().any(|k| depth_feed_format(k.format).is_some() && self.depth_feeds.contains(&k.base))
    }

    /// Depth images among `keys` whose depth plane GPU work stored through a
    /// color view of the same address and size since it was last copied in:
    /// the title builds its half-resolution depth with a compute shader
    /// (R32 float storage), then depth-tests particles against that memory
    /// as a D32 surface (after a full-screen HTILE resummarize, which has no
    /// counterpart here). Images here do not share memory, so the D32 image
    /// kept its clear (0, the far plane of a reversed-Z depth) and every
    /// particle passed the test: fire and embers showed through walls.
    /// Once per write generation of the source (its storage bindings), so
    /// depth the attachment's own draws write afterwards is kept.
    pub fn depth_copies(&self, keys: &[ImageKey]) -> Vec<DepthCopy> {
        let mut out: Vec<DepthCopy> = Vec::new();
        for k in keys {
            let Some(feed) = depth_feed_format(k.format) else { continue };
            if k.ty3d || !self.depth_feeds.contains(&k.base) || out.iter().any(|c| c.key == *k) {
                continue;
            }
            let Some(dst) = self.map.get(k) else { continue };
            let src = self
                .by_base
                .get(&k.base)
                .into_iter()
                .flatten()
                .filter_map(|kk| self.map.get(kk))
                .filter(|i| i.gpu_written && i.key.format == feed && !i.key.ty3d && i.key.width == k.width && i.key.height == k.height && i.write_gen > dst.block_gen)
                .max_by_key(|i| i.write_gen);
            let Some(src) = src else { continue };
            out.push(DepthCopy {
                key: *k,
                dst: dst.image,
                src: src.image,
                width: k.width,
                height: k.height,
                layers: k.layers.min(src.key.layers).max(1),
                texel: if feed == vk::Format::R32_SFLOAT { 4 } else { 2 },
                seq: src.write_gen,
            });
        }
        out
    }

    pub fn take_depth_copies(&mut self, keys: &[ImageKey]) -> Vec<DepthCopy> {
        let out = self.depth_copies(keys);
        for c in &out {
            if let Some(i) = self.map.get_mut(&c.key) {
                i.block_gen = c.seq;
            }
        }
        out
    }

    /// The mip tail of block-compressed texture `k` stored through a view
    /// of the tail block (`BlockCopy::tail`), newer than its last copy: the
    /// view, its write generation and the tail mips with whole blocks.
    fn tail_blocks(&self, k: &ImageKey, dst: &Image, bytes: u32) -> Option<(vk::Image, u64, Vec<(u32, u32, u32, u32, u32)>)> {
        let t = dst.layout.as_ref()?;
        let el = elem_log2(t.format)?;
        let lay = tiling::layout(t.sw_mode, el, k.width, k.height, k.levels, true)?;
        let regions: Vec<(u32, u32, u32, u32, u32)> = lay
            .mips
            .iter()
            .enumerate()
            .filter(|(_, m)| m.texel_w % 4 == 0 && m.texel_h % 4 == 0)
            .filter_map(|(i, m)| m.tail.map(|(x, y)| (i as u32, x, y, m.width, m.height)))
            .collect();
        if regions.is_empty() {
            return None;
        }
        let src = self
            .by_base
            .get(&k.base)
            .into_iter()
            .flatten()
            .filter_map(|kk| self.map.get(kk))
            .filter(|i| i.gpu_written && !is_compressed(i.key.format) && !i.key.ty3d && i.key.levels == 1)
            .filter(|i| texel_bytes(i.key.format) == bytes && i.key.width == lay.block_w && i.key.height == lay.block_h && i.write_gen > dst.block_gen)
            .max_by_key(|i| i.write_gen)?;
        Some((src.image, src.write_gen, regions))
    }

    /// The block copies `take_block_copies` would take (nothing marked).
    pub fn block_copies(&self, keys: &[ImageKey]) -> Vec<BlockCopy> {
        let mut out: Vec<BlockCopy> = Vec::new();
        for k in keys {
            if !is_compressed(k.format) || k.ty3d || out.iter().any(|c| c.key == *k) {
                continue;
            }
            if !self.gpu_bases.contains(&k.base) {
                continue;
            }
            let Some(dst) = self.map.get(k) else { continue };
            let (bw, bh) = (k.width.div_ceil(4), k.height.div_ceil(4));
            let bytes = texel_bytes(k.format);
            let src = self
                .by_base
                .get(&k.base)
                .into_iter()
                .flatten()
                .filter_map(|kk| self.map.get(kk))
                .filter(|i| i.gpu_written && !is_compressed(i.key.format) && !i.key.ty3d)
                .filter(|i| texel_bytes(i.key.format) == bytes && i.key.width == bw && i.key.height == bh && i.write_gen > dst.block_gen)
                .max_by_key(|i| i.write_gen);
            let tail = if super::tune::block_tails() { self.tail_blocks(k, dst, bytes) } else { None };
            if src.is_none() && tail.is_none() {
                continue;
            }
            // Each mip from the view that wrote it since the last copy: the
            // whole chain's view where it wrote that level (each storage
            // binding writes its base level), else the tail block's view for
            // a tail mip. (Until 2026-11-11 the chain's view gave only the
            // mips before the tail and the tail view the rest: the title's
            // character texture encoders write mips 4-8 through the chain's
            // view and only the one-block mips through the tail view, so a
            // soldier's face had stale bytes from mip 4 on.)
            let whole = |m: u32| (k.width >> m) > 0 && (k.height >> m) > 0 && (k.width >> m) % 4 == 0 && (k.height >> m) % 4 == 0;
            let main_gen = |m: u32| src.filter(|s| m < s.key.levels).map_or(0, |s| s.mip_gens.get(m as usize).copied().unwrap_or(0));
            let tail_gen = tail.as_ref().map_or(0, |t| t.1);
            let mut mips: Vec<u32> = Vec::new();
            let mut tail_mips: Vec<u32> = Vec::new();
            let mut seq = 0;
            for m in 0..k.levels {
                let g = if whole(m) { main_gen(m) } else { 0 };
                if g > dst.block_gen {
                    mips.push(m);
                    seq = seq.max(g);
                } else if tail_gen > dst.block_gen && tail.as_ref().is_some_and(|t| t.2.iter().any(|r| r.0 == m)) {
                    tail_mips.push(m);
                    seq = seq.max(tail_gen);
                }
            }
            if mips.is_empty() && tail_mips.is_empty() {
                continue;
            }
            let tail = tail.filter(|_| !tail_mips.is_empty()).map(|(img, _, r)| (img, r.into_iter().filter(|x| tail_mips.contains(&x.0)).collect::<Vec<_>>()));
            out.push(BlockCopy {
                key: *k,
                src: if mips.is_empty() { vk::Image::null() } else { src.map_or(vk::Image::null(), |s| s.image) },
                dst: dst.image,
                width: bw,
                height: bh,
                mips,
                layers: src.map_or(1, |s| s.key.layers.min(k.layers).max(1)),
                tail,
                seq,
            });
        }
        out
    }

    /// Notes writes (GPU buffer writes, CP DMA) that cover the metadata of a
    /// rendered surface: fast clears, resolved at the surface's next use.
    /// GPU buffer writes: fast-clear metadata (`note_meta_writes`) and
    /// GPU-owned images overwritten (`note_overwrites`, the sized writes).
    pub fn note_writes(&mut self, writes: &[(u64, u64)], sized: &[(u64, u64)]) {
        let lookup = LOOKUP_GEN.load(std::sync::atomic::Ordering::Acquire);
        if let Some((w, s, g, n)) = &self.last_note {
            if *g == lookup && *n == self.note_gen && w == writes && s == sized {
                return;
            }
        }
        self.note_meta_writes(writes);
        self.note_overwrites(sized);
        self.last_note = Some((writes.to_vec(), sized.to_vec(), lookup, self.note_gen));
    }

    /// The fast-clear metadata a write (`fill`: a fill, which must cover a block's first dword)
    /// over [b, e) reaches: (the surface's metadata address, its image, the layers reached; 0 the
    /// whole image). A layered depth surface's later layers' metadata lies past its address
    /// (`meta_layer_bytes`), so the entries just below `b` are looked at too.
    fn meta_reach(&self, b: u64, e: u64, fill: bool) -> Vec<(u64, ImageKey, u128)> {
        let mut out = Vec::new();
        for (&m, &(k, _)) in self.meta_of.range(..b).rev().take(8) {
            if let Some(per) = meta_layer_bytes(&k) {
                let mask = layers_reached(m, per, k.layers, b, e, fill);
                if mask != 0 {
                    out.push((m, k, mask));
                }
            }
        }
        for (&m, &(k, _)) in self.meta_of.range(b..e) {
            match meta_layer_bytes(&k) {
                Some(per) => {
                    let mask = layers_reached(m, per, k.layers, b, e, fill);
                    if mask != 0 {
                        out.push((m, k, mask));
                    }
                }
                None if !fill || m + 4 <= e => out.push((m, k, 0)),
                None => {}
            }
        }
        out
    }

    pub fn note_meta_writes(&mut self, writes: &[(u64, u64)]) {
        if self.meta_of.is_empty() {
            return;
        }
        for &(b, n) in writes {
            for (meta, key, layers) in self.meta_reach(b, b.saturating_add(n), false) {
                // (A fill's value it replaces is kept for the use: `merge_clear`.)
                merge_clear(&mut self.pending_clears, key, meta, None, layers);
                self.pending_change += 1;
                if trace_clears() {
                    crate::info!("trace clears: {:#x} ({:?} {}x{}) pending: write {b:#x}+{n:#x} covers its metadata {meta:#x} (layers {layers:#x}) (for flip {}, {})", key.base, key.format, key.width, key.height, crate::hle::libs::videoout::QUEUED_FLIP.load(std::sync::atomic::Ordering::Relaxed) + 1, std::thread::current().name().unwrap_or("?"));
                }
            }
        }
    }

    /// `note_meta_writes` for the writes of a fill of [base, base+len) whose `note_meta_fill`
    /// follows (`fill_note_mode`): the metadata the fill answers for (the same surface, metadata and layers) is
    /// left to the fill, its stored value known. Noted first as a write of unknown data, each
    /// layer's fill of a layered depth surface made the value its earlier layers' fills had
    /// given unknown (one value an image: `merge_clear`), so the surface's uses drained the GPU
    /// to read its metadata back: the shadow atlas three times a flip at 1-2's Gatehouse Hall,
    /// ~1.6 ms each (2026-11-21).
    pub fn note_meta_writes_but_fill(&mut self, writes: &[(u64, u64)], base: u64, len: u64) {
        if self.meta_of.is_empty() {
            return;
        }
        let mode = fill_note_mode();
        if mode == 0 {
            self.note_meta_writes(writes);
            return;
        }
        let filled: Vec<_> = self.meta_reach(base, base.saturating_add(len), true).into_iter().filter(|(_, k, _)| left_to_fill(mode, k)).collect();
        for &(b, n) in writes {
            for (meta, key, layers) in self.meta_reach(b, b.saturating_add(n), false) {
                if filled.iter().any(|&(m, k, l)| m == meta && k == key && l == layers) {
                    continue;
                }
                merge_clear(&mut self.pending_clears, key, meta, None, layers);
                self.pending_change += 1;
                if trace_clears() {
                    crate::info!("trace clears: {:#x} ({:?} {}x{}) pending: write {b:#x}+{n:#x} beside a fill covers its metadata {meta:#x} (layers {layers:#x}) (for flip {}, {})", key.base, key.format, key.width, key.height, crate::hle::libs::videoout::QUEUED_FLIP.load(std::sync::atomic::Ordering::Relaxed) + 1, std::thread::current().name().unwrap_or("?"));
                }
            }
        }
    }

    /// What `note_writes` (then `note_meta_fill` for a fill of `fill`)
    /// would note, without noting it (held executors keep their
    /// own: `Exec::held_note`): the fast clears made pending (image,
    /// metadata address, the value when the fill gives it; later entries
    /// win; the writes' metadata the fill answers for left to the fill, as
    /// `note_meta_writes_but_fill`) and the GPU-owned images overwritten.
    pub fn notes_for(&self, writes: &[(u64, u64)], sized: &[(u64, u64)], fill: Option<((u64, u64), u32)>) -> (Vec<(ImageKey, u64, Option<u32>, u128)>, Vec<ImageKey>) {
        let mut clears = Vec::new();
        if !self.meta_of.is_empty() {
            let mode = fill_note_mode();
            // Every entry the fill reaches takes its value after the writes' unknown ones; the
            // writes leave out only those `left_to_fill` (e97f75a's mode 0 had pushed no fill
            // values here, so a held stretch's clears all went unknown: drains at their uses).
            let filled: Vec<_> = fill.map(|((base, len), _)| self.meta_reach(base, base.saturating_add(len), true)).unwrap_or_default();
            for &(b, n) in writes {
                for (meta, key, layers) in self.meta_reach(b, b.saturating_add(n), false) {
                    if left_to_fill(mode, &key) && filled.iter().any(|&(m, k, l)| m == meta && k == key && l == layers) {
                        continue;
                    }
                    clears.push((key, meta, None, layers));
                }
            }
            if let Some(((_, _), v)) = fill {
                for (meta, key, layers) in filled {
                    clears.push((key, meta, Some(v), layers));
                }
            }
        }
        let mut over = Vec::new();
        if !self.owned.is_empty() {
            for &(b, n) in sized {
                let end = b.saturating_add(n);
                for (_, v) in self.owned.range(b.saturating_sub(self.max_span)..end) {
                    for &k in v {
                        let Some(img) = self.map.get(&k) else { continue };
                        let span = img.span;
                        if k.base + span <= b || !img.gpu_written {
                            continue;
                        }
                        if b <= k.base && k.base + span <= end && n <= 2 * span + (64 << 10) && !over.contains(&k) {
                            over.push(k);
                        }
                    }
                }
            }
        }
        (clears, over)
    }

    /// The GPU-owned images a zero fill of [base, base+len) clears as
    /// `zero_filled` chooses them, nothing noted (held executors).
    pub fn zero_fill_targets(&self, base: u64, len: u64) -> Vec<(ImageKey, vk::Image, vk::Format)> {
        let mut out = Vec::new();
        if len == 0 || self.owned.is_empty() {
            return out;
        }
        let end = base.saturating_add(len);
        for (_, v) in self.owned.range(base.saturating_sub(self.max_span)..end) {
            for &k in v {
                let Some(img) = self.map.get(&k) else { continue };
                let span = img.span;
                if !img.gpu_written || is_compressed(k.format) || !(base <= k.base && k.base + span <= end && len <= 2 * span + (64 << 10)) {
                    continue;
                }
                if !out.iter().any(|x: &(ImageKey, vk::Image, vk::Format)| x.0 == k) {
                    out.push((k, img.image, k.format));
                }
            }
        }
        out
    }

    /// Notes a held executor's own fast clears and overwritten images once
    /// its batches go to the GPU (`Exec::past_commit`): after the notes of
    /// the work before them.
    pub fn apply_held(&mut self, clears: &std::collections::HashMap<ImageKey, PendingClear>, over: &std::collections::HashSet<ImageKey>) {
        for (k, &e) in clears {
            merge_pending(&mut self.pending_clears, *k, e);
            self.pending_change += 1;
        }
        for k in over {
            let Some(img) = self.map.get_mut(k) else { continue };
            if !img.overwritten {
                img.overwritten = true;
                self.overwritten_keys.insert(*k);
                self.pending_change += 1;
                self.pending_overwrites += 1;
            }
        }
        self.note_gen += 1;
        NOTE_GEN.fetch_add(1, std::sync::atomic::Ordering::Release);
    }

    /// The fast clear of image `k` with metadata at `meta` (`take_clears`'s
    /// entry for a clear a held executor noted itself).
    pub fn fast_clear_of(&self, k: ImageKey, meta: u64, known: Option<u32>, kept: Option<u32>, layers: u128) -> Option<FastClear> {
        let (Some(img), Some(&(_, clear))) = (self.map.get(&k), self.meta_of.get(&meta)) else { return None };
        Some(FastClear { key: k, image: img.image, meta, clear, known, kept, layers })
    }

    /// The fill kernel stored `v` in every word of [base, base+len): the fast
    /// clears pending on metadata inside it read `v` (after `note_writes`
    /// for the same dispatch, which marked them unknown).
    pub fn note_meta_fill(&mut self, base: u64, len: u64, v: u32) {
        if self.meta_of.is_empty() {
            return;
        }
        for (meta, key, layers) in self.meta_reach(base, base.saturating_add(len), true) {
            if trace_clears() {
                crate::info!("trace clears: {:#x} ({:?} {}x{}) pending: fill {base:#x}+{len:#x} of {v:#x} covers its metadata {meta:#x} (layers {layers:#x}) (for flip {}, {})", key.base, key.format, key.width, key.height, crate::hle::libs::videoout::QUEUED_FLIP.load(std::sync::atomic::Ordering::Relaxed) + 1, std::thread::current().name().unwrap_or("?"));
            }
            merge_clear(&mut self.pending_clears, key, meta, Some(v), layers);
            // Published for the draws' check without the lock (`pending_hit`). A fill's entry
            // with no write noted before it in the same `with` (`fill_note_mode` 1 and 2) was
            // missing from the published set: draws passed the image by, and the next parallel
            // run's start (`Cp`, every pending clear) applied the clear over what they had drawn
            // (1-1's Start lost the world's geometry in alternate flips, 2026-11-21).
            self.pending_change += 1;
            // A write noted again over it makes it unknown again: not a
            // repeat for the memos of `note_writes` and `Exec::last_notes`.
            self.note_gen += 1;
            NOTE_GEN.fetch_add(1, std::sync::atomic::Ordering::Release);
        }
    }

    /// Whether every pending fast clear among `keys` has a known value.
    pub fn clears_known(&self, keys: &[ImageKey]) -> bool {
        keys.iter().all(|k| self.pending_clears.get(k).is_none_or(|&(_, v, _, _)| v.is_some()))
    }

    /// The images with a fast clear pending.
    pub fn pending_clear_keys(&self) -> Vec<ImageKey> {
        self.pending_clears.keys().copied().collect()
    }

    /// Whether any of `keys` has a fast clear pending.
    pub fn clears_pending(&self, keys: &[ImageKey]) -> bool {
        !self.pending_clears.is_empty() && keys.iter().any(|k| self.pending_clears.contains_key(k))
    }

    /// Takes the pending fast clears of `keys`.
    pub fn take_clears(&mut self, keys: &[ImageKey]) -> Vec<FastClear> {
        let mut out = Vec::new();
        for k in keys {
            let Some((meta, known, kept, layers)) = self.pending_clears.remove(k) else { continue };
            self.pending_change += 1;
            self.note_gen += 1;
            NOTE_GEN.fetch_add(1, std::sync::atomic::Ordering::Release);
            let (Some(img), Some(&(_, clear))) = (self.map.get(k), self.meta_of.get(&meta)) else { continue };
            if out.iter().any(|c: &FastClear| c.key == *k) {
                continue;
            }
            out.push(FastClear { key: *k, image: img.image, meta, clear, known, kept, layers });
        }
        out
    }

    /// The cached image a T# would use now (fast-clear checks; no side effects).
    /// Whether a T# binds as a depth/stencil view (a rendered depth surface):
    /// the translation compares in hardware there and emulates comparisons
    /// on color views.
    pub fn depth_view(&mut self, t: &Tsharp) -> bool {
        self.resolved_key(t, false).is_some_and(|k| is_depth(k.format))
    }

    pub fn resolved_key(&mut self, t: &Tsharp, storage: bool) -> Option<ImageKey> {
        // (Generations read before the lookup; it holds while they do.)
        use std::sync::atomic::Ordering::Acquire;
        let (gpu_gen, bgen) = (GPU_IMAGES_GEN.load(Acquire), base_gen(t.base).load(Acquire));
        let mk = memo_raw(t);
        if let Some(&(k, g, b)) = self.key_memo.get(&(mk, storage)) {
            if g == gpu_gen && b == bgen {
                tls(|m| {
                    if m.keys.len() >= m.cap {
                        m.keys.clear();
                    }
                    m.keys.insert((mk, storage), (k, g, b));
                });
                return k;
            }
        }
        let k = self.lookup(&Self::key_of(t)?, storage).map(|(k, _)| k);
        if self.key_memo.len() >= MEMO_CAP {
            self.key_memo.clear();
        }
        self.key_memo.insert((mk, storage), (k, gpu_gen, bgen));
        tls(|m| {
            if m.keys.len() >= m.cap {
                m.keys.clear();
            }
            m.keys.insert((mk, storage), (k, gpu_gen, bgen));
        });
        k
    }

    /// Image for a T#: an existing GPU image at that address with a
    /// compatible size, or a texture uploaded from guest memory.
    fn image_for(&mut self, t: &Tsharp, frame: u64, storage: bool) -> Result<ImageKey, String> {
        let key = Self::key_of(t).ok_or_else(|| format!("image format {} unsupported", t.format))?;
        let timed = || holds_on().then(std::time::Instant::now);
        if let Some((k, how)) = self.lookup(&key, storage) {
            if how == "exact" && !self.map[&k].gpu_written {
                let ts = timed();
                self.refresh(key, t, frame)?;
                note_step(0, ts);
            } else if let Some(w) = self.refreshing.get(&k).filter(|_| REFRESH_DEFER.with(|d| d.get())) {
                // (Another view of an image whose upload a thread left for
                // after the lock: waited for as `refresh` does.)
                REFRESH_OUT[3].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let w = w.clone();
                REFRESH_WAITS.with(|v| v.borrow_mut().push(w));
            }
            if how == "exact" {
                self.note_layout(k, t);
            }
            return Ok(k);
        }
        let ts = timed();
        let t_first = std::time::Instant::now();
        let (image, mem, bytes) = self.create_texture(key, storage)?;
        let t_created = std::time::Instant::now();
        note_step(1, ts);
        // A texture detiled on the CPU: its first upload after the lock, as
        // its re-uploads are (`RefreshJob`; since 2026-11-02, Castlewall's
        // cold arrival +6% flips/s and p99 -26%). The image is in the cache
        // from now on, its write tracking armed before its bytes are read;
        // threads binding it meanwhile wait for its copy before their work
        // can be submitted, and its hashes are stored with the copy.
        let deferred = REFRESH_DEFER.with(|d| d.get()) && reupload_on_cpu(&key, t);
        let size = deferred.then(|| elem_log2(t.format).and_then(|el| tiling::layout(t.sw_mode, el, key.width, key.height, key.levels, is_block(t.format)))).flatten().map(|lay| lay.size(key.layers));
        if let Some(region) = size.and_then(|n| super::track::watch(t.base, n)) {
            let n = size.unwrap_or(0);
            let wait = std::sync::Arc::new(RefreshWait::default());
            self.refreshing.insert(key, wait.clone());
            FIRST_OUT[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            REFRESH_JOBS.with(|j| j.borrow_mut().push(RefreshJob { key, image, t: *t, region, range: (t.base, n), parts: None, all: true, cause: 0, remap: None, wait, first: true, keep: Vec::new() }));
            self.map_insert(key, Image { image, key, views: HashMap::default(), gpu_written: false, source: Some(region), source_gen: super::track::gen_of(region), hashes: [0; SLICES], checked_frame: frame, write_gen: 0, block_gen: 0, gpu_parts: Vec::new(), mip_gens: Vec::new(), span: 0, layout: None, overwritten: false, writer: 0, mem, bytes, used: UseStamp::new(frame) });
            self.note_layout(key, t);
            return Ok(key);
        }
        let ts = timed();
        let t_up = std::time::Instant::now();
        let uploaded = {
            let r = self.upload(image, key, t);
            note_step(3, ts);
            r
        };
        let t_uploaded = std::time::Instant::now();
        let source = match uploaded {
            Ok(s) => Some(s),
            Err(e) => {
                self.report(format!("texture {:#x} ({}x{} format {} sw_mode {}): {e}; contents undefined", t.base, t.width, t.height, t.format, t.sw_mode));
                self.to_general(image, key.format);
                None
            }
        };
        let _ = frame;
        // CPU writes to the source re-upload it (write tracking).
        let ts = timed();
        let region = source.and_then(|(b, n)| super::track::watch(b, n));
        note_step(4, ts);
        if source.is_some() && region.is_none() {
            self.report(format!("texture {:#x} is not write-tracked (read-only or unmapped pages): later CPU updates are not seen", t.base));
        }
        let ts = timed();
        let t_watched = std::time::Instant::now();
        let hashes = source.map(|(b, n)| all_hashes(b, n)).unwrap_or([0; SLICES]);
        note_step(5, ts);
        // First uploads of 30 ms or more: where their time went (diagnosis; the upload's own
        // line, `images: GPU-detiled upload of` or `images: upload of`, splits its part).
        let ms = |a: std::time::Instant, b: std::time::Instant| (b - a).as_secs_f64() * 1e3;
        let now = std::time::Instant::now();
        if ms(t_first, now) >= 30.0 {
            crate::info!(
                "images: first upload of {:#x} {}x{}x{} {:?} ({} KiB) took {:.0} ms: the image {:.0}, the upload {:.0}, its write tracking {:.0}, its sampled hashes {:.0}",
                t.base,
                key.width,
                key.height,
                key.layers,
                key.format,
                source.map(|s| s.1 >> 10).unwrap_or(0),
                ms(t_first, now),
                ms(t_first, t_created),
                ms(t_up, t_uploaded),
                ms(t_uploaded, t_watched),
                ms(t_watched, now)
            );
        }
        self.map_insert(key, Image { image, key, views: HashMap::default(), gpu_written: false, source: region, source_gen: region.and_then(super::track::gen_of), hashes, checked_frame: frame, write_gen: 0, block_gen: 0, gpu_parts: Vec::new(), mip_gens: Vec::new(), span: 0, layout: None, overwritten: false, writer: 0, mem, bytes, used: UseStamp::new(frame) });
        self.note_layout(key, t);
        Ok(key)
    }

    /// Remembers the guest layout of an image from a T# that describes it
    /// exactly (its span and a T# to upload it with again).
    fn note_layout(&mut self, k: ImageKey, t: &Tsharp) {
        let Some(img) = self.map.get_mut(&k) else { return };
        if img.layout.is_some() || k.ty3d {
            return;
        }
        let Some(el) = elem_log2(t.format) else { return };
        if let Some(lay) = tiling::layout(t.sw_mode, el, k.width, k.height, k.levels, is_block(t.format)) {
            img.span = lay.size(k.layers);
            img.layout = Some(*t);
            self.index_owned(k);
        }
    }

    /// Indexes a GPU-owned image with a known span (`note_overwrites`).
    fn index_owned(&mut self, k: ImageKey) {
        let Some(img) = self.map.get(&k) else { return };
        if !img.gpu_written || img.span == 0 {
            return;
        }
        let span = img.span;
        let v = self.owned.entry(k.base).or_default();
        if !v.contains(&k) {
            mark_note_pages(k.base, span);
            v.push(k);
            self.max_span = self.max_span.max(span);
            self.note_gen += 1;
            NOTE_GEN.fetch_add(1, std::sync::atomic::Ordering::Release);
        }
    }

    /// Notes GPU buffer writes and CP DMA writes (sized ranges, not unbounded
    /// pointers) over GPU-owned images. The title clears some of its render
    /// targets and volumes through a buffer view of their memory (the
    /// volumetric fog's density volume is filled with zeros by a compute
    /// shader every frame, then fog volumes are drawn into its slices);
    /// images here do not share memory with buffers, so an image whose guest
    /// bytes a write covers entirely takes its contents from guest memory at
    /// its next use (`reload_overwritten`). Writes over part of an image are
    /// reported, not applied: the rest of its guest bytes are stale.
    pub fn note_overwrites(&mut self, writes: &[(u64, u64)]) {
        if writes.is_empty() || self.owned.is_empty() {
            return;
        }
        let mut partial = Vec::new();
        for &(b, n) in writes {
            let end = b.saturating_add(n);
            let keys: Vec<ImageKey> = self.owned.range(b.saturating_sub(self.max_span)..end).flat_map(|(_, v)| v.iter().copied()).collect();
            for k in keys {
                let Some(img) = self.map.get_mut(&k) else { continue };
                let span = img.span;
                if k.base + span <= b || !img.gpu_written {
                    continue;
                }
                // Covering it, and not a much larger range that happens to
                // contain it (a pool or heap binding).
                if b <= k.base && k.base + span <= end && n <= 2 * span + (64 << 10) {
                    if !img.overwritten {
                        img.overwritten = true;
                        self.overwritten_keys.insert(k);
                        self.pending_change += 1;
                        self.pending_overwrites += 1;
                    }
                } else if self.partial_seen.len() < 64 && self.partial_seen.insert(k.base) {
                    partial.push(format!(
                        "GPU buffer write {b:#x}+{n:#x} covers part of GPU-owned image {:#x}+{span:#x} ({}x{}x{} {:?}): not applied",
                        k.base, k.width, k.height, k.layers, k.format
                    ));
                }
            }
        }
        for m in partial {
            crate::info!("images: {m}");
        }
    }

    /// The fill kernel stores zeros over [base, base+len) (`tune
    /// zero_fill_clears`): the GPU-owned images it covers as
    /// `note_overwrites` counts covering are cleared by the caller right after
    /// the fill, in its batch (returned), instead of being marked overwritten
    /// (which made their next user wait for the GPU, scan their guest bytes
    /// for zeros and clear them: ~6,300 per gate run, ~0.4 ms each on the
    /// graphics command processor). Compressed formats keep that path.
    /// The parts of guest [base, base+len) under GPU-written uncompressed
    /// images used in the last FILL_FRESH frames and under no image uploaded
    /// from guest memory (merged, clipped to the range): bytes nothing reads
    /// from guest memory (such images are never written back there), which a
    /// zero fill need not store (`tune fill_partial`; an image not used
    /// lately may be memory the title uses otherwise now, so its bytes are
    /// stored). Empty when an image over the range has no known span. Every
    /// image is looked at, as `fill_covered` does.
    pub fn gpu_written_spans(&self, base: u64, len: u64) -> Vec<(u64, u64)> {
        const FILL_FRESH: u64 = 8;
        let mut plus: Vec<(u64, u64)> = Vec::new();
        let mut minus: Vec<(u64, u64)> = Vec::new();
        if len == 0 || self.owned.is_empty() {
            return plus;
        }
        let end = base.saturating_add(len);
        let frame = super::FRAMES.load(std::sync::atomic::Ordering::Relaxed);
        for (k, img) in self.map.iter() {
            if k.base >= end || k.base.saturating_add(img.span.max(1)) <= base {
                continue;
            }
            if img.span == 0 {
                return Vec::new();
            }
            let part = (k.base.max(base), k.base.saturating_add(img.span).min(end));
            if !img.gpu_written || is_compressed(k.format) {
                minus.push(part);
            } else if frame.saturating_sub(img.used.get()) <= FILL_FRESH {
                plus.push(part);
            }
        }
        let merge = |mut v: Vec<(u64, u64)>| -> Vec<(u64, u64)> {
            v.sort_unstable();
            let mut m: Vec<(u64, u64)> = Vec::with_capacity(v.len());
            for (a, b) in v {
                match m.last_mut() {
                    Some(x) if a <= x.1 => x.1 = x.1.max(b),
                    _ => m.push((a, b)),
                }
            }
            m
        };
        let (plus, minus) = (merge(plus), merge(minus));
        // plus minus minus: the parts no other image shares.
        let mut out = Vec::with_capacity(plus.len());
        for (a, b) in plus {
            let mut at = a;
            for &(x, y) in &minus {
                if y <= at || x >= b {
                    continue;
                }
                if x > at {
                    out.push((at, x));
                }
                at = at.max(y);
                if at >= b {
                    break;
                }
            }
            if at < b {
                out.push((at, b));
            }
        }
        out
    }

    pub fn zero_filled(&mut self, base: u64, len: u64) -> Vec<(vk::Image, vk::Format)> {
        let mut out = Vec::new();
        if len == 0 || self.owned.is_empty() {
            return out;
        }
        let end = base.saturating_add(len);
        let keys: Vec<ImageKey> = self.owned.range(base.saturating_sub(self.max_span)..end).flat_map(|(_, v)| v.iter().copied()).collect();
        for k in keys {
            let Some(img) = self.map.get_mut(&k) else { continue };
            let span = img.span;
            if !img.gpu_written || is_compressed(k.format) || !(base <= k.base && k.base + span <= end && len <= 2 * span + (64 << 10)) {
                continue;
            }
            if img.overwritten {
                img.overwritten = false;
                self.overwritten_keys.remove(&k);
                self.pending_overwrites = self.pending_overwrites.saturating_sub(1);
            }
            self.pending_change += 1;
            self.note_gen += 1;
            NOTE_GEN.fetch_add(1, std::sync::atomic::Ordering::Release);
            self.overwrite_clears += 1;
            ZERO_FILL_CLEARS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if trace_clears() {
                crate::info!("trace clears: {:#x} ({:?} {}x{}) zero-filled by a fill {base:#x}+{len:#x} (for flip {}, {})", k.base, k.format, k.width, k.height, crate::hle::libs::videoout::QUEUED_FLIP.load(std::sync::atomic::Ordering::Relaxed) + 1, std::thread::current().name().unwrap_or("?"));
            }
            out.push((img.image, k.format));
        }
        out
    }

    /// Evicts uploaded images not used for EVICT_AGE frames, least recently
    /// used first, until `want` bytes are freed: (the images to destroy,
    /// bytes, images left). A texture evicted is uploaded again from guest
    /// memory at its next use.
    /// The `n` largest cached images, one clause each (eviction reports).
    /// Images the GPU wrote (kept for the session): [their bytes, those unused for `age` flips
    /// or more] (the over-budget warning: what moving cold ones to system memory could free).
    fn gpu_written_census(&self, frame: u64, age: u64) -> (u64, u64) {
        self.map.values().filter(|i| i.gpu_written).fold((0, 0), |(all, cold), i| (all + i.bytes, cold + if frame.saturating_sub(i.used.get()) >= age { i.bytes } else { 0 }))
    }

    /// Bytes in video memory of [images the GPU wrote, textures], and the `n` largest images there
    /// (the over-budget census: what holds video memory on a small card).
    fn video_census(&self, n: usize) -> (u64, u64, String) {
        let (mut gw, mut tex) = (0u64, 0u64);
        let mut v: Vec<(&ImageKey, &Image)> = Vec::new();
        for (k, i) in self.map.iter() {
            if i.mem.in_system() {
                continue;
            }
            if i.gpu_written { gw += i.bytes } else { tex += i.bytes }
            v.push((k, i));
        }
        v.sort_by(|a, b| b.1.bytes.cmp(&a.1.bytes));
        let top = v.iter().take(n).map(|(k, i)| format!("{}x{}x{} {:?} {} MiB{}", k.width, k.height, k.layers, k.format, i.bytes >> 20, if i.gpu_written { " GPU-written" } else { "" })).collect::<Vec<_>>().join(", ");
        (gw, tex, top)
    }

    /// Textures (not GPU-written): [their bytes, about those of the top mip of the large ones (1024
    /// texels or more on a side, two levels or more: 3/4 of a full chain's bytes)] (the over-budget
    /// census: what fewer mips on small cards could free).
    fn texture_census(&self) -> (u64, u64) {
        self.map.values().filter(|i| !i.gpu_written).fold((0, 0), |(all, top), i| {
            let k = &i.key;
            let large = (k.width >= 1024 || k.height >= 1024) && k.levels >= 2 && !k.ty3d;
            (all + i.bytes, top + if large { i.bytes / 4 * 3 } else { 0 })
        })
    }

    fn largest(&self, n: usize) -> String {
        let mut v: Vec<(&ImageKey, &Image)> = self.map.iter().collect();
        v.sort_by(|a, b| b.1.bytes.cmp(&a.1.bytes));
        let parts: Vec<String> = v
            .iter()
            .take(n)
            .map(|(k, i)| format!("{} MiB at {:#x} {}x{}x{} {:?} {} mips{}{}", i.bytes >> 20, k.base, k.width, k.height, k.layers, k.format, k.levels, if k.ty3d { " 3D" } else { "" }, if i.gpu_written { " GPU-written" } else { "" }))
            .collect();
        parts.join(", ")
    }

    /// Textures in system memory (`slab::Mem::in_system`), none written by the GPU or holding
    /// GPU-written blocks, none with an upload pending (`refreshing`: its commit would find the
    /// image gone and release its waiters, a batch that bound it sampling what was there) or used
    /// in the last `PROMOTE_AGE` flips, evicted up to `room` bytes, most recently used first: they
    /// come back to video memory at their next use (`create_texture`, no pressure then).
    fn promote(&mut self, frame: u64, room: u64) -> Vec<Evicted> {
        let mut cands: Vec<(u64, ImageKey, u64)> = self
            .map
            .iter()
            .filter(|(k, i)| i.mem.in_system() && !i.gpu_written && i.gpu_parts.is_empty() && !self.refreshing.contains_key(*k) && frame.saturating_sub(i.used.get()) >= PROMOTE_AGE && !big_on_small_card(k))
            .map(|(k, i)| (i.used.get(), *k, i.bytes))
            .collect();
        cands.sort_unstable_by_key(|c| std::cmp::Reverse(c.0));
        let (mut out, mut moved) = (Vec::new(), 0u64);
        for (_, k, bytes) in cands {
            if moved + bytes > room {
                continue;
            }
            if let Some((e, b, _)) = self.drop_image(&k) {
                moved += b;
                out.push(e);
            }
        }
        out
    }

    pub fn evict(&mut self, frame: u64, want: u64) -> (Vec<Evicted>, u64, usize) {
        // (Not a texture holding blocks GPU work wrote through block views (`gpu_parts`): guest
        // memory never holds those, so a re-upload after eviction would lose them, as evicting the
        // images the GPU wrote lost 4-2's architecture, 2026-10-19. Added 2026-11-22 small hours;
        // the 7 GB tours could not measure it either way: TRIED.md.)
        let demote = under_pressure() && super::vk::system_can_take(1 << 20);
        let mut cands: Vec<(u64, ImageKey)> = self
            .map
            .iter()
            .filter(|(_, i)| !i.gpu_written && i.gpu_parts.is_empty() && frame.saturating_sub(i.used.get()) >= if !i.mem.in_system() && demote { EVICT_AGE_PRESSURE } else { EVICT_AGE })
            .map(|(k, i)| (i.used.get(), *k))
            .collect();
        cands.sort_unstable_by_key(|c| c.0);
        let (mut out, mut freed) = (Vec::new(), 0u64);
        for (_, k) in cands {
            if freed >= want {
                break;
            }
            if let Some((e, bytes, _)) = self.drop_image(&k) {
                // (An image in system memory frees none of the video memory wanted.)
                if !e.mem.in_system() {
                    freed += bytes;
                }
                out.push(e);
            }
        }
        EVICTED[0].fetch_add(out.len() as u64, std::sync::atomic::Ordering::Relaxed);
        EVICTED[1].fetch_add(freed, std::sync::atomic::Ordering::Relaxed);
        (out, freed, self.map.len())
    }

    /// Removes an image from the cache and every index naming it: (the
    /// image to destroy, its bytes, whether GPU-written).
    fn drop_image(&mut self, k: &ImageKey) -> Option<(Evicted, u64, bool)> {
        let img = self.map_remove(k)?;
        if let Some(v) = self.owned.get_mut(&k.base) {
            v.retain(|x| x != k);
            if v.is_empty() {
                self.owned.remove(&k.base);
            }
        }
        self.overwritten_keys.remove(k);
        if img.overwritten {
            self.pending_overwrites = self.pending_overwrites.saturating_sub(1);
        }
        self.pending_clears.remove(k);
        self.pending_change += 1;
        self.stencil_of.retain(|_, v| *v != *k);
        self.meta_of.retain(|_, v| v.0 != *k);
        if img.gpu_written && !self.by_base.get(&k.base).is_some_and(|ks| ks.iter().any(|x| self.map.get(x).is_some_and(|i| i.gpu_written))) {
            self.gpu_bases.remove(&k.base);
        }
        if let Some(id) = img.source {
            super::track::unwatch(id);
        }
        self.last_note = None;
        self.note_gen += 1;
        NOTE_GEN.fetch_add(1, std::sync::atomic::Ordering::Release);
        let views = img.views.values().copied().collect();
        Some((Evicted { image: img.image, mem: img.mem, views }, img.bytes, img.gpu_written))
    }

    /// Whether [base, base+len) is wholly the memory of GPU-written images
    /// that `zero_filled` clears, with no other image over any of it: a
    /// zero fill of the range then need not store into guest memory (`tune
    /// fill_skip`), whose bytes such images do not keep current anyway.
    pub fn fill_covered(&self, base: u64, len: u64) -> bool {
        if len == 0 || self.owned.is_empty() {
            return false;
        }
        let end = base.saturating_add(len);
        let mut spans: Vec<(u64, u64)> = Vec::new();
        for (k, img) in self.map.iter() {
            let (a, b) = (k.base, k.base.saturating_add(img.span.max(1)));
            if b <= base || a >= end {
                continue;
            }
            if !img.gpu_written || img.span == 0 || is_compressed(k.format) || !(base <= a && b <= end && len <= 2 * img.span + (64 << 10)) {
                return false;
            }
            spans.push((a, b));
        }
        spans.sort_unstable();
        let mut at = base;
        for (a, b) in spans {
            if a > at {
                return false;
            }
            at = at.max(b);
        }
        at >= end
    }

    /// Diagnosis: the images over [base, base+len) with what `fill_covered`
    /// checks (written by the GPU, span, compressed, 3D, layers).
    pub fn fill_cover_report(&self, base: u64, len: u64) -> String {
        let end = base.saturating_add(len);
        let parts: Vec<String> = self
            .map
            .iter()
            .filter(|(k, img)| {
                let b = k.base.saturating_add(img.span.max(1));
                k.base < end && b > base
            })
            .take(8)
            .map(|(k, img)| format!("{:#x} {}x{}x{} {:?}{} gpu_written {} span {:#x}", k.base, k.width, k.height, k.layers, k.format, if k.ty3d { " 3D" } else { "" }, img.gpu_written, img.span))
            .collect();
        if parts.is_empty() { "no image".into() } else { parts.join(", ") }
    }

    /// The guest bytes of the images among `keys` overwritten through a
    /// buffer (`reload_overwritten` reads them).
    pub fn overwritten_spans(&self, keys: &[ImageKey]) -> Vec<(u64, u64)> {
        keys.iter().filter_map(|k| self.map.get(k).filter(|i| i.overwritten && i.span > 0).map(|i| (k.base, i.span))).collect()
    }

    /// Whether any of `keys` was overwritten through a buffer.
    pub fn overwrites_pending(&self, keys: &[ImageKey]) -> bool {
        self.pending_overwrites > 0 && keys.iter().any(|k| self.map.get(k).is_some_and(|i| i.overwritten))
    }

    /// Gives the images among `keys` that `note_overwrites` marked their
    /// guest contents (the writers have completed): all-zero bytes clear the
    /// image, other contents are uploaded with the layout of its T#.
    pub fn reload_overwritten(&mut self, keys: &[ImageKey]) {
        if trace_clears() {
            for k in keys.iter().filter(|k| self.map.get(k).is_some_and(|i| i.overwritten)) {
                crate::info!("trace clears: {:#x} ({:?} {}x{}) reloaded from guest memory (overwritten; for flip {}, {})", k.base, k.format, k.width, k.height, crate::hle::libs::videoout::QUEUED_FLIP.load(std::sync::atomic::Ordering::Relaxed) + 1, std::thread::current().name().unwrap_or("?"));
            }
        }
        for k in keys {
            let Some(img) = self.map.get_mut(k) else { continue };
            if !img.overwritten {
                continue;
            }
            img.overwritten = false;
            self.overwritten_keys.remove(k);
            self.pending_change += 1;
            self.pending_overwrites = self.pending_overwrites.saturating_sub(1);
            self.note_gen += 1;
            NOTE_GEN.fetch_add(1, std::sync::atomic::Ordering::Release);
            let (image, span, layout) = (img.image, img.span, img.layout);
            if !super::cp::guest_readable(k.base, span) {
                self.report(format!("image {:#x} overwritten through a buffer: guest memory not readable", k.base));
                continue;
            }
            let bytes = unsafe { std::slice::from_raw_parts(k.base as *const u8, span as usize) };
            let zero = bytes.chunks(4096).all(|c| c.iter().fold(0u8, |a, &x| a | x) == 0);
            let how = if zero {
                let format = k.format;
                self.one_shot(|cmd| unsafe {
                    let d = &ctx().device;
                    let range = vk::ImageSubresourceRange { aspect_mask: aspects(format), base_mip_level: 0, level_count: vk::REMAINING_MIP_LEVELS, base_array_layer: 0, layer_count: vk::REMAINING_ARRAY_LAYERS };
                    let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
                    d.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
                    if is_depth(format) {
                        d.cmd_clear_depth_stencil_image(cmd, image, vk::ImageLayout::GENERAL, &vk::ClearDepthStencilValue { depth: 0.0, stencil: 0 }, &[range]);
                    } else {
                        d.cmd_clear_color_image(cmd, image, vk::ImageLayout::GENERAL, &vk::ClearColorValue::default(), &[range]);
                    }
                    let a = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
                    d.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[a], &[], &[]);
                });
                self.overwrite_clears += 1;
                "cleared (zeros)"
            } else if let (Some(t), false) = (layout, is_depth(k.format)) {
                if let Err(e) = self.upload(image, *k, &t) {
                    self.report(format!("image {:#x} overwritten through a buffer: upload failed: {e}", k.base));
                    continue;
                }
                self.overwrite_uploads += 1;
                "uploaded"
            } else {
                self.report(format!("image {:#x} ({:?}) overwritten through a buffer with non-zero data: not applied", k.base, k.format));
                continue;
            };
            static SEEN: Mutex<Option<std::collections::HashSet<u64>>> = Mutex::new(None);
            if SEEN.lock().get_or_insert_with(Default::default).insert(k.base) {
                crate::info!("images: {:#x} {}x{}x{} {:?} overwritten through a buffer view of its memory: {how}", k.base, k.width, k.height, k.layers, k.format);
            }
        }
    }

    /// Re-uploads a CPU-sourced texture whose guest bytes were written
    /// since its upload (write tracking: CPU faults, file reads, GPU writes).
    fn refresh(&mut self, key: ImageKey, t: &Tsharp, frame: u64) -> Result<(), String> {
        // Staging buffers of completed uploads go back at once (not only at
        // the next upload).
        if !self.inflight.is_empty() {
            self.reap(usize::MAX, u64::MAX);
        }
        let img = &self.map[&key];
        if img.gpu_written {
            return Ok(());
        }
        let (image, Some(region)) = (img.image, img.source) else { return Ok(()) };
        let (b, n) = super::track::range(region).unwrap_or((0, 0));
        // A re-upload of it left for after the lock (`RefreshJob`): waited
        // for there; its marks are not taken again meanwhile.
        if let Some(w) = self.refreshing.get(&key) {
            if REFRESH_DEFER.with(|d| d.get()) {
                REFRESH_OUT[3].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let w = w.clone();
                REFRESH_WAITS.with(|v| v.borrow_mut().push(w));
            }
            return Ok(());
        }
        if super::track::take_gpu_written(region) {
            // GPU buffer writes (e.g. per-frame clears): re-upload only if
            // the contents changed.
            let h = all_hashes(b, n);
            if h != self.map[&key].hashes {
                self.reuploads += 1;
                self.report(format!("texture {:#x} {}x{} re-uploaded (GPU write)", t.base, t.width, t.height));
                REUPLOAD_CAUSE.with(|c| c.set(5));
                // (`DS_SKIP_REUPLOADS`: timing only, the image keeps its contents.)
                let r = self.upload_parts(image, key, t, None, false);
                REUPLOAD_CAUSE.with(|c| c.set(0));
                r?;
                self.map.get_mut(&key).unwrap().hashes = h;
            }
        }
        if !super::track::protected() && !scrubbing() && self.map[&key].checked_frame != frame {
            // CPU writes: one slice of the sampled hash per frame.
            let img = self.map.get_mut(&key).unwrap();
            img.checked_frame = frame;
            let s = (frame as usize) % SLICES;
            if slice_hash(b, n, s) != img.hashes[s] {
                self.reuploads += 1;
                self.report(format!("texture {:#x} {}x{} re-uploaded (CPU write)", t.base, t.width, t.height));
                REUPLOAD_CAUSE.with(|c| c.set(4));
                // (`DS_SKIP_REUPLOADS`: timing only, the image keeps its contents.)
                let r = self.upload_parts(image, key, t, Some(&[]), false);
                REUPLOAD_CAUSE.with(|c| c.set(0));
                r?;
                self.map.get_mut(&key).unwrap().hashes = all_hashes(b, n);
            }
        }
        if let Some((why, parts)) = super::track::take_dirty(region) {
            // Re-armed first: writes during the upload mark it dirty again.
            super::track::rearm(region);
            self.reuploads += 1;
            if self.reuploads <= 12 {
                crate::info!(
                    "images: re-upload of {:#x} ({}x{}x{} format {} type {} sw_mode {} levels {} key {:?}): {why:?}",
                    t.base, t.width, t.height, t.depth, t.format, t.ty, t.sw_mode, t.max_mip + 1, key
                );
            }
            let msg = format!("texture {:#x} {}x{} re-uploaded ({})", t.base, t.width, t.height, match why {
                super::track::Why::Fault(_) => "CPU write",
                super::track::Why::HostWrite(_) => "file read or import",
                super::track::Why::GpuWrite(..) => "GPU write",
                super::track::Why::Remap(..) => "mapping change",
                super::track::Why::Unmapped => "unmapped",
                super::track::Why::Forced => "requested",
                super::track::Why::Sampled => "CPU write, sampled",
                super::track::Why::Written(_) => "CPU write",
                super::track::Why::None => "?",
            });
            self.report(msg);
            let t0 = std::time::Instant::now();
            REUPLOAD_CAUSE.with(|c| {
                c.set(match why {
                    super::track::Why::Remap(_, _, 0) => 1,
                    super::track::Why::Remap(..) => 2,
                    super::track::Why::HostWrite(_) => 3,
                    super::track::Why::Fault(_) | super::track::Why::Written(_) | super::track::Why::Sampled => 4,
                    super::track::Why::GpuWrite(..) => 5,
                    _ => 6,
                })
            });
            // Detiled after the lock (CPU-detiled re-uploads), the copy
            // recorded under it again.
            if REFRESH_DEFER.with(|d| d.get()) && reupload_on_cpu(&key, t) {
                let cause = REUPLOAD_CAUSE.with(|c| c.replace(0));
                let wait = std::sync::Arc::new(RefreshWait::default());
                self.refreshing.insert(key, wait.clone());
                REFRESH_OUT[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let remap = match why {
                    super::track::Why::Remap(rb, rl, op) => Some((rb, rl, op)),
                    _ => None,
                };
                let all = matches!(why, super::track::Why::Forced);
                let keep = self.map[&key].gpu_parts.clone();
                REFRESH_JOBS.with(|j| j.borrow_mut().push(RefreshJob { key, image, t: *t, region, range: (b, n), parts, all, cause, remap, wait, first: false, keep }));
                return Ok(());
            }
            // (`DS_SKIP_REUPLOADS`: timing only, the image keeps its contents.)
            let r = self.upload_parts(image, key, t, parts.as_deref(), matches!(why, super::track::Why::Forced));
            REUPLOAD_CAUSE.with(|c| c.set(0));
            r?;
            let h = all_hashes(b, n);
            if let super::track::Why::Remap(rb, rl, op) = why {
                use std::sync::atomic::Ordering::Relaxed;
                let s = &REMAP_UPLOADS[(op as usize).min(2)];
                s[0].fetch_add(1, Relaxed);
                if h == self.map[&key].hashes {
                    s[1].fetch_add(1, Relaxed);
                }
                if rb > b || rb.saturating_add(rl) < b + n {
                    s[2].fetch_add(1, Relaxed);
                }
                s[3].fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
            }
            self.map.get_mut(&key).unwrap().hashes = h;
        }
        Ok(())
    }

    /// A deferred re-upload's commit (`run_refresh`): its copy recorded into
    /// the uploads' shared command buffer and its hashes stored, unless its
    /// image was replaced, dropped or written by the GPU meanwhile.
    fn commit_refresh(&mut self, job: &RefreshJob, prepared: Result<Prepared, String>, hashes: Option<[u64; SLICES]>, t0: std::time::Instant) -> Result<(), String> {
        use std::sync::atomic::Ordering::Relaxed;
        self.refreshing.remove(&job.key);
        let out = if job.first { &FIRST_OUT[1..4] } else { &REFRESH_OUT[1..3] };
        let p = match prepared {
            Ok(p) => p,
            // A first upload that failed leaves its image as the upload
            // under the lock would (reported, cleared, in its general layout).
            Err(e) if job.first => {
                FIRST_OUT[3].fetch_add(1, Relaxed);
                if self.map.get(&job.key).is_some_and(|i| i.image == job.image) {
                    self.report(format!("texture {:#x} ({}x{} format {} sw_mode {}): {e}; contents undefined", job.t.base, job.t.width, job.t.height, job.t.format, job.t.sw_mode));
                    self.to_general(job.image, job.key.format);
                }
                return Ok(());
            }
            Err(e) => return Err(e),
        };
        for n in p.notes {
            self.report(n);
        }
        // (A first upload lands even when GPU work bound the image for
        // writing meanwhile, e.g. as a storage image in the same stage: that
        // work was recorded after it and waits for its copy.)
        let live = self.map.get(&job.key).is_some_and(|i| i.image == job.image && (job.first || !i.gpu_written));
        if let Some((st, regions, _)) = p.copy {
            if !live {
                out[1].fetch_add(1, Relaxed);
                free_batched(vec![(st.buf, st.mem)]);
                return Ok(());
            }
            let (image, buf, base, first) = (job.image, st.buf, job.t.base, job.first);
            let record = |cmd: vk::CommandBuffer| record_upload(cmd, image, first, buf, &regions, base);
            if let Err(e) = record_batched((st.buf, st.mem), st.cap, &record) {
                static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                if !WARNED.swap(true, Relaxed) {
                    crate::warn!("images: upload not batched: {e}");
                }
                self.one_shot(record);
                free_batched(vec![(st.buf, st.mem)]);
            }
            out[0].fetch_add(1, Relaxed);
            self.uploads += 1;
        } else if !live {
            out[1].fetch_add(1, Relaxed);
            return Ok(());
        }
        self.upload_bytes += p.size;
        self.upload_ns += t0.elapsed().as_nanos() as u64;
        if let Some(h) = hashes {
            if let Some((rb, rl, op)) = job.remap {
                let s = &REMAP_UPLOADS[(op as usize).min(2)];
                s[0].fetch_add(1, Relaxed);
                if h == self.map[&job.key].hashes {
                    s[1].fetch_add(1, Relaxed);
                }
                let (b, n) = job.range;
                if rb > b || rb.saturating_add(rl) < b + n {
                    s[2].fetch_add(1, Relaxed);
                }
                s[3].fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
            }
            self.map.get_mut(&job.key).unwrap().hashes = h;
        }
        Ok(())
    }

    /// The guest ranges of the next `max` uploaded textures not marked
    /// written yet (write-watch scrubbing).
    fn scrub_regions(&mut self, max: usize) -> Vec<(u64, u64)> {
        let g = LOOKUP_GEN.load(std::sync::atomic::Ordering::Acquire);
        if self.scrub_gen != g || self.scrub_keys.is_empty() {
            self.scrub_keys = self.map.iter().filter(|(_, i)| !i.gpu_written && i.source.is_some()).map(|(k, _)| *k).collect();
            self.scrub_gen = g;
        }
        let n = self.scrub_keys.len();
        let mut out = Vec::new();
        for _ in 0..max.min(n) {
            self.scrub_at = (self.scrub_at + 1) % n;
            let Some(img) = self.map.get(&self.scrub_keys[self.scrub_at]) else { continue };
            let (false, Some(region)) = (img.gpu_written, img.source) else { continue };
            if super::track::is_dirty(region) {
                continue;
            }
            if let Some(r) = super::track::range(region) {
                out.push(r);
            }
        }
        out
    }

    /// The next `max` uploaded textures to sample: (tracking region, guest
    /// range, slice, the slice's hash at upload); each visit samples the
    /// texture's next slice (`checked_frame` counts them while scrubbing).
    fn scrub_batch(&mut self, max: usize) -> Vec<(u32, u64, u64, usize, u64)> {
        let g = LOOKUP_GEN.load(std::sync::atomic::Ordering::Acquire);
        if self.scrub_gen != g || self.scrub_keys.is_empty() {
            self.scrub_keys = self.map.iter().filter(|(_, i)| !i.gpu_written && i.source.is_some()).map(|(k, _)| *k).collect();
            self.scrub_gen = g;
        }
        let n = self.scrub_keys.len();
        let mut out = Vec::new();
        for _ in 0..max.min(n) {
            self.scrub_at = (self.scrub_at + 1) % n;
            let k = self.scrub_keys[self.scrub_at];
            // (An upload left for after the lock stores its hashes later.)
            if self.refreshing.contains_key(&k) {
                continue;
            }
            let Some(img) = self.map.get_mut(&k) else { continue };
            let (false, Some(region)) = (img.gpu_written, img.source) else { continue };
            let s = (img.checked_frame % SLICES as u64) as usize;
            img.checked_frame += 1;
            let stored = img.hashes[s];
            // Already marked: re-uploaded at its next use (textures whose
            // memory streaming reused are often never used again).
            if super::track::is_dirty(region) {
                continue;
            }
            if let Some((b, len)) = super::track::range(region) {
                out.push((region, b, len, s, stored));
            }
        }
        out
    }

    /// Adds an image (keeps the by-address index).
    fn map_insert(&mut self, key: ImageKey, img: Image) {
        if img.gpu_written || self.gpu_bases.contains(&key.base) {
            gpu_images_changed();
        }
        IMAGE_BYTES.fetch_add(img.bytes, std::sync::atomic::Ordering::Relaxed);
        base_gen(key.base).fetch_add(1, std::sync::atomic::Ordering::Release);
        match self.map.insert(key, img) {
            None => self.by_base.entry(key.base).or_default().push(key),
            Some(old) => {
                // Replaced: destroyed once no batch can name it.
                IMAGE_BYTES.fetch_sub(old.bytes, std::sync::atomic::Ordering::Relaxed);
                super::exec::bury_images(vec![Evicted { image: old.image, mem: old.mem, views: old.views.values().copied().collect() }]);
            }
        }
        LOOKUP_GEN.fetch_add(1, std::sync::atomic::Ordering::Release);
    }

    fn map_remove(&mut self, key: &ImageKey) -> Option<Image> {
        let img = self.map.remove(key)?;
        base_gen(key.base).fetch_add(1, std::sync::atomic::Ordering::Release);
        IMAGE_BYTES.fetch_sub(img.bytes, std::sync::atomic::Ordering::Relaxed);
        LOOKUP_GEN.fetch_add(1, std::sync::atomic::Ordering::Release);
        if img.gpu_written || self.gpu_bases.contains(&key.base) {
            gpu_images_changed();
        }
        if let Some(v) = self.by_base.get_mut(&key.base) {
            v.retain(|k| k != key);
            if v.is_empty() {
                self.by_base.remove(&key.base);
            }
        }
        Some(img)
    }

    /// An existing image at `key`'s address with the same size that can be
    /// viewed in `key`'s format (optionally only GPU-written ones).
    fn alias(&self, key: &ImageKey, gpu_written_only: bool) -> Option<ImageKey> {
        self.by_base
            .get(&key.base)?
            .iter()
            .filter(|k| {
                self.map.get(k).is_some_and(|i| !gpu_written_only || i.gpu_written)
                    && k.width == key.width
                    && k.height == key.height
                    && k.layers >= key.layers
                    && k.levels >= key.levels
                    && k.ty3d == key.ty3d
                    && view_compatible(k.format, key.format)
            })
            .copied()
            .max_by_key(|k| (k.format == key.format, k.levels, k.layers))
    }

    /// Image for a render target or depth surface: the image of that exact
    /// surface, an alias of it (same address and size, compatible format), or
    /// a new zero-filled image. Marked GPU-written.
    fn surface_image(&mut self, s: &Surface) -> Result<ImageKey, String> {
        let key = ImageKey { base: s.base, format: s.format, width: s.width, height: s.height, layers: s.layers.max(1), levels: s.levels.max(1), ty3d: s.ty3d, cube: false };
        let k = if self.map.contains_key(&key) {
            key
        } else if let Some(k) = self.alias(&key, false) {
            k
        } else {
            let (image, mem, bytes) = self.create_in(key, big_on_small_card(&key))?;
            self.to_general(image, key.format);
            self.map_insert(key, Image { image, key, views: HashMap::default(), gpu_written: true, source: None, source_gen: None, hashes: [0; SLICES], checked_frame: 0, write_gen: 0, block_gen: 0, gpu_parts: Vec::new(), mip_gens: Vec::new(), span: 0, layout: None, overwritten: false, writer: 0, mem, bytes, used: UseStamp::new(super::FRAMES.load(std::sync::atomic::Ordering::Relaxed)) });
            self.gpu_bases.insert(key.base);
            crate::info!("images: new render target {:#x} {}x{} {:?} at flip {}", key.base, key.width, key.height, key.format, super::FRAMES.load(std::sync::atomic::Ordering::Relaxed));
            // DS_CAPTURE_NEW_RT=<hex base>: capture the frame after a render
            // target at or above that address is created.
            static TRIGGER: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
            if let Some(t) = *TRIGGER.get_or_init(|| std::env::var("DS_CAPTURE_NEW_RT").ok().and_then(|v| u64::from_str_radix(v.trim_start_matches("0x"), 16).ok())) {
                if key.base >= t {
                    super::capture::arm_next();
                }
            }
            key
        };
        let img = self.map.get_mut(&k).unwrap();
        if !img.gpu_written {
            crate::info!("images: {:#x} {}x{} {:?} becomes a render target", k.base, k.width, k.height, k.format);
        }
        // GPU-owned from now on: guest memory is no longer its source.
        if !img.gpu_written {
            LOOKUP_GEN.fetch_add(1, std::sync::atomic::Ordering::Release);
            gpu_images_changed();
        }
        img.gpu_written = true;
        self.gpu_bases.insert(k.base);
        // Generations move only on a change (every draw with a depth/stencil
        // target came here: lookups cached by generation never hit).
        if s.stencil_base != 0 && has_stencil(k.format) {
            let old = self.stencil_of.insert(s.stencil_base, k);
            if old != Some(k) {
                LOOKUP_GEN.fetch_add(1, std::sync::atomic::Ordering::Release);
                gpu_images_changed();
                self.gpu_bases.insert(s.stencil_base);
                // (Diagnosis: a stencil plane's owner changing; the first 400 logged.)
                static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 400 {
                    crate::info!("images: stencil plane {:#x} now of {:#x} {}x{} {:?} (was {}) at flip {} on {}", s.stencil_base, k.base, k.width, k.height, k.format, old.map_or("none".to_string(), |o| format!("{:#x} {}x{}", o.base, o.width, o.height)), super::FRAMES.load(std::sync::atomic::Ordering::Relaxed), crate::log::thread_name());
                }
            }
        }
        if s.meta != 0 {
            mark_note_pages(s.meta, 4);
        }
        self.meta_now.insert(k, (s.meta, super::FRAMES.load(std::sync::atomic::Ordering::Relaxed)));
        // `DS_TRACE_LAYER_BINDS=1` (diagnosis): binds of layered depth surfaces, each distinct
        // (base, slice, layers, metadata) once, with the flip.
        static TRACE_LAYER_BINDS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if k.layers > 1 && is_depth(k.format) && *TRACE_LAYER_BINDS.get_or_init(|| std::env::var("DS_TRACE_LAYER_BINDS").ok().as_deref() == Some("1")) {
            static SEEN: Mutex<Option<std::collections::HashSet<(u64, u32, u32, u64)>>> = Mutex::new(None);
            if SEEN.lock().get_or_insert_with(Default::default).insert((s.base, s.slice, s.layers, s.meta)) {
                crate::info!("layer bind: {:#x} {}x{} {:?} slice {} of {} (image {} layers), metadata {:#x}, flip {}", s.base, s.width, s.height, s.format, s.slice, s.layers, k.layers, s.meta, super::FRAMES.load(std::sync::atomic::Ordering::Relaxed));
            }
        }
        if s.meta != 0 && bind_clear_metadata(&mut self.meta_of, s.meta, (k, s.clear)) {
            self.note_gen += 1;
            NOTE_GEN.fetch_add(1, std::sync::atomic::Ordering::Release);
        }
        img.source = None;
        img.source_gen = None;
        self.index_owned(k);
        Ok(k)
    }

    /// Attachment for a color or depth/stencil surface (one mip level, one layer).
    pub fn target(&mut self, s: &Surface) -> Result<Target, String> {
        let _cpu = crate::gpu::cpu_profile::scope(crate::gpu::cpu_profile::Part::ImageLookup);
        let k = self.surface_image(s)?;
        let img = self.map.get_mut(&k).unwrap();
        img.used.stamp(super::FRAMES.load(std::sync::atomic::Ordering::Relaxed));
        let mip = s.mip.min(img.key.levels - 1);
        let layer = if img.key.ty3d { 0 } else { s.slice.min(img.key.layers - 1) };
        let vkey = (s.format, vk::ImageViewType::TYPE_2D, mip, 1, layer, 1, [4u8, 5, 6, 7], 2u8);
        let view = match img.views.get(&vkey) {
            Some(&v) => v,
            None => {
                let depth = is_depth(s.format);
                let ci = vk::ImageViewCreateInfo::default()
                    .image(img.image)
                    .view_type(if img.key.ty3d { vk::ImageViewType::TYPE_2D_ARRAY } else { vk::ImageViewType::TYPE_2D })
                    .format(s.format)
                    .subresource_range(vk::ImageSubresourceRange { aspect_mask: aspects(s.format), base_mip_level: mip, level_count: 1, base_array_layer: layer, layer_count: 1 });
                let mut usage = vk::ImageViewUsageCreateInfo::default().usage(if depth { vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT } else { vk::ImageUsageFlags::COLOR_ATTACHMENT });
                let ci = if depth { ci } else { ci.push_next(&mut usage) };
                let v = unsafe { ctx().device.create_image_view(&ci, None) }.map_err(|e| format!("attachment view {vkey:?}: {e}"))?;
                img.views.insert(vkey, v);
                v
            }
        };
        Ok(Target { key: k, image: img.image, view, used: img.used, format: s.format, width: (img.key.width >> mip).max(1), height: (img.key.height >> mip).max(1) })
    }

    /// Every image at `base` (diagnosis of the image a display buffer
    /// presents): format, size, layers, GPU-written, frame of last use.
    pub fn at_base_report(&self, base: u64) -> String {
        let mut v: Vec<String> = self
            .map
            .values()
            .filter(|i| i.key.base == base)
            .map(|i| format!("{:?} {}x{}x{} gpu {} used {}", i.key.format, i.key.width, i.key.height, i.key.layers, i.gpu_written, i.used.get()))
            .collect();
        v.sort();
        v.join("; ")
    }

    /// Every fast-clear metadata address and its surface (`metamap`, diagnosis): the image's
    /// key, whether GPU-written, its last use, the metadata it was last bound with and when.
    pub fn meta_report(&self) -> Vec<String> {
        self.meta_of
            .iter()
            .map(|(&meta, &(k, clear))| {
                let (used, gpu) = self.map.get(&k).map(|i| (i.used.get(), i.gpu_written)).unwrap_or((0, false));
                let (now, at) = self.meta_now.get(&k).copied().unwrap_or((0, 0));
                format!(
                    "{meta:#x} -> {:#x} {}x{}x{} {:?} gpu {gpu} used {used}; bound last with {now:#x} at flip {at}{}; clear {:#x}/{:#x}",
                    k.base, k.width, k.height, k.layers, k.format, if now != meta { " (another)" } else { "" }, clear[0], clear[1]
                )
            })
            .collect()
    }

    /// Removes the surface of a metadata address and its pending clear (`metadrop`, diagnosis).
    pub fn meta_drop(&mut self, meta: u64) -> Option<String> {
        let (k, _) = self.meta_of.remove(&meta)?;
        if self.pending_clears.get(&k).is_some_and(|c| c.0 == meta) {
            self.pending_clears.remove(&k);
        }
        self.pending_change += 1;
        self.note_gen += 1;
        NOTE_GEN.fetch_add(1, std::sync::atomic::Ordering::Release);
        Some(format!("{meta:#x}: {:#x} {}x{} {:?}", k.base, k.width, k.height, k.format))
    }

    /// `imgprobe` (diagnosis): the mean of each channel of a 4x4 block at (u, v) (0-1 of the
    /// image's size) in every GPU-written 2D image used in the last two frames, read in one copy
    /// (depth images: their depth plane). One line per image: base, size, format, the means.
    /// `atlasprobe` (diagnosis): each layer of the largest GPU-written image with 4-byte texels at
    /// `base`, every 16th row read in one copy: per layer its texels not zero and an FNV hash of
    /// the words read (`L<k> <covered> <hash>`; layers with none covered counted, not listed).
    pub fn probe_layers(&mut self, base: u64) -> String {
        let pick = self
            .map
            .values()
            .filter(|i| i.key.base == base && i.gpu_written && !i.key.ty3d && matches!(i.key.format, vk::Format::D32_SFLOAT | vk::Format::D32_SFLOAT_S8_UINT | vk::Format::R32_SFLOAT | vk::Format::R32_UINT))
            .max_by_key(|i| (i.key.layers, i.key.width))
            .map(|i| (i.key, i.image));
        let Some((k, image)) = pick else { return "no such image".into() };
        let layers = k.layers.clamp(1, 128);
        let rows: Vec<u32> = (0..k.height).step_by(16).collect();
        let row_bytes = 4 * k.width as u64;
        let per_layer = row_bytes * rows.len() as u64;
        let size = per_layer * layers as u64;
        let c = ctx();
        let Ok((buf, mem, ptr)) = c.host_buffer(size, vk::BufferUsageFlags::TRANSFER_DST) else { return "no staging buffer".into() };
        let aspect = if is_depth(k.format) { vk::ImageAspectFlags::DEPTH } else { vk::ImageAspectFlags::COLOR };
        let mut regions = Vec::with_capacity(rows.len() * layers as usize);
        for l in 0..layers {
            for (r, &y) in rows.iter().enumerate() {
                regions.push(
                    vk::BufferImageCopy::default()
                        .buffer_offset(l as u64 * per_layer + r as u64 * row_bytes)
                        .image_subresource(vk::ImageSubresourceLayers { aspect_mask: aspect, mip_level: 0, base_array_layer: l, layer_count: 1 })
                        .image_offset(vk::Offset3D { x: 0, y: y as i32, z: 0 })
                        .image_extent(vk::Extent3D { width: k.width, height: 1, depth: 1 }),
                );
            }
        }
        self.one_shot(|cmd| unsafe {
            let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
            c.device.cmd_copy_image_to_buffer(cmd, image, vk::ImageLayout::GENERAL, buf, &regions);
            let hb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[hb], &[], &[]);
        });
        let raw = unsafe { std::slice::from_raw_parts(ptr, size as usize).to_vec() };
        unsafe {
            c.device.destroy_buffer(buf, None);
            c.device.unmap_memory(mem);
            c.device.free_memory(mem, None);
        }
        let mut items = Vec::new();
        let mut empty = 0u32;
        for l in 0..layers as u64 {
            let s = &raw[(l * per_layer) as usize..((l + 1) * per_layer) as usize];
            let mut covered = 0u32;
            let mut h: u32 = 0x811c_9dc5;
            for w in s.chunks_exact(4) {
                let v = u32::from_le_bytes([w[0], w[1], w[2], w[3]]);
                covered += (v != 0) as u32;
                h = (h ^ v).wrapping_mul(0x0100_0193);
            }
            if covered == 0 {
                empty += 1;
                continue;
            }
            items.push(format!("L{l} {covered} {h:08x}"));
        }
        format!("{:#x} {}x{}x{} {:?}: {} ({empty} layers empty)", k.base, k.width, k.height, k.layers, k.format, items.join(", "))
    }

    /// Diagnosis (`DS_ATPROBE_MEM`): `len` bytes of `buf` from `off` as the GPU reads them, copied
    /// after the work submitted before.
    pub fn read_buffer(&mut self, buf: vk::Buffer, off: u64, len: u64) -> Option<Vec<u8>> {
        let c = ctx();
        let (hb, mem, ptr) = c.host_buffer(len, vk::BufferUsageFlags::TRANSFER_DST).ok()?;
        self.one_shot(|cmd| unsafe {
            let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
            c.device.cmd_copy_buffer(cmd, buf, hb, &[vk::BufferCopy { src_offset: off, dst_offset: 0, size: len }]);
            let hm = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[hm], &[], &[]);
        });
        let raw = unsafe { std::slice::from_raw_parts(ptr, len as usize).to_vec() };
        unsafe {
            c.device.destroy_buffer(hb, None);
            c.device.unmap_memory(mem);
            c.device.free_memory(mem, None);
        }
        Some(raw)
    }

    /// `DS_ATPROBE_DUMP=<layer>:<w>:<h>` (diagnosis, 2026-11-20): a w x h region of one layer of
    /// the GPU-written 4-byte-texel image at `base`, centred at (u, v), as the GPU holds it now
    /// (texels row by row), for `atprobe`'s per-frame dumps.
    pub fn dump_region(&mut self, base: u64, layer: u32, u: f32, v: f32, w: u32, h: u32) -> Option<Vec<u8>> {
        use vk::Format as F;
        let i = self.map.values().filter(|i| i.key.base == base && i.gpu_written && !i.key.ty3d && layer < i.key.layers.max(1)).max_by_key(|i| i.used.get())?;
        let k = i.key;
        if !matches!(k.format, F::R32_UINT | F::R32_SFLOAT | F::R8G8B8A8_UNORM | F::B8G8R8A8_UNORM | F::A2B10G10R10_UNORM_PACK32 | F::B10G11R11_UFLOAT_PACK32) {
            return None;
        }
        let img = i.image;
        let (w, h) = (w.clamp(1, k.width), h.clamp(1, k.height));
        let x = ((u.clamp(0.0, 1.0) * k.width as f32) as u32).saturating_sub(w / 2).min(k.width - w);
        let y = ((v.clamp(0.0, 1.0) * k.height as f32) as u32).saturating_sub(h / 2).min(k.height - h);
        let size = 4 * w as u64 * h as u64;
        let c = ctx();
        let (buf, mem, ptr) = c.host_buffer(size, vk::BufferUsageFlags::TRANSFER_DST).ok()?;
        self.one_shot(|cmd| unsafe {
            let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
            let region = vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: layer, layer_count: 1 })
                .image_offset(vk::Offset3D { x: x as i32, y: y as i32, z: 0 })
                .image_extent(vk::Extent3D { width: w, height: h, depth: 1 });
            c.device.cmd_copy_image_to_buffer(cmd, img, vk::ImageLayout::GENERAL, buf, &[region]);
            let hb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[hb], &[], &[]);
        });
        let raw = unsafe { std::slice::from_raw_parts(ptr, size as usize).to_vec() };
        unsafe {
            c.device.destroy_buffer(buf, None);
            c.device.unmap_memory(mem);
            c.device.free_memory(mem, None);
        }
        Some(raw)
    }

    pub fn probe_used(&mut self, u: f32, v: f32) -> Vec<String> {
        use vk::Format as F;
        let frame = super::FRAMES.load(std::sync::atomic::Ordering::Relaxed);
        // (Bytes a texel takes in the copy, channels, decoder kind.)
        let shape = |f: vk::Format| -> Option<(u64, usize, u8)> {
            Some(match f {
                F::R16G16B16A16_SFLOAT => (8, 4, 1),
                F::R16G16_SFLOAT => (4, 2, 1),
                F::R16_SFLOAT => (2, 1, 1),
                F::R32G32B32A32_SFLOAT => (16, 4, 2),
                F::R32G32_SFLOAT => (8, 2, 2),
                F::R32_SFLOAT | F::D32_SFLOAT | F::D32_SFLOAT_S8_UINT => (4, 1, 2),
                F::R8G8B8A8_UNORM | F::B8G8R8A8_UNORM | F::R8G8B8A8_SRGB | F::B8G8R8A8_SRGB => (4, 4, 3),
                F::R8G8_UNORM => (2, 2, 3),
                F::R8_UNORM | F::R8_UINT => (1, 1, 3),
                F::R16_UNORM => (2, 1, 4),
                F::R16G16_UNORM => (4, 2, 4),
                F::B10G11R11_UFLOAT_PACK32 => (4, 3, 5),
                F::A2B10G10R10_UNORM_PACK32 => (4, 4, 6),
                F::R32_UINT => (4, 1, 7),
                F::R32G32_UINT => (8, 2, 7),
                _ => return None,
            })
        };
        // (Each of the first 16 layers of a layered image: `layer` the eighth field.)
        let mut picks: Vec<(ImageKey, vk::Image, u32, u32, u32, u32, u64, (u64, usize, u8), u32)> = Vec::new();
        let mut size = 0u64;
        let mut keys: Vec<&Image> = self.map.values().filter(|i| i.gpu_written && !i.key.ty3d && frame.saturating_sub(i.used.get()) <= 1).collect();
        keys.sort_by_key(|i| (i.key.base, i.key.format.as_raw(), i.key.width));
        for i in keys {
            let k = i.key;
            let Some(sh) = shape(k.format) else { continue };
            let (bw, bh) = (4u32.min(k.width), 4u32.min(k.height));
            let x = ((u.clamp(0.0, 1.0) * k.width as f32) as u32).min(k.width - bw);
            let y = ((v.clamp(0.0, 1.0) * k.height as f32) as u32).min(k.height - bh);
            for layer in 0..k.layers.clamp(1, 16) {
                picks.push((k, i.image, x, y, bw, bh, size, sh, layer));
                size += (sh.0 * (bw * bh) as u64 + 15) & !15;
            }
        }
        if picks.is_empty() {
            return Vec::new();
        }
        let c = ctx();
        let Ok((buf, mem, ptr)) = c.host_buffer(size, vk::BufferUsageFlags::TRANSFER_DST) else { return vec!["no staging buffer".into()] };
        self.one_shot(|cmd| unsafe {
            let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
            for p in &picks {
                let aspect = if is_depth(p.0.format) { vk::ImageAspectFlags::DEPTH } else { vk::ImageAspectFlags::COLOR };
                let region = vk::BufferImageCopy::default()
                    .buffer_offset(p.6)
                    .image_subresource(vk::ImageSubresourceLayers { aspect_mask: aspect, mip_level: 0, base_array_layer: p.8, layer_count: 1 })
                    .image_offset(vk::Offset3D { x: p.2 as i32, y: p.3 as i32, z: 0 })
                    .image_extent(vk::Extent3D { width: p.4, height: p.5, depth: 1 });
                c.device.cmd_copy_image_to_buffer(cmd, p.1, vk::ImageLayout::GENERAL, buf, &[region]);
            }
            let hb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[hb], &[], &[]);
        });
        let raw = unsafe { std::slice::from_raw_parts(ptr, size as usize).to_vec() };
        unsafe {
            c.device.destroy_buffer(buf, None);
            c.device.unmap_memory(mem);
            c.device.free_memory(mem, None);
        }
        let mut out = Vec::with_capacity(picks.len());
        for (k, _, _, _, bw, bh, off, (tb, ch, kind), layer) in picks {
            let n = (bw * bh) as usize;
            let mut sum = [0f64; 4];
            for t in 0..n {
                let p = &raw[off as usize + t * tb as usize..off as usize + (t + 1) * tb as usize];
                let word = |o: usize| u32::from_le_bytes([p[o], p[o + 1], p[o + 2], p[o + 3]]);
                for (c, s) in sum.iter_mut().enumerate().take(ch) {
                    *s += match kind {
                        1 => half_to_f32(u16::from_le_bytes([p[2 * c], p[2 * c + 1]])) as f64,
                        2 => f32::from_le_bytes([p[4 * c], p[4 * c + 1], p[4 * c + 2], p[4 * c + 3]]) as f64,
                        3 => p[c] as f64 / 255.0,
                        4 => u16::from_le_bytes([p[2 * c], p[2 * c + 1]]) as f64 / 65535.0,
                        5 => {
                            // B10G11R11: R 11 bits, G 11, B 10 (5-bit exponents, no sign).
                            let w = word(0);
                            let (bits, shift, mant) = [(11u32, 0u32, 6u32), (11, 11, 6), (10, 22, 5)][c];
                            let x = (w >> shift) & ((1 << bits) - 1);
                            let (e, m) = (x >> mant, x & ((1 << mant) - 1));
                            if e == 0 { (m as f64 / (1u64 << mant) as f64) * 2f64.powi(-14) } else { (1.0 + m as f64 / (1u64 << mant) as f64) * 2f64.powi(e as i32 - 15) }
                        }
                        6 => {
                            let w = word(0);
                            if c < 3 { ((w >> (10 * c)) & 1023) as f64 / 1023.0 } else { (w >> 30) as f64 / 3.0 }
                        }
                        _ => word(4 * c) as f64,
                    };
                }
            }
            let means: Vec<String> = sum.iter().take(ch).map(|s| format!("{:.4}", s / n as f64)).collect();
            let at = if k.layers > 1 { format!(" layer {layer}") } else { String::new() };
            out.push(format!("{:#x} {}x{}x{}{at} {:?}: {}", k.base, k.width, k.height, k.layers, k.format, means.join(" ")));
        }
        out
    }

    /// The GPU image last rendered at `base` (display buffers).
    /// `size`: the size the buffer is registered with: an image of that
    /// size first (the title draws a display buffer at another size after
    /// it changes its display mode, and the larger image of the old size
    /// stays at the address, no longer drawn), else the largest.
    pub fn rendered_at(&self, base: u64, size: Option<(u32, u32)>) -> Option<(vk::Image, vk::Format, u32, u32)> {
        let at = || self.map.values().filter(|i| i.key.base == base && i.gpu_written && !is_depth(i.key.format));
        let area = |i: &&Image| i.key.width as u64 * i.key.height as u64;
        size.and_then(|(w, h)| at().filter(|i| (i.key.width, i.key.height) == (w, h)).max_by_key(area))
            .or_else(|| at().max_by_key(area))
            .map(|i| {
                // Presented: in use even when not rendered again (a static
                // screen flipped as it is), so never evicted meanwhile.
                i.used.stamp(super::FRAMES.load(std::sync::atomic::Ordering::Relaxed));
                (i.image, i.key.format, i.key.width, i.key.height)
            })
    }

    /// Reads mip 0 / layer 0 of a color image back (tightly packed texels).
    pub fn read_back(&mut self, image: vk::Image, format: vk::Format, width: u32, height: u32) -> Result<Vec<u8>, String> {
        if texel_bytes(format) == 0 || is_compressed(format) || is_depth(format) {
            return Err(format!("read back of {format:?} unsupported"));
        }
        self.read_back_aspect(image, format, vk::ImageAspectFlags::COLOR, width, height)
    }

    /// One texel at (x, y) of mip 0 / layer 0 of one aspect, in the layout
    /// of `read_back_aspect` (pixel watches of frame captures).
    pub fn read_back_texel(&mut self, image: vk::Image, format: vk::Format, aspect: vk::ImageAspectFlags, x: u32, y: u32) -> Result<Vec<u8>, String> {
        let bpp = if aspect == vk::ImageAspectFlags::STENCIL {
            1
        } else if aspect == vk::ImageAspectFlags::DEPTH {
            if matches!(format, vk::Format::D16_UNORM | vk::Format::D16_UNORM_S8_UINT) { 2 } else { 4 }
        } else {
            texel_bytes(format) as u64
        };
        if bpp == 0 || is_compressed(format) {
            return Err(format!("texel read back of {format:?} unsupported"));
        }
        let c = ctx();
        let (buf, mem, ptr) = c.host_buffer(bpp.max(4), vk::BufferUsageFlags::TRANSFER_DST)?;
        self.one_shot(|cmd| unsafe {
            let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
            let region = vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers { aspect_mask: aspect, mip_level: 0, base_array_layer: 0, layer_count: 1 })
                .image_offset(vk::Offset3D { x: x as i32, y: y as i32, z: 0 })
                .image_extent(vk::Extent3D { width: 1, height: 1, depth: 1 });
            c.device.cmd_copy_image_to_buffer(cmd, image, vk::ImageLayout::GENERAL, buf, &[region]);
            let hb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[hb], &[], &[]);
        });
        let out = unsafe { std::slice::from_raw_parts(ptr, bpp as usize).to_vec() };
        unsafe {
            c.device.destroy_buffer(buf, None);
            c.device.unmap_memory(mem);
            c.device.free_memory(mem, None);
        }
        Ok(out)
    }

    /// Reads mip 0 / layer 0 of one aspect back: color texels, depth as 32-bit
    /// float (D32 formats) or 16-bit unorm (D16), stencil as bytes.
    pub fn read_back_aspect(&mut self, image: vk::Image, format: vk::Format, aspect: vk::ImageAspectFlags, width: u32, height: u32) -> Result<Vec<u8>, String> {
        self.read_back_level(image, format, aspect, width, height, 0)
    }

    /// `read_back_aspect` of one mip level (`width`, `height`: that level's size).
    pub fn read_back_level(&mut self, image: vk::Image, format: vk::Format, aspect: vk::ImageAspectFlags, width: u32, height: u32, level: u32) -> Result<Vec<u8>, String> {
        self.read_back_slice(image, format, aspect, width, height, level, 0, 1, 1)
    }

    /// Readback of consecutive array layers, or a whole 3D mip (depth > 1).
    #[allow(clippy::too_many_arguments)]
    fn read_back_slice(&mut self, image: vk::Image, format: vk::Format, aspect: vk::ImageAspectFlags, width: u32, height: u32, level: u32, layer: u32, depth: u32, layers: u32) -> Result<Vec<u8>, String> {
        let bpp = if aspect == vk::ImageAspectFlags::STENCIL {
            1
        } else if aspect == vk::ImageAspectFlags::DEPTH {
            if matches!(format, vk::Format::D16_UNORM | vk::Format::D16_UNORM_S8_UINT) { 2 } else { 4 }
        } else {
            texel_bytes(format) as u64
        };
        let size = bpp * width as u64 * height as u64 * depth as u64 * layers as u64;
        let c = ctx();
        let (buf, mem, ptr) = c.host_buffer(size, vk::BufferUsageFlags::TRANSFER_DST)?;
        self.one_shot(|cmd| unsafe {
            let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
            let region = vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers { aspect_mask: aspect, mip_level: level, base_array_layer: layer, layer_count: layers })
                .image_extent(vk::Extent3D { width, height, depth });
            c.device.cmd_copy_image_to_buffer(cmd, image, vk::ImageLayout::GENERAL, buf, &[region]);
            let hb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[hb], &[], &[]);
        });
        let out = unsafe { std::slice::from_raw_parts(ptr, size as usize).to_vec() };
        unsafe {
            c.device.destroy_buffer(buf, None);
            c.device.unmap_memory(mem);
            c.device.free_memory(mem, None);
        }
        Ok(out)
    }

    /// Read an existing cache incarnation only: observing a draw must not create a new
    /// target, upload guest bytes or change alias ownership. Return the actual selected key.
    pub fn diagnostic_read(&mut self, wanted: ImageKey, level: u32, slice: u32, attachment: bool, storage: bool) -> Result<(ImageKey, Vec<u8>), String> {
        let key = if attachment {
            if self.map.contains_key(&wanted) { Some(wanted) } else { self.alias(&wanted, false) }
        } else { self.lookup(&wanted, storage).map(|(k, _)| k) }.ok_or("no cached image (not created/uploaded yet)")?;
        if level >= key.levels || (!key.ty3d && slice >= key.layers) { return Err("subresource outside cached image".into()); }
        let (w, h) = ((key.width >> level).max(1), (key.height >> level).max(1));
        let depth = if key.ty3d { (key.layers >> level).max(1) } else { 1 };
        // Sampled/storage array views can read/write every layer, not only layer zero.
        let layers = if !attachment && !key.ty3d { key.layers - slice } else { 1 };
        let aspect = if is_depth(key.format) { vk::ImageAspectFlags::DEPTH } else { vk::ImageAspectFlags::COLOR };
        if is_compressed(key.format) || texel_bytes(key.format) == 0 { return Err("compressed/unknown-format diagnostic read skipped".into()); }
        if w as u64 * h as u64 * depth as u64 * layers as u64 * texel_bytes(key.format) as u64 > 64 << 20 { return Err("diagnostic image exceeds 64 MiB limit".into()); }
        let raw = self.read_back_slice(self.map[&key].image, key.format, aspect, w, h, level, if key.ty3d { 0 } else { slice }, depth, layers)?;
        Ok((key, raw))
    }

    /// Share of zero bytes in the cached image a T# resolves to, at the T#'s
    /// base mip (layer 0), as the GPU holds it (frame captures; compare with
    /// the guest data).
    pub fn gpu_zero_stats(&mut self, t: &Tsharp) -> Option<String> {
        let (k, _) = self.lookup(&Self::key_of(t)?, false)?;
        if k.ty3d || is_depth(k.format) {
            return None;
        }
        let image = self.map.get(&k)?.image;
        let mip = t.base_level.min(k.levels.saturating_sub(1));
        let (w, h) = ((k.width >> mip).max(1), (k.height >> mip).max(1));
        let bytes = texel_bytes(k.format) as u64;
        let size = if is_compressed(k.format) { w.div_ceil(4) as u64 * h.div_ceil(4) as u64 * bytes } else { w as u64 * h as u64 * bytes };
        if size == 0 || size > 64 << 20 {
            return None;
        }
        let c = ctx();
        let (buf, mem, ptr) = c.host_buffer(size, vk::BufferUsageFlags::TRANSFER_DST).ok()?;
        self.one_shot(|cmd| unsafe {
            let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
            let region = vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: mip, base_array_layer: 0, layer_count: 1 })
                .image_extent(vk::Extent3D { width: w, height: h, depth: 1 });
            c.device.cmd_copy_image_to_buffer(cmd, image, vk::ImageLayout::GENERAL, buf, &[region]);
            let hb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[hb], &[], &[]);
        });
        let data = unsafe { std::slice::from_raw_parts(ptr, size as usize) };
        let z = data.iter().filter(|&&b| b == 0).count() * 100 / data.len();
        // The same mip of the guest data, detiled now: elements (blocks of
        // block-compressed formats) that differ from the GPU's copy.
        let mut cmp = String::new();
        let mut per_mip = false;
        let gpu_written = self.map.get(&k).is_some_and(|i| i.gpu_written);
        if let (Some(el), false) = (elem_log2(t.format), gpu_written) {
            if let Some(lay) = tiling::layout(t.sw_mode, el, k.width, k.height, k.levels, is_block(t.format)) {
                let e = 1u64 << el;
                let (s, n) = part_range(&lay, e, mip as usize, 0);
                let m = &lay.mips[mip as usize];
                if m.width as u64 * m.height as u64 * e != size {
                    cmp = format!("; not compared with the guest data (its mip is {}x{} elements of {e} bytes)", m.width, m.height);
                } else if !super::cp::guest_readable(t.base + s, n) {
                    cmp = "; guest mip not readable now".into();
                } else {
                    let src = unsafe { std::slice::from_raw_parts(t.base as *const u8, (s + n) as usize) };
                    let mut lin = Vec::new();
                    lay.detile(src, mip as usize, 0, &mut lin);
                    let mut differ = 0usize;
                    let mut first = None;
                    for (i, (a, b)) in lin.chunks_exact(e as usize).zip(data.chunks_exact(e as usize)).enumerate() {
                        if a != b {
                            differ += 1;
                            first.get_or_insert(i);
                        }
                    }
                    let total = (lin.len() / e as usize).max(1);
                    // Both copies' raw elements, for offline decoding (the
                    // first 16 differing textures of a session).
                    static RAW_DUMPS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                    if first.is_some() && RAW_DUMPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 16 {
                        if let Some(dir) = crate::run_dir() {
                            let tdir = dir.join("textures");
                            let _ = std::fs::create_dir_all(&tdir);
                            let name = format!("{:010x}_f{}_{}x{}_m{mip}", t.base, t.format, m.width, m.height);
                            let _ = std::fs::write(tdir.join(format!("{name}_gpu.raw")), data);
                            let _ = std::fs::write(tdir.join(format!("{name}_guest.raw")), &lin);
                        }
                    }
                    per_mip = first.is_some();
                    cmp = match first {
                        None => format!("; equal to the guest data ({total} elements)"),
                        Some(i) => format!(
                            "; {differ} of {total} elements ({}%) differ from the guest data detiled now, first at element {},{}",
                            differ * 100 / total,
                            i % m.width as usize,
                            i / m.width as usize
                        ),
                    };
                }
            }
        }
        unsafe {
            c.device.destroy_buffer(buf, None);
            c.device.unmap_memory(mem);
            c.device.free_memory(mem, None);
        }
        // Its base mip differs from the guest data: every mip's share.
        if per_mip {
            let shares: Vec<String> = (0..k.levels).filter_map(|m| self.mip_vs_guest(image, &k, t, m).map(|(d, n)| format!("m{m} {}%", d * 100 / n.max(1)))).collect();
            cmp.push_str(&format!("; per mip, elements differing from the guest data: {}", shares.join(" ")));
            cmp.push_str(&self.tail_view_report(&k, t));
        }
        Some(format!("gpu image mip {mip}: {z}% zero{cmp}"))
    }

    /// Captures: where GPU work wrote a compressed texture's mip-tail block
    /// through its tail view (the GPU-written single-level view of the
    /// block at the base: elements that differ from the guest block), next
    /// to where the layout puts each tail mip ("" without such a view).
    fn tail_view_report(&mut self, k: &ImageKey, t: &Tsharp) -> String {
        let Some(el) = elem_log2(t.format) else { return String::new() };
        let Some(lay) = tiling::layout(t.sw_mode, el, k.width, k.height, k.levels, true) else { return String::new() };
        let bytes = texel_bytes(k.format);
        let Some(view) = self
            .by_base
            .get(&k.base)
            .into_iter()
            .flatten()
            .filter_map(|kk| self.map.get(kk))
            .find(|i| i.gpu_written && !is_compressed(i.key.format) && !i.key.ty3d && i.key.levels == 1 && texel_bytes(i.key.format) == bytes && i.key.width == lay.block_w && i.key.height == lay.block_h)
            .map(|i| (i.image, i.key))
        else {
            return String::new();
        };
        let (vw, vh) = (view.1.width, view.1.height);
        let Some(vlay) = tiling::layout(t.sw_mode, el, vw, vh, 1, false) else { return String::new() };
        let e = 1u64 << el;
        let size = vw as u64 * vh as u64 * e;
        if !super::cp::guest_readable(t.base, vlay.size(1)) {
            return "; tail view: guest block not readable".into();
        }
        let c = ctx();
        let Ok((buf, mem, ptr)) = c.host_buffer(size, vk::BufferUsageFlags::TRANSFER_DST) else { return String::new() };
        self.one_shot(|cmd| unsafe {
            let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
            let region = vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 })
                .image_extent(vk::Extent3D { width: vw, height: vh, depth: 1 });
            c.device.cmd_copy_image_to_buffer(cmd, view.0, vk::ImageLayout::GENERAL, buf, &[region]);
            let hb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[hb], &[], &[]);
        });
        let data = unsafe { std::slice::from_raw_parts(ptr, size as usize) }.to_vec();
        unsafe {
            c.device.destroy_buffer(buf, None);
            c.device.unmap_memory(mem);
            c.device.free_memory(mem, None);
        }
        let src = unsafe { std::slice::from_raw_parts(t.base as *const u8, vlay.size(1) as usize) };
        let mut lin = Vec::new();
        vlay.detile(src, 0, 0, &mut lin);
        // Rows of written elements as x ranges (a written element differs
        // from the guest block).
        let mut rows: Vec<String> = Vec::new();
        let mut written = 0usize;
        for y in 0..vh as usize {
            let mut runs: Vec<(usize, usize)> = Vec::new();
            for x in 0..vw as usize {
                let i = (y * vw as usize + x) * e as usize;
                if lin[i..i + e as usize] != data[i..i + e as usize] {
                    written += 1;
                    match runs.last_mut() {
                        Some(r) if r.1 == x => r.1 = x + 1,
                        _ => runs.push((x, x + 1)),
                    }
                }
            }
            if !runs.is_empty() && rows.len() < 24 {
                rows.push(format!("y{y}: {}", runs.iter().map(|r| format!("{}..{}", r.0, r.1)).collect::<Vec<_>>().join(",")));
            }
        }
        let tails: Vec<String> = lay.mips.iter().enumerate().filter_map(|(i, m)| m.tail.map(|(x, y)| format!("m{i} at {x},{y} {}x{}", m.width, m.height))).collect();
        format!("; tail view {vw}x{vh}: {written} elements written by GPU work (rows {}); the layout's tail mips: {}", rows.join(" | "), tails.join(", "))
    }

    /// One mip (layer 0) of a cached image read back and compared with the
    /// guest data detiled now (frame captures): (differing elements, all).
    fn mip_vs_guest(&mut self, image: vk::Image, k: &ImageKey, t: &Tsharp, mip: u32) -> Option<(usize, usize)> {
        let el = elem_log2(t.format)?;
        let lay = tiling::layout(t.sw_mode, el, k.width, k.height, k.levels, is_block(t.format))?;
        let m = lay.mips.get(mip as usize)?;
        let e = 1u64 << el;
        let (w, h) = ((k.width >> mip).max(1), (k.height >> mip).max(1));
        let bytes = texel_bytes(k.format) as u64;
        let size = if is_compressed(k.format) { w.div_ceil(4) as u64 * h.div_ceil(4) as u64 * bytes } else { w as u64 * h as u64 * bytes };
        if size == 0 || size > 64 << 20 || m.width as u64 * m.height as u64 * e != size {
            return None;
        }
        let (s, n) = part_range(&lay, e, mip as usize, 0);
        if !super::cp::guest_readable(t.base + s, n) {
            return None;
        }
        let c = ctx();
        let (buf, mem, ptr) = c.host_buffer(size, vk::BufferUsageFlags::TRANSFER_DST).ok()?;
        self.one_shot(|cmd| unsafe {
            let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[mb], &[], &[]);
            let region = vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: mip, base_array_layer: 0, layer_count: 1 })
                .image_extent(vk::Extent3D { width: w, height: h, depth: 1 });
            c.device.cmd_copy_image_to_buffer(cmd, image, vk::ImageLayout::GENERAL, buf, &[region]);
            let hb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[hb], &[], &[]);
        });
        let data = unsafe { std::slice::from_raw_parts(ptr, size as usize) };
        let src = unsafe { std::slice::from_raw_parts(t.base as *const u8, (s + n) as usize) };
        let mut lin = Vec::new();
        lay.detile(src, mip as usize, 0, &mut lin);
        let differ = lin.chunks_exact(e as usize).zip(data.chunks_exact(e as usize)).filter(|(a, b)| a != b).count();
        let total = lin.len() / e as usize;
        unsafe {
            c.device.destroy_buffer(buf, None);
            c.device.unmap_memory(mem);
            c.device.free_memory(mem, None);
        }
        Some((differ, total))
    }

    /// An upscaler's view of a resource (`gpu::upscale`): the image behind `t` as a binding
    /// finds it (GENERAL layout), with its format, size, usage and the aspect its view shows.
    pub fn upscale_tex(&mut self, t: &Tsharp, storage: bool, frame: u64) -> Result<super::upscale::Tex, String> {
        let (view, image) = self.view(t, storage, frame)?;
        let key = self.map.values().find(|i| i.image == image).map(|i| i.key).ok_or("its image is not in the cache")?;
        let c = ctx();
        let f = unsafe { c.instance.get_physical_device_format_properties(c.pdev, key.format) }.optimal_tiling_features;
        let mut usage = vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC;
        if f.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE) {
            usage |= vk::ImageUsageFlags::SAMPLED;
        }
        if !is_depth(key.format) {
            let sprops = unsafe { c.instance.get_physical_device_format_properties(c.pdev, storage_format(key.format)) };
            if sprops.optimal_tiling_features.contains(vk::FormatFeatureFlags::STORAGE_IMAGE) {
                usage |= vk::ImageUsageFlags::STORAGE;
            }
        }
        if f.contains(vk::FormatFeatureFlags::COLOR_ATTACHMENT) {
            usage |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
        }
        if f.contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT) {
            usage |= vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
        }
        let aspect = if is_depth(key.format) { vk::ImageAspectFlags::DEPTH } else { vk::ImageAspectFlags::COLOR };
        Ok(super::upscale::Tex {
            image,
            view,
            range: vk::ImageSubresourceRange { aspect_mask: aspect, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 },
            format: key.format,
            size: (key.width, key.height),
            usage,
            layout: vk::ImageLayout::GENERAL,
        })
    }

    /// Diagnosis (`imgpng <address> <name>`): the largest GPU-written color image at `base`
    /// (mip 0, layer 0) to `<dir>/<name>.png`; its description.
    pub fn dump_one(&mut self, base: u64, dir: &std::path::Path, name: &str) -> Result<String, String> {
        // A depth surface: its depth as R32 floats (`<name>.raw`).
        if let Some(k) = self.map.values().filter(|i| i.key.base == base && is_depth(i.key.format)).max_by_key(|i| i.gpu_written).map(|i| i.key) {
            // (Also when the color images there were never written by the GPU: a depth
            // surface's R32F alias the cache keeps apart.)
            let only_depth = self.map.values().all(|i| i.key.base != base || is_depth(i.key.format));
            if only_depth || self.map[&k].gpu_written && self.map.values().all(|i| i.key.base != base || is_depth(i.key.format) || !i.gpu_written) {
                let image = self.map[&k].image;
                let raw = self.read_back_aspect(image, k.format, vk::ImageAspectFlags::DEPTH, k.width, k.height)?;
                let mut out = Vec::with_capacity(raw.len() + 8);
                out.extend_from_slice(&k.width.to_le_bytes());
                out.extend_from_slice(&k.height.to_le_bytes());
                out.extend_from_slice(&raw);
                std::fs::write(dir.join(format!("{name}.raw")), out).map_err(|e| e.to_string())?;
                return Ok(format!("{}x{} {:?} depth", k.width, k.height, k.format));
            }
        }
        let k = self.map.values().filter(|i| i.key.base == base && !is_depth(i.key.format)).max_by_key(|i| (i.key.width as u64 * i.key.height as u64, i.gpu_written, i.key.levels)).map(|i| i.key).ok_or("no color image there")?;
        // Every image cached at that address (writers and readers keyed apart would show here).
        let all: Vec<String> = self.map.values().filter(|i| i.key.base == base).map(|i| format!("{}x{}x{} {:?} {} levels{}", i.key.width, i.key.height, i.key.layers, i.key.format, i.key.levels, if i.gpu_written { " gpu-written" } else { "" })).collect();
        let image = self.map[&k].image;
        let raw = self.read_back(image, k.format, k.width, k.height)?;
        // Float images as their raw texels too (`<name>.raw`: width, height as u32, then the
        // texels): a PNG clamps their range. R32F images with mips: each further level as
        // `<name>_m<level>.raw` (depth pyramids).
        if matches!(k.format, vk::Format::R16G16B16A16_SFLOAT | vk::Format::R32G32B32A32_SFLOAT | vk::Format::R16G16_SFLOAT | vk::Format::B10G11R11_UFLOAT_PACK32 | vk::Format::R32_SFLOAT) {
            let mut out = Vec::with_capacity(raw.len() + 8);
            out.extend_from_slice(&k.width.to_le_bytes());
            out.extend_from_slice(&k.height.to_le_bytes());
            out.extend_from_slice(&raw);
            std::fs::write(dir.join(format!("{name}.raw")), out).map_err(|e| e.to_string())?;
            if k.format == vk::Format::R32_SFLOAT && !k.ty3d && k.layers == 1 {
                for level in 1..k.levels {
                    let (w, h) = ((k.width >> level).max(1), (k.height >> level).max(1));
                    let raw = self.read_back_level(image, k.format, vk::ImageAspectFlags::COLOR, w, h, level)?;
                    let mut out = Vec::with_capacity(raw.len() + 8);
                    out.extend_from_slice(&w.to_le_bytes());
                    out.extend_from_slice(&h.to_le_bytes());
                    out.extend_from_slice(&raw);
                    std::fs::write(dir.join(format!("{name}_m{level}.raw")), out).map_err(|e| e.to_string())?;
                }
            }
        }
        if let Some(rgba) = to_rgba8(k.format, &raw) {
            std::fs::write(dir.join(format!("{name}.png")), super::png::encode(k.width, k.height, &rgba)).map_err(|e| e.to_string())?;
        }
        Ok(format!("{}x{} {:?} {} levels (cached there: {})", k.width, k.height, k.format, k.levels, all.join("; ")))
    }

    /// Writes the GPU-written color images (mip 0, layer 0) to `dir` as PNG
    /// files named <address>_<width>x<height>_<format>.png (frame captures).
    pub fn dump_targets(&mut self, dir: &std::path::Path) -> usize {
        self.dump_targets_inner(dir, None)
    }

    /// Targets bound for writing in one captured frame, including raw texels and a manifest.
    /// No address-sorted cap: it could exclude the scene while retaining old menu targets.
    pub fn dump_frame_targets(&mut self, dir: &std::path::Path, bases: &std::collections::BTreeSet<u64>) -> usize {
        self.dump_targets_inner(dir, Some(bases))
    }

    fn dump_targets_inner(&mut self, dir: &std::path::Path, bases: Option<&std::collections::BTreeSet<u64>>) -> usize {
        let selected = |i: &&Image| i.gpu_written && bases.is_none_or(|b| b.contains(&i.key.base));
        let mut keys: Vec<ImageKey> = self.map.values().filter(selected).filter(|i| !is_depth(i.key.format)).map(|i| i.key).collect();
        keys.sort_by_key(|k| (k.base, k.format.as_raw()));
        let _ = std::fs::create_dir_all(dir);
        let mut n = 0;
        let mut stats = String::from("Snapshots at frame end, mip 0 / layer 0. Raw files: little-endian u32 width,height then native-format texels. Match addresses to cb/depth/written image bindings in capture-<flip>.txt; targets may be reused within a frame.
");
        for k in keys.into_iter().take(if bases.is_some() { usize::MAX } else { 256 }) {
            let image = self.map[&k].image;
            let name = if bases.is_some() {
                format!("{:010x}_{}x{}_{:?}_levels{}_layers{}_3d{}_cube{}", k.base, k.width, k.height, k.format, k.levels, k.layers, k.ty3d, k.cube)
            } else { format!("{:010x}_{}x{}_{:?}", k.base, k.width, k.height, k.format) };
            let raw = match self.read_back(image, k.format, k.width, k.height) {
                Ok(raw) => raw,
                Err(e) => { stats.push_str(&format!("{name}: readback failed: {e}\n")); continue; }
            };
            if bases.is_some() {
                let mut bytes = Vec::with_capacity(8 + raw.len());
                bytes.extend_from_slice(&k.width.to_le_bytes());
                bytes.extend_from_slice(&k.height.to_le_bytes());
                bytes.extend_from_slice(&raw);
                if let Err(e) = std::fs::write(dir.join(format!("{name}.raw")), bytes) { stats.push_str(&format!("{name}: write failed: {e}\n")); }
                stats.push_str(&format!("{name}: {} bytes, {} nonzero bytes, {} levels, {} layers\n", raw.len(), raw.iter().filter(|&&b| b != 0).count(), k.levels, k.layers));
            }
            if let Some(s) = float_stats(k.format, &raw) {
                stats.push_str(&format!("{name}: {s}\n"));
            }
            let Some(rgba) = to_rgba8(k.format, &raw) else { continue };
            if std::fs::write(dir.join(format!("{name}.png")), super::png::encode(k.width, k.height, &rgba)).is_ok() {
                n += 1;
            }
        }
        // Depth surfaces: depth statistics and the stencil values present.
        let mut dkeys: Vec<ImageKey> = self.map.values().filter(selected).filter(|i| is_depth(i.key.format)).map(|i| i.key).collect();
        dkeys.sort_by_key(|k| k.base);
        for k in dkeys.into_iter().take(if bases.is_some() { usize::MAX } else { 64 }) {
            let image = self.map[&k].image;
            let name = if bases.is_some() {
                format!("{:010x}_{}x{}_{:?}_levels{}_layers{}_3d{}_cube{}", k.base, k.width, k.height, k.format, k.levels, k.layers, k.ty3d, k.cube)
            } else { format!("{:010x}_{}x{}_{:?}", k.base, k.width, k.height, k.format) };
            if aspects(k.format).contains(vk::ImageAspectFlags::DEPTH) {
                if let Ok(raw) = self.read_back_aspect(image, k.format, vk::ImageAspectFlags::DEPTH, k.width, k.height) {
                    if bases.is_some() {
                        let mut bytes = Vec::with_capacity(8 + raw.len());
                        bytes.extend_from_slice(&k.width.to_le_bytes());
                        bytes.extend_from_slice(&k.height.to_le_bytes());
                        bytes.extend_from_slice(&raw);
                        if let Err(e) = std::fs::write(dir.join(format!("{name}_depth.raw")), bytes) { stats.push_str(&format!("{name}: depth write failed: {e}\n")); }
                    }
                    if raw.len() as u64 == 4 * k.width as u64 * k.height as u64 {
                        if let Some(s) = float_stats(vk::Format::R32_SFLOAT, &raw) {
                            stats.push_str(&format!("{name} depth: {s}\n"));
                        }
                        // Depth as gray, scaled to the largest value.
                        let d: Vec<f32> = raw.chunks_exact(4).map(|p| f32::from_le_bytes([p[0], p[1], p[2], p[3]])).collect();
                        let max = d.iter().copied().filter(|v| v.is_finite()).fold(0f32, f32::max).max(f32::MIN_POSITIVE);
                        let gray: Vec<u8> = d.iter().flat_map(|&v| {
                            let g = ((v / max).clamp(0.0, 1.0) * 255.0) as u8;
                            [g, g, g, 255]
                        }).collect();
                        let _ = std::fs::write(dir.join(format!("{name}_depth.png")), super::png::encode(k.width, k.height, &gray));
                    }
                }
            }
            if has_stencil(k.format) {
                if let Ok(raw) = self.read_back_aspect(image, k.format, vk::ImageAspectFlags::STENCIL, k.width, k.height) {
                    let mut hist = [0u64; 256];
                    for &v in &raw {
                        hist[v as usize] += 1;
                    }
                    let mut top: Vec<(u64, usize)> = hist.iter().enumerate().filter(|&(_, &c)| c > 0).map(|(v, &c)| (c, v)).collect();
                    top.sort_unstable_by(|a, b| b.cmp(a));
                    let list: Vec<String> = top.iter().take(12).map(|(c, v)| format!("{v:#04x}: {c}")).collect();
                    stats.push_str(&format!("{name} stencil: {}\n", list.join(", ")));
                    let gray: Vec<u8> = raw.iter().flat_map(|&v| [v, v, v, 255]).collect();
                    let _ = std::fs::write(dir.join(format!("{name}_stencil.png")), super::png::encode(k.width, k.height, &gray));
                }
            }
        }
        let _ = std::fs::write(dir.join("stats.txt"), stats);
        n
    }

    /// Whether a T# resolves to an image the GPU owns (written by GPU work;
    /// never re-uploaded from guest memory).
    pub fn gpu_owned(&self, t: &Tsharp) -> bool {
        let Some(key) = Self::key_of(t) else { return false };
        match self.map.get(&key) {
            Some(i) => i.gpu_written,
            None => self.alias(&key, true).is_some(),
        }
    }

    /// The cached image a T# resolves to (mip 0, layer 0): raw texels in its
    /// Vulkan format, width, height and bytes per texel (diagnostics).
    pub fn snapshot_raw(&mut self, t: &Tsharp) -> Result<(Vec<u8>, u32, u32, u32), String> {
        self.snapshot_raw_level(t, 0)
    }

    /// `snapshot_raw` of one mip level (absolute; color images only past level 0).
    pub fn snapshot_raw_level(&mut self, t: &Tsharp, level: u32) -> Result<(Vec<u8>, u32, u32, u32), String> {
        let want = Self::key_of(t).ok_or("unsupported format")?;
        let key = match self.lookup(&want, false) {
            Some((k, _)) => k,
            None => self.alias(&want, false).ok_or("not cached")?,
        };
        let image = self.map[&key].image;
        if level > 0 {
            if is_depth(key.format) || is_compressed(key.format) || texel_bytes(key.format) == 0 || level >= key.levels {
                return Err(format!("level {level} of {:?} ({} levels) not read", key.format, key.levels));
            }
            let (w, h) = ((key.width >> level).max(1), (key.height >> level).max(1));
            let raw = self.read_back_level(image, key.format, vk::ImageAspectFlags::COLOR, w, h, level)?;
            return Ok((raw, w, h, texel_bytes(key.format)));
        }
        if is_depth(key.format) {
            // Rendered depth or stencil read as a texture (R32F / R16 / R8 uint).
            let aspect = if want.format == vk::Format::R8_UINT { vk::ImageAspectFlags::STENCIL } else { vk::ImageAspectFlags::DEPTH };
            let raw = self.read_back_aspect(image, key.format, aspect, key.width, key.height)?;
            let texel = (raw.len() as u64 / (key.width as u64 * key.height as u64).max(1)) as u32;
            return Ok((raw, key.width, key.height, texel));
        }
        let raw = self.read_back(image, key.format, key.width, key.height)?;
        Ok((raw, key.width, key.height, texel_bytes(key.format)))
    }

    /// One texel of the cached image a T# resolves to (mip 0, layer 0; depth
    /// images: the aspect the T# reads), with the image's size and format
    /// (pixel watches of captures).
    pub fn texel_of(&mut self, t: &Tsharp, x: u32, y: u32) -> Result<(Vec<u8>, u32, u32, vk::Format), String> {
        let want = Self::key_of(t).ok_or("unsupported format")?;
        let key = match self.lookup(&want, false) {
            Some((k, _)) => k,
            None => self.alias(&want, false).ok_or("not cached")?,
        };
        let image = self.map[&key].image;
        let (x, y) = (x.min(key.width.saturating_sub(1)), y.min(key.height.saturating_sub(1)));
        let aspect = if !is_depth(key.format) {
            vk::ImageAspectFlags::COLOR
        } else if want.format == vk::Format::R8_UINT {
            vk::ImageAspectFlags::STENCIL
        } else {
            vk::ImageAspectFlags::DEPTH
        };
        Ok((self.read_back_texel(image, key.format, aspect, x, y)?, key.width, key.height, key.format))
    }

    /// The cached image a T# resolves to (mip 0, layer 0) as RGBA8, and the
    /// statistics of float formats (captures).
    pub fn snapshot(&mut self, t: &Tsharp) -> Result<(Vec<u8>, u32, u32, Option<String>), String> {
        let key = Self::key_of(t).ok_or("unsupported format")?;
        let key = if self.map.contains_key(&key) { key } else { self.alias(&key, false).ok_or("not cached")? };
        let image = self.map[&key].image;
        let raw = self.read_back(image, key.format, key.width, key.height)?;
        Ok((to_rgba8(key.format, &raw).ok_or("format")?, key.width, key.height, float_stats(key.format, &raw)))
    }

    /// Which cached image a T# resolves to now (frame captures).
    pub fn describe(&self, t: &Tsharp) -> String {
        let Some(key) = Self::key_of(t) else { return "unsupported format".into() };
        let all: Vec<String> = self
            .map
            .values()
            .filter(|i| i.key.base == t.base)
            .map(|i| format!("{}x{}x{} {:?} l{} m{}{}{}", i.key.width, i.key.height, i.key.layers, i.key.format, i.key.layers, i.key.levels, if i.gpu_written { " gpu" } else { "" }, if i.key == key { " =key" } else { "" }))
            .collect();
        let pick = match self.lookup(&key, false) {
            Some((k, "exact")) => format!("exact{}", if self.map[&k].gpu_written { "" } else { " (uploaded)" }),
            Some((k, how)) => format!("{how} {:#x} {:?} l{} m{}", k.base, k.format, k.layers, k.levels),
            None => "new upload".into(),
        };
        // Share of zero bytes in the first 64 KiB of the guest data (texture
        // data not loaded yet reads as zeros).
        let n = 64u64 << 10;
        let zeros = if super::cp::guest_readable(t.base, n) {
            let s = unsafe { std::slice::from_raw_parts(t.base as *const u8, n as usize) };
            format!("; guest data {}% zero", s.iter().filter(|&&b| b == 0).count() * 100 / s.len())
        } else {
            "; guest data unreadable".into()
        };
        // Per mip in the sampled range (upload layout): share of zero bytes
        // where each mip's elements are read from.
        let mut per_mip = String::new();
        if let (Some(el), false) = (elem_log2(t.format), key.ty3d) {
            if let Some(lay) = tiling::layout(t.sw_mode, el, key.width, key.height, key.levels, is_block(t.format)) {
                let e = 1u64 << el;
                for m in (t.base_level as usize)..lay.mips.len().min(t.base_level as usize + 4) {
                    let ml = &lay.mips[m];
                    let (mut z, mut n) = (0u64, 0u64);
                    for y in (0..ml.height).step_by((ml.height as usize / 16).max(1)) {
                        for x in (0..ml.width).step_by((ml.width as usize / 16).max(1)) {
                            let a = t.base + lay.offset(m, 0, x, y);
                            if super::cp::guest_readable(a, e) {
                                let s = unsafe { std::slice::from_raw_parts(a as *const u8, e as usize) };
                                z += s.iter().filter(|&&b| b == 0).count() as u64;
                                n += e;
                            }
                        }
                    }
                    if n > 0 {
                        per_mip.push_str(&format!(" m{m}:{}%", z * 100 / n));
                    }
                }
            }
        }
        if !per_mip.is_empty() {
            per_mip = format!("; zero bytes per mip{per_mip}");
        }
        // Uploaded textures: whether the sampled hashes of the source bytes
        // changed since the last upload (what the CPU-write check compares).
        if let Some(img) = self.map.get(&key).filter(|i| !i.gpu_written) {
            if let Some((b, n)) = img.source.and_then(super::track::range) {
                let now = all_hashes(b, n);
                let changed = (0..SLICES).filter(|&s| now[s] != img.hashes[s]).count();
                per_mip.push_str(&format!("; hashed slices changed since upload {changed}/{SLICES} (checked frame {})", img.checked_frame));
            }
        }
        format!("{pick}; images at base: [{}]{zeros}{per_mip}", all.join(", "))
    }

    /// View of a T# (format, type, levels/layers, swizzle) for sampling or storage.
    /// Bindings of a view in its translation's dimension, not the T#'s (`view_as`): [2D textures
    /// as 1-layer arrays, arrays' base slices as 2D] (status).
    pub fn keyed_views() -> [u64; 2] {
        [KEYED_VIEWS[0].load(std::sync::atomic::Ordering::Relaxed), KEYED_VIEWS[1].load(std::sync::atomic::Ordering::Relaxed)]
    }

    pub fn view(&mut self, t: &Tsharp, storage: bool, frame: u64) -> Result<(vk::ImageView, vk::Image), String> {
        let _cpu = crate::gpu::cpu_profile::scope(crate::gpu::cpu_profile::Part::ImageLookup);
        self.view_as(t, storage, frame, None)
    }

    /// `view` as `want` where the binding's translation keys the image by its instructions'
    /// dimension (`shaders::d2_resources`): a 2D texture as a 1-layer 2D-array view, a 2D
    /// array's base slice as a 2D view (what the hardware reads for such addresses).
    pub fn view_as(&mut self, t: &Tsharp, storage: bool, frame: u64, want: Option<vk::ImageViewType>) -> Result<(vk::ImageView, vk::Image), String> {
        // (A view of another type than the T#'s: the full lookup, no memo (keyed by the T#).)
        if want.is_some() {
            KEYED_VIEWS[(want == Some(vk::ImageViewType::TYPE_2D)) as usize].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        if storage || want.is_some() {
            return self.view_full(t, storage, frame, want).map(|r| (r.0, r.1));
        }
        // Generations read first, the image's region's too when the image
        // exists: a change during the full lookup (its refresh) makes the
        // entry stale at its next use.
        use std::sync::atomic::Ordering::Acquire;
        let (gpu_gen, bgen) = (GPU_IMAGES_GEN.load(Acquire), base_gen(t.base).load(Acquire));
        let key = Self::key_of(t);
        let region_before = key.and_then(|k| self.map.get(&k)).and_then(|i| i.source_gen);
        let rgen = region_before.map_or(0, |g| g.load(Acquire));
        let mk = memo_raw(t);
        if let Some(m) = self.view_memo.get(&mk).copied() {
            // (An image whose upload a thread left for after the lock goes
            // the full way, whose `refresh` waits for it: a memo another
            // waiting thread made would hand it out unrecorded.)
            if m.miss(t, frame, scrubbing()) == 0 && !self.upload_pending(m.image) {
                m.used.stamp(frame);
                // This thread's memo too: without it every thread but the one
                // that made the entry took this lock for each lookup (64% of
                // view lookups missed their memo, run 20261012-073715).
                tls(|tm| {
                    if tm.views.len() >= tm.cap {
                        tm.views.clear();
                    }
                    tm.views.insert(mk, ViewMemo { frame, ..m });
                });
                return Ok((m.view, m.image));
            }
        }
        let r = self.view_full(t, storage, frame, None)?;
        // The region of the image found (an alias of another format, or a
        // GPU-written image, has none): known before the lookup, or the
        // entry is good for this call only.
        let found = key.and_then(|k| self.lookup(&k, false)).and_then(|(k, _)| self.map.get(&k)).filter(|i| i.image == r.1);
        let region = found.and_then(|i| i.source_gen);
        let stale = found.is_none() || region.is_some() && !region.zip(region_before).is_some_and(|(a, b)| std::ptr::eq(a, b));
        let m = ViewMemo { frame, gpu_gen, base_gen: bgen, region, region_gen: rgen, stale, view: r.0, image: r.1, used: r.2 };
        if self.view_memo.len() >= MEMO_CAP {
            self.view_memo.clear();
        }
        self.view_memo.insert(mk, m);
        tls(|tm| {
            if tm.views.len() >= tm.cap {
                tm.views.clear();
            }
            tm.views.insert(mk, m);
        });
        Ok((r.0, r.1))
    }

    /// Whether `image`'s upload was left for after the lock and is not
    /// recorded yet (`refreshing`).
    fn upload_pending(&self, image: vk::Image) -> bool {
        !self.refreshing.is_empty() && self.refreshing.keys().any(|k| self.map.get(k).is_some_and(|m| m.image == image))
    }

    fn view_full(&mut self, t: &Tsharp, storage: bool, frame: u64, want: Option<vk::ImageViewType>) -> Result<(vk::ImageView, vk::Image, UseStamp), String> {
        if storage {
            // DS_TRACE_IMAGE_WRITERS=<hex base> or <w>x<h>x<depth>: each
            // shader that binds such an image for writing, once.
            static WATCH: std::sync::OnceLock<Option<(u64, [u32; 3])>> = std::sync::OnceLock::new();
            let w = *WATCH.get_or_init(|| {
                let v = std::env::var("DS_TRACE_IMAGE_WRITERS").ok()?;
                let d: Vec<u32> = v.split('x').filter_map(|x| x.parse().ok()).collect();
                if d.len() == 3 {
                    Some((0, [d[0], d[1], d[2]]))
                } else {
                    u64::from_str_radix(v.trim_start_matches("0x"), 16).ok().map(|b| (b, [0; 3]))
                }
            });
            if let Some((base, dims)) = w {
                static SEEN: Mutex<Option<std::collections::HashSet<(u64, u64)>>> = Mutex::new(None);
                let sh = BINDING_SHADER.with(|s| s.get());
                let hit = if base != 0 { t.base == base } else { [t.width, t.height, t.depth] == dims };
                if hit && SEEN.lock().get_or_insert_with(Default::default).insert((sh, t.base)) {
                    crate::info!("images: shader {sh:#x} writes {:#x} ({}x{}x{} format {}, frame {frame})", t.base, t.width, t.height, t.depth, t.format);
                }
            }
        }
        let key = self.image_for(t, frame, storage)?;
        let req = vk_format(t.format).unwrap();
        let depth = is_depth(key.format);
        let stencil = depth && req == vk::Format::R8_UINT;
        if depth && storage {
            return Err(format!("texture {:#x} aliases a depth surface (storage access not supported)", t.base));
        }
        // Rendered depth is sampled through a view of its depth aspect (the
        // value arrives in R, like the 32-bit float texture the title reads).
        let fmt = if depth { key.format } else if storage { storage_format(req) } else { req };
        let levels = (t.last_level + 1).max(1);
        let _ = levels;
        let (vt, base_layer, layer_count) = if key.ty3d {
            (vk::ImageViewType::TYPE_3D, 0, 1)
        } else if t.ty == 11 && want == Some(vk::ImageViewType::TYPE_2D_ARRAY) {
            // Exact cube texel loads: the same image, format, mip range and swizzle, with
            // one array layer per face. Integer coordinates are relative to BASE_ARRAY.
            (vk::ImageViewType::TYPE_2D_ARRAY, t.base_array, t.depth.saturating_sub(t.base_array).max(1))
        } else if t.ty == 9 && want == Some(vk::ImageViewType::TYPE_2D_ARRAY) {
            (vk::ImageViewType::TYPE_2D_ARRAY, 0, 1)
        } else if t.ty == 13 && want == Some(vk::ImageViewType::TYPE_2D) {
            (vk::ImageViewType::TYPE_2D, t.base_array, 1)
        } else {
            match t.ty {
                11 if t.depth > 6 && !storage => (vk::ImageViewType::CUBE_ARRAY, 0, t.depth / 6 * 6),
                11 if !storage => (vk::ImageViewType::CUBE, 0, 6),
                11 => (vk::ImageViewType::TYPE_2D_ARRAY, 0, t.depth),
                12 | 13 | 15 => (vk::ImageViewType::TYPE_2D_ARRAY, t.base_array, t.depth.saturating_sub(t.base_array).max(1)),
                _ => (vk::ImageViewType::TYPE_2D, 0, 1),
            }
        };
        let sel = |s: u16| -> u8 { (s & 7) as u8 };
        let swz = if storage { [4, 5, 6, 7] } else { [sel(t.dst_sel), sel(t.dst_sel >> 3), sel(t.dst_sel >> 6), sel(t.dst_sel >> 9)] };
        let vkey = (fmt, vt, t.base_level, levels.saturating_sub(t.base_level).max(1), base_layer, layer_count, swz, storage as u8 | (stencil as u8) << 1);
        if storage {
            self.write_seq += 1;
        }
        let seq = self.write_seq;
        if storage && matches!(key.format, vk::Format::R32_SFLOAT | vk::Format::R16_UNORM) && self.depth_feeds.insert(key.base) {
            // A possible compute view of a depth plane: memos of
            // `depth_copies_possible` see it from now on.
            LOOKUP_GEN.fetch_add(1, std::sync::atomic::Ordering::Release);
        }
        let img = self.map.get_mut(&key).unwrap();
        if storage {
            img.write_gen = seq;
            let m = t.base_level as usize;
            if img.mip_gens.len() <= m {
                img.mip_gens.resize(m + 1, 0);
            }
            img.mip_gens[m] = seq;
            img.writer = BINDING_SHADER.with(|s| s.get());
            if !img.gpu_written {
                crate::info!("images: {:#x} {}x{} {:?} becomes a storage image (was uploaded: {}; shader {:#x})", key.base, key.width, key.height, key.format, img.source.is_some(), BINDING_SHADER.with(|s| s.get()));
            }
            if !img.gpu_written {
                LOOKUP_GEN.fetch_add(1, std::sync::atomic::Ordering::Release);
                gpu_images_changed();
            }
            img.gpu_written = true;
            img.source = None;
            img.source_gen = None;
            self.gpu_bases.insert(key.base);
            self.index_owned(key);
        }
        let img = self.map.get_mut(&key).unwrap();
        img.used.stamp(frame);
        if let Some(&v) = img.views.get(&vkey) {
            return Ok((v, img.image, img.used));
        }
        let comp = |s: u8| match s {
            0 => vk::ComponentSwizzle::ZERO,
            1 => vk::ComponentSwizzle::ONE,
            4 => vk::ComponentSwizzle::R,
            5 => vk::ComponentSwizzle::G,
            6 => vk::ComponentSwizzle::B,
            _ => vk::ComponentSwizzle::A,
        };
        let levels_avail = img.key.levels;
        let base_level = t.base_level.min(levels_avail - 1);
        let ci = vk::ImageViewCreateInfo::default()
            .image(img.image)
            .view_type(vt)
            .format(fmt)
            .components(vk::ComponentMapping { r: comp(swz[0]), g: comp(swz[1]), b: comp(swz[2]), a: comp(swz[3]) })
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: if stencil { vk::ImageAspectFlags::STENCIL } else if depth { vk::ImageAspectFlags::DEPTH } else { vk::ImageAspectFlags::COLOR },
                base_mip_level: base_level,
                level_count: if storage { 1 } else { (levels.saturating_sub(base_level)).clamp(1, levels_avail - base_level) },
                base_array_layer: base_layer,
                layer_count,
            });
        let mut usage = vk::ImageViewUsageCreateInfo::default().usage(if storage { vk::ImageUsageFlags::STORAGE } else { vk::ImageUsageFlags::SAMPLED });
        let ci = ci.push_next(&mut usage);
        let v = unsafe { ctx().device.create_image_view(&ci, None) }.map_err(|e| format!("image view {vkey:?}: {e}"))?;
        img.views.insert(vkey, v);
        Ok((v, img.image, img.used))
    }

    /// Sampler of an S#; `compare`: used by depth-compare instructions (the
    /// S# compare function applies only to those).
    pub fn sampler(&mut self, s: [u32; 4], compare_use: bool) -> Result<vk::Sampler, String> {
        let aniso_pc = ANISO.load(std::sync::atomic::Ordering::Relaxed);
        let key = (s, compare_use, aniso_pc);
        if let Some(&v) = self.samplers.get(&key) {
            tls(|m| {
                m.samplers.insert(key, v);
            });
            return Ok(v);
        }
        let b = |v: u32, lo: u32, n: u32| (v >> lo) & ((1 << n) - 1);
        // (Mirror once without the device feature: mirrored repeat, `vk::mirror_clamp`.)
        let mirror_once = if super::vk::mirror_clamp() { vk::SamplerAddressMode::MIRROR_CLAMP_TO_EDGE } else { vk::SamplerAddressMode::MIRRORED_REPEAT };
        let addr = |m: u32| match m {
            0 => vk::SamplerAddressMode::REPEAT,
            1 => vk::SamplerAddressMode::MIRRORED_REPEAT,
            2 => vk::SamplerAddressMode::CLAMP_TO_EDGE,
            3 => mirror_once,
            4 | 6 => vk::SamplerAddressMode::CLAMP_TO_BORDER,
            _ => mirror_once,
        };
        let filt = |m: u32| if m & 1 != 0 { vk::Filter::LINEAR } else { vk::Filter::NEAREST };
        let aniso = b(s[0], 9, 3);
        let compare = b(s[0], 12, 3);
        let mip = b(s[2], 26, 2);
        let border = match b(s[3], 30, 2) {
            1 => vk::BorderColor::FLOAT_OPAQUE_BLACK,
            2 => vk::BorderColor::FLOAT_OPAQUE_WHITE,
            _ => vk::BorderColor::FLOAT_TRANSPARENT_BLACK,
        };
        let lod_bias = ((b(s[2], 0, 14) << 18) as i32 >> 18) as f32 / 256.0;
        let min_lod = b(s[1], 0, 12) as f32 / 256.0;
        let max_lod = b(s[1], 12, 12) as f32 / 256.0;
        // The PC texture filtering (`ANISO`): anisotropic at that level for the title's
        // mipmapped, linearly filtered samplers that do not compare (its surface textures).
        let raise = aniso_pc > 1 && !compare_use && mip == 2 && b(s[2], 22, 2) >= 1 && b(s[2], 20, 2) >= 1;
        let ci = vk::SamplerCreateInfo::default()
            .mag_filter(filt(b(s[2], 20, 2)))
            .min_filter(filt(b(s[2], 22, 2)))
            .mipmap_mode(if mip == 2 { vk::SamplerMipmapMode::LINEAR } else { vk::SamplerMipmapMode::NEAREST })
            .address_mode_u(addr(b(s[0], 0, 3)))
            .address_mode_v(addr(b(s[0], 3, 3)))
            .address_mode_w(addr(b(s[0], 6, 3)))
            .mip_lod_bias(lod_bias.clamp(-15.0, 15.0))
            .anisotropy_enable((aniso > 0 && b(s[2], 22, 2) >= 2) || raise)
            .max_anisotropy(((1u32 << aniso.min(4)) as f32).max(if raise { aniso_pc as f32 } else { 1.0 }))
            .compare_enable(compare_use)
            // (`DS_CMP_ALWAYS=1`, diagnosis: every comparison passes, so depth-compare samples
            // of depth views read 1: no shadow from any shadow map.)
            .compare_op(match if cmp_always() { 7 } else { compare } {
                1 => vk::CompareOp::LESS,
                2 => vk::CompareOp::EQUAL,
                3 => vk::CompareOp::LESS_OR_EQUAL,
                4 => vk::CompareOp::GREATER,
                5 => vk::CompareOp::NOT_EQUAL,
                6 => vk::CompareOp::GREATER_OR_EQUAL,
                7 => vk::CompareOp::ALWAYS,
                _ => vk::CompareOp::NEVER,
            })
            .min_lod(min_lod)
            .max_lod(if mip == 0 { min_lod + 0.25 } else { max_lod.max(min_lod) })
            .border_color(border)
            .unnormalized_coordinates(false);
        let v = unsafe { ctx().device.create_sampler(&ci, None) }.map_err(|e| format!("sampler: {e}"))?;
        self.samplers.insert(key, v);
        tls(|m| {
            m.samplers.insert(key, v);
        });
        Ok(v)
    }

    pub fn count(&self) -> usize {
        self.map.len()
    }

    /// One status line (hang reports).
    pub fn status(&self) -> String {
        use std::sync::atomic::Ordering::Relaxed;
        let remaps: Vec<String> = ["commit", "decommit", "protection"]
            .iter()
            .zip(&REMAP_UPLOADS)
            .map(|(what, s)| format!("{what} {} ({} unchanged, {} partial, {} ms)", s[0].load(Relaxed), s[1].load(Relaxed), s[2].load(Relaxed), s[3].load(Relaxed) / 1_000_000))
            .collect();
        let p = &PART_UPLOADS;
        let ww = &crate::mem::WRITE_WATCH_STATS;
        let check = if upload_check() {
            format!(
                " (checked: {} left out, {} changed; {} uploaded again, {} of them unchanged)",
                SUB_CHECKS[0].load(Relaxed),
                SUB_CHECKS[1].load(Relaxed),
                SUB_CHECKS[2].load(Relaxed),
                SUB_CHECKS[3].load(Relaxed)
            )
        } else {
            String::new()
        };
        let g: Vec<u64> = GPU_DETILE.iter().map(|c| c.load(Relaxed)).collect();
        let gpu = if g[0] + g[4] > 0 {
            let checked = if gpu_detile_check() { format!(", {} (mip, layer)s compared with the CPU's detiling of the bytes copied, {} differing, {} tiled ranges changed in guest memory during their upload", g[6], g[7], g[8]) } else { String::new() };
            format!("; detiled on the GPU: {} uploads ({} (mip, layer)s, {} MiB copied, {} MiB detiled, {} ms recording and submitting them; video-memory buffers made {}; transfers on the copy queue {}; 3D volumes {}), {} left to the CPU{checked}", g[0], g[1], g[2] >> 20, g[3] >> 20, g[5] / 1_000_000, g[9], g[10], g[11], g[4])
        } else {
            String::new()
        };
        let o: Vec<u64> = REFRESH_OUT.iter().map(|c| c.load(Relaxed)).collect();
        let f: Vec<u64> = FIRST_OUT.iter().map(|c| c.load(Relaxed)).collect();
        let gpu = if o[0] + o[3] + f[0] > 0 {
            format!("{gpu}; re-uploads detiled after the lock: {} ({} recorded, {} dropped: their image gone or GPU-written meanwhile; {} ms outside the lock), first uploads detiled after the lock: {} ({} recorded, {} dropped, {} failed; {} ms outside the lock), bindings that waited for one {} ({} ms)", o[0], o[1], o[2], o[4] / 1_000_000, f[0], f[1], f[2], f[3], f[4] / 1_000_000, o[3], o[5] / 1_000_000)
        } else {
            gpu
        };
        format!(
            "    images: {} cached ({} MiB; GPU-written {} ({} MiB), kept for the session; {} textures evicted, {} MiB, in {} rounds), {} uploads ({} re-uploads; {} MiB, {} ms detiling and copying: 2D {} MiB, {} ms detiling ({} ms choosing (mapping queries, hashes); CPU-detiled: staging buffers taken {} ms ({} from the pool, {} made for it, {} outside it: over 64 MiB), detiling into them {} ms, first uploads' hashes {} ms), {} ms staging, {} ms transfers; waits for uploads in flight {}; batched (`tune upload_batch`) {} in {} command buffers, {} waits; pages checked in the mapping table {} ({} of them while other mappings changed), again with the operating system {}); partial re-uploads {} ({} (mip, layer)s uploaded, {} left out, {} of them unchanged written ones, {} MiB){check}; write-watch takes (all) {} ({:.1} us each, {} pages, {} found clean by a probe); GPU-owned images overwritten through buffers: {} cleared ({} right after a zero fill, no wait), {} uploaded; re-uploads after mapping changes: {}{gpu}; {}\n",
            self.map.len(),
            IMAGE_BYTES.load(Relaxed) >> 20,
            self.map.values().filter(|i| i.gpu_written).count(),
            self.map.values().filter(|i| i.gpu_written).map(|i| i.bytes).sum::<u64>() >> 20,
            EVICTED[0].load(Relaxed),
            EVICTED[1].load(Relaxed) >> 20,
            EVICTED[2].load(Relaxed),
            self.uploads,
            self.reuploads,
            self.upload_bytes >> 20,
            self.upload_ns / 1_000_000,
            DETILED.load(Relaxed) >> 20,
            UPLOAD_SPLIT[0].load(Relaxed) / 1_000_000,
            UPLOAD_SPLIT[3].load(Relaxed) / 1_000_000,
            UPLOAD_PHASES[0].load(Relaxed) / 1_000_000,
            STAGING_STATS[0].load(Relaxed),
            STAGING_STATS[1].load(Relaxed),
            STAGING_STATS[2].load(Relaxed),
            UPLOAD_PHASES[1].load(Relaxed) / 1_000_000,
            UPLOAD_PHASES[2].load(Relaxed) / 1_000_000,
            UPLOAD_SPLIT[1].load(Relaxed) / 1_000_000,
            UPLOAD_SPLIT[2].load(Relaxed) / 1_000_000,
            UPLOAD_WAITS.load(Relaxed),
            BATCHED_STATS[0].load(Relaxed),
            BATCHED_STATS[1].load(Relaxed),
            BATCHED_STATS[2].load(Relaxed),
            UPLOAD_TABLE[0].load(Relaxed),
            UPLOAD_TABLE[2].load(Relaxed),
            UPLOAD_TABLE[1].load(Relaxed),
            p[0].load(Relaxed),
            p[1].load(Relaxed),
            p[2].load(Relaxed),
            SUB_SAME.load(Relaxed),
            p[3].load(Relaxed) >> 20,
            ww[0].load(Relaxed),
            ww[1].load(Relaxed) as f64 / ww[0].load(Relaxed).max(1) as f64 / 1000.0,
            ww[2].load(Relaxed),
            crate::mem::PROBE_CLEAN.load(Relaxed),
            self.overwrite_clears,
            ZERO_FILL_CLEARS.load(std::sync::atomic::Ordering::Relaxed),
            self.overwrite_uploads,
            remaps.join(", "),
            super::slab::status()
        ) + &zero_census_line()
    }
}

/// How a fill dispatch's writes are noted against the fast clears' metadata
/// (`note_meta_writes_but_fill`, `notes_for`; `DS_FILL_NOTE`): 2 the fill's own range left to the
/// fill for every image, so per-layer fills of one value keep it known (619c3a5's form, the
/// default again since 2026-11-21 night: 1-2's Gatehouse Hall 54.4 against 51.6 flips/s at
/// 240 Hz); 1 the same for layered (array) images only; 0 every write noted, the fill's own
/// range too. Its evening's flicker at 1-1's Start was a fill's clear left out of the draws'
/// published set (`note_meta_fill`), not this noting (FINDINGS.md 2026-11-21 night).
pub fn fill_note_mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| std::env::var("DS_FILL_NOTE").ok().and_then(|v| v.trim().parse::<u8>().ok()).filter(|&m| m <= 2).unwrap_or(2))
}

/// Whether a fill's own write over `key`'s metadata is left to the fill in `mode`.
fn left_to_fill(mode: u8, key: &ImageKey) -> bool {
    match mode {
        2 => true,
        1 => key.layers > 1 && !key.ty3d,
        _ => false,
    }
}
