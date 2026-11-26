use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// The scene's TAA: the largest the title ran within `MAIN_VIEW_HOLD` flips (width, height, the
/// last flip it ran in). The title also runs its TAA on small views (the equipment menu's
/// character preview at 640x360, in turn with the scene's); those keep the title's own result.
/// Until 2026-11-24 the upscaler followed both: its output image and its feature were remade at
/// every change, a frame or two apart, and the images were never freed (issue #4).
static MAIN_VIEW: parking_lot::Mutex<(u32, u32, u64)> = parking_lot::Mutex::new((0, 0, 0));
const MAIN_VIEW_HOLD: u64 = 120;
/// The title's TAA runs left to it as a smaller view than the scene's.
pub static SMALL_VIEWS: AtomicU64 = AtomicU64::new(0);

/// Whether the title's TAA at `w`x`h` in flip `frame` is the scene's (`MAIN_VIEW`): the size it
/// last had, a larger one, or any once the scene's has not run for `MAIN_VIEW_HOLD` flips.
pub fn main_view(w: u32, h: u32, frame: u64) -> bool {
    let mut m = MAIN_VIEW.lock();
    let (mw, mh, at) = *m;
    if (w, h) == (mw, mh) || u64::from(w) * u64::from(h) > u64::from(mw) * u64::from(mh) || frame > at + MAIN_VIEW_HOLD {
        if (w, h) != (mw, mh) && mw != 0 {
            crate::info!("upscaler: the scene's view is {w}x{h} now (it was {mw}x{mh}, last at flip {at})");
        }
        *m = (w, h, frame);
        return true;
    }
    drop(m);
    if SMALL_VIEWS.fetch_add(1, Relaxed) == 0 {
        crate::info!("upscaler: the title's TAA at {w}x{h} keeps its own result (a smaller view beside the scene's {mw}x{mh}, such as a menu's character preview)");
    }
    false
}

#[cfg(test)]
mod tests {
    #[test]
    fn scene_inputs_survive_preview_taa_and_eventually_allow_resize() {
        assert!(super::main_view(1920, 1080, 1));
        assert!(!super::main_view(640, 360, 1));
        assert!(super::main_view(1920, 1080, 2));
        assert!(!super::main_view(640, 360, 3));
        assert!(super::main_view(3840, 2160, 3));
        assert!(!super::main_view(1920, 1080, 4));
        assert!(super::main_view(1920, 1080, 124));
    }
}

