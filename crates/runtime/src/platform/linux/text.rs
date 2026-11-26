//! Host font rasterization for the IME and keyboard/mouse prompt labels. No game font data.
use std::sync::OnceLock;

fn font() -> Option<&'static fontdue::Font> {
    static FONT: OnceLock<Option<fontdue::Font>> = OnceLock::new();
    FONT.get_or_init(|| {
        let path = std::env::var_os("DS_LINUX_FONT")
            .map(std::path::PathBuf::from)
            .or_else(|| {
                let out = std::process::Command::new("fc-match")
                    .args(["-f", "%{file}", "sans"])
                    .output()
                    .ok()?;
                out.status
                    .success()
                    .then(|| std::path::PathBuf::from(String::from_utf8_lossy(&out.stdout).trim()))
            })
            .unwrap_or_else(|| "/usr/share/fonts/TTF/DejaVuSans.ttf".into());
        match std::fs::read(&path)
            .ok()
            .and_then(|data| fontdue::Font::from_bytes(data, fontdue::FontSettings::default()).ok())
        {
            Some(f) => {
                crate::info!("Linux overlay font: {}", path.display());
                Some(f)
            }
            None => {
                crate::warn!(
                    "Linux overlay font unavailable: {}; set DS_LINUX_FONT to a TTF/OTF file",
                    path.display()
                );
                None
            }
        }
    })
    .as_ref()
}

pub fn width(text: &str, size: f32) -> f32 {
    font().map_or(0.0, |f| {
        text.chars().map(|c| f.metrics(c, size).advance_width).sum()
    })
}

/// Coverage fitted to a rectangle, centered on a fixed canvas. Positions are physical pixels.
pub fn mask(text: &str, w: usize, h: usize, rect: (f32, f32, f32, f32)) -> Option<Vec<u8>> {
    let f = font()?;
    let (cx, cy, max_w, max_h) = rect;
    let mut size = max_h.max(1.0);
    let advance = |size| {
        text.chars()
            .map(|c| f.metrics(c, size).advance_width)
            .sum::<f32>()
    };
    let width = advance(size);
    if width > max_w {
        size *= max_w / width;
    }
    let metrics = f.horizontal_line_metrics(size)?;
    let mut x = cx - advance(size) / 2.0;
    let baseline = cy + (metrics.ascent + metrics.descent) / 2.0;
    let mut out = vec![0; w * h];
    for ch in text.chars() {
        let (m, bitmap) = f.rasterize(ch, size);
        let left = x.round() as i32 + m.xmin;
        let top = baseline.round() as i32 - m.ymin - m.height as i32;
        for y in 0..m.height {
            for col in 0..m.width {
                let (dx, dy) = (left + col as i32, top + y as i32);
                if dx >= 0 && dx < w as i32 && dy >= 0 && dy < h as i32 {
                    let p = &mut out[dy as usize * w + dx as usize];
                    *p = (*p).max(bitmap[y * m.width + col]);
                }
            }
        }
        x += m.advance_width;
    }
    Some(out)
}

pub struct Canvas {
    pub pixels: Vec<u8>,
    pub w: usize,
    pub h: usize,
    pub scale: f32,
}
impl Canvas {
    pub fn fill(&mut self, rect: (f32, f32, f32, f32), color: [u8; 3]) {
        let (x, y, w, h) = rect;
        let (a, b, c, d) = (
            (x * self.scale) as usize,
            (y * self.scale) as usize,
            ((x + w) * self.scale) as usize,
            ((y + h) * self.scale) as usize,
        );
        for y in b.min(self.h)..d.min(self.h) {
            for x in a.min(self.w)..c.min(self.w) {
                self.pixels[(y * self.w + x) * 4..(y * self.w + x) * 4 + 4]
                    .copy_from_slice(&[color[2], color[1], color[0], 255]);
            }
        }
    }
    pub fn text(&mut self, text: &str, rect: (f32, f32, f32, f32), color: [u8; 3]) {
        let (x, y, w, h) = rect;
        if let Some(mask) = mask(
            text,
            self.w,
            self.h,
            (
                (x + w / 2.0) * self.scale,
                (y + h / 2.0) * self.scale,
                w * self.scale,
                h * self.scale,
            ),
        ) {
            for (p, a) in self.pixels.chunks_exact_mut(4).zip(mask) {
                for i in 0..3 {
                    p[i] = ((p[i] as u32 * (255 - a as u32) + color[2 - i] as u32 * a as u32) / 255)
                        as u8;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn readable_label_coverage() {
        let m = super::mask("Enter / Confirm", 256, 64, (128.0, 32.0, 240.0, 32.0))
            .expect("install a system font or set DS_LINUX_FONT");
        assert!(m.iter().filter(|&&p| p > 128).count() > 100);
        assert!(m[..256].iter().all(|&p| p == 0));
    }
}
