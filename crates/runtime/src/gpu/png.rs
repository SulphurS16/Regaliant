//! Minimal PNG encoder (RGBA8, zlib-compressed) for screenshots and image dumps.

fn crc32(data: &[u8]) -> u32 {
    let mut c = 0xFFFF_FFFFu32;
    for &b in data {
        c ^= b as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
        }
    }
    !c
}

pub fn encode(w: u32, h: u32, rgba: &[u8]) -> Vec<u8> {
    let mut raw = Vec::with_capacity((w * 4 + 1) as usize * h as usize);
    for y in 0..h as usize {
        raw.push(0);
        raw.extend_from_slice(&rgba[y * w as usize * 4..(y + 1) * w as usize * 4]);
    }
    let z = miniz_oxide::deflate::compress_to_vec_zlib(&raw, 3);
    let mut out = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    let mut chunk = |ty: &[u8; 4], data: &[u8]| {
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        let mut c = ty.to_vec();
        c.extend_from_slice(data);
        out.extend_from_slice(&c);
        out.extend_from_slice(&crc32(&c).to_be_bytes());
    };
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&w.to_be_bytes());
    ihdr.extend_from_slice(&h.to_be_bytes());
    ihdr.extend_from_slice(&[8, 6, 0, 0, 0]);
    chunk(b"IHDR", &ihdr);
    chunk(b"IDAT", &z);
    chunk(b"IEND", &[]);
    out
}

/// A display screenshot has opaque coverage, even when a shader uses alpha as scratch
/// data (FSR interpolation stores its inpainting weight there). Keep RGB unchanged.
/// Raw image dumps and ordinary encode() retain the original alpha.
#[cfg(any(target_os = "linux", test))]
pub fn encode_opaque(w: u32, h: u32, rgba: &[u8]) -> Vec<u8> {
    let mut pixels = rgba.to_vec();
    for p in pixels.chunks_exact_mut(4) { p[3] = 255; }
    encode(w, h, &pixels)
}

#[cfg(test)]
mod tests {
    #[test]
    fn opaque_display_preserves_rgb_and_leaves_raw_mask_untouched() {
        let rgba = [108,134,156,0, 185,185,185,255, 0,0,0,0];
        let png = super::encode_opaque(3,1,&rgba);
        // IHDR follows the signature. Read the encoder's IDAT, then its filter0 row.
        let mut at = 8;
        let mut row = Vec::new();
        while at < png.len() {
            let n = u32::from_be_bytes(png[at..at+4].try_into().unwrap()) as usize;
            if &png[at+4..at+8] == b"IDAT" {
                row = miniz_oxide::inflate::decompress_to_vec_zlib(&png[at+8..at+8+n]).unwrap();
            }
            at += n + 12;
        }
        assert_eq!(row, [0,108,134,156,255,185,185,185,255,0,0,0,255]);
        assert_eq!(rgba[3],0);
        assert_eq!(rgba[11],0);
    }
}
