//! PPSA01341 01.007.000 constants, captured with the TAA dispatch, not by flip-counter guess.
//! See docs/linux-frame-generation.md for shader PCs and CPU writer evidence.
#[derive(Clone, Debug)]
pub struct FrameConstants {
    pub near: f32,
    pub far: f32,
    pub fov: f32,
    pub infinite: bool,
    pub jitter: [f32; 2],
    pub history: bool,
    pub reprojection: [f32; 16],
    pub globals: Vec<u32>,
    pub taa: Vec<u32>,
}

impl FrameConstants {
    pub fn decode(globals: Vec<u32>, taa: Vec<u32>, size: (u32, u32)) -> Result<Self, String> {
        if globals.len() < 0x2b0 / 4 || taa.len() < 0x64 / 4 {
            return Err("short per-frame constants".into());
        }
        let f = |byte: usize| f32::from_bits(globals[byte / 4]);
        // Global +0x40 is inverse projection, row-major. The depth consumers use
        // viewZ = (m[10]*depth+m[11]) / (m[14]*depth+m[15]).
        let m: [f32; 16] = std::array::from_fn(|i| f(0x40 + 4*i));
        let prev: [f32; 16] = std::array::from_fn(|i| f(0x180 + 4*i));
        if !m.iter().chain(prev.iter()).all(|x| x.is_finite())
            || m[0] <= 0.0 || m[5] <= 0.0 || m[11].abs() < 1e-6 || m[14].abs() < 1e-6
            || [1,3,4,7,8,9,10,12,13].iter().any(|&i| m[i].abs() > 1e-5)
        {
            return Err("unsupported/non-perspective inverse projection".into());
        }
        let near = (m[11] / (m[14] + m[15])).abs();
        let infinite = m[15] == 0.0;
        // SDK sorts near/far even with INFINITE set, so use finite FLT_MAX there.
        let far = if infinite { f32::MAX } else { (m[11] / m[15]).abs() };
        let fov = 2.0 * (m[5] / m[11]).abs().atan();
        if !near.is_finite() || !(1e-5..1e4).contains(&near) || !far.is_finite()
            || far <= near || !(0.01..3.13).contains(&fov)
        {
            return Err("projection is not supported reversed-Z depth".into());
        }
        // The title's depthsharedprep subtracts +0x1c0.xy / 2 from SCREEN UV.
        // Thus these are twice the UV jitter, already Y-down. No Halton lookup,
        // guessed frame pairing, or extra Y flip belongs here.
        let jitter = [f(0x1c0) * size.0 as f32 * 0.5, f(0x1c4) * size.1 as f32 * 0.5];
        let dims = [f(0x2a0), f(0x2a4), f(0x2a8), f(0x2ac)];
        if !jitter.iter().all(|x| x.is_finite() && x.abs() <= 2.0)
            || (dims[0] - size.0 as f32).abs() > 0.5 || (dims[1] - size.1 as f32).abs() > 0.5
            || (dims[2] * size.0 as f32 - 1.0).abs() > 1e-4
            || (dims[3] * size.1 as f32 - 1.0).abs() > 1e-4
        {
            return Err("per-view resolution/jitter does not match TAA depth/MV".into());
        }
        let reprojection = std::array::from_fn(|i| {
            (0..4).map(|k| prev[i/4*4+k] * m[k*4+i%4]).sum()
        });
        if !reprojection.iter().all(|x: &f32| x.is_finite()) {
            return Err("non-finite camera reprojection".into());
        }
        let history = taa[0x60/4] & 1 != 0;
        Ok(Self { near, far, fov, infinite, jitter, history, reprojection, globals, taa })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture(far: Option<f32>) -> (Vec<u32>, Vec<u32>) {
        let mut g = vec![0; 0x2b0/4];
        let n = 0.25;
        let b = far.map_or(n, |f| n*f/(f-n));
        let a = far.map_or(0.0, |f| n/(f-n));
        let inv = [1.0,0.0,0.0,0.0, 0.0,0.75,0.0,0.0, 0.0,0.0,0.0,-1.0, 0.0,0.0,1.0/b,a/b];
        for (i,v) in inv.into_iter().enumerate() { g[0x40/4+i] = f32::to_bits(v); }
        for i in 0..4 { g[0x180/4+5*i] = 1.0f32.to_bits(); }
        for (o,v) in [(0x1c0,0.5/1920.0), (0x1c4,-0.5/1080.0), (0x2a0,1920.0),
            (0x2a4,1080.0), (0x2a8,1.0/1920.0), (0x2ac,1.0/1080.0)] { g[o/4]=f32::to_bits(v); }
        let mut t = vec![0;0x64/4]; t[0x60/4]=1; (g,t)
    }
    #[test]
    fn real_planes_jitter_and_history() {
        for far in [Some(1200.0),None] {
            let (g,t)=fixture(far); let c=FrameConstants::decode(g,t,(1920,1080)).unwrap();
            assert!((c.near-0.25).abs()<1e-5);
            assert!((c.fov-2.0*0.75f32.atan()).abs()<1e-5);
            assert_eq!(c.infinite,far.is_none());
            if let Some(f)=far { assert!((c.far-f).abs()<0.01); }
            assert_eq!(c.jitter,[0.25,-0.25]); assert!(c.history);
        }
        let (g,mut t)=fixture(None); t[0x60/4]=2;
        assert!(!FrameConstants::decode(g,t,(1920,1080)).unwrap().history);
    }
    #[test]
    fn reject_wrong_target_and_bad_constants() {
        let (mut g,t)=fixture(None);
        assert!(FrameConstants::decode(g.clone(),t.clone(),(3840,2160)).is_err());
        g[0x40/4+14]=f32::NAN.to_bits();
        assert!(FrameConstants::decode(g,t,(1920,1080)).is_err());
        assert!(FrameConstants::decode(vec![],vec![],(1920,1080)).is_err());
    }
}
