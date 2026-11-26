# Building from source

[Back to the main page](../README.md)

For the experimental Linux native bootstrap, see [Linux build and limitations](linux.md). The steps below build the Windows release.

1. Install Rust with rustup (the x86_64-pc-windows-msvc toolchain) and the Visual Studio Build Tools with the C++ workload.
2. Run `scripts\upscalers.ps1` once. It downloads the upscalers' libraries (FSR, XeSS and NVIDIA Streamline) with their licenses into `local-data\upscalers` and checks their signatures. NVIDIA's own DLSS file is never packaged.
3. Run `scripts\package.ps1`. It builds both programs with the release settings and writes `dist\Regaliant-<version>.zip`. `-NoUpscalers` leaves the upscalers out.

`scripts\cargo.ps1 build --release` builds the programs without packaging them. The game itself is never part of the build. The programs read it from the folder you choose in the launcher.
