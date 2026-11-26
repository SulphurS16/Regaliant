<p align="center"><img src="assets/regaliant-logo.png" alt="Regaliant" width="560"></p>

> **This is a Linux-focused fork of [SulphurS16/Regaliant](https://github.com/SulphurS16/Regaliant).**
> It adds a native Linux build. The work happens on the `linux-support` branch, which is this fork's default.
> Windows support still comes from the original project, and this fork follows it. For Windows builds, releases and the Discord, see the original repository.
> Please report Linux problems in this fork's issues, not upstream.

Regaliant is an unofficial PC port of Demon's Souls (PS5). It runs the game's own PS5 program from your copy of the game, with no CPU emulation. The PS5's system libraries are reimplemented for the PC, and the game's graphics run on Vulkan. The original project targets Windows. This fork adds a native Linux build.

**You need your own copy of the game.** Regaliant contains no game data, no game code and no keys, and it decrypts nothing. It needs Demon's Souls for PS5 (PPSA01341) at version 01.007.000, with its executable and two libraries decrypted. See the [install guide](docs/install.md).

Not affiliated with, endorsed or sponsored by Sony Interactive Entertainment, Bluepoint Games, FromSoftware or any other rights holder. Demon's Souls is a trademark of Sony Interactive Entertainment.

This is an early version.

## Linux

The Linux build is experimental. It runs the game natively on x86-64 Linux, with Vulkan graphics and SDL3 windowing, input and audio. No Wine or Proton is involved. Regaliant builds from source: see the [Linux guide](docs/linux.md) for the build and play commands.

What works:

- Gameplay and resuming saves.
- Fullscreen on fractional-scale displays.
- The full shader-preparation tour.
- Intro movies at their own 30 fps on any refresh rate.
- Optional [FSR3 frame generation](docs/linux-frame-generation.md).

Still in progress:

- Frame rate in busy areas.
- Stutter.
- Graphical glitches: oversized glows, stray geometry, some black textures.
- Audio dropouts in long sessions.

Testing so far has been on one NVIDIA PC. The Features, Performance and Known issues sections below describe the Windows build. Not everything there applies to Linux yet.

## Features

- No CPU emulation: the game's own x86-64 code runs directly on your processor.
- A PC tab in the game's own SETTINGS screen. Most settings take effect at once.
- Windowed, borderless or exclusive fullscreen with a choice of monitor, resolution and refresh rate.
- 16:9, 16:10, 21:9 and 32:9 without stretching. A wider screen shows more to the sides.
- HDR output through the game's own HDR mode.
- NVIDIA DLSS, AMD FSR and Intel XeSS as anti-aliasing, fed the game's own motion vectors and jitter. DLSS needs your own copy of NVIDIA's DLSS file.
- Render scale, sharpening, quality presets, geometry detail, shadow distance, texture filtering, bloom and vignette.
- A frame rate limit of 30, 60 or uncapped. The game keeps its normal speed at any frame rate.
- Even frame pacing at any refresh rate, with G-SYNC and FreeSync.
- Keyboard and mouse with raw mouse look, a mouse pointer in menus and prompts for the keys you bound. Every key can be changed in the game.
- Xbox and PlayStation controllers with their own button prompts. A DualSense works directly over USB or Bluetooth, without Steam Input.
- 5.1 and 7.1 sound, and headphone output of the game's 3D sound, also through Windows Sonic or Dolby Atmos for Headphones.
- Trophies as achievements with a pop-up and a list.
- A launcher that checks your game files, sets the display before the game starts, prepares the shaders ahead (the game plays itself in the background, hidden and silent, once: about an hour on a fast PC, longer on others), backs up and restores saves, and offers a safe mode.
- The game can also be one .zar archive, about two thirds of the folder's size and read in place. The launcher makes one from your game folder ([what you trade](docs/install.md#save-disk-space-with-a-zar-archive)).

## Performance

The game runs at 60 fps or close to it in many places, and lower in the busiest ones: about 40 to 55 fps there on the main test PC (Ryzen 9 5900X, RTX 4090). The processor's single-core speed matters most.

Shader preparation happens once. The first time you press Play, the game plays itself in the background, hidden and silent, to prepare its shaders: about an hour on a fast PC, longer on others. After that the game starts right away. After a graphics driver or port update, the launcher only compiles them again, without the self-play.

## Known issues

- In a window on a 144 or 165 Hz screen the game shows an even 48 or 55 fps. Borderless or fullscreen with G-SYNC or FreeSync, or a 120 Hz screen, gives 60 ([more](docs/settings.md#frame-rate-and-refresh-rate)).
- Shaders the preparation did not meet (rare effects, spells) still compile on first sight, and a level's first arrival can pause briefly.
- A full playthrough has not been tested.
- AMD and Intel graphics cards, HDR screens and 16 GB of memory have not been tested.
- Online features are not available. The game plays offline.

See [all known issues](docs/known-issues.md). Found a bug? Here is [how to report a bug](docs/report-a-bug.md).

## Roadmap

Goals, roughly in order. They are aims, not promises or dates.

- A steady 60 fps everywhere, the busiest places included, then higher frame rates for high-refresh screens.
- Shorter shader preparation, then none. Shaders get compiled in the background while you play, without stutter.
- Linux (this fork): a steady frame rate in the busiest places, no graphical glitches, and testing on AMD and Intel graphics.
- Mod support: a mods folder whose files the game loads in place of its own, so mods never change your game files.
- Online play: messages, bloodstains, co-op and invasions through a community server. This is the biggest goal and the furthest off.

## More

- [Install, update and uninstall](docs/install.md)
- [Controls](docs/controls.md)
- [Settings and saves](docs/settings.md)
- [System requirements](docs/requirements.md)
- [How it works](docs/how-it-works.md)
- [Building from source](docs/building.md)
- [License](docs/license.md)

Regaliant is free software under the GNU General Public License version 3 or later, with an additional permission for DLSS and XeSS. See [License](docs/license.md).

Please don't share game files, keys, decrypted files or download links in this project's issues or on the Discord.
