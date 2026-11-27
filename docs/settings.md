# Settings and saves

[Back to the main page](../README.md)

## Settings

- In the game: SETTINGS > PC. The tab has window mode, resolution, render scale, Anti-Aliasing / Upscaler, Upscaling, the frame rate limit and mouse sensitivity. Its lists hold the rest: Graphics & Display, Controls, Sound & Saves, and the key and controller bindings. Most settings take effect at once.
- In the launcher: the **Display** page sets the window mode, monitor, resolution, refresh rate, render scale and frame rate limit before the game starts.
- Motion Blur Fix is No Blur by default: the game's motion blur is off, so plants and detail stay sharp while the camera moves. Game's Blur uses the game's own Motion Blur setting, which smears every camera move. Reduced Blur keeps a tenth of it.
- Shadow Distance (Graphics & Display) shortens how far the sun's shadows reach. A shorter distance raises the frame rate in busy places.
- Performance Overlay (Graphics & Display) shows in the bottom-right corner the frame rate and its 1% and 0.1% lows over the last 20 seconds, the CPU use of the game's busiest thread (100% is a whole core, the limit when one thread holds the frame rate back), the GPU's use and clock, and video memory and memory, the game's and all in use. Frame Rate shows the first line only. The GPU clock needs an NVIDIA card.

## Frame rate and refresh rate

The game runs at the same speed at every frame rate. With the 60 limit each frame stays on the screen for the same number of refreshes, so motion stays even:

- At 60, 120 or 240 Hz the game shows 60 frames a second.
- With G-SYNC or FreeSync it shows 60 in borderless or exclusive fullscreen.
- In a window on a 165 Hz or 144 Hz screen it shows an even 55 or 48 (G-SYNC is often on only for fullscreen). For 60 there, use borderless or exclusive fullscreen with G-SYNC or FreeSync, or set the screen to 120 Hz.
- Exclusive fullscreen with Refresh Rate on Automatic picks a rate where 60 fits, unless the screen has G-SYNC or FreeSync.

## DLSS, FSR and XeSS

Upscaling (SETTINGS > PC) sets how far below your screen's resolution DLSS, FSR or XeSS render: Quality, Balanced, Performance or Ultra Performance. The upscaler then fills in to your screen's resolution. Performance runs faster, Quality looks sharper. Native AA renders at your screen's resolution and only smooths edges. With the game's TAA, Render Scale sets the resolution instead.

NVIDIA's 'nvngx_dlss.dll' is under NVIDIA's own license and not included. On the launcher's **Display** page, **Download DLSS...** fetches it from NVIDIA's DLSS SDK on GitHub once you accept NVIDIA's license, and checks it is NVIDIA's file as released. You can also copy your own into the 'upscalers' folder beside 'regaliant.exe' (games with DLSS have it in their installation folder). Then DLSS appears in the Anti-Aliasing list. XeSS, FSR and NVIDIA Streamline ship in the 'upscalers' folder with their licenses, and each is loaded only after its maker's digital signature checks out. DLSS is first tried in a short separate check: if your 'nvngx_dlss.dll' cannot be loaded, the game starts without DLSS and its log says why. It is tried again when the file changes.

NVIDIA and DLSS are trademarks of NVIDIA Corporation. Intel and XeSS are trademarks of Intel Corporation. AMD and FSR are trademarks of Advanced Micro Devices, Inc.

## Saves and files

- Settings and saves are in '%APPDATA%\Regaliant'. Before a session first changes a save, the game copies it to 'save-backups' there. The launcher's **Saves** page restores a backup.
- Saves made in another program, or a decrypted PS5 save: on the launcher's **Saves** page, **Import saves...** asks for the folder that holds them, such as that program's own folder. Save folders copied into the save folder by hand are found at the next start too. The saves an import replaces are kept in 'save-backups'.
- To play on with a decrypted PS5 save: put its folder (for example 'PPSA01341', holding 'SAVEDATA0PlayerProfile0' and 'SAVEDATA0OptionsProfile0', each with a 'USR-DATA' file, or the same folders named 'sdimg_SAVEDATA...') into a folder named 'PS5 Saves' in '%APPDATA%\Regaliant'. Turn on **Import PS5 Saves** in SETTINGS > PC, then quit and start the game again. The saves it replaces are kept in 'save-backups'.
- The pre-order and Digital Deluxe add-ons: put your own add-on packages (the .pkg files, such as 'EP9000-PPSA01341_00-DEMONSSOULSGIFT1.pkg' and '...GIFT2.pkg', or the folders unpacked from them) into a folder named 'Add-ons' in your settings folder, '%APPDATA%\Regaliant\Add-ons' (that is 'AppData\Roaming\Regaliant\Add-ons' in your user folder). Typing '%APPDATA%\Regaliant' into File Explorer's address bar opens that 'Roaming\Regaliant' folder. An 'Add-ons' folder in '%LOCALAPPDATA%\Regaliant' (under 'AppData\Local') or next to 'regaliant.exe' works too. The game finds them when it starts, and its log says what it found. Without them it finds none.
- The add-ons' items go to Stockpile Thomas's storage in the Nexus when a character loads (Organise storage). The pre-order's weapon is under the weapons tab, not under items. The game gives the Digital Deluxe items only to characters created while the add-ons are in place: a character made before you added them gets the pre-order's weapon but not those.
- Logs and prepared shaders are in '%LOCALAPPDATA%\Regaliant'. The launcher's **Open logs** button opens the logs.
- An empty file named 'portable.txt' next to 'regaliant.exe' keeps everything in a 'user' folder beside it.
- When the game cannot go on, it says so and names its log folder. Logs leave out your user name, your computer's name and your profile folder.
