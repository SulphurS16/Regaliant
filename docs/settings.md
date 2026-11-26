# Settings and saves

[Back to the main page](../README.md)

## Settings

- In the game: SETTINGS > PC. The tab has window mode, resolution, render scale, Anti-Aliasing / Upscaler, the frame rate limit, the Motion Blur Fix and mouse sensitivity. Its lists hold the rest: Graphics & Display, Controls, Sound & Saves, and the key and controller bindings. Most settings take effect at once.
- In the launcher: the **Display** page sets the window mode, monitor, resolution, refresh rate, render scale and frame rate limit before the game starts.
- Motion Blur Fix is No Blur by default: the game's motion blur is off, so plants and detail stay sharp while the camera moves. Game's Blur uses the game's own Motion Blur setting, which smears every camera move. Reduced Blur keeps a tenth of it.
- Shadow Distance (Graphics & Display) shortens how far the sun's shadows reach. A shorter distance raises the frame rate in busy places.

## Frame rate and refresh rate

The game runs at the same speed at every frame rate. With the 60 limit each frame stays on the screen for the same number of refreshes, so motion stays even:

- At 60, 120 or 240 Hz the game shows 60 frames a second.
- With G-SYNC or FreeSync it shows 60 in borderless or exclusive fullscreen.
- In a window on a 165 Hz or 144 Hz screen it shows an even 55 or 48 (G-SYNC is often on only for fullscreen). For 60 there, use borderless or exclusive fullscreen with G-SYNC or FreeSync, or set the screen to 120 Hz.
- Exclusive fullscreen with Refresh Rate on Automatic picks a rate where 60 fits, unless the screen has G-SYNC or FreeSync.

## DLSS, FSR and XeSS

NVIDIA's 'nvngx_dlss.dll' is under NVIDIA's own license and not included. Copy your own into the 'upscalers' folder beside 'regaliant.exe' (games with DLSS have it in their installation folder). Then DLSS appears in the Anti-Aliasing list. XeSS, FSR and NVIDIA Streamline ship in the 'upscalers' folder with their licenses, and each is loaded only after its maker's digital signature checks out. DLSS is first tried in a short separate check: if your 'nvngx_dlss.dll' cannot be loaded, the game starts without DLSS and its log says why. It is tried again when the file changes.

## DLSS Frame Generation and NVIDIA Reflex

These are in SETTINGS > PC > Graphics & Display.

- **Frame Generation**: Off or DLSS Frame Generation. It needs a GeForce RTX 40 or newer, a recent driver, Windows' Hardware-accelerated GPU scheduling turned on, and NVIDIA's 'nvngx_dlssg.dll'. Like 'nvngx_dlss.dll', that file is not included: copy your own into the 'upscalers' folder (games with DLSS Frame Generation have it in their installation folder). Without it the list shows only Off, and the log says why.
- **Frame Generation Mode**: how many frames are shown for each frame the game renders. The list goes from 2x up to what your GPU can do: 2x on RTX 40, up to 6x on RTX 50 with a recent Frame Generation file. The game itself still renders 60 frames a second, so with 3x you see 180.
- Frame Generation uses the game's own depth and motion vectors, the same ones DLSS gets. It also uses the picture without the menus and HUD, so text and markers do not smear. It works with any Anti-Aliasing choice.
- In HDR the generated frames are made from the 10-bit HDR10 picture the screen gets. Highlights are not clipped and there is no extra banding.
- Frame Generation uses the newest full model in your 'nvngx_dlssg.dll'. Neither DLSS nor Streamline lets a game pick another model. The NVIDIA App's DLSS override can still change it for the game, and that choice is honored.
- **NVIDIA Reflex Low Latency**: Off, On (the default) or On + Boost. It lowers the delay between your input and the screen on GeForce GPUs, with or without Frame Generation. NVIDIA requires Reflex with Frame Generation, so while Frame Generation is on, Off counts as On. On + Boost also keeps the GPU's clocks up, which can lower latency a little more when the game is limited by the CPU.
- For the smoothest picture with Frame Generation, turn on G-SYNC in the NVIDIA Control Panel and play in borderless or exclusive fullscreen. With Frame Generation the game always presents without V-Sync. To avoid tearing above the screen's refresh rate, turn on V-Sync for the game in the NVIDIA Control Panel, not in the game. Frame Generation then keeps the frame rate under the refresh rate.

### Window modes and presentation

- Borderless fullscreen hands its frames straight to the screen (independent flip), like exclusive fullscreen, so the desktop compositor adds no latency. G-SYNC and FreeSync work there too.
- Exclusive fullscreen works as before.
- Without G-SYNC, in a window or in borderless mode, the game presents new frames as soon as they are ready and drops none of them early (mailbox). The game's own pacing keeps the frame rate even. With G-SYNC or FreeSync it waits for the screen as before, so the screen follows the game's frames.

NVIDIA, DLSS, G-SYNC and Reflex are trademarks of NVIDIA Corporation. Intel and XeSS are trademarks of Intel Corporation. AMD and FSR are trademarks of Advanced Micro Devices, Inc.

## Saves and files

- Settings and saves are in '%APPDATA%\Regaliant'. Before a session first changes a save, the game copies it to 'save-backups' there. The launcher's **Saves** page restores a backup.
- To play on with a decrypted PS5 save: put its folder (for example 'PPSA01341', holding 'SAVEDATA0PlayerProfile0' and 'SAVEDATA0OptionsProfile0', each with a 'USR-DATA' file; folders named 'sdimg_SAVEDATA...' work too) into a folder named 'PS5 Saves' in '%APPDATA%\Regaliant'. Turn on **Import PS5 Saves** in SETTINGS > PC, then quit and start the game again. The saves it replaces are kept in 'save-backups'.
- The pre-order and Digital Deluxe add-ons: put your own add-on packages (the .pkg files, such as 'EP9000-PPSA01341_00-DEMONSSOULSGIFT1.pkg' and '...GIFT2.pkg', or the folders unpacked from them) into a folder named 'Add-ons' in your settings folder, '%APPDATA%\Regaliant\Add-ons' (that is 'AppData\Roaming\Regaliant\Add-ons' in your user folder). Typing '%APPDATA%\Regaliant' into File Explorer's address bar opens that 'Roaming\Regaliant' folder. An 'Add-ons' folder in '%LOCALAPPDATA%\Regaliant' (under 'AppData\Local') or next to 'regaliant.exe' works too. The game finds them when it starts, and its log says what it found. Without them it finds none.
- The add-ons' items go to Stockpile Thomas's storage in the Nexus when a character loads (Organise storage). The pre-order's weapon is under the weapons tab, not under items. The game gives the Digital Deluxe items only to characters created while the add-ons are in place: a character made before you added them gets the pre-order's weapon but not those.
- Logs and prepared shaders are in '%LOCALAPPDATA%\Regaliant'. The launcher's **Open logs** button opens the logs.
- An empty file named 'portable.txt' next to 'regaliant.exe' keeps everything in a 'user' folder beside it.
- When the game cannot go on, it says so and names its log folder. Logs leave out your user name, your computer's name and your profile folder.
