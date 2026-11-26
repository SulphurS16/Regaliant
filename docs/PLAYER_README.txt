Regaliant 0.1.2 - an unofficial PC port of Demon's Souls for Windows (64-bit)
=============================================================================

Regaliant is an unofficial PC port of Demon's Souls (PS5, PPSA01341). It is not rebuilt from the game's
source code. It runs the game's own PS5 program from your copy of the game's files. The
program's x86-64 code runs directly on your processor. The PS5 system libraries it calls are
reimplemented for Windows, and its graphics commands and shaders are translated to Vulkan.
Nothing of the game is included.

Not affiliated with, endorsed or sponsored by Sony Interactive Entertainment, Bluepoint Games,
FromSoftware or any other rights holder.

This is an early version. Read "Known issues" below before you play.

News, help and more ports: join the Discord at https://discord.gg/rSr5JhxxtZ
Bug reports: https://github.com/SulphurS16/Regaliant/issues


What you need
-------------
- Your own copy of Demon's Souls for PS5 (PPSA01341) at version 01.007.000, as a folder or as
  a .zar archive of that folder. The folder holds sce_sys, the game's data folders and
  eboot.bin. The executable and its two libraries must be decrypted: a decrypted eboot.bin (in
  the folder or in its "decrypted" folder) and decrypted libc.prx and libSceNpCppWebApi.prx in
  sce_module, as plain ELF files or in their SELF containers with the segments decrypted.
- Windows 10 or 11, 64-bit.
- A graphics card and driver with Vulkan 1.3.
- A processor with AVX2 (Intel from 2013 on, AMD Ryzen).
- Disk space: up to 4 GB for prepared shaders, besides the game files.

Tested: Ryzen 9 5900X (12 cores), 32 GB of memory and an NVIDIA RTX 4090. See "System
requirements" for what was measured and what is estimated.


Getting started
---------------
1. Unpack this zip into a folder, for example C:\Games\Regaliant.
2. Start regaliant-launcher.exe. If Windows says "Windows protected your PC", click
   "More info", then "Run anyway". The programs are not code-signed.
3. On the Game page, choose the game folder (Folder...) or a .zar archive of it
   (.zar file...). The launcher checks it and says what is missing.
4. Click Play. The first time, the launcher prepares the game's shaders before the game
   starts. This takes about an hour once on a fast PC, longer on others (the Shaders page
   says how long). You can turn it off on the Shaders page.


Controls
--------
Controllers: Xbox and other XInput controllers, and PlayStation controllers (DualSense,
DualSense Edge, DUALSHOCK 4) over USB or Bluetooth, without Steam Input. Connect them at any
time. All connected controllers play as one. The prompts show the controller's own buttons.
Vibration works on both, and a PlayStation controller's light bar follows the game. If you
remapped buttons in Steam Input or DS4Windows, turn it off for this game: the game also reads
the controller itself, so the original buttons press too.

Keyboard and mouse (defaults, Dark Souls III-style):
  W A S D            move                      Left Alt         walk
  mouse              camera                    I J K L          camera (keys)
  Space              dash (hold), roll, back in menus
  left button        attack                    Shift+left button  strong attack
  right button       guard                     Left Ctrl        parry
  F                  two-hand                  R                use item
  E, Enter           interact, confirm         Q, middle button lock on
  G                  tool belt                 Escape           menu
  arrow keys, wheel  switch spells, items and weapons
  F1                 free the mouse            Alt+Enter        fullscreen or window

In menus the mouse pointer selects what it points at, the left button confirms and the right
button goes back. The prompts show the keys you bound. Every key can be changed in the game:
SETTINGS > PC > Key Bindings.

To enter your character's name the game shows a text box. Type the name and press Enter.
Escape cancels. With a controller the box shows a keyboard: the d-pad or left stick moves,
A types, X erases, Y adds a space, LB is Shift, RB shows accents and symbols, Menu confirms
and B cancels.


PC settings
-----------
The game's SETTINGS screen has a PC tab with the port's settings. Most take effect at once.
- On the tab: window mode (windowed, borderless or exclusive fullscreen), resolution, render
  scale, Anti-Aliasing / Upscaler (the game's TAA, off, DLSS, FSR or XeSS), frame rate limit
  (30, 60 or uncapped), Motion Blur Fix and mouse sensitivity.
- Graphics & Display: quality preset, geometry detail, texture filtering, sharpening, bloom,
  vignette, HDR, aspect ratio (16:9, 16:10, 21:9, 32:9), monitor and refresh rate.
- Controls, Sound & Saves: mouse inversion, menu and HUD size, HUD margin, button prompts,
  sound output (headphones, stereo, 5.1, 7.1, Windows Sonic or Dolby Atmos) and device, the
  controller speaker's sounds through your speakers (Pad Speaker), skip intro logos, save
  folder, import of a PS5 save.
- Key Bindings and Controller Bindings: every action's key, mouse button and controller button.

Motion Blur Fix is No Blur by default: the game's motion blur is off, so plants and detail
stay sharp while the camera moves. Game's Blur uses the game's own Motion Blur setting, which
smears every camera move. Reduced Blur keeps a tenth of it.

DLSS needs one more file. NVIDIA's nvngx_dlss.dll is under NVIDIA's own license and does not
come with the port. Copy your own into the "upscalers" folder beside regaliant.exe. Games with DLSS
have the file in their installation folder. Then DLSS appears in the Anti-Aliasing list. XeSS and
FSR are in the "upscalers" folder already. Each library is checked for its maker's digital
signature before it is loaded. NVIDIA and DLSS are trademarks of NVIDIA Corporation. Intel and
XeSS are trademarks of Intel Corporation. AMD and FSR are trademarks of Advanced Micro Devices,
Inc.


Frame rate and refresh rate
---------------------------
The game runs at the same speed at every frame rate. With the 60 limit each frame stays on
the screen for the same number of refreshes, so motion stays even:
- At 60, 120 or 240 Hz the game shows 60 frames a second.
- With G-Sync or FreeSync it shows 60 in borderless or exclusive fullscreen.
- In a window on a 165 Hz or 144 Hz screen it shows an even 55 or 48 (G-Sync is often on
  only for fullscreen). For 60 there, use borderless or exclusive fullscreen with G-Sync or
  FreeSync, or set the screen to 120 Hz.
- Exclusive fullscreen with the Automatic refresh rate picks a rate where 60 fits, unless the
  screen has G-Sync or FreeSync.


The launcher
------------
- Game: the game folder or its .zar archive and its check, Compress to .zar (see "Saving disk
  space"), what your PC brings, and safe mode for the next start.
- Display: window mode, monitor, fullscreen resolution and refresh rate, render scale and
  frame rate limit for the next start. The game's PC tab changes the same settings.
  Safe mode starts the game in a window with the default settings and without an upscaler,
  Frame Generation or Reflex. Use it when the game does not start.
- Shaders: prepares the game's shaders ahead. Without them the game compiles each shader the
  first time it needs it. An object can then be missing for a moment, and some compiles make
  play stutter. Prepared shaders make the stutters much shorter, but some remain on first
  visits. They hold for one graphics card and driver. After a graphics driver update or a port
  update, the launcher compiles them again without the game playing again. Some drivers stop
  taking prepared shaders after Windows events. The launcher notices this when it starts and
  compiles them again too. On battery, Play starts the game without preparing, and a
  preparation stops when the battery runs low. If preparing cannot finish on your PC, the
  Shaders page says why, and Play starts the game.
- Saves: the save folder and the backups the game makes. The game copies each save before a
  session first changes it. "Restore..." puts a backup back. The save as it is now is kept
  as a backup first.
- Reset settings puts the PC settings, display choices and key bindings back to their
  defaults. The old files are kept as .bak files.
- Open logs opens the logs folder with the newest session's folder selected. The launcher's own
  quick checks keep their logs in its checks folder.


Saving disk space
-----------------
The launcher can pack the game folder into one .zar archive, which the game reads in place.
With the tested copy the archive takes 54 GB instead of the folder's 82 GB.

1. On the Game page, with the game folder chosen, click "Compress to .zar..." and choose where
   to save the archive. That drive needs about two thirds of the folder's size free.
2. Wait. On the tested PC it took about 3 to 4 minutes. You can stop it, and closing the
   launcher stops it too. An unfinished archive is removed.
3. When it is done, the launcher offers to play from the archive. Prepared shaders, settings
   and saves carry over.
4. Once the game runs from the archive, delete the game folder yourself to free its space. Keep
   it if you want to switch back with "Folder...".

What you trade, measured on the tested PC (both copies on the same NVMe drive): the game starts
about 2 seconds sooner from the archive, and a level's first load in a session is the same or
faster (up to 3 seconds on one level). Reloading a level whose data Windows still holds in
memory takes about 0.3 seconds longer (8.3 instead of 8.0). Frame rate and stutter while
playing are the same. Slower drives and processors were not measured.


Settings, saves and logs
------------------------
- Settings, key bindings and saves: %APPDATA%\Regaliant.
- Logs, prepared shaders and the game's own temporary files: %LOCALAPPDATA%\Regaliant.
- Portable use: an empty file named portable.txt next to regaliant.exe keeps everything in a
  "user" folder beside it.
- Logs leave out your user name, your computer's name and your profile folder.


When something goes wrong
-------------------------
The game tells you when it cannot go on, for example when the graphics card stops responding.
It then closes and names its log folder. Please report the problem on the project's GitHub
issues page and attach that session's log folder (dsport.log, and crash.txt when there is
one). If the game does not start at all, try safe mode in the launcher. For a problem with the
shader preparation, attach the whole logs folder (Open logs): launcher.log and a folder for
each of its steps.


System requirements
-------------------
Tested on one PC: Ryzen 9 5900X (12 cores), 32 GB of memory, NVIDIA RTX 4090 with driver
610.88, Windows 11.

Memory, measured there: the game reserves about 35 to 40 GB of memory and page file. While you
play an area it keeps about 4 to 5 GB in active use. 32 GB is recommended. The graphics card
reads part of the game's memory directly, and Windows lets it use about half of the installed
memory for that. 16 GB has not been tested on a real PC. In a simulation of a 16 GB PC on the
tested PC, the game played every tested level at the same frame rates, and a first session's
level loads took about as long. With 16 GB, keep Windows' page file on automatic and about 30 GB
free on the system drive.

Graphics card: Vulkan 1.3. Only an RTX 4090 was tested. Smaller cards were simulated on it by
giving the game less video memory. That shows how much memory the game needs, but not how fast
a smaller card is.
- 12 GB or more: the game fits.
- 8 to 12 GB: in busy places the game keeps some textures in system memory. The picture stays
  right. The frame rate there can drop.
- 6 GB: the game fits once some textures go to system memory. In the tested levels it drew
  right, at the same frame rates as with 7 GB on the 4090.
- 4 GB: the game runs and draws right in the tested levels, but even on the 4090 it ran at
  about 9 to 30 frames a second with what a 4 GB card leaves a game. Cards with 4 GB are
  usually slow graphics chips too, so expect low frame rates. The Low preset in the game's PC
  tab lowers the graphics card's work. A 4 GB card in a PC with 16 GB of memory has not been
  tested.
- Less than 4 GB: not supported. Parts of some levels drew wrong in the simulation.
AMD and Intel graphics cards have not been tested.

Processor: limited to 6 of its 12 cores, the tested processor kept 91 to 94% of its frame
rate. A slower processor lowers the frame rate in proportion to its single-core speed.


Known issues
------------
- The frame rate is below 60 in the busiest places: about 40 to 55 fps on the tested PC.
- In a window on a 144 or 165 Hz screen the game shows an even 48 or 55 fps. See "Frame rate
  and refresh rate" for how to get 60.
- On a first visit some objects can be missing for a moment while their shaders compile, and
  some compiles still stutter, even with prepared shaders. A level's first load can pause for
  a few seconds behind the loading screen. Later visits do not compile them again.
- Areas, bosses and the story were reached by the developer's shortcuts in testing. A full
  playthrough has not been made.
- Water outside the Nexus can show bands. Some other rare picture faults are known.
- The sky in the opening cutscene is a little darker than on PS5. Some light shafts look
  softer too.
- At 32:9 the character preview in the equipment menu stands at the right edge.
- In character creation the header can turn bright cyan after the button prompts change, for
  example when you switch between the controller and the keyboard. It is only cosmetic.
- In character creation the character fills only part of the screen when Render Scale is set
  to a percentage. Visual Mode shows it whole.
- HDR has not been seen on an HDR screen yet.
- AMD and Intel graphics cards have not been tested.
- On graphics cards with 12 GB or less the game can run short of graphics memory in busy
  places. It then keeps some textures in system memory, and the frame rate there can drop.
- On cards with 4 or 5 GB the game keeps its large images in system memory from the start. It
  runs, but slowly.
- PCs with 16 GB of memory have not been tested on a real PC. A simulation of one played the
  tested levels. 32 GB is recommended.
- Graphics cards without fragment shader barycentrics draw some surfaces through an extra
  shader stage, which costs some speed. NVIDIA's GTX 10 series and older and Intel Arc A-series,
  Meteor Lake and Arrow Lake graphics are among them. This was only simulated on an NVIDIA card.
- Online features (messages, bloodstains, co-op and invasions) are not available. The game
  plays offline.
- Linux, Proton and the Steam Deck are not supported yet.


License
-------
Copyright (C) 2026 SulphurS16 and the Regaliant contributors.

The port is free software: you can redistribute it and/or modify it under the terms of the
GNU General Public License as published by the Free Software Foundation, either version 3 of
the License, or (at your option) any later version. It is distributed in the hope that it will
be useful, but WITHOUT ANY WARRANTY, without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License, whose full text follows
at the end of this file, for more details.

Additional permission under GNU GPL version 3 section 7: If you modify this program, or any
covered work, by linking or combining it with NVIDIA DLSS (the NGX runtime nvngx_dlss.dll and
the NVIDIA Streamline libraries) or Intel XeSS (libxess.dll), or modified versions of those
libraries, containing parts covered by the terms of their own licenses, the licensors of this
program grant you additional permission to convey the resulting work. The Corresponding Source
for a non-source form of such a combination need not include the source code of those
libraries.

The license covers the port's own code only, not the game. The notices of the third-party
components in the two programs follow below.
