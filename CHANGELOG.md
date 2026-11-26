# Changelog

## Unreleased

- NVIDIA DLSS Frame Generation (SETTINGS > PC > Graphics & Display): 2x up to 6x, as far as
  the GPU allows. It uses the game's own depth and motion vectors and the picture without the
  menus and HUD, and it works in SDR and HDR (10-bit HDR10). It needs your own
  'nvngx_dlssg.dll' in the 'upscalers' folder ([more](docs/settings.md#dlss-frame-generation-and-nvidia-reflex)).
- NVIDIA Reflex Low Latency: Off, On (the default) or On + Boost, with or without Frame
  Generation. Frame Generation always runs with at least On.
- Borderless fullscreen now presents straight to the screen (independent flip) instead of
  through the desktop compositor. On screens without G-SYNC, windowed and borderless modes no
  longer wait for the screen's refresh (mailbox).
- DLSS (upscaling) now shares its NVIDIA Streamline setup with Frame Generation and Reflex.

## 0.1.7

- Fixed the crash with Aspect Ratio 21:9 or 32:9 and Visual Mode Cinematic, also at every
  start after it (issue #20). At shapes wider than 16:9 the game now renders at most 1512
  lines, which its render targets have room for.
- Fixed DLSS, FSR and XeSS not working after a restart of the game. The Anti-Aliasing setting
  still showed the upscaler, but the game used its own TAA. Now the chosen upscaler takes over
  as soon as it is ready.
- Fixed a crash inside the graphics driver a few seconds after DLSS started, seen here with the
  newest DLSS file (310.9.16). When the game combined two batches of graphics work, it gave the
  upscaler the wrong command buffer. DLSS, FSR and XeSS now draw into the right one. This likely
  also fixes the game closing without a message with FSR on (issue #5).
- The shader preparation now always uses the game's own TAA, even with an upscaler saved. Its
  hidden sessions show nothing, so a saved DLSS choice cannot get in its way.
- A crash inside the graphics driver now leaves a line in the log that names where it happened.
- Ctrl+F10 in the game's window writes what the game is waiting for into the log. If the game
  hangs but keeps drawing, such as a loading screen that never ends (issue #14), press it,
  wait a few seconds before closing the game, then attach the log.
- Motion Blur Fix's choices are now named by what they give: No Blur (the default), Reduced
  Blur and Game's Blur. They were On, Reduced Blur and Off, and "Off" read as no blur while it
  turned on the game's full motion blur, which smears every camera move (issue #6). Your saved
  choice stays as it was.
- When another program holds part of the memory the game needs ("status 10") with memory that
  names no owner, the message and `early-failure.txt` now also list the modules other programs
  put into the game at that moment, to help find the program (issue #15).

## 0.1.6

- Fixed the shader preparation and the game ending when the graphics driver stops on a debug
  break while it compiles a shader (seen with NVIDIA's driver 621.40 as "crashed (status
  0x80000003)", and likely the silent ends of issue #5). The game runs under its own helper,
  which counts as a debugger, so the driver broke where it would otherwise go on. It now goes
  on. Should the driver still fail there, the shader is left out, the step runs again, and the
  log names the shader.
- On PCs where Vulkan lists the graphics card twice, for example through Windows' OpenCL,
  OpenGL and Vulkan Compatibility Pack, the game now takes the card's own driver. Before, it
  could take the other entry and stop with "vkCreateDevice: Extension specified does not
  exist". The log lists every device with its driver and what it lacks.
- When another program holds part of the memory the game needs ("status 10"), the game tries
  again for a moment. If it still cannot start, a file in the logs folder names what holds the
  memory (the launcher's Open logs shows it).

## 0.1.5

- Fixed the game not starting at all for players who put their own DLSS file
  ('nvngx_dlss.dll') in the 'upscalers' folder and whose DLSS version could not be loaded: every
  start, and the shader preparation, ended with an error at once. DLSS is now first tried in a
  separate short check; if that check fails, the game starts without DLSS and its log says why.
  It tries again when the DLSS files or Regaliant change. On PCs without an NVIDIA driver DLSS is
  not loaded at all.
- When a graphics driver crashes while it compiles one of the game's shaders, the game now
  remembers which compiles were running even when the crash leaves no error report, and the
  next start leaves them out of the background compiles (a compile that crashes again is left
  out everywhere). Before, such a crash could repeat at every start, as seen with NVIDIA's
  driver 621.40. The log now names the shaders on that list.
- Fixed video memory filling up and the game slowing down the longer it ran with DLSS, XeSS or
  FSR chosen. In menus that show your character (the equipment menu), the game runs its
  anti-aliasing on the small preview too, and the upscaler followed it: it switched size every
  frame or two and kept a new 30 MB image each time, never freed. A 16 GB card filled within
  minutes. The upscaler now stays on the game's view and frees images it no longer needs.
- Fixed HUD trails, menus that stayed on screen after closing, and likely some black screens
  (seen with HDR on at 2160p). After the game rebuilt its images, for example on a display mode
  change, it could use one block of the graphics card's compression data for two images. The
  port then cleared the wrong image each frame: the HUD layer kept its old picture, and the
  scene's lighting was cleared before the HUD was drawn over it.
- Load Game now shows your characters to choose from, as the PS5 does. Before, it loaded the
  first one without asking. Choose with up and down or the mouse wheel. Load with Cross, A,
  Enter or a left click. Go back with Circle, B, Esc or a right click.
- Add-ons: the game also looks for your add-on packages in an 'Add-ons' folder in
  '%LOCALAPPDATA%\Regaliant' and next to 'regaliant.exe', besides the one in
  '%APPDATA%\Regaliant' (AppData\Roaming), and finds them by the content ID inside them,
  whatever their names. Its log now says what it found where. Their items go to Stockpile
  Thomas's storage in the Nexus when a character loads (the pre-order's weapon under the
  weapons tab). The game gives the Digital Deluxe items only to characters created while the
  add-ons are in place.
- A controller the port does not see now leaves a line in the log after a minute: which
  controllers Windows lists and which programs that hide or translate controllers are running.

## 0.1.4

- Fixed the mostly black picture in play on RTX 50-series graphics cards (black shapes and streaks
  across the view from the first tunnel on). The game gives the unused slots of some draws
  "not a number" positions so that the graphics card drops them, and the port moves them out of
  view; the driver for these cards was allowed to skip that check and drew them across the
  screen. The game's shaders now tell the driver to keep it, on RTX 50-series cards and on AMD and
  Intel ones. Confirmed on an RTX 5090. On these cards the launcher prepares the shaders again
  once; RTX 40-series and older cards keep their prepared shaders.

## 0.1.3

- Game folders whose executable and libraries are decrypted but still in their SELF containers
  (as some dumps keep them) now work as they are, and so do .zar archives made from them.
  Before, the launcher called them encrypted and they had to be converted to ELF files first. A
  container the port cannot read, for example one cut short by a copy, now says why instead of
  being called encrypted.
- Fixed a timing fault on the graphics card in the texture streaming's statistics: resetting them
  could overtake reading them, and the game then kept textures at low detail. Vulkan's
  synchronization checks found it. It was not seen to cause problems on the tested cards.
- Your own pre-order and Digital Deluxe add-on packages count: put them in a folder named
  'Add-ons' in '%APPDATA%\Regaliant', and the game finds them as owned when it starts. Before,
  it was told that no add-ons were installed.
- Import PS5 Saves also takes save folders named 'sdimg_SAVEDATA...', as PS5 save tools name
  them. Before, such a save was not found. The PS5's own backups beside them are left out.
- Fixed a rare hang after slow first shader compiles: a part of the graphics work that fell
  behind could wait forever for a frame that had already passed, and the game stopped.

## 0.1.2

- On RTX 50-series graphics cards (and AMD and Intel ones) the game's shaders now tell the driver
  to keep each group of shader threads together after they take different paths. The game's
  translated shaders rely on it, and RTX 40-series cards do it on their own. On an RTX 5090, an
  RTX 5080 and an RTX 5060 Ti the tunnel at the game's start showed black shapes, stretched
  geometry or an almost black picture, in 0.1.1 too. This is the likely cause, not yet confirmed
  on such a card. On these cards the launcher prepares the shaders again once.
- **Open logs** in the launcher now selects the newest session's folder, the one to send with a
  bug report. The launcher's own quick checks keep their logs in a `checks` folder.
- When Windows reports more video memory in use by other programs than the graphics card has (a
  12 GB card reported 38 GB), the game no longer shrinks its texture memory to half the card
  because of it. That caused stutter, and restarts of the shader preparation, on such PCs. The
  log also names the programs that hold the most video memory.

## 0.1.1

- The game can be one .zar archive instead of its folder, read in place: about two thirds of the
  folder's size (54 GB instead of 82 GB for the tested copy). On the launcher's Game page,
  "Compress to .zar..." makes one from the game folder and then offers to play from it, and
  ".zar file..." chooses an existing one. The game folder is left as it is. On the tested PC the
  game started about 2 s sooner from the archive and loaded levels from the drive as fast or
  faster; reloading a level still in memory took about 0.3 s longer. Frame rate and stutter were
  the same.
- PlayStation controllers work directly, without Steam Input: DualSense, DualSense Edge and
  DUALSHOCK 4, over USB or Bluetooth, also when connected while the game runs. While you use one,
  the game shows its own PlayStation buttons. Vibration and the light bar work on them.
- All connected controllers play together as one pad. Before, only the first Xbox controller was
  read.
- The picture fills the window and the screen in every display mode when Windows display scaling is
  above 100%. Before, it could cover only the top-left part, as on a 4K screen at 175% or a 1440p
  screen at 200%.
- On RTX 50-series graphics cards (and AMD and Intel ones) every compute shader now runs in its
  portable form. The faster forms some shaders used were only checked on RTX 40-series cards, and
  on an RTX 5090 and an RTX 5080 the tunnel at the game's start was drawn black. This is the likely
  cause, not yet confirmed on such a card.

## 0.1.0

The first version of Regaliant. It runs the PS5 game Demon's Souls (PPSA01341, version 01.007.000) from the
player's own game files on Windows.

- The game's own program runs on the PC's processor. Its system calls, audio, saves, controllers
  and video output are reimplemented for Windows. Its graphics commands and shaders run on Vulkan.
- A launcher checks the game files and their version, sets the display (window mode, monitor,
  resolution, refresh rate, render scale, frame rate limit), prepares the shaders ahead (the game
  plays a new game and then its levels, enemies, bosses and cutscenes by itself in the background,
  hidden and silent: about an hour once on a fast PC, longer on others), lists save backups and restores them, and starts the game. It also offers a safe mode, a settings reset and
  a button that opens the Discord invite.
- PC settings inside the game's own menus (SETTINGS > PC): window mode with monitor, resolution and
  refresh rate, HDR, aspect ratios up to 32:9, render scale, sharpening, quality presets,
  geometry detail, shadow distance, texture filtering, anti-aliasing with DLSS, FSR or XeSS, bloom, vignette, a
  frame rate limit of 30, 60 or uncapped, menu and HUD size, HUD margin, button prompts, key and
  controller bindings, mouse sensitivity and inversion, a motion blur fix that keeps plants sharp
  while the camera moves, sound output and device, the save folder and the import of a decrypted
  PS5 save.
- Keyboard and mouse with raw mouse look, key prompts in the game's style, a mouse pointer that
  points and clicks in the menus (it selects what it points at at once and hides in cutscenes), a text box in the game's window for the character's name
  (with a keyboard of its own for controllers), and Xbox prompts for Xbox controllers.
- 5.1 and 7.1 sound, headphone and Windows Sonic or Dolby Atmos output of the game's 3D sound.
  The controller speaker's sounds can play through the PC's speakers (Pad Speaker).
- The game keeps its normal speed at any frame rate, and its frames are paced evenly at any
  refresh rate, with G-SYNC and FreeSync in borderless and exclusive fullscreen.
- Shaders the launcher did not prepare compile in the background where that is safe: an object
  can be missing for a moment instead of the game pausing.
- The shader preparation looks after itself: a step that stops responding or crashes runs once
  more, and when preparing cannot finish on a PC (no usable graphics card, a full drive, steps
  that keep failing) the launcher says why and Play starts the game. An interrupted preparation
  goes on where it stopped, and after a graphics driver or port update only the compile is done
  again. Some graphics drivers stop taking prepared shaders after Windows events. The launcher
  notices this when it starts and compiles them again. On battery, Play starts the game without
  preparing, and a preparation stops when the battery runs low.
- Trophies as achievements with a pop-up and a list.
- Saves are backed up before each session changes them.

Fixed and added while 0.1.0 was prepared, before it was published:

- A shorter freeze when the world first appears after loading a save: large textures the game
  creates while arriving are prepared on the graphics card again.
- 3D textures are prepared on the graphics card, which takes work off the processor while playing.
- Fewer crashes while shaders are prepared: a shader whose preparation crashed the graphics
  driver is not prepared ahead again.
- No more flicker right after loading: the world no longer vanishes in every other frame for
  a moment.
- Smoother frame rates where many shadows are drawn: clearing shadows no longer stalls the
  graphics card.
- No more flicker while turning the camera at the start of the first level.
- The shader preparation finishes on more PCs. Pressing Prepare or Play while the launcher
  checks the game now waits for the check. An old preparation is never taken as done.
- When the graphics driver crashes on a shader, the preparation leaves that shader out and goes
  on. Play does the same.
- When the graphics card runs out of memory, the game no longer slows to a crawl or stops
  responding.
- When graphics memory runs short, textures now go to system memory instead of being left out,
  so the picture stays whole. The frame rate can drop in busy places.
- Graphics cards with 4 to 6 GB of video memory: the game runs there and draws the whole
  picture, slowly. On cards with 4 or 5 GB, large images stay in system memory from the start.
- Fixed a crash when graphics memory ran out while the game made room for its work.
- The launcher says what to expect on a card with little video memory.
- A long preparation that runs short of graphics memory goes on in a fresh session.
- PCs with 16 GB of memory: in a simulation of one, the game no longer freezes at level loads.
  When the graphics card's share of memory runs short, the game frees memory it has not used
  lately.
- A preparation step that finished its work is no longer run again when it is slow to close.
- Prepare now on prepared shaders gives shaders that once crashed the graphics driver another
  chance.
