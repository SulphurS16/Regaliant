# Install, update and uninstall

[Back to the main page](../README.md)

## What you need

- Windows 10 or 11, 64-bit.
- A graphics card with a Vulkan 1.3 driver.
- A processor with AVX2: Intel from the Haswell generation (2013) on, or AMD Ryzen.
- Your own copy of Demon's Souls for PS5 (PPSA01341) at version 01.007.000, as a folder that holds sce_sys, the game's data folders and eboot.bin, or as a .zar archive of that folder (see [Save disk space](#save-disk-space-with-a-zar-archive)). The executable eboot.bin and the libraries libc.prx and libSceNpCppWebApi.prx in sce_module must be decrypted (in the folder itself or in its decrypted folder), as plain ELF files or in their SELF containers with the segments decrypted.
- Up to 5 GB of disk space for prepared shaders.

See [System requirements](requirements.md) for what was tested.

## Install

1. Download `Regaliant-<version>.zip` from the [latest release](https://github.com/SulphurS16/Regaliant/releases/latest).
2. Right click the zip, choose **Extract All...**, and pick a folder, for example 'C:\Games\Regaliant'.
3. Start 'regaliant-launcher.exe'. If Windows says "Windows protected your PC" (the programs are not code-signed), click **More info**, then **Run anyway**.
4. On **Game**, choose the game folder with **Folder...**, or a .zar archive of it with **.zar file...**. The launcher checks it and says what is missing.
5. Click **Play**. The first time, the launcher prepares the game's shaders before the game starts: the game plays a new game and its levels by itself in the background, hidden and silent, to learn them. This takes about an hour once on a fast PC and longer on others, and the launcher says how long. It can be stopped and continued later, and turned off on the **Shaders** page. On battery, Play starts the game without preparing: plug in the PC, then press **Play** or **Prepare now**.

## Save disk space with a .zar archive

The launcher can pack the game folder into one .zar archive, which the game reads in place. With the tested copy the archive takes 54 GB instead of the folder's 82 GB.

1. On **Game**, with the game folder chosen, click **Compress to .zar...** and choose where to save the archive. That drive needs about two thirds of the folder's size free.
2. Wait. On the tested PC it took about 3 to 4 minutes. You can stop it, and closing the launcher stops it too. An unfinished archive is removed.
3. When it is done, the launcher offers to play from the archive. Prepared shaders, settings and saves carry over.
4. Once the game runs from the archive, delete the game folder yourself to free its space. Keep it if you want to switch back with **Folder...**.

An archive made by another tool should work too if it holds the game folder (the one with eboot.bin and sce_sys) at its top or one folder down. Choose it with **.zar file...**.

**What you trade.** Measured on the tested PC (Ryzen 9 5900X, RTX 4090, both copies on the same NVMe drive):

| | Game folder | .zar archive |
| --- | --- | --- |
| Disk space | 82 GB | 54 GB |
| Starting the game | | about 2 s faster |
| A level's first load in a session | | the same or faster: up to 3 s on one level, 0.5 s on average |
| Reloading a level whose data is still in memory | 8.0 s | 8.3 s |
| Frame rate while playing | 47.6 fps | 47.7 fps |
| Stutter | the same | the same |

The processor decompresses the archive while the game reads it. From the drive it still loads at least as fast, because there is a third less to read and one file instead of thousands. When Windows still holds the data in memory, decompressing makes a reload about 0.3 s slower. Slower drives and processors were not measured.

## Update

Unzip the new release over the old one or into a new folder, and start its 'regaliant-launcher.exe'. Settings, key bindings and saves are in '%APPDATA%\Regaliant', which every version uses. Prepared shaders in '%LOCALAPPDATA%\Regaliant' carry on too.

## Uninstall

Delete the folder. Settings and saves are in '%APPDATA%\Regaliant', logs and prepared shaders in '%LOCALAPPDATA%\Regaliant'.

To keep everything in one place instead, put an empty file named 'portable.txt' next to 'regaliant.exe'. Settings, saves, logs and prepared shaders then go to a 'user' folder beside it, and deleting the game's folder removes them all. To keep your saves and settings, first copy what '%APPDATA%\Regaliant' holds into that 'user' folder.
