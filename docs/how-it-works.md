# How it works

[Back to the main page](../README.md)

Regaliant is not rebuilt from the game's source code. It runs the game's own PS5 program from your copy of the game's files.

The PS5 has an x86-64 processor like a PC, so the game's program can run on a PC processor. The port loads that program (eboot.bin and the two libraries in sce_module) from your files and links it to its own implementation of the PS5's system libraries: memory, threads, files, saves, audio, movies, video output and controllers. While loading, it adjusts a few kinds of instructions. Thread-local storage goes through Windows, and the AMD-only SSE4A instructions are rewritten on processors that lack them.

The game drives the PS5's graphics chip directly, without a graphics API. The port executes the game's graphics command stream with Vulkan and translates its shaders to SPIR-V. A few patches to the game's code, applied as it loads to the tested 01.007.000 program, fix crashes and add PC features. Each patch checks the game's bytes before it applies: where they differ (as in another region's release), that patch is skipped and the difference is logged, so nothing is patched at the wrong address.

Nothing from the game is in this repository.
