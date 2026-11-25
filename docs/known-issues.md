# Known issues

[Back to the main page](../README.md)

- The frame rate is below 60 in the busiest places: about 40 to 55 fps on the tested PC.
- In a window on a 144 or 165 Hz screen the game shows an even 48 or 55 fps. Borderless or fullscreen with G-SYNC or FreeSync, or a 120 Hz screen, gives 60 ([more](settings.md#frame-rate-and-refresh-rate)).
- The shader preparation covers what the game meets while it plays itself: the levels, their enemies and bosses, and the cutscenes. Shaders it did not meet (rare effects, spells) still compile on first sight, and some of those compiles stutter. On a slower PC the preparation takes longer than an hour, and the launcher says how long. If a graphics driver does not keep compiled shaders from one session to the next, the launcher says so and skips the preparation. A level's first arrival can pause briefly. Later visits do not compile them again.
- Areas, bosses and the story were reached by the developer's shortcuts in testing. A full playthrough has not been made.
- Water outside the Nexus can show bands. Some other rare picture faults are known.
- The sky in the opening cutscene is a little darker than on PS5. Some light shafts look softer too.
- At 32:9 the character preview in the equipment menu stands at the right edge.
- In character creation the header can turn bright cyan after the button prompts change, for example when you switch between the controller and the keyboard. It is only cosmetic.
- In character creation and in the equipment menu the character preview can fill only part of the screen, and look dark in the menu, when Render Scale is set to a percentage (seen at small render sizes such as 100% of a 1280x720 window). Visual Mode shows it right.
- HDR has not been seen on an HDR screen yet.
- AMD and Intel graphics cards have not been tested.
- On graphics cards with 12 GB or less the game can run short of graphics memory in busy places. It then keeps some textures in system memory, and the frame rate there can drop.
- On cards with 4 or 5 GB the game keeps its large images in system memory from the start. It runs, but slowly. Cards with less than 4 GB have not been tested.
- PCs with 16 GB of memory have not been tested on a real PC. A simulation of one played the tested levels. 32 GB is recommended.
- Online features (messages, bloodstains, co-op and invasions) are not available. The game plays offline.
- Linux, Proton and the Steam Deck are not supported yet.

Found a bug that is not here? See [how to report a bug](report-a-bug.md).
