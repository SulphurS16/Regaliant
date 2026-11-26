# 3. A dated history of the shared runtime

The Windows line supplied the loader, high-level libraries, shader translator and renderer that Linux inherited. Read the five periods in sequence to see which assumptions survived. Commit references in these chapters refer to changes reachable from the Linux tip unless explicitly identified otherwise. The local Windows branch continued after the last upstream merge.

| Period | Question the developers had to answer |
| --- | --- |
| [8–14 October](03a-boot.md) | If the title reaches its draw loop, why can a room still be unlit, missing its walls or full of the wrong geometry? |
| [15–31 October](03b-residency.md) | Which bytes can stay on the device, and which parts of the next command stream can be prepared early? |
| [1–10 November](03c-coherence.md) | How can compute overlap graphics while both queues and the CPU continue to observe one address space? |
| [11–18 November](03d-recording.md) | Which of the hundreds of small preparations per frame can actually be eliminated? |
| [19–25 November](03e-release.md) | What changes when the first shader cache is empty, the GPU is smaller, or the game must recover on a player's machine? |

The [Linux chapter](04-linux.md) picks up the second line of development on 23 November. Its first-parent sequence runs alongside upstream release work. A merge date establishes when that work became available to the port.

**How to interpret the measurements.** A flip is a guest display request. A flip rate measured with 240 Hz vblanks is a throughput experiment with extra headroom. A number from an uncapped frame window therefore answers a different question from the number of frames displayed on a 60 Hz monitor. A tour crosses areas and stresses remapping, streaming and compilation. A fixed-location comparison holds more variables steady but can miss a defect that appears only while walking or loading. Replay can repeat the same submitted work to price the renderer while leaving out the live interaction with guest producers. The chapters identify the kind of experiment wherever the record supplies it.

For a cited commit, `git show -s --format=%B HASH` reads the author's full observation. Many early commits put most of the evidence in a long subject. `git show HASH -- path/to/file` gives the change at the time, including experiments since removed. References in old comments to STATUS, FINDINGS or RENDERER describe the working documents of those revisions. Present-day source links in this chronicle point to files available in this worktree.
