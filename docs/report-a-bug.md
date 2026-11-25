# How to report a bug

[Back to the main page](../README.md)

1. Check the [known issues](known-issues.md) first.
2. If the game hangs but keeps drawing (a loading screen that never ends, for example), press Ctrl+F10 in the game's window and wait a few seconds before you close it. The log then says what the game was waiting for.
3. In the launcher, click **Open logs**. It opens the logs folder with the newest session's folder selected. Zip that folder, together with `launcher.log`. For a problem with the shader preparation, zip the whole logs folder: it has a folder for each of its steps. Logs leave out your user name, your computer's name and your profile folder.
   On Linux, use the cache folder's `linux-runs` directory; there is no graphical launcher.
   Ctrl+F10 writes a `threads-N` folder there. See [Linux troubleshooting](linux.md#troubleshooting-and-developer-verification).
   Review raw thread/mapping reports for personal paths and memory contents before sharing.
4. Open a [bug report](https://github.com/SulphurS16/Regaliant/issues/new?template=bug_report.yml). Say what happened and where, add your PC's specs and attach the zip.

Ideas are welcome too: open a [feature request](https://github.com/SulphurS16/Regaliant/issues/new?template=feature_request.yml). For questions and help, ask on the [Discord](https://discord.gg/rSr5JhxxtZ).

Please don't share game files, keys, decrypted files or download links in issues.
