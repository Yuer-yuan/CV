Options defined at Makefile switch execution of the script to different modes.
Examples to use:
- `make INTER=1 LINEAR=1 LINK=1 IMG_PATH=./assets/Lenna.png run` This command run the script in interactive mode, with bilinear interpolation, and edge-pixel prediction to detect edge of `./assets/Lenna.png` .
- `make INTER=0 LINEAR=1 LINK=1 IMG_PATH=./assets/Lenna.png gdb` Same as above, but run in gdb, which is useful for examine details. To achieve this goal, refer to installation and use of [OpenImageDebugger](https://github.com/openimagedebugger/openimagedebugger)