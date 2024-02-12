Options defined at Makefile switch execution of the script to different modes.
Examples to use:
- `make INTER=1 LINEAR=1 LINK=1 IMG_PATH=./assets/Lenna.png run` This command runs the script in interactive mode, with bilinear interpolation, and edge-pixel prediction to detect edge of `./assets/Lenna.png` .
- `make INTER=0 LINEAR=1 LINK=1 IMG_PATH=./assets/Lenna.png gdb` Same as above, but run in gdb, which is useful for examine details. To achieve this goal, refer to installation and use of [OpenImageDebugger](https://github.com/openimagedebugger/openimagedebugger)
while setting up `Makefile` is trivial, setting up `meson` is rather easy.
Examples to use:
- set up build directory (under root directory): `meson setup build`
- then compile: `meson compile -C build`
- checkout help message to find the usage and the meaning of arguments: `./build/src/canny -H`
- run and save what we got: `./build/src/canny -i ./assets/Lenna.png -l 15 -h 20 -s ./save` this is to run the program with setting low-threshold to 15, high-threshold to 20, and where to save images to `./save`
- checkout `Makefile` or help message for more usage and options
