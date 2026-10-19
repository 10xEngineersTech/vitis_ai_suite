# PSMNet application (KV260)

Stereo depth with PSMNet on the Kria KV260. The DPU (DPUCZDX8G, B512) runs the
2D feature extraction and most of the 3D hourglass. The 3D convolutions that
the DPU cannot run go to the `conv3d-fix` / `transposed-conv3d-fix` op
libraries, which use either the PL kernel (see [../kernel/fpga](../kernel/fpga/README.md))
or the CPU kernel (see [../kernel/cpu](../kernel/cpu/README.md)). Softmax and
disparity regression run on the CPU with an exp lookup table (`LutSfm`).

Two programs are built from the same `PsmNet` class:

| program | input | output |
|---|---|---|
| `demo_psmnet` | one left/right image pair (arguments) | `result_psmnet_0.jpg` (gray disparity) |
| `demo_psmnet_live` | two USB cameras (or an image folder) | live view + colour disparity on the display (`/dev/fb0`) |

## Pipeline (one frame)

```
left/right image -> resize to model input
  kernel 0 (left)  : DpuTask subgraph 0 -> 4 x Resize (CPU, NEON) -> DpuTask subgraph 1
  kernel 1 (right) : same, separate DpuTask
  CostVolume       : CPU, NEON, int8 [H][W][48][2C]
  kernel 2         : GraphRunner: DPU subgraphs + conv3d-fix / transposed-conv3d-fix
                     CPU-op subgraphs (libvart_op_imp_*conv3d-fix.so) + fix2float
  LutSfm           : softmax over D + sum(d * p_d) per pixel, CPU threads
  get_result       : disparity * 1.7 -> 8-bit gray image
```

## Files

| file | role |
|---|---|
| `psmnet.hpp` | public interface: `PsmNet::create()`, `run()`, `get_result()`, `get_input_batch/width/height()` |
| `psmnet.cpp` | `PsmNetImp`: loads the three models, wires the tensors, runs Resize, CostVolume, kernel 2 and LutSfm, makes the result image |
| `resize.hpp` / `resize.cpp` | `Resize`: NEON bilinear upsampling + requantisation between the two subgraphs of kernels 0/1 |
| `cost_volume.hpp` / `cost_volume.cpp` | `CostVolume`: NEON requantisation of both feature maps + cost volume `[H][W][48][2C]` (kernel 2 input) |
| `lut_sfm.hpp` / `lut_sfm.cpp` | `LutSfm`: softmax + disparity regression with a 256-entry exp table |
| `demo_psmnet.cpp` | single image pair demo |
| `demo_psmnet_live.cpp` | live two-camera demo |
| `Makefile`, `build.sh` | cross build with the PetaLinux 2022.2 SDK (`build.sh` runs `make all`) |
| `demo_psmnet_left.png`, `demo_psmnet_right.png` | sample stereo pair |

## Build (on the host, cross compile)

Needs the PetaLinux 2022.2 SDK with the Vitis AI 3.0 libraries in its sysroot.

```sh
make            # or ./build.sh -> demo_psmnet, demo_psmnet_live (aarch64)
make clean
```

Another SDK location:

```sh
make SDK=/path/to/petalinux_sdk_2022.2
```

Compile flags: `--sysroot=<sysroot> -I<sysroot>/usr/include -I<sysroot>/usr/include/opencv4 -std=c++17 -O3 -g -Wall`.
Link libraries: `glog vitis_ai_library-model_config protobuf vitis_ai_library-graph_runner vitis_ai_library-dpu_task vart-runner xir opencv_{core,imgproc,highgui,imgcodecs,videoio,video} pthread`.
The application itself has no XRT dependency; only the conv3d op libraries use XRT.

## Board setup

Board: KV260, PetaLinux 2022.2 image with Vitis AI 3.0 runtime (VART, XIR,
Vitis AI Library), XRT 2.14 / zocl, OpenCV 4.5, GStreamer (live demo).

1. Firmware: the `dpu512_c` app (DPU + `conv3d_fix` xclbin, bitstream,
   overlay) is installed in `/lib/firmware/xilinx/dpu512_c/` and loaded. How to
   build, install and load it:
   [../README.md, Building the firmware](../README.md#building-the-firmware).

2. Op libraries: kernel 2 needs `libvart_op_imp_conv3d-fix.so` and
   `libvart_op_imp_transposed-conv3d-fix.so` in `/usr/lib`. Use the FPGA build
   (`../kernel/fpga/ops/aarch64/`, PL kernel with CPU fallback) or the CPU-only
   build (`../kernel/cpu/aarch64/`). Build and install them as described in
   [../kernel/fpga/README.md, steps 6 and 7](../kernel/fpga/README.md#6-op-libraries-libvartso)
   (FPGA build) or
   [../kernel/cpu/README.md, Install on the board](../kernel/cpu/README.md#install-on-the-board-cpu-only-path)
   (CPU-only build).

3. Models: the compiled B512 models are not in this folder. Build
   `PSMNet_pruned_{0,1,2}_pt/` with [../model/README.md](../model/README.md);
   they land in `../model/compiled/`.

4. Application: from this folder on the host:

   ```sh
   make
   mkdir -p psmnet
   cp -r demo_psmnet demo_psmnet_live demo_psmnet_left.png demo_psmnet_right.png ../model/compiled/PSMNet_pruned_0_pt ../model/compiled/PSMNet_pruned_1_pt ../model/compiled/PSMNet_pruned_2_pt psmnet/
   ```
   Now, send `psmnet` on the board using `scp`.

### `demo_psmnet`

```sh
cd ~/psmnet
./demo_psmnet demo_psmnet_left.png demo_psmnet_right.png
```

- Arguments: `<left image> <right image>` (any size; resized to the model input).
- Output: `result_psmnet_0.jpg` in the current directory.
- Profile per op: `DEEPHI_PROFILING=1 ./demo_psmnet demo_psmnet_left.png demo_psmnet_right.png`.

### `demo_psmnet_live`

```sh
cd ~/psmnet
./demo_psmnet_live
```

- Two USB cameras with MJPG at `IMG_W`×`IMG_H`@`CAM_FPS` on `CAM_LEFT` / `CAM_RIGHT`.
- The display (`DISP_W`×`DISP_H`, refreshed at `DISP_FPS`) shows both live views
  (top row), the last left input and its disparity (JET colour map, bottom row), written through
  the GStreamer pipeline `DEMO_VIDEO_WRITER` (default: `fbdevsink /dev/fb0`).
- Inference takes the newest camera pair, runs, publishes the result and
  takes the next pair; the log prints `result #<n> ... in <seconds>`.
- `Ctrl-C` (SIGINT/SIGTERM) stops after the current inference finishes.
- Without cameras: `DEBUG=1 DEBUG_DIR=data ./demo_psmnet_live` loops over
  `data/left/*` and the right image with the same file name in `data/right/`.

## Environment variables

Application (`grep -rn "getenv\|DEF_ENV_PARAM"` in this folder):

| file | variable | default | effect |
|---|---|---|---|
| `psmnet.cpp` | `PSMNET_MODEL_0` | `PSMNet_pruned_0_pt/PSMNet_pruned_0_pt.xmodel` | kernel 0 (left features) |
| `psmnet.cpp` | `PSMNET_MODEL_1` | `PSMNet_pruned_1_pt/PSMNet_pruned_1_pt.xmodel` | kernel 1 (right features) |
| `psmnet.cpp` | `PSMNET_MODEL_2` | `PSMNet_pruned_2_pt/PSMNet_pruned_2_pt.xmodel` | kernel 2 (3D hourglass, GraphRunner) |
| `psmnet.cpp` | `PSMNET_SFM_FIX` | `-1` | fix point of kernel 2's logits for LutSfm; `-1` = read it from the model's tail `fix2float` input |
| `lut_sfm.cpp` | `PSMNET_SFM_THREADS` | number of cores | threads for softmax + regression |
| `resize.cpp` | `DEBUG_RESIZE` | `0` | `1` = log each Resize: tensors, sizes, channels, scale |
| `demo_psmnet_live.cpp` | `DEBUG` | `0` | `1` = read images from `DEBUG_DIR` instead of the cameras |
| `demo_psmnet_live.cpp` | `DEBUG_DIR` | `data` | folder with `left/` and `right/` |
| `demo_psmnet_live.cpp` | `IMG_W`, `IMG_H` | `640`, `480` | camera capture size (MJPG mode) |
| `demo_psmnet_live.cpp` | `CAM_LEFT`, `CAM_RIGHT` | `/dev/video0`, `/dev/video2` | v4l2 devices |
| `demo_psmnet_live.cpp` | `CAM_FPS` | `30` | camera frame rate |
| `demo_psmnet_live.cpp` | `DISP_W`, `DISP_H` | `1280`, `720` | display canvas |
| `demo_psmnet_live.cpp` | `DISP_FPS` | `5` | display refresh rate |
| `demo_psmnet_live.cpp` | `DEMO_VIDEO_WRITER` | `appsrc ! queue max-size-buffers=2 leaky=downstream ! videoconvert n-threads=2 ! video/x-raw,format=RGB16 ! fbdevsink device=/dev/fb0 sync=false` | GStreamer output pipeline, must start with `appsrc` |

conv3d op libraries (read inside `libvart_op_imp_*conv3d-fix.so`, sources in `../kernel`):

| file | variable | default | effect |
|---|---|---|---|
| `kernel/cpu/ops/common/conv3d_fix_op.hpp`, `kernel/fpga/conv3d_fix_op_fpga.hpp` | `VAI_CONV3D_FIX_NUM_THREADS` | number of cores | threads of the CPU kernel |
| `kernel/fpga/conv3d_fix_op_fpga.hpp` | `VAI_CONV3D_FPGA` | on | `0` = force the CPU kernel |
| `kernel/fpga/conv3d_fix_op_fpga.hpp` | `DEBUG_CONV3D_FPGA` | `0` | `1` = log FPGA/CPU per layer and the xclbin used |
| `kernel/fpga/conv3d_fix_op_fpga.hpp` | `VAI_CONV3D_FPGA_VERIFY` | `0` | `1` = also run the CPU kernel and log `VERIFY ... mismatches <n>`; `2` = also replace the FPGA output with the CPU one |
| `kernel/fpga/conv3d_fix_fpga_host.hpp` | `VAI_CONV3D_FPGA_TIMEOUT_MS` | `30000` | kernel run timeout |
| `kernel/fpga/conv3d_fix_fpga_host.hpp` | `XLNX_VART_FIRMWARE` | `firmware:` line of `/etc/vart.conf` | xclbin whose uuid is used to open the CU |

Vitis AI runtime: `DEEPHI_PROFILING=1` prints `CPU_OP_EXEC : <us>` per CPU op and the DPU run times.

## Expected output

```
$ ./demo_psmnet demo_psmnet_left.png demo_psmnet_right.png
...
I... demo_psmnet.cpp:...] BYEBYE
$ ls result_psmnet_0.jpg
result_psmnet_0.jpg
```

## Troubleshooting

| symptom | cause / fix |
|---|---|
| `VAILIB_CPU_RUNNER_OPEN_LIB_ERROR` with `lib=libvart_op_imp_conv3d-fix.so` | op libraries not in `/usr/lib` (Board setup step 2) |
| `ldd /usr/lib/libvart_op_imp_conv3d-fix.so` shows `not found` | wrong SDK / Vitis AI version; build with the 2022.2 SDK (`.so.3` libraries) |
| model load fails with a fingerprint / target mismatch | the loaded firmware is not the B512 DPU; `cat /etc/vart.conf`, `sudo xmutil loadapp dpu512_c` |
| `DEBUG_CONV3D_FPGA=1` shows `CPU (<reason>)` for every layer | conv3d_fix not in the loaded xclbin, or the library is the CPU-only build: `strings /usr/lib/libvart_op_imp_conv3d-fix.so \| grep -c VAI_CONV3D_FPGA_VERIFY` must print `1` |
| disparity noisy or distorted | run `VAI_CONV3D_FPGA_VERIFY=1 ./demo_psmnet demo_psmnet_left.png demo_psmnet_right.png 2>&1 \| grep VERIFY`; every layer must show `mismatches 0` (see [../kernel/fpga/README.md](../kernel/fpga/README.md)) |
| `logits fix point unknown: LutSfm falls back to expf` | kernel 2 does not end in `fix2float`; set `PSMNET_SFM_FIX` |
| hang / timeout in the conv3d kernel, then errors in the next run | reboot the board; a hung CU stays hung |
| errors about `/tmp/DPU_*` after a run as root | `sudo rm -f /tmp/DPU_*` |
| live demo: `no frame from /dev/video0` | check `v4l2-ctl --list-devices` and the MJPG modes; set `CAM_LEFT`, `CAM_RIGHT`, `IMG_W`, `IMG_H`, `CAM_FPS` |
| live demo: nothing on the screen | check `DEMO_VIDEO_WRITER` and that `/dev/fb0` exists |
