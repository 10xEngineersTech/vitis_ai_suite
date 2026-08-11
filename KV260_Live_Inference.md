# Live AI Inference on Xilinx Kria KV260
## Running Pre-Compiled Models with USB Camera Output to HDMI Display

---

## Overview

This guide walks through everything needed to run a live AI inference demo on the Kria KV260 board using a USB camera and HDMI display — without X11 or a desktop environment.

**Why is this non-trivial?**
The `zynqmp-display` DRM driver on KV260 is broken for GStreamer `kmssink` without X11. Running `modetest -f` reveals zero framebuffers allocated at boot, causing `kmssink` to fail silently. The solution is to bypass the DRM layer entirely and write directly to `/dev/fb0` using `fbdevsink` with `RGB16` pixel format.

---

## Prerequisites

- Xilinx Kria KV260 board running PetaLinux (for this [see](https://github.com/10xEngineersTech/vitis_ai_suite/blob/main/Kria-KV260-DPU-TRD-VIVADO.md))
- USB camera (tested with a 4K-capable UVC camera)
- HDMI display connected to the board's DisplayPort output
- Host PC with Vitis AI SDK cross-compiler installed (for this [see Phase 2: Setup Host](https://github.com/10xEngineersTech/vitis_ai_suite/blob/main/README.md))
- SSH access to the board

---

# Example 1
## Part 1 — Prepare the Application Code (on Host PC)

### Step 1 — Choose Your Model and Locate the Test Application
> See: Vitis-AI/examples/vai_library/samples/ for example application. 

For this guide we use **YOLOv3**: `yolov3_coco_416_tf2`

Every Vitis AI model comes with a test application. Find the one matching your model:

```bash
# On host PC — inside the Vitis AI repository
find . -name "test_video_yolov3.cpp"
# → ./examples/vai_library/samples/yolov3/test_video_yolov3.cpp
```

Open `test_video_yolov3.cpp` and trace what it calls:

```cpp
#include <vitis/ai/demo.hpp>   // ← the key file we need to modify
...
return vitis::ai::main_for_video_demo(...);  // ← defined in demo.hpp
```

The core logic lives in `demo.hpp`.

---

### Step 2 — Get demo.hpp from the Board

The version of `demo.hpp` on the board matches your PetaLinux build exactly. Always use this version rather than the one from the GitHub repo.

```bash
# Copy demo.hpp from board to host PC
scp petalinux@<board-ip>:/usr/include/vitis/ai/demo.hpp \
    Vitis-AI/examples/vai_library/samples/yolov3/demo.hpp
```

---

### Step 3 — Add Camera Resolution Environment Variables to demo.hpp

By default, `demo.hpp` hardcodes the camera resolution:

```cpp
// Original — hardcoded, cannot be changed at runtime
cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
cap.set(cv::CAP_PROP_FRAME_HEIGHT, 360);
```

This prevents you from changing resolution without recompiling. Fix it by adding two environment variable declarations at the top of `demo.hpp`, alongside the existing ones:

```cpp
// Add these two lines near the other DEF_ENV_PARAM declarations at the top
DEF_ENV_PARAM(DEMO_CAMERA_WIDTH, "640")
DEF_ENV_PARAM(DEMO_CAMERA_HEIGHT, "360")
```

Then replace the hardcoded lines in `DecodeThread::open_stream()`:

```cpp
// Replace this:
cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
cap.set(cv::CAP_PROP_FRAME_HEIGHT, 360);

// With this:
cap.set(cv::CAP_PROP_FRAME_WIDTH, ENV_PARAM(DEMO_CAMERA_WIDTH));
cap.set(cv::CAP_PROP_FRAME_HEIGHT, ENV_PARAM(DEMO_CAMERA_HEIGHT));
```

Then replace the hardcoded line in `main_for_video_demo`:

```cpp
// Replace this:
gui_layout() = {{0, 0, 640, 360}};

// With this:
gui_layout() = {{0, 0, DEMO_CAMERA_WIDTH, DEMO_CAMERA_HEIGHT}};
```

---

### Step 4 — Fix the Include Path in test_video_yolov3.cpp

The original include uses angle brackets which searches system paths, ignoring your local `demo.hpp`:

```cpp
// Change this (searches system path — ignores your local edits):
#include <vitis/ai/demo.hpp>

// To this (searches local directory first):
#include "demo.hpp"
```

---

### Step 5 — Recompile on Host PC

```bash
cd ./examples/vai_library/samples/yolov3/
bash build.sh
```

The `build.sh` script uses the cross-compiler targeting `cortexa72-cortexa53` (ARM64) for the KV260.

---

### Step 6 — Deploy Binary to Board

```bash
scp test_video_yolov3 petalinux@<board-ip>:~/app/samples/yolov3/
```

---

## Part 2 — Prepare the Board

### Step 7 — Get the Compiled Model

Download the pre-compiled `.xmodel` for your target board and model from the Vitis AI Model Zoo:

> 📦 For that you can take help from [*Issue#2*](https://github.com/10xEngineersTech/vitis_ai_suite/issues/2).

Place the model files under:
```
/usr/share/vitis_ai_library/models/yolov3_coco_416_tf2/
```

Verify:
```bash
ls /usr/share/vitis_ai_library/models/yolov3_coco_416_tf2/
# Should show: yolov3_coco_416_tf2.xmodel  (and config files)
```

---

### Step 8 — Install Required GStreamer Plugin

The default PetaLinux image may be missing `fbdevsink`. Install it:

```bash
sudo dnf install -y gstreamer1.0-plugins-bad-meta
```

**Why this matters:** `fbdevsink` is the only reliable way to write video frames to the display on KV260 without X11. Without it, you have no display output.

---

### Step 9 — Discover Your Framebuffer Format

```bash
cat /sys/class/graphics/fb0/bits_per_pixel   # → 16
```

**Key findings:**
- `/dev/fb0` uses **RGB16** (16-bit, 2 bytes per pixel) — not RGB24 or BGR

**Why this matters:** If you write the wrong pixel format to `/dev/fb0`, the display shows garbage or nothing. Always force `format=RGB16` in your GStreamer pipeline.

---

## Part 3 — Debugging Tools

Use these commands to diagnose issues before running the model.

### Verify DisplayPort is Connected
```bash
cat /sys/class/drm/card*-DP-*/status
# Expected output: connected
```
**When to use:** If display shows nothing. Confirms the physical cable and DRM detection are working.

---

### Check What Formats the Display Planes Accept
```bash
# Install if needed:
sudo dnf install -y libdrm-tests

modetest -M xlnx -p
```
**When to use:** To see supported pixel formats for each DRM plane and the active CRTC resolution. Helps confirm why `kmssink` fails.

---

### Watch for DRM Driver Errors in Real Time
```bash
dmesg -w | grep -i "layer\|zynqmp\|display"
```
**When to use:** Run this in a second SSH terminal while launching GStreamer. If you see:
```
zynqmp-display fd4a0000.display: Layer width:height must be 1920:1080
```
It means `kmssink` is being used and the DRM driver is rejecting frames. Switch to `fbdevsink`.

---

### Check CMA Memory (DPU Buffer Pool)
```bash
cat /proc/meminfo | grep CmaFree
dmesg | grep "cma_alloc.*failed"
```
**When to use:** If the app crashes with `cannot allocate memory` or the display goes completely black at higher resolutions. Reboot the board.

---

### Check What Resolutions Your USB Camera Supports
```bash
# Install if needed:
sudo dnf install -y v4l-utils

v4l2-ctl --device=/dev/video0 --list-formats-ext
```
**When to use:** Before choosing `DEMO_CAMERA_WIDTH/HEIGHT`. Only use resolutions listed here. Requesting an unsupported resolution causes the camera to silently fall back to a different size, causing a mismatch with `DEMO_VIDEO_WRITER_WIDTH/HEIGHT` and a black or distorted display.

---

### Verify the Full Display Chain Works (Without GStreamer)
```bash
dd if=/dev/urandom of=/dev/fb0 bs=3840 count=1080
```
**When to use:** If nothing appears on the HDMI display. This writes random noise directly to the framebuffer. If the entire screen fills with noise, the display chain (board → cable → monitor) is working correctly and the problem is in your GStreamer pipeline.

---

### Test Live Camera Stream Without AI Model
```bash
gst-launch-1.0 -e v4l2src device=/dev/video0 ! \
  videoconvert ! \
  video/x-raw,format=RGB16,width=1920,height=1080 ! \
  fbdevsink device=/dev/fb0 sync=false max-lateness=-1
```
**When to use:** Before running the model. If this shows live camera feed on your display, the entire chain (camera → GStreamer → fb0 → HDMI) is confirmed working. If this fails, fix it before attempting to run the model.

---

## Part 4 — Run Live Inference

Once the camera stream test works, run the model.

### Navigate to Application Directory
```bash
cd ~/app/samples/yolov3/
```

### Set Environment Variables and Run

```bash
# Enable verbose debug logging (optional but useful first time)
export DEBUG_DEMO=1

# Tell demo.hpp to use GStreamer VideoWriter instead of cv::imshow
# This bypasses the GTK/X11 requirement entirely
export DEMO_USE_VIDEO_WRITER=1

# Camera capture resolution
# Must match a resolution your camera actually supports (check v4l2-ctl output)
export DEMO_CAMERA_WIDTH=640
export DEMO_CAMERA_HEIGHT=360

# Must exactly match DEMO_CAMERA_WIDTH x DEMO_CAMERA_HEIGHT
# This tells cv::VideoWriter what size frames to expect from the DPU pipeline
export DEMO_VIDEO_WRITER_WIDTH=640
export DEMO_VIDEO_WRITER_HEIGHT=360

# GStreamer pipeline:
# - appsrc: receives frames from cv::VideoWriter
# - videoconvert: converts OpenCV BGR format to RGB16
# - videoscale: upscales from camera resolution to display resolution
# - video/x-raw: caps filter enforcing RGB16 at display resolution
# - fbdevsink: writes directly to /dev/fb0, bypassing broken DRM layer
export DEMO_VIDEO_WRITER="appsrc ! videoconvert ! videoscale ! video/x-raw,format=RGB16,width=1920,height=1080 ! fbdevsink device=/dev/fb0 sync=false max-lateness=-1"

# Run the model
# -E flag passes all exported environment variables through sudo
sudo -E ./test_video_yolov3 yolov3_coco_416_tf2 0
```

**What each variable does:**

| Variable | Purpose |
|---|---|
| `DEBUG_DEMO=1` | Prints thread activity, FPS, queue sizes to terminal |
| `DEMO_USE_VIDEO_WRITER=1` | Switches from `cv::imshow` (needs X11) to GStreamer pipeline (no X11 needed) |
| `DEMO_CAMERA_WIDTH/HEIGHT` | Controls USB camera capture resolution (our custom addition to demo.hpp) |
| `DEMO_VIDEO_WRITER_WIDTH/HEIGHT` | Must match camera size — tells VideoWriter what frame size to expect |
| `DEMO_VIDEO_WRITER` | The GStreamer pipeline string used for display output |
| `sudo -E` | Runs with root privileges while preserving (`-E`) all exported environment variables |

---

### Expected Output on Terminal

```
video writer is created: 640x360 appsrc ! videoconvert ! ...
DPU model size=416x416
thread [DedodeThread-0] is started
thread [DPU-0] is started
thread [SORT-0] is started
thread [SORT-0] frame id 1 FPS: -1
thread [SORT-0] frame id 2 FPS: 10.87
```

**Expected output on HDMI display:** Live camera feed upscaled to 1920×1080 with YOLOv3 bounding boxes drawn around detected objects and an FPS counter in the top-left corner.

---

# Example 2: Running the Multi-Task Demo Application

### 1. Build the Application

1. Navigate to the application directory:
```bash
cd Vitis-AI/example/vai_library/apps/multitask_v3_quad_windows/

```


2. Copy the `demo.hpp` file created in **Example 1** into this directory and update the include path in `multitaskv3_quad_windows.cpp`.
3. Compile the application by executing the build script:
```bash
./build.sh

```



---

### 2. Download the Model

Download the required model package (`multi_task_v3_pt`) from the [Xilinx Model Zoo](https://www.xilinx.com/bin/public/openDownload?filename=multi_task_v3_pt-zcu102_zcu104_kv260-r3.0.0.tar.gz).

---

### 3. Deploy to the Target Board

Transfer the compiled binary (`multitaskv3_quad_windows_x`) and the downloaded model archive (`multi_task_v3_pt-zcu102_zcu104_kv260-r3.0.0.tar.gz`) to your board using `scp`.

---

### 4. Setup and Run on the Board

Run the following commands on the board to move files, extract the model, set environment variables, and start the application:

```bash
# Move model package and application binary
sudo mv multi_task_v3_pt-zcu102_zcu104_kv260-r3.0.0.tar.gz /usr/share/vitis_ai_library/models/
sudo mv multitaskv3_quad_windows_x ~/app/samples/multitaskv3/

# Extract the model
cd /usr/share/vitis_ai_library/models/
sudo tar -xzvf multi_task_v3_pt-zcu102_zcu104_kv260-r3.0.0.tar.gz
sudo rm multi_task_v3_pt-zcu102_zcu104_kv260-r3.0.0.tar.gz

# Navigate to application directory and configure environment variables
cd ~/app/samples/multitaskv3/
export DEBUG_DEMO=1
export DEMO_USE_VIDEO_WRITER=1
export DEMO_CAMERA_WIDTH=640
export DEMO_CAMERA_HEIGHT=360
export DEMO_VIDEO_WRITER_WIDTH=1280
export DEMO_VIDEO_WRITER_HEIGHT=720
export DEMO_VIDEO_WRITER="appsrc ! videoconvert ! video/x-raw,format=RGB16,width=1280,height=720 ! fbdevsink device=/dev/fb0 sync=false max-lateness=-1"

# Execute the application with root privileges (preserving environment variables)
sudo -E ./multitaskv3_quad_windows_x 0 -t 4

```

> **Note:** Do not modify any width or height resolution values in the commands above.
---

## Troubleshooting

### Complete Black Screen
**Cause:** Camera resolution or DPU buffer size exceeds available resolution range.

**Fix:** Reduce `DEMO_CAMERA_WIDTH` and `DEMO_CAMERA_HEIGHT` to model's native resolution (`416x416` for YOLOv3).

---

### Distorted or Resolution-Mismatch Display
**Cause:** `DEMO_VIDEO_WRITER_WIDTH/HEIGHT` does not match the actual frame size coming from the camera.

**Fix:** Ensure `DEMO_VIDEO_WRITER_WIDTH` = `DEMO_CAMERA_WIDTH` and `DEMO_VIDEO_WRITER_HEIGHT` = `DEMO_CAMERA_HEIGHT` exactly.

---

### Model Only Runs at One Specific Resolution
**Cause:** Some models (e.g. `densebox_640_320`) have fixed input dimensions baked into the `.xmodel` at compile time. They only accept exactly that resolution regardless of camera settings.

**Fix:** Set camera resolution to match the model name exactly:
```bash
# For densebox_640_320:
export DEMO_CAMERA_WIDTH=640
export DEMO_CAMERA_HEIGHT=320
export DEMO_VIDEO_WRITER_WIDTH=640
export DEMO_VIDEO_WRITER_HEIGHT=320
```

---

### `Can't initialize GTK backend` Error
```
error: (-2) Can't initialize GTK backend in function 'cvInitSystem'
```
**Cause:** `DEMO_USE_VIDEO_WRITER` is not set, so the app tries to use `cv::imshow` which requires X11/GTK.

**Fix:**
```bash
export DEMO_USE_VIDEO_WRITER=1
```

---
