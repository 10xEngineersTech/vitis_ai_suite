# Vitis AI Deployment on Kria KV260

Welcome to the internal onboarding and development repository for deploying AI models on the **AMD Xilinx Kria KV260 Vision AI Starter Kit**.

This repository acts as a step-by-step track to guide engineers from a complete beginner level to an advanced deployment level. It structured around the three deployment flows defined in the **UG1354 (v3.0)** user guide.

---

## 🗺️ Repository Roadmap (The 3 User Types)

Our goal is to cover all three distinct workflows defined by AMD Xilinx. Progress through them sequentially:

* [x] **Phase 1: Pre-trained Models (Plug & Play)** – Use out-of-the-box optimized models from the Xilinx Model Zoo. *No Host PC setup required.*
* [ ] **Phase 2: Custom Datasets (Standard Architectures)** – Train supported models (like standard ResNet/YOLO) with our own data on a Host PC, compile them, and run. *(In Progress)*
* [ ] **Phase 3: Custom Models (Advanced Tweak)** – Modify standard model architectures and build customized inference pipelines using the Vitis AI post-processing libraries. *(In Progress)*

---

## 🛠️ Phase 1: Running Pre-Compiled Examples (Edge Only)

This guide takes you through executing optimized, pre-compiled model examples directly on the Kria KV260 board. **You do not need to set up a Docker container or configure a powerful Host Linux PC for this phase.**

### Prerequisites & Downloads

1. **Board Image:** Download the pre-built Kria KV260 DPU Image [v2022.2-v3.0.0.img.gz](https://account.amd.com/en/forms/downloads/design-license-xef.html?filename=xilinx-kv260-dpu-v2022.2-v3.0.0.img.gz).
   * *Struggling with regional download blocks? Check [Issue #2](https://github.com/10xEngineersTech/vitis_ai_suite/issues/2).*

2. **Flashing Tool:** Download and install [Balena Etcher](https://etcher.balena.io/).
3. **Sample Assets:** Download the official evaluation datasets to your laptop:
   * [vitis_ai_library_r3.0.0_images.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_library_r3.0.0_images.tar.gz)
   * [vitis_ai_library_r3.0.0_video.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_library_r3.0.0_video.tar.gz)

---

### Step-by-Step Execution Guide

#### Step 1: Flash the SD Card

1. Launch **Balena Etcher** on your computer.
2. Select the downloaded `.img.gz` file, choose your MicroSD card, and click **Flash**.
3. Insert the flashed MicroSD card into your KV260 board slot.

#### Step 2: Establish Serial Connection (UART)

1. Connect the KV260 to your laptop using a micro-USB data cable.
2. Open your terminal on your Linux/Mac laptop and scan for the serial ports:
```bash
sudo dmesg | grep tty
```

3. Identify the active ports. Connect using Putty (adjusting `/dev/ttyUSB1` based on your output):
```bash
sudo putty /dev/ttyUSB1 -serial -sercfg 115200,8,n,1,N
```

> ⚠️ **Important:** The board exposes 4 virtual COM ports. Ensure you connect to the **2nd serial port** in the sequence to access the primary boot logs and terminal shell.

#### Step 3: First Power Boot

1. Power on the board.
2. If the board boots successfully, proceed to Step 4.
   * *If the board skips your SD card and runs an old image or enters a firmware recovery screen, follow the hardware override steps in [Issue #3](https://github.com/10xEngineersTech/vitis_ai_suite/issues/3).*

#### Step 4: Network Pairing & SSH Remote Control

1. Connect the KV260 to the internet via an Ethernet cable. It must reside on the **same local area network (LAN)** as your laptop.
2. Find the IP address assigned to the board, then establish a remote terminal connection from your laptop:
```bash
ssh root@<IP_OF_BOARD>
```

   * **Default Password:** `root`

#### Step 5: Transfer Environment Evaluation Files

From your laptop terminal, push the downloaded sample image and video archives directly onto the board's storage file system:

```bash
scp vitis_ai_library_r3.0.0_images.tar.gz root@<IP_OF_BOARD>:~/
scp vitis_ai_library_r3.0.0_video.tar.gz root@<IP_OF_BOARD>:~/
```

#### Step 6: Unpack Assets on the Board

Return to your board's SSH window and unpack the evaluation files into the sample application workspaces:

```bash
cd ~
tar -xzvf vitis_ai_library_r3.0*_images.tar.gz -C Vitis-AI/examples/vai_library
tar -xzvf vitis_ai_library_r3.0*_video.tar.gz -C Vitis-AI/examples/vai_library
```

#### Step 7: Run Face Detection Inference

Navigate into the native compiled Vitis AI sample directory to test a model:

```bash
cd ~/Vitis-AI/examples/vai_library/samples/facedetect
./test_jpeg_facedetect densebox_320_320 sample_facedetect.jpg
```

The application will execute inference on the hardware DPU and output a new image file (`sample_facedetect_result.jpg`) containing predicted bounding boxes in the same folder.

> 💡 **Tip:** You can inspect the individual `readme.md` files located within each model sample subfolder to test different variations like ADAS, segmentation, or posture analysis.

---

## 🛠️ Troubleshooting & Known Bottlenecks

We track setup exceptions inside the repository Issue tracker. If you hit a wall, look at these standard resolutions:

1. **[Issue #1] Git Environment Mismatch:** Why you must use the **Vitis AI Branch 3.0** instead of `main`/`master` to prevent compilation errors.
2. **[Issue #2] Region Access Blocks:** Alternative local mirror links for pulling pre-built target firmware files.
3. **[Issue #3] QSPI Persistence Failure:** How to use the physical `FWUEN` switch and local static IP browser configurations to correct an invalid boot sequence layout.
