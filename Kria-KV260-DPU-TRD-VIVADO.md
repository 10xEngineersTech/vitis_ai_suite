# Kria KV260 DPU-TRD Vivado Flow — Vitis AI 3.0 Tutorial

This guide explains how to build a DPU hardware design for the Kria KV260 Vision AI Starter Kit using the Vivado flow, then deploy it with PetaLinux and Vitis AI 3.0.

---

## A. Prerequisites

1. Install Vivado/Vitis 2022.2 and PetaLinux 2022.2.
2. Download the Vitis AI 3.0 DPU-TRD files: `DPUCZDX8G_VAI_v3.0.tar.gz`
   [Download link](https://www.xilinx.com/bin/public/openDownload?filename=DPUCZDX8G_VAI_v3.0.tar.gz)
3. Download the PetaLinux 2022.2 BSP for KV260.
   [Download link](https://drive.google.com/drive/u/0/folders/13qcbGNd81VAhOV7NYlbpO_9hlCN2tkJL)
4. A Linux PC running Ubuntu 20.04 LTS (recommended for Vivado, Vitis, and PetaLinux 2022.2).

---

## B. Vivado IP Design

Rename `DPUCZDX8G_VAI_v3.0.tar.gz` to `DPU_TRD` and place it inside your Vitis-AI folder.

```bash
tar -xzvf DPUCZDX8G_VAI_v3.0.tar.gz
mv -r ~/Downloads/DPUCZDX8G_VAI_v3.0 <Vitis_AI_path>/DPU_TRD
export TRD_HOME=<Vitis_AI_path>/DPU_TRD
cd $TRD_HOME
source <Vivado_install_path>/Vivado/2022.2/settings64.sh
```

### 1. Customize the DPU-TRD project settings

Open the project settings file:

```bash
nano ./prj/Vivado/hw/scripts/trd_prj.tcl
```

Edit the following values:

```tcl
dict set dict_prj dict_sys prj_name                 {KV260}
dict set dict_prj dict_sys prj_part                 {xck26-sfvc784-2LV-c}
dict set dict_prj dict_sys prj_board                {KV260}
dict set dict_prj dict_param DPU_CLK_MHz            {275}
dict set dict_prj dict_param DPU_NUM                {1}
dict set dict_prj dict_param DPU_ARCH               {4096}
dict set dict_prj dict_param DPU_SFM_NUM            {0}
dict set dict_prj dict_param DPU_URAM_PER_DPU       {50}
```

Save and exit (`Ctrl+O`, `Enter`, `Ctrl+X`).

> **Important:** Use `DPU_ARCH = {4096}`. All the precompiled Vitis AI models available online are compiled for the B4096 architecture. Using a different value will cause model compatibility errors later.

Also edit the base block design script:

```bash
nano ./prj/Vivado/hw/scripts/base/trd_bd.tcl
```

Set:

```tcl
dict set dict_prj dict_param HP_CLK_MHz {274}
```

Save and exit.

### 2. Generate the Vivado project

```bash
cd $TRD_HOME/prj/Vivado/hw
vivado -source scripts/trd_prj.tcl
```

Vivado will open with the generated project.

<img width="1542" height="869" alt="Vivado project opened" src="https://github.com/user-attachments/assets/482dcf36-2efd-4a05-8d2a-16c98a734616" />

### 3. Fix the project settings

There are several settings to correct. Follow these steps in order.

**a. Open Settings**

<img width="1808" height="981" alt="Settings window" src="https://github.com/user-attachments/assets/f6116666-210f-4f9a-9442-c1fae06bd95e" />

**b. Set the project device and board**

Click the three dots next to "Project Device." Search for `kv260`, click **Connections**, click the dropdown, and select **AI Starter Kit Carrier Card**.

<img width="1808" height="981" alt="Selecting KV260 carrier card" src="https://github.com/user-attachments/assets/36001403-2e06-4cd4-a9c2-50b124323dcf" />

**c. Enable extensible platform**

Check the box **"Project is an extensible Vitis platform."**

Your General settings tab should now look like this:

<img width="1808" height="981" alt="General settings tab" src="https://github.com/user-attachments/assets/51f396c4-571b-4d53-a41a-5438709de5d0" />

**d. Enable bin file generation**

Go to the **Bitstream** tab in Settings and select the `bin_file` option.

<img width="1808" height="981" alt="Bitstream settings tab" src="https://github.com/user-attachments/assets/f32f15d2-e3b0-48d4-bb5a-01ad3ed72143" />

Click **Apply**, then **OK**.

### 4. Update IP blocks

A yellow bar will appear at the top of the Vivado window saying something like *"The design has 6 blocks that should be upgraded."*

Click the blue **"6 blocks"** link. In the console at the bottom, select all six blocks and click **Update Selected**.

<img width="1808" height="981" alt="Updating IP blocks" src="https://github.com/user-attachments/assets/abe7ee94-44a2-4875-a716-61ebca78b903" />

### 5. Edit the block design

Delete the wire highlighted in the image below:

<img width="1097" height="511" alt="image" src="https://github.com/user-attachments/assets/74a20c33-fc01-415e-bad2-4be32a89b752" />

Remove all extra ports on the Zynq UltraScale+ block, then click **Run Connection Automation** and click **OK**.

<img width="1294" height="803" alt="image" src="https://github.com/user-attachments/assets/2d129ba8-7f18-4b01-915e-079f2686ed2a" />

Click **Save**, then **Regenerate Layout**. Your design should now look like this:

<img width="1810" height="918" alt="Regenerated block design" src="https://github.com/user-attachments/assets/f22941fa-6660-4209-9515-aced6620a047" />

### 6. Configure the platform clock and AXI ports
 
This step tells Vivado which AXI port and which clock the platform should use as its main connection to the DPU. Without this, Vivado has no default clock or port assigned to the platform, and it cannot generate the hardware handoff file (`hpfm`) needed later.
 
> **If you skip this step**, the bitstream generation step will fail with an error similar to: *"select at least one clock and AXI port."* Come back to this step and complete it before trying to generate the bitstream again.

Click **Platform Setup** (next to the Diagram tab). Go to the **AXI Port** section and select a master AXI port under the Zynq section.

<img width="1808" height="981" alt="AXI Port setup" src="https://github.com/user-attachments/assets/4b472c8b-1efa-4185-b02c-bdbebc99e5f8" />

Go to the **AXI Port to Clock** section. Enable `pl_clk0` and set it as the default clock.

<img width="1808" height="981" alt="Clock setup" src="https://github.com/user-attachments/assets/63e0a1b0-c10a-4a28-8bc2-fcc1cbcb650c" />

### 7. Generate the bitstream

Save and validate the project, then click **Generate Bitstream**.

### 8. Export the hardware platform

Once the bitstream is generated, go to **File → Export → Export Platform → Hardware**, check **Include Bitstream**, and export.

<img width="1810" height="918" alt="Export hardware platform" src="https://github.com/user-attachments/assets/5b3c93ca-443d-4a46-b744-492f1aee0b08" />

---

## C. PetaLinux Build

For reference, `./prj/Vivado/sw/README.md` (inside the DPU-TRD package) covers the full PetaLinux build process. Some of its steps are not required for this guide.

### 1. Create the PetaLinux project

```bash
cd $TRD_HOME/
petalinux-create -t project -s <BSP_directory>/xilinx-kv260-starterkit-v2022.2-10141622.bsp --name kv260-dpu-trd
```

### 2. Load the hardware XSA

```bash
cd kv260-dpu-trd
petalinux-config --get-hw-description=$TRD_HOME/prj/Vivado/hw/prj/ --silentconfig
petalinux-config
```

This opens the PetaLinux system configuration menu:

<img width="1034" height="838" alt="PetaLinux config menu" src="https://github.com/user-attachments/assets/975d61ee-9e4d-499d-8af6-8113af08cb12" />

### 3. Configure the project

- **Enable FPGA Manager:** go to `FPGA Manager`, press Enter, then `Y` to select it.
- **Disable TFTP boot copy:** go to `Image Packaging Configuration` and unselect `Copy final images to tftpboot`.

Your Image Packaging Configuration section should look like this:

<img width="1034" height="838" alt="Image packaging configuration" src="https://github.com/user-attachments/assets/c97773f4-24fe-446d-8fef-e864c0c12fd9" />

Exit and save.

### 4. Configure the kernel

Enable the DPU driver:

```bash
petalinux-config -c kernel
```

Navigate to:

```
Device Drivers  --->
  Misc devices  --->
    <*> Xilinx Deep learning Processing Unit (DPU) Driver
```

<img width="1782" height="927" alt="Kernel config DPU driver" src="https://github.com/user-attachments/assets/cadc98b7-498b-45a9-9ce7-92ea30332393" />

Exit and save.

### 5. Copy the DPU-TRD recipes

Copy `recipes-apps` and `recipes-vitis-ai`, then merge `recipes-kernel`:

```bash
cp -r ../prj/Vivado/sw/meta-vitis/recipes-apps ./project-spec/meta-user/
cp -r ../prj/Vivado/sw/meta-vitis/recipes-vitis-ai ./project-spec/meta-user/
cp -rut ../prj/Vivado/sw/meta-vitis/recipes-kernel ./project-spec/meta-user/
```

### 6. Add Vitis AI packages to the image

Edit `project-spec/meta-user/conf/petalinuxbsp.conf`:

```bash
nano ./project-spec/meta-user/conf/petalinuxbsp.conf
```

Add these lines at the end:

```
IMAGE_INSTALL:append = " vitis-ai-library "
IMAGE_INSTALL:append = " vitis-ai-library-dev "
IMAGE_INSTALL:append = " resnet50 "
```

Save and exit (`Ctrl+O`, `Enter`, `Ctrl+X`).

### 7. Register the packages with rootfs config

Edit `project-spec/meta-user/conf/user-rootfsconfig`:

```bash
nano ./project-spec/meta-user/conf/user-rootfsconfig
```

Add these lines at the end:

```
CONFIG_vitis-ai-library
CONFIG_vitis-ai-library-dev
CONFIG_vitis-ai-library-dbg
CONFIG_dnf
CONFIG_nfs-utils
```

Save and exit.

### 8. Select the packages in rootfs config

```bash
petalinux-config -c rootfs
```

Navigate to **user packages** and select the packages shown below:

<img width="1028" height="585" alt="Rootfs package selection" src="https://github.com/user-attachments/assets/c7d6bf45-beb1-450e-a47e-806ddc4caf51" />

Exit and save.

### 9. Build

```bash
petalinux-build
```

### 10. Create the WIC image

```bash
petalinux-package --wic --images-dir images/linux/ --bootfiles "ramdisk.cpio.gz.u-boot,boot.scr,Image,system.dtb,system-zynqmp-sck-kv-g-revB.dtb" --disk-name "mmcblk1" --wic-extra-args "-c gzip"
```

---

## D. Preparing the SD Card

Use Balena Etcher to flash the WIC image onto an SD card (16GB or larger recommended).

```bash
cd ./images/linux/
```

Open Balena Etcher, select `petalinux-sdimage.wic.gz`, and flash it to the SD card.

<img width="1028" height="585" alt="Balena Etcher flashing SD card" src="https://github.com/user-attachments/assets/68b8541d-a5bd-44ee-9996-e2d4ff087770" />

---

## E. Creating the Firmware Files (DTBO)

This step uses XSCT to create a device tree source file (DTSI) from the XSA, then DTC to compile it into a DTBO.

### 1. Open XSCT

```bash
source <Vivado_install_path>/Vivado/2022.2/settings64.sh
xsct
```

### 2. Create the device tree

```tcl
createdts -hw $TRD_HOME/prj/Vivado/hw/prj/top_wrapper.xsa -zocl -platform-name KV260 -git-branch xlnx_rel_v2022.2 -overlay -compile -out $TRD_HOME/prj/Vivado/sw/kv260-dpu-trd/dt
exit
```

```bash
cd ../../
```

### 3. Compile the DTSI into a DTBO

```bash
dtc -@ -O dtb -o ./kv260.dtbo ./dt/KV260/psu_cortexa53_0/device_tree_domain/bsp/pl.dtsi
cp ./build/tmp/sysroots-components/xilinx_k26_kv/k26-starter-kits/lib/firmware/xilinx/k26-starter-kits/shell.json .
cp ../prj/Vivado/hw/prj/KV260.runs/impl_1_01/top_wrapper.bin ./kv260.bit.bin
```

At this point you should have these files ready:

1. `kv260.bit.bin`
2. `kv260.dtbo`
3. `shell.json`
4. An SD card flashed with `petalinux-sdimage.wic.gz`

---

## F. Copying Firmware Files to the KV260 Board

Boot the custom image on the KV260 board. For help with first boot, see the
[Vitis AI Suite setup guide](https://github.com/10xEngineersTech/vitis_ai_suite/blob/main/README.md).

Find the board's IP address by connecting it to your network over Ethernet and running `ifconfig` in the serial terminal. In this example the board IP is `10.42.0.27`.

Connect over SSH:

```bash
ssh petalinux@10.42.0.27
```

Default login user is `petalinux`, default password is `root`.

Copy the firmware files to the board:

```bash
scp shell.json petalinux@10.42.0.27:~/
scp kv260.dtbo petalinux@10.42.0.27:~/
scp kv260.bit.bin petalinux@10.42.0.27:~/
```

---

## G. Loading Firmware on the KV260

1. The three firmware files (`shell.json`, `kv260.dtbo`, `kv260.bit.bin`) should now be in `/home/petalinux/`.

2. Create the firmware directory:

   ```bash
   sudo mkdir /lib/firmware/xilinx/kv260-dpu-trd
   ```

3. Copy the files into it:

   ```bash
   sudo cp ./* /lib/firmware/xilinx/kv260-dpu-trd/
   ```

4. List available firmware apps:

   ```bash
   sudo xmutil listapps
   ```

5. Unload the default app:

   ```bash
   sudo xmutil unloadapp
   ```

6. Load the DPU-TRD firmware:

   ```bash
   sudo xmutil loadapp kv260-dpu-trd
   ```

---

## H. Checking DPU Availability

Confirm the DPU driver is installed and available:

```bash
sudo show_dpu
sudo xdputil query
```

You should see an `app` folder under `/home/root/`. Copy it to your home directory:

```bash
sudo cp -r /home/root/app /home/petalinux/
cd /home/petalinux/app
sudo mkdir images
sudo cp ./img/* ./images/
```

Run the example:

```bash
cd model
sudo ../samples/bin/resnet50 ../images/jinrikisha-911722.JPEG
```

> **Note:** Always run DPU-related commands as `sudo`. Make sure the arguments are in the correct order — the `resnet50` sample takes an **image file** as its argument and looks for `resnet50.xmodel` in the current directory.
