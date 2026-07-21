

# Running SimpleCNN on the Kria KV260 Board (Vitis AI)

This guide shows you how to train, quantize, compile, and run a simple Convolutional Neural Network (SimpleCNN) on the Xilinx Kria KV260 board using Vitis AI.

---

## Prerequisites & Setup

Before running the model steps, make sure your computer and board are set up correctly:

1. **Set up the Host Computer:**
Follow the [Phase 2: Host Setup Guide](https://github.com/10xEngineersTech/vitis_ai_suite/blob/main/README.md).
> **Note:** Make sure you checkout **branch 2.5** before installing the cross-compilation environment.


2. **Set up the KV260 Board:**
Follow the [KV260 Board Setup Guide](https://github.com/10xEngineersTech/vitis_ai_suite/blob/main/Kria-KV260-DPU-TRD-VIVADO.md).
3. **Project Location:**
Place your `SimpleCNN` folder inside the main `Vitis-AI` directory on your computer.

---

## Step 1: Run the Model Pipeline (Inside Docker)

Open a terminal on your host computer and start the Vitis AI Docker container:

```bash
docker pull xilinx/vitis-ai-pytorch-cpu:latest
./docker_run.sh xilinx/vitis-ai-pytorch-cpu:latest
conda activate vitis-ai-pytorch

```

Inside the Docker container, run the scripts in this order:

1. **Train the Model (`train.py`):**
Downloads the MNIST dataset, trains the model, and saves the trained file inside the `float_model/` folder.
```bash
python train.py

```


2. **Inspect the Architecture (`inspector.py`):** *(Optional)*
Checks which parts of the model can run on the board's DPU hardware versus the CPU. Results are saved in the `inspect_output/` folder.
```bash
python inspector.py

```


3. **Quantize the Model (`quantize.py`):**
Converts the trained model into INT8 format required by the KV260 board. The output is saved in the `quantize_result/` folder.
```bash
python quantize.py

```


4. **Compile the Model (`vai_c_xir`):**
Converts the quantized model into an `.xmodel` file compiled specifically for the KV260 DPU.
```bash
vai_c_xir -x quantize_result/SimpleCNN_int.xmodel -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json -o compiled_model -n simple_cnn_kv260

```



---

## Step 2: Build the App & Prepare Samples (Outside Docker)

Open a **new terminal window** on your host computer (do not use the Docker terminal) and navigate to your `SimpleCNN` folder:

1. **Create Sample Images (`datasee.py`):**
Converts test images from the dataset into `.png` files and saves them in the `samples/` folder.
```bash
python datasee.py

```


2. **Compile the C++ Application (`build.sh`):**
Compiles `app.cpp` to create the executable binary (`app`) that runs on the KV260 board.
```bash
bash build.sh

```



---

## Step 3: Copy Files to the KV260 Board

Transfer the compiled files and sample images from your computer to the board using `scp`. Replace `<board-ip>` with your board's IP address:

```bash
scp -r compiled_model app samples petalinux@<board-ip>:~/SimpleCNN/

```

---

## Step 4: Run the Model on the Board

Log in to your KV260 board via SSH, go to the project directory, and run the executable on a test image:

```bash
cd ~/SimpleCNN
sudo ./app ./compiled_model/simple_cnn_kv260.xmodel ./samples/5_label4.png

```

---

## Troubleshooting

* **Fingerprint Mismatch or Missing `.so` Files:**
If you get an error saying shared libraries (`.so` files) cannot be loaded or a fingerprint mismatch occurs then do visit [*Issue#5*](https://github.com/10xEngineersTech/vitis_ai_suite/issues/5).
