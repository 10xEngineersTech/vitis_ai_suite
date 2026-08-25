# Conv3D Custom Operator for Vitis AI

This folder contains a custom CPU operator implementation for **3D Convolution** and **Transposed 3D Convolution**, written to run unsupported layers on the CPU/APU of the **Kria KV260** board. It was built specifically to run **PSMNet** (a stereo depth estimation network), which uses Conv3D layers that the DPU hardware cannot handle natively.

---


## Folder Structure

```
conv3D_CustomOP/
│
├── conv3d_fix_op.hpp           ← Bridge between Vitis AI runtime and your math
├── conv3d_fix_kernel.hpp       ← The actual convolution math (MAC loops, NEON)
├── conv3d-fix.cpp              ← Entry point / wrapper for regular Conv3D
├── transposed-conv3d-fix.cpp   ← Entry point / wrapper for Transposed Conv3D
└── Makefile                    ← Cross-compilation build script
```

---

## File Descriptions

### `conv3d_fix_kernel.hpp` — The Math

This is where the actual convolution computation lives. It does not depend on any Vitis AI types, so you can compile and test it on your laptop without a VART installation.

Key things it handles:
- **5D tensor layout** — `[N, H, W, D, C]` for input/output, `[OC, KH, KW, KD, IC]` for weights
- **Fixed-point int8 arithmetic** — multiplies int8 values and accumulates into int32, then rescales back to int8
- **Tap precomputation** — for each output position, it figures out in advance which input positions to read, handling padding and out-of-bounds as zero automatically
- **Multi-threading** — splits output rows across CPU cores (configurable via `VAI_CONV3D_FIX_NUM_THREADS`)
- **NEON acceleration** — on AArch64 (the Kria board), uses ARM NEON SIMD intrinsics to process 16 input channels at a time, with 4-way output channel unrolling
- **Activation functions** — supports ReLU, PReLU, LeakyReLU, and ReLU6

### `conv3d_fix_op.hpp` — The Bridge

This file connects the kernel math to the Vitis AI runtime. It reads the operator's configuration from the compiled `.xmodel` file and sets up the `Param` struct that the kernel uses.

This is the file you would modify if you wanted to support a different operator type. See the "How the Op Contract Works" section below for a full explanation.

### `conv3d-fix.cpp` — Conv3D Entry Point

A small wrapper file. It just aliases the op class with `TRANSPOSED = false` and registers it with the runtime using `DEF_XIR_OP_IMP`. This compiles into `libvart_op_imp_conv3d-fix.so`.

```cpp
using MyOpImp = vitis::ai::cpu_task::conv3d::Conv3dFixOpImp<false>;
DEF_XIR_OP_IMP(MyOpImp)
```

### `transposed-conv3d-fix.cpp` — Transposed Conv3D Entry Point

Same as above but with `TRANSPOSED = true`. Compiles into `libvart_op_imp_transposed-conv3d-fix.so`.

```cpp
using MyOpImp = vitis::ai::cpu_task::conv3d::Conv3dFixOpImp<true>;
DEF_XIR_OP_IMP(MyOpImp)
```

---

## How the Custom Op Contract Works

To register any custom CPU op in Vitis AI, you must follow four steps. Here is how this implementation does it.

### Step 1 — Inherit from `OpImpBase`

Every custom op class must inherit from `vart::experimental::OpImpBase`. This is non-negotiable — without it, the Vitis runtime simply won't recognize the class.

```cpp
template <bool TRANSPOSED>
struct Conv3dFixOpImp : public vart::experimental::OpImpBase {
```

The `template <bool TRANSPOSED>` trick lets one class handle both regular and transposed convolution — `<false>` for conv3d, `<true>` for transposed conv3d.

---

### Step 2 — Read Op Configuration in the Constructor

The constructor receives a pointer to the XIR graph node (`xir::Op*`). This is a description of the operation as stored in the `.xmodel` file — things like kernel size, stride, padding, and activation type. You read all of it here, because this graph node may not be available later during inference.

```cpp
Conv3dFixOpImp(const xir::Op* op, xir::Attrs* attrs)
    : vart::experimental::OpImpBase{op, attrs} {

    // Required — the model is broken without these
    param_.kernel = op->get_attr<std::vector<int>>("kernel");  // e.g. {3,3,3}
    param_.stride = op->get_attr<std::vector<int>>("stride");  // e.g. {1,1,1}

    // Optional — use has_attr() to check first, provide safe defaults
    param_.dilation = {1, 1, 1};
    if (op->has_attr("dilation"))
        param_.dilation = op->get_attr<std::vector<int>>("dilation");

    param_.pad = {0, 0, 0, 0, 0, 0};  // {left, right, top, bottom, near, far}
    if (op->has_attr("pad"))
        param_.pad = op->get_attr<std::vector<int>>("pad");
}
```

A separate op object is constructed for each layer in the model, so each layer's kernel size, stride, etc. are correctly stored independently.

---

### Step 3 — Implement `calculate()`

This is the method the runtime calls on every inference frame. It receives the actual memory buffers — the real input data, weights, and where to write the output.

```cpp
int calculate(
    vart::simple_tensor_buffer_t<int8_t> output,
    vart::simple_tensor_buffer_t<int8_t> input,
    vart::simple_tensor_buffer_t<int8_t> weight,
    std::unique_ptr<vart::simple_tensor_buffer_t<int8_t>> bias
)
```

Inside, it does three things:

**A) Read tensor shapes.** Shapes can be dynamic (e.g. batch size might not be known at graph load time), so they are read here from the live buffer rather than the graph node.

```cpp
param_.input_shape  = input.tensor->get_shape();
param_.weight_shape = weight.tensor->get_shape();
param_.output_shape = output.tensor->get_shape();
```

**B) Read fixed-point quantization info.** Every tensor has a `fix_point` value that describes its quantization scale. These are used to correctly convert the int32 accumulator back to int8 output range.

```cpp
auto input_fp  = input.tensor->get_attr<int>("fix_point");
auto weight_fp = weight.tensor->get_attr<int>("fix_point");
auto output_fp = output.tensor->get_attr<int>("fix_point");
param_.set_fix_point(input_fp, weight_fp, output_fp);
// computes: shift_cut_scale = 2^(input_fp + weight_fp - output_fp)
```

**C) Run the kernel.**

```cpp
Kernel(param_).run(input.data, weight.data, bias_data_ptr, output.data);
return 0;  // must return 0 on success
```

---

### Step 4 — Register with `DEF_XIR_OP_IMP`

This macro, placed in the `.cpp` wrapper file, is what actually tells the Vitis runtime "this class handles this op type." Each op type gets its own `.cpp` file and its own compiled `.so` library.

```cpp
namespace {
    using MyOpImp = Conv3dFixOpImp<false>;  // give it a local name
}
DEF_XIR_OP_IMP(MyOpImp)  // register it with the runtime
```

The anonymous `namespace { }` keeps `MyOpImp` local to that file, so both `.cpp` files can use the same alias name without conflicting at link time.

---

### The Full Flow

```
.xmodel file
    │
    │  Op attributes: kernel, stride, pad, dilation, group, nonlinear
    │  Quantization:  fix_point values for input, weight, bias, output
    ▼
Vitis Runtime loads the graph
    │
    ├──► Conv3dFixOpImp(op, attrs)        [once per layer, at load time]
    │        reads attributes, sets up param_
    │
    │   
    │
    └──► calculate(output, input, weight, bias)
             reads shapes + fix_points from live buffers
             │
             └──► Kernel(param_).run(...)
                      NEON-accelerated int8 MAC loops
                      bias add + quantization rescale
                      activation (ReLU / ReLU6 / etc.)
                      write int8 output
```

---

## How to Compile

> This is **cross-compilation** — you build on your host PC but the output runs on the Kria board (AArch64).

**1. Place this folder on the host under:**
```
Vitis-AI/src/vai_library/cpu_task/conv3D_CustomOP/
```

**2. Edit the `Makefile` and set your correct paths:**
```makefile
SYSROOT ?= /<your path>/petalinux_sdk_2022.2/sysroots/cortexa72-cortexa53-xilinx-linux
CXX     ?= /tools/X22/Vitis/2022.2/gnu/aarch64/lin/aarch64-linux/bin/aarch64-linux-gnu-g++
```

**3. Build:**
```bash
cd conv3D_CustomOP/
make all
```

**4. Output — two shared libraries under `./aarch64/`:**
```
libvart_op_imp_conv3d-fix.so
libvart_op_imp_transposed-conv3d-fix.so
```

---

## Deploy to the Board

**Copy the libraries to the board** (replace `<board-ip>` with your board's IP):
```bash
scp ./aarch64/libvart_op_imp_*.so root@<board-ip>:~/
```

**On the board, install and reload:**
```bash
sudo cp ~/libvart_op_imp_*.so /usr/lib/
sudo ldconfig
```

Once the `.so` files are in `/usr/lib/`, the Vitis AI runtime will find and load them automatically when it encounters a `conv3d-fix` or `transposed-conv3d-fix` op in your model.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `VAI_CONV3D_FIX_NUM_THREADS` | All CPU cores | Number of threads for the conv3d kernel. Set to `1` to disable threading. |

---

## Notes

- This implementation supports `FLOOR`, `CEIL`, `SAME`, and `VALID` pad modes.
- Grouped convolution (`group > 1`) is not implemented — the kernel will throw an error if it encounters one.
- `HSIGMOID` and `HSWISH` activations are not implemented for 3D convolution.
- On AArch64, the NEON path is used automatically. On other architectures it falls back to scalar code with no change in output values.