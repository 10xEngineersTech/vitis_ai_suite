# PSMNet Depth Estimation on Kria KV260 (Vitis-AI 3.0)

This guide shows how to port `Vitis-AI/examples/waa/app/psmnet` to run on the Kria KV260 board.

---

## Prerequisites

Complete these steps before making any code changes:

1. **Set up your host machine** — [see guide](https://github.com/10xEngineersTech/vitis_ai_suite/blob/main/README.md)
2. **Set up the KV260 board** — [see guide](https://github.com/10xEngineersTech/vitis_ai_suite/blob/main/Kria-KV260-DPU-TRD-VIVADO.md)
3. **Download the pre-compiled models of PSMNet for KV260** — [see guide](https://github.com/10xEngineersTech/vitis_ai_suite/issues/6)
4. **Build the `conv3D` custom operators** — [see guide](/conv3D_CustomOP/README.md)

---

## File Changes

Navigate to `Vitis-AI/examples/waa/app/psmnet/` and apply the following changes.

---

### 1. `cpu_op.hpp`

**Change 1 — Replace the `inputs_` member and add `is_float_` and `sfm_row_`**

```cpp
// BEFORE
std::vector<int8_t*> inputs_;
```

```cpp
// AFTER
bool is_float_;
std::vector<const void*> inputs_;
std::vector<float> sfm_row_;
```

**Why:** On KV260, kernel 2 runs through a `GraphRunner` who involve the subgraph of conve3D op, which is a custom op for KV260. So the output buffer holds `float32`, not `int8` as on VCK190. `is_float_` tracks which type the buffer holds so the softmax can pick the right overload. `inputs_` changes to `const void*` so it can hold either type without a bad cast. `sfm_row_` is a scratch buffer for one row of softmax output — this avoids allocating a full 424 MB buffer for the whole tensor.

---

### 2. `cpu_op.cpp`

**Change 1 — Detect the input data type and set `is_float_`**

Inside `CPUsfm::CPUsfm(vitis::ai::library::OutputTensor& input_)` function.
Add this line in the cpu softmax and sum part, after the input tensor is bound:

```cpp
// ADD
is_float_ = (input_.dtype == vitis::ai::library::DT_FLOAT);
```

**Why:** This reads the actual data type from the tensor descriptor at runtime, so the code works correctly for both `int8` (VCK190) and `float32` (KV260).

---

**Change 2 — Extend the log message**

Inside `CPUsfm::CPUsfm(vitis::ai::library::OutputTensor& input_)` function.
Just after the above fix:

```cpp
// BEFORE
<< "sft size is: " << size;
```

```cpp
// AFTER
<< "sft size is: " << size
<< ", dtype " << (is_float_ ? "float32" : "int8")
<< ", fixpos " << input_.fixpos
<< ", scale " << tensor_scale(input_);
```

**Why:** Adds dtype, fixpos, and scale to the log so you can verify the tensor is being interpreted correctly when debugging.

---

**Change 3 — Store the raw pointer without casting**

Inside `CPUsfm::CPUsfm(vitis::ai::library::OutputTensor& input_)` function.
Just after the above fix:

```cpp
// BEFORE
inputs_.push_back((int8_t*)input_.get_data(i));
```

```cpp
// AFTER
inputs_.push_back(input_.get_data(i));
```

**Why:** Casting a `float32` pointer to `int8_t*` would misread four bytes of one float as four separate values, corrupting the data and only covering a quarter of the tensor.

---

**Change 4 — Resize the `sfm_row_` scratch buffer**

Inside `CPUsfm::CPUsfm(vitis::ai::library::OutputTensor& input_)` function.
Add this line after the inputs are populated (at the end of function):

```cpp
// ADD
sfm_row_.resize(i_shape_[2] * i_shape_[3]);
```

**Why:** Pre-allocates the per-row scratch buffer once so it can be reused across all rows without repeated allocation.

---

**Change 5 — Replace the full-tensor softmax with a row-by-row**

```cpp
// BEFORE
void CPUsfm::run() {
  auto fea_size = i_shape_[1] * i_shape_[2];
  auto out_size = i_shape_[3] * fea_size;
  std::vector<float> output(out_size);
  for(size_t b = 0; b < i_shape_[0]; ++b) {
    vitis::ai::softmax(inputs_[b], fix_scale, i_shape_[3],
		       fea_size, output.data());
    for(size_t h = 0; h < i_shape_[1]; ++h) {
      for(size_t w = 0; w < i_shape_[2]; ++w) {
        float sum = 0.0;
        int pos = h*i_shape_[2] + w;
        for(size_t c = 0; c < i_shape_[3]; ++c) {
          sum += disp_[c] * output[pos*i_shape_[3]+c];
        }
        outputs_[b*fea_size + pos] = sum;
      }
    }
  }
}
```

```cpp
// AFTER
void CPUsfm::run() {
  const auto H = i_shape_[1];
  const auto W = i_shape_[2];
  const auto C = i_shape_[3];
  const auto fea_size = H * W;
  for(size_t b = 0; b < i_shape_[0]; ++b) {
    for(size_t h = 0; h < H; ++h) {
      const size_t off = h * W * C;
      if (is_float_) {
        vitis::ai::softmax(static_cast<const float*>(inputs_[b]) + off,
                           fix_scale, C, W, sfm_row_.data());
      } else {
        vitis::ai::softmax(static_cast<const int8_t*>(inputs_[b]) + off,
                           fix_scale, C, W, sfm_row_.data());
      }
      for(size_t w = 0; w < W; ++w) {
        float sum = 0.0;
        for(size_t c = 0; c < C; ++c) {
          sum += disp_[c] * sfm_row_[w*C + c];
        }
        outputs_[b*fea_size + h*W + w] = sum;
      }
    }
  }
}
```

**Why:** The full output tensor is 576×960×192 values, its about 424 MB. The named variables `H`, `W`, `C` make the index math clearer and are used in the row-by-row loop below. Processing one row at a time with `sfm_row_` avoids the 424 MB scratch buffer. The `is_float_` branch calls the correct `softmax` overload for the actual data type. For `int8`, the fix-point scale dequantises the logits. For `float32`, fixpos is 0 so scale is 1.0 and values are used as-is. The result is bit-identical to a single call over the whole tensor.

---

### 3. `psmnet.cpp`

**Change 1 — Add new `#include` headers**

After the existing includes, add:

```cpp
// ADD
#include <arm_neon.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include <vitis/ai/graph_runner.hpp>
#include <xir/attrs/attrs.hpp>
```

**Why:** `arm_neon.h` is needed for NEON intrinsics. The standard headers are needed for the new cost-volume functions. `graph_runner.hpp` and `xir/attrs/attrs.hpp` are needed to use `GraphRunner` for kernel 2.

---

**Change 2 — Add the `PSMNET_CV_NEON` environment parameter**

```cpp
// ADD (after existing DEF_ENV_PARAM lines)
DEF_ENV_PARAM(PSMNET_CV_NEON, "0");
```

**Why:** Both the NEON and scalar cost-volume paths are always compiled. This flag lets you switch between them at runtime on the board without rebuilding, so you can compare their outputs.

---

**Change 3 — Add tensor conversion helper functions**

After the existing includes and before the class definition, add these two functions:

```cpp
// ADD
static vitis::ai::library::InputTensor convert_tensor_buffer_to_input_tensor(
    vart::TensorBuffer* tb, vart::Runner::TensorFormat fmt) {
  auto ret = vitis::ai::library::InputTensor{};
  auto tensor = tb->get_tensor();
  auto dims = tensor->get_shape();
  auto dim_num = dims.size();
  auto batch = dim_num <= 0 ? 1 : dims[0];
  ret.batch = batch;
  ret.size = tensor->get_element_num() *
             std::ceil(tensor->get_data_type().bit_width / 8.f);
  if (fmt == vart::Runner::TensorFormat::NHWC) {
    ret.height = dim_num <= 1 ? 1 : dims[1];
    ret.width = dim_num <= 2 ? 1 : dims[2];
    ret.channel = 1;
    for (auto i = 3u; i < dims.size(); ++i) ret.channel *= dims[i];
    if (tensor->get_data_type().type == xir::DataType::XINT) {
      ret.fixpos = tensor->template get_attr<int>("fix_point");
      ret.dtype = vitis::ai::library::DT_INT8;
    } else {
      ret.fixpos = 0;
      ret.dtype = vitis::ai::library::DT_FLOAT;
    }
  } else {
    ret.height = dim_num <= 2 ? 1 : dims[2];
    ret.width = dim_num <= 3 ? 1 : dims[3];
    ret.channel = dim_num <= 1 ? 1 : dims[1];
    for (auto i = 4u; i < dims.size(); ++i) ret.channel *= dims[i];
    ret.fixpos = 0;
    ret.dtype = vitis::ai::library::DT_FLOAT;
  }
  ret.name = tensor->get_name();
  auto tb_ext = dynamic_cast<vart::TensorBufferExt*>(tb);
  auto batch_bytes = ret.size / batch;
  for (auto batch_idx = 0; batch_idx < batch; ++batch_idx) {
    auto idx = std::vector<int32_t>(dims.size(), 0);
    idx[0] = batch_idx;
    auto data = tb->data(idx);
    ret.get_data(batch_idx) = (void*)data.first;
    CHECK_GE(data.second, batch_bytes);
    ret.xcl_bo[batch_idx] = vitis::ai::library::XclBoInfo{0, nullptr, 0u};
    if (tb_ext) {
      auto bo = tb_ext->get_xcl_bo(batch_idx);
      ret.xcl_bo[batch_idx].xcl_handle = bo.xcl_handle;
      ret.xcl_bo[batch_idx].bo_handle = bo.bo_handle;
      ret.xcl_bo[batch_idx].offset =
          (unsigned int)tensor->template get_attr<int>("ddr_addr");
    }
  }
  return ret;
}

static vitis::ai::library::OutputTensor convert_tensor_buffer_to_output_tensor(
    vart::TensorBuffer* tb, vart::Runner::TensorFormat fmt) {
  auto ret = vitis::ai::library::OutputTensor{};
  // ... (same pattern as Input tensor function)
  if (fmt == vart::Runner::TensorFormat::NHWC) {
    if (dim_num == 2) {
      ret.height = 1;
      ret.width = 1;
      ret.channel = dims[1];
    } else {
      ret.height = dim_num <= 1 ? 1 : dims[1];
      ret.width = dim_num <= 2 ? 1 : dims[2];
      ret.channel = 1;
      for (auto i = 3u; i < dims.size(); ++i) {
        ret.channel *= dims[i];
      }
    }
    if (tensor->get_data_type().type == xir::DataType::XINT) {
      // ... (same pattern as Input tensor function)
}
```

**Why:** `GraphRunner` exposes tensors as `vart::TensorBuffer*` objects, but the rest of the PSMNet code uses `vitis::ai::library::InputTensor` / `OutputTensor` structs. These helpers bridge that gap and also detect the data type (`XINT` vs float) so `fixpos` and `dtype` are set correctly.

---

**Change 4 — Add new member variables for the kernel 2 `GraphRunner`**

In the class definition, add:

```cpp
// ADD
std::unique_ptr<xir::Graph> graph2_;
std::unique_ptr<xir::Attrs> attrs2_;
std::unique_ptr<vart::RunnerExt> graph_runner2_;
vitis::ai::library::InputTensor input_tensor_k2_;
vitis::ai::library::OutputTensor final_tensor2_;
```

**Why:** Kernel 2 is no longer driven by a `DpuTask`. These members hold the `GraphRunner` and its cached tensor descriptors for kernel 2.

---

**Change 5 — Replace the kernel 2 `DpuTask` setup with a `GraphRunner`**

Inside 'PsmNetImp::PsmNetImp()' function.

```cpp
// BEFORE
tasks_.emplace_back(vitis::ai::DpuTask::create(
    ENV_PARAM(PSMNET_MODEL_2)));
```

```cpp
// AFTER
graph2_ = xir::Graph::deserialize(ENV_PARAM(PSMNET_MODEL_2));
attrs2_ = xir::Attrs::create();
graph_runner2_ = vitis::ai::GraphRunner::create_graph_runner(
    graph2_.get(), attrs2_.get());
CHECK(graph_runner2_ != nullptr);
auto graph_inputs = graph_runner2_->get_inputs();
auto graph_outputs = graph_runner2_->get_outputs();
CHECK(!graph_inputs.empty());
CHECK(!graph_outputs.empty());
input_tensor_k2_ = convert_tensor_buffer_to_input_tensor(
    graph_inputs[0], graph_runner2_->get_tensor_format());
final_tensor2_ = convert_tensor_buffer_to_output_tensor(
    graph_outputs[0], graph_runner2_->get_tensor_format());
```

**Why:** KV260 uses `GraphRunner` for kernel 2 instead of `DpuTask`. The `GraphRunner` manages its own tensor buffers and needs explicit sync calls, which `DpuTask` handled internally.

---

**Change 6 — Update the `CPUsfm` initialisation**

At the end of 'PsmNetImp::PsmNetImp()' function.

```cpp
// BEFORE
cpu_sfm_ = std::make_unique<CPUsfm>(tasks_[2]->getOutputTensor(0u)[0]);
```

```cpp
// AFTER
cpu_sfm_ = std::make_unique<CPUsfm>(final_tensor2_);
```

**Why:** `final_tensor2_` now holds the output tensor for kernel 2 instead of `tasks_[2]`.

---

**Change 7 — Add following functions before cost-volume function**

Just Before the cost-volume function, add:

```cpp
// ADD
static inline int8_t cv_requant(int8_t v, int shift_num) {
  int x = v;
  if (shift_num > 0) {
    x = (x >> shift_num) + ((x >> (shift_num - 1)) & 1);
  } else if (shift_num < 0) {
    x <<= -shift_num;
    if (x > 127) x = 127;
    if (x < -128) x = -128;
  }
  return static_cast<int8_t>(x);
}

void cost_volume_cpu(const int8_t* input_l_ptr, const int8_t* input_r_ptr,
                     int8_t* output_ptr, size_t height, size_t width,
                     size_t channel, int act_fix, size_t disp) {
  LOG_IF(INFO, ENV_PARAM(DEBUG_COST_VOLUME))
      << "height " << height << " "    //
      << "width " << width << " "      //
      << "channel " << channel << " "  //
      << "act_fix " << act_fix << " "  //
      << "disp " << disp;

  __TIC__(COST_VOLUME_C)
  const size_t pair = channel * 2;
  for (size_t h = 0; h < height; ++h) {
    for (size_t w = 0; w < width; ++w) {
      const int8_t* l = input_l_ptr + (h * width + w) * channel;
      int8_t* o = output_ptr + (h * width + w) * disp * pair;
      for (size_t m = 0; m < disp; ++m, o += pair) {
        if (w < m) {
          memset(o, 0, pair);
          continue;
        }
        const int8_t* r = input_r_ptr + (h * width + w - m) * channel;
        if (act_fix == 0) {
          memcpy(o, l, channel);
          memcpy(o + channel, r, channel);
        } else {
          for (size_t c = 0; c < channel; ++c) {
            o[c] = cv_requant(l[c], act_fix);
            o[channel + c] = cv_requant(r[c], act_fix);
          }
        }
      }
    }
  }
  __TOC__(COST_VOLUME_C)
}

static void requant_neon(int8_t* dst, const int8_t* src, size_t n,
                         int shift_num) {
  if (shift_num == 0) {
    memcpy(dst, src, n);
    return;
  }
  size_t i = 0;
  if (shift_num > 0) {
    // round half up: (v >> s) + ((v >> (s - 1)) & 1)
    const int8x16_t all_1 = vdupq_n_s8(1);
    for (; i + 16 <= n; i += 16) {
      int8x16_t v = vld1q_s8(src + i);
      int8x16_t q = vshrq_n_s8(v, shift_num);
      int8x16_t r = vandq_s8(vshrq_n_s8(v, shift_num - 1), all_1);
      vst1q_s8(dst + i, vaddq_s8(q, r));
    }
  } else {
    // SQSHL saturates, matching cv_requant()'s clamp to [-128, 127].
    const int8x16_t cnt = vdupq_n_s8(static_cast<int8_t>(-shift_num));
    for (; i + 16 <= n; i += 16) {
      vst1q_s8(dst + i, vqshlq_s8(vld1q_s8(src + i), cnt));
    }
  }
  for (; i < n; ++i) dst[i] = cv_requant(src[i], shift_num);
}

void cost_volume_neon_hoisted(const int8_t* input_l_ptr,
                              const int8_t* input_r_ptr, int8_t* output_ptr,
                              size_t height, size_t width, size_t channel,
                              int act_fix, size_t disp) {
  LOG_IF(INFO, ENV_PARAM(DEBUG_COST_VOLUME))
      << "height " << height << " "    //
      << "width " << width << " "      //
      << "channel " << channel << " "  //
      << "act_fix " << act_fix << " "  //
      << "disp " << disp;

  __TIC__(COST_VOLUME_NEON)
  // ~1.1 MB each for this model. Sized on first use and reused across frames;
  // thread_local so concurrent PsmNet runners do not share them.
  static thread_local std::vector<int8_t> scratch_l;
  static thread_local std::vector<int8_t> scratch_r;
  const size_t fea_size = height * width * channel;
  const int8_t* l_base = input_l_ptr;
  const int8_t* r_base = input_r_ptr;
  if (act_fix != 0) {
    if (scratch_l.size() < fea_size) scratch_l.resize(fea_size);
    if (scratch_r.size() < fea_size) scratch_r.resize(fea_size);
    __TIC__(COST_VOLUME_REQUANT)
    requant_neon(scratch_l.data(), input_l_ptr, fea_size, act_fix);
    requant_neon(scratch_r.data(), input_r_ptr, fea_size, act_fix);
    __TOC__(COST_VOLUME_REQUANT)
    l_base = scratch_l.data();
    r_base = scratch_r.data();
  }

  const size_t pair = channel * 2;
  for (size_t h = 0; h < height; ++h) {
    for (size_t w = 0; w < width; ++w) {
      const int8_t* l = l_base + (h * width + w) * channel;
      int8_t* o = output_ptr + (h * width + w) * disp * pair;
      for (size_t m = 0; m < disp; ++m, o += pair) {
        if (w < m) {
          memset(o, 0, pair);
          continue;
        }
        memcpy(o, l, channel);
        memcpy(o + channel, r_base + (h * width + w - m) * channel, channel);
      }
    }
  }
  __TOC__(COST_VOLUME_NEON)
}
```

---

**Change 8 — Add batch-size and cost-volume shape checks**

Before the cost-volume computation for loop, add:
Just after `size_t input_r_size = input_r.size / input_r.batch;`

```cpp
// ADD
CHECK_EQ(input_l.batch, output.batch)
    << "feature-map batch and cost-volume batch must match.";
```

And inside the cost-volume loop, add:
Just before `#ifdef ENABLE_NEON`

```cpp
// ADD
CHECK_EQ(out_size, height * width * disp * channel * 2)
    << "cost volume must be [H][W][" << disp << "][2*" << channel << "].";
```

**Why:** Kernels 0/1 run on the DPU while kernel 2 runs through `GraphRunner`. Nothing forces their batch sizes to agree, so a mismatch would cause an out-of-bounds memory access without this check.

---

**Change 9 — Replace the `ENABLE_NEON` compile-time switch with a runtime switch**

```cpp
// BEFORE
#ifdef ENABLE_NEON
    if (act_fix == 0 && channel == 32) {
      cost_volume_neon(input_l_ptr, input_r_ptr, output_ptr, height, width, 32,
                       0, 48);
    } else {
      cost_volume_neon(input_l_ptr, input_r_ptr, output_ptr, height, width,
                       channel, act_fix, disp);
    }
#else
    __TIC__(COST_VOLUME_C)
    auto o_i = 0;
    for (auto h = 0u; h < height; ++h) {
      for (auto w = 0u; w < width; ++w) {
        for (auto m = 0u; m < disp; ++m) {
          for (auto c = 0u; c < channel * 2; ++c) {
            o_i = o_i + 1;
          }
        }
      }
    }
    __TOC__(COST_VOLUME_C)
#endif
```

```cpp
// AFTER
if (ENV_PARAM(PSMNET_CV_NEON)) {
  cost_volume_neon_hoisted(input_l_ptr, input_r_ptr, output_ptr, height,
                           width, channel, act_fix, disp);
} else {
  cost_volume_cpu(input_l_ptr, input_r_ptr, output_ptr, height, width,
                  channel, act_fix, disp);
}
```

**Why:** The old `#else` branch was a stub (it only incremented a counter and did nothing). The new `cost_volume_cpu` is the real scalar implementation. Both paths are always compiled. The env flag lets you switch at runtime to compare their outputs on the board without rebuilding.

---

**Change 10 — Replace `tasks_[2]->getInputTensor` with the cached descriptor**

```cpp
// BEFORE
auto input_tensor_k2 = tasks_[2]->getInputTensor(0u)[0];
```

```cpp
// AFTER
auto input_tensor_k2 = input_tensor_k2_;
```

---

**Change 11 — Replace `tasks_[2]->run()` with the `GraphRunner` execute cycle**

```cpp
// BEFORE
tasks_[2]->run(0u);
```

```cpp
// AFTER
{
  auto graph_inputs = graph_runner2_->get_inputs();
  auto graph_outputs = graph_runner2_->get_outputs();
  for (auto input : graph_inputs) {
    input->sync_for_write(0,
        input->get_tensor()->get_data_size() / input->get_tensor()->get_shape()[0]);
  }
  auto job = graph_runner2_->execute_async(graph_inputs, graph_outputs);
  CHECK_EQ(job.second, 0);
  CHECK_EQ(graph_runner2_->wait(job.first, -1), 0);
  for (auto output : graph_outputs) {
    output->sync_for_read(0,
        output->get_tensor()->get_data_size() / output->get_tensor()->get_shape()[0]);
  }
}
```

**Why:** `GraphRunner` manages its own tensor buffers and requires explicit sync calls when crossing the CPU/DPU boundary. `DpuTask::run()` handled this internally.

---

**Change 12 — Replace the debug dump with a correct one**

```cpp
// BEFORE
  if (ENV_PARAM(DUMP_PSMNET)) {
    auto final_tensor = tasks_[2]->getOutputTensor(0u)[0];
    for (size_t b = 0; b < final_tensor.batch; ++b) {
      std::ofstream ofs("tensor_res_" + to_string(b) + ".bin", ios::binary);
      ofs.write((char*)final_tensor.get_data(b),
                final_tensor.width * final_tensor.height);
      ofs.close();
    }
  }
```

```cpp
// AFTER
  if (ENV_PARAM(DUMP_PSMNET)) {
    auto final_tensor = final_tensor2_;
    const size_t hw = final_tensor.height * final_tensor.width;
    float* sfm_output = cpu_sfm_->get_output();
    for (size_t b = 0; b < final_tensor.batch; ++b) {
      auto filename = "disparity_" + to_string(b) + ".f32";
      LOG(INFO) << "write to " << filename << " (" << final_tensor.height << "x"
                << final_tensor.width << " float32)";
      std::ofstream ofs(filename, ios::binary);
      ofs.write((char*)(sfm_output + b * hw), hw * sizeof(float));
      ofs.close();
    }
  }
```

**Why:** The old code wrote `width * height` bytes, which is one byte per pixel — but the output is `float32` so each pixel needs 4 bytes. It was only capturing 1/768 of the actual data. The new code writes the correct `hw * sizeof(float)` bytes.

---

**Change 13 — Replace the remaining `tasks_[2]->getOutputTensor` reference**

```cpp
// BEFORE
auto final_tensor = tasks_[2]->getOutputTensor(0u)[0];
```

```cpp
// AFTER
auto final_tensor = final_tensor2_;
```

---

### 4. `Makefile`

**Change 1 — Replace the include flags and add the KV260 cross-compiler**

```makefile
# BEFORE
CXX_FLAGS += -I=/usr/include/
CXX_FLAGS += -I=/usr/include/xrt
CXX_FLAGS += -I=/install/Release/include
CXX_FLAGS += -DENABLE_CV_ACCEL=1
```

```makefile
# AFTER
SYSROOT ?= /path/to/petalinux_sdk_2022.2/sysroots/cortexa72-cortexa53-xilinx-linux
override CXX := /path/to/petalinux_sdk_2022.2/sysroots/x86_64-petalinux-linux/usr/bin/aarch64-xilinx-linux/aarch64-xilinx-linux-g++
override LD := $(CXX)

CXX_FLAGS += --sysroot=$(SYSROOT)
CXX_FLAGS += -I$(SYSROOT)/usr/include
CXX_FLAGS += -I$(SYSROOT)/usr/include/opencv4
CXX_FLAGS += -I$(SYSROOT)/usr/include/xrt
CXX_FLAGS += -I/install/Release/include
# ENABLE_CV_ACCEL requires the VCK190 PL xclbin — not available on KV260.
#CXX_FLAGS += -DENABLE_CV_ACCEL=1
```

> Replace `/path/to/petalinux_sdk_2022.2` with the actual path to your SDK.

**Why:** The original paths target a native VCK190 build. For KV260 you need to cross-compile with the PetaLinux 2022.2 SDK for the Cortex-A72/A53 cores. `ENABLE_CV_ACCEL` enables the VCK190 PL cost-volume kernel, which does not exist on KV260, so it is commented out.

---

**Change 2 — Add debug symbols**

```makefile
# ADD
CXX_FLAGS += -g
```

---

**Change 3 — Remove the duplicate OpenCV include**

```makefile
# REMOVE this line
CXX_FLAGS += -I=/usr/include/opencv4
```

**Why:** OpenCV is now included via the sysroot path added above.

---

**Change 4 — Replace the library search paths**

```makefile
# BEFORE
LD_FLAGS += -L=/install/Release/lib
LD_FLAGS += -L/opt/petalinux/2021.1/sysroots/cortexa72-cortexa53-xilinx-linux/install/Release/lib/
```

```makefile
# AFTER
LD_FLAGS += -L$(SYSROOT)/usr/lib
```

**Why:** Vitis-AI 3.0 ships its libraries under `<sysroot>/usr/lib` in the PetaLinux 2022.2 SDK. The old paths do not exist in this layout.

---

**Change 5 — Add the `GraphRunner` libraries**

```makefile
# ADD (after existing LD_FLAGS lines)
LD_FLAGS += -lvitis_ai_library-graph_runner
LD_FLAGS += -lvart-runner
LD_FLAGS += -lxir
```

**Why:** These are required to link the `GraphRunner` used for kernel 2.

---

**Change 6 — Remove the XRT libraries**

```makefile
# BEFORE
LD_FLAGS += -lxrt_coreutil
LD_FLAGS += -lxrt_core
```

```makefile
# AFTER
#LD_FLAGS += -lxrt_coreutil
#LD_FLAGS += -lxrt_core
```

**Why:** XRT is only needed by the VCK190 PL cost-volume path (`ENABLE_CV_ACCEL`). With that flag off, the code is unreachable. The KV260 PetaLinux image also does not ship `libxrt_core.so.2` or `libxrt_coreutil.so.2`, so linking them would cause a build error. Note: the XRT *headers* are still needed to compile `cost_volume.hpp` — only the libraries are removed.

---

**Change 7 — Remove `cost_volume.o` from the build targets**

```makefile
# BEFORE
$(OUTPUT_DIR)/demo_psmnet: $(OUTPUT_DIR)/demo_psmnet.o $(OUTPUT_DIR)/psmnet.o $(OUTPUT_DIR)/cpu_op.o $(OUTPUT_DIR)/cost_volume.o

$(OUTPUT_DIR)/test_performance_psmnet: $(OUTPUT_DIR)/test_performance_psmnet.o $(OUTPUT_DIR)/psmnet.o $(OUTPUT_DIR)/cpu_op.o $(OUTPUT_DIR)/cost_volume.o
```

```makefile
# AFTER
$(OUTPUT_DIR)/demo_psmnet: $(OUTPUT_DIR)/demo_psmnet.o $(OUTPUT_DIR)/psmnet.o $(OUTPUT_DIR)/cpu_op.o

$(OUTPUT_DIR)/test_performance_psmnet: $(OUTPUT_DIR)/test_performance_psmnet.o $(OUTPUT_DIR)/psmnet.o $(OUTPUT_DIR)/cpu_op.o
```

**Why:** `cost_volume.o` implements the VCK190 PL accelerator path. It is not used on KV260, and linking it would pull in the XRT libraries that are not available.

---

## Build

After applying all changes, build from the `psmnet/` directory:

```bash
make
```
To Run the Depth-Estimation on board send the following files on the board using `scp`.
1. PSMNet_pruned_0_pt
2. PSMNet_pruned_1_pt 
3. PSMNet_pruned_2_pt
4. demo_psmnet
5. demo_psmnet_left.png
6. demo_psmnet_right.png

To use the scaler cost-volume path on board:

```bash
sudo env DEEPHI_PROFILING=1 PSMNET_CV_NEON=0 ./demo_psmnet_neon demo_psmnet_left.png demo_psmnet_right.png
```
To use the NEON cost-volume path instead of the scalar one:

```bash
sudo env DEEPHI_PROFILING=1 PSMNET_CV_NEON=1 ./demo_psmnet_neon demo_psmnet_left.png demo_psmnet_right.png
```
