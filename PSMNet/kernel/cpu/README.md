# conv3d-fix on the CPU (golden reference and fallback)

The Vitis AI 3.0 DPU cannot run PSMNet's 3D convolutions, so the compiler puts
them in CPU subgraphs as `conv3d-fix` and `transposed-conv3d-fix` ops. At
graph-runner init, `cpu_task` loads `libvart_op_imp_<op type>.so` for each
CPU op type. This folder builds those two libraries with a multi-threaded,
NEON-accelerated int8 kernel.

It has two roles:

- **Golden reference.** The PL kernel in [../fpga](../fpga/README.md) must be
  bit-exact with `ops/common/conv3d_fix_kernel.hpp`. The FPGA C-sim and the
  on-board verify mode (`VAI_CONV3D_FPGA_VERIFY=1`) both compare against it.
- **Fallback.** The FPGA op libraries include this kernel and use it for any
  layer the PL kernel does not support (`FpgaKernel::supported()`), or for all
  layers with `VAI_CONV3D_FPGA=0`. The libraries built here are the CPU-only
  version.

## What it computes

XIR layout and conventions:

```
input   : [N, H, W, D, IC]        int8, fix point in_fp
weights : [OC, KH, KW, KD, IC]    int8, fix point w_fp
bias    : [OC]                    int8, fix point bias_fp (optional)
output  : [N, OH, OW, OD, OC]     int8, fix point out_fp
kernel / stride / dilation : {W, H, D}
pad                        : {left, right, top, bottom, near, far}, pad_mode FLOOR / CEIL / VALID / SAME
```

`transposed-conv3d-fix` uses the same kernel with stride holes and a mirrored
kernel (XIR stores transposed weights mirrored). Per output element,
the kernel accumulates `acc = Σ input · weight` over all taps and IC in an
integer, then runs the epilogue.

### Epilogue (`run_rows` in `ops/common/conv3d_fix_kernel.hpp`)

```cpp
float filout = static_cast<float>(acc[oc]);
if (bias != nullptr) filout += static_cast<float>(bias[oc]) * p_.shift_bias_scale;
output = util::fix(nonlinear_fn(filout / p_.shift_cut_scale));
```

- `shift_cut_scale = 2^(in_fp + w_fp − out_fp)` = `2^n`, and
  `shift_bias_scale = 2^(in_fp + w_fp − bias_fp)` = `2^b`
  (`Param::set_fix_point`, `Param::set_bias_fix_point`).
- `nonlinear_fn`:
  - RELU: `max(x, 0)`
  - RELU6: `min(max(x, 0), 6·2^out_fp)`
  - LEAKYRELU: `x · 0.01` for `x < 0`
  - PRELU: `x · prelu` for `x < 0`
  - NONE
- `util::fix` (`ops/common/util.hpp`): clamp to [−128, 127], then round to
  nearest with ties toward +∞ (`ceil` for negative `x.5`, `round` otherwise).

The PL kernel computes the same result in integers. The host precomputes
`bterm[oc] = floor(bias·2^b + 2^(n−1))` (plus `thr6` for RELU6), and the
kernel computes `clamp((acc + bterm) >> n)` with ReLU/ReLU6. This is exact
because `floor((acc + f)/2^n) == floor((acc + floor(f))/2^n)` for integer
`acc`. The float form only rounds when |value| ≥ 256 after scaling, and then
both sides saturate. Details:
[../../skills/conv3d_kernel_skill/references/epilogue-exactness.md](../../skills/conv3d_kernel_skill/references/epilogue-exactness.md).

### Speed

- Rows of the output are split across cores (`VAI_CONV3D_FIX_NUM_THREADS`,
  default all cores). Each worker writes its own rows, so the result does not
  depend on the thread count.
- On AArch64 the IC loop uses NEON (`vmull_s8` / `vmlal_s8`, 4 OC unrolled).
  The result is bit-identical to the scalar loop.

## Files

| file | role |
|---|---|
| `ops/common/conv3d_fix_kernel.hpp` | the kernel: tap tables, threading, NEON MAC, epilogue (`Param`, `Kernel`) |
| `ops/common/conv3d_fix_op.hpp` | `vart::experimental::OpImpBase` wrapper: reads op attrs and fix points into `Param`, runs the kernel |
| `ops/common/util.hpp` | `util::fix`, `NONLINEAR`, helpers (from cpu_task) |
| `ops/conv3d-fix/conv3d-fix.cpp` | `libvart_op_imp_conv3d-fix.so` entry (`DEF_XIR_OP_IMP`) |
| `ops/transposed-conv3d-fix/transposed-conv3d-fix.cpp` | `libvart_op_imp_transposed-conv3d-fix.so` entry |
| `include/vart/op_imp.h` | Vitis AI 3.0 `vart/op_imp.h`  |
| `test/conv3d_fix/test_conv3d_fix.cpp`, `gen_and_check.py` | 3D kernel vs PyTorch, bit-exact on the int8 output |
| `test/conv3d_fix/stub/` | stub `glog`, so the test builds with plain `g++` |
| `test/conv3d_fix/Makefile`, `build_aarch64.sh` | original test Makefile and library build script |
| `Makefile` | builds the aarch64 libraries and the host tests |

## Build

From this folder:

```sh
make            # aarch64/libvart_op_imp_conv3d-fix.so, aarch64/libvart_op_imp_transposed-conv3d-fix.so + host test
make lib        # only the two aarch64 libraries
make test       # only the host test binary
make clean
```

Tools:

- Defaults: Vitis 2022.2 aarch64 g++
  (`/tools/X22/Vitis/2022.2/gnu/aarch64/lin/aarch64-linux/bin/aarch64-linux-gnu-g++`)
  and the PetaLinux 2022.2 SDK sysroot. Override them with
  `make SYSROOT=... AARCH64_CXX=...`.
- The libraries link `vitis_ai_library-runner_helper`, `vart-runner`, `xir`,
  `glog` and `pthread`, with `-Wl,--no-undefined`.
- The host test needs only `g++`.

## Tests

```sh
make check
```

`gen_and_check.py` (needs `python3`, `numpy` and `torch`) drives
`test_conv3d_fix` and compares the int8 output with PyTorch, bit-exact:

- [A] `util::fix` over a float sweep;
- [B] 19 conv/tconv configurations: PSMNet shapes, axis-distinct
  kernels/strides/pads, FLOOR/CEIL/VALID, no bias, saturation, RELU6,
  LEAKYRELU, batch;
- [C] 1 vs 7 threads bit-identical;

It ends with `ALL CHECKS PASSED`.

## Install on the board (CPU-only path)

```sh
make lib
scp aarch64/libvart_op_imp_conv3d-fix.so aarch64/libvart_op_imp_transposed-conv3d-fix.so petalinux@<Board IP>:~
```

On the board:

```sh
sudo cp ~/libvart_op_imp_conv3d-fix.so ~/libvart_op_imp_transposed-conv3d-fix.so /usr/lib/
```

The FPGA build from [../fpga](../fpga/README.md) installs under the same names
and replaces these.
