# conv3d_fix: int8 Conv3D / transposed Conv3D on the KV260 PL

An HLS kernel for PSMNet's `conv3d-fix` and `transposed-conv3d-fix` layers,
linked into the same xclbin as the DPU (DPUCZDX8G). The host side builds the
same two op libraries as [../cpu](../cpu/README.md):
`libvart_op_imp_conv3d-fix.so` and `libvart_op_imp_transposed-conv3d-fix.so`.
Each layer runs on the PL when the kernel supports it, otherwise on the CPU
kernel. The output is bit-exact with the CPU kernel.

## Files

| file | role |
|---|---|
| `conv3d_fix_hls.h` / `conv3d_fix_hls.cpp` | HLS kernel `conv3d_fix`, limits, register/cfg layout (`CONV3D_RED`, default 24) |
| `conv3d_fix_fpga_host.hpp` | host driver `conv3d::FpgaKernel`: `supported()`, the planner (tap/phase tables, weight packing, `bterm`/`thr6`), XRT C-API backend |
| `conv3d_fix_op_fpga.hpp` | op wrapper: FPGA path with CPU fallback, verify mode (replaces `../cpu/ops/common/conv3d_fix_op.hpp`) |
| `ops/conv3d-fix_fpga.cpp`, `ops/transposed-conv3d-fix_fpga.cpp` | op library entry points |
| `ops/build_ops_fpga.sh` | cross build of the two op libraries with `-DUSE_XRT` → `ops/aarch64/` |
| `test_conv3d_fix.cpp` | C-sim: kernel vs the CPU kernel, 25 cases (13 generic + 12 PSMNet kernel-2 layer types: real C, k, stride, pad, small H×W), PSMNet/RELU6 cases repeated for 4 fix-point sets |
| `sim_threaded/hls_stream.h` | bounded, blocking `hls::stream` for the threaded (deadlock) simulation |
| `Makefile` | `check`, `check-threaded`, `hls`, `clean` |
| `run_hls.tcl` | Vitis HLS 2022.2 script: csynth, `.xo` export, optional Vivado synthesis |
| `system.cfg` | `v++ --link` lines for the kernel: `nk=`, `sp=` per AXI port, `[clock] id=2` |
| `check_util.py` | fails if the kernel uses more than a given % of any K26 resource |

The kernel includes the CPU sources from `../cpu` (`ops/common`, `include`, the test stubs).

## Architecture

```
reader -> aligner -> mac_engine -> epilogue -> writer      (HLS DATAFLOW, 200 MHz)
 DDR      16 B align  16 oc x RED lanes x 2   int8 out    16 B bursts
```

- **Jobs.** One job is one output line `(n, oh, ow)`: all OD × OC outputs.
  The host plans everything and passes a `cfg` block (32 words) plus `tab`
  (tap and phase tables, `bterm`, `thr6`). Ports:
  - `input`: gmem0
  - `output`: gmem1
  - `weight`, `tab`, `cfg`: gmem2, one bundle
  - all three are 128-bit AXI.
- **Flattened reduction.** The RED lanes run over the `(kd, ic)` bytes of the
  staged line, so small IC (6, 10, 20, 26) wastes few lanes and needs no padding.
- **Transposed conv by phases.** The host groups output positions by stride
  residue and packs one weight set per phase, so no cycles go to stride holes.
- **DSP packing.** Two output positions share one DSP48E2: `(b<<18 + a) · w`.
  Up to 6 products are summed before unpacking, which stays exact.
  - RED=24: 16 × 24 × 2 = 768 MACs/cycle; MAC array 16 × 24 = 384 DSP
    (`conv3d_fix_hls.h`), 448 DSP for the whole kernel in the routed link.
  - RED=16: 512 MACs/cycle; MAC array 256 DSP, 320 DSP routed.
- **Weights.** Loaded once per call into URAM (≤ `W_ROWS` rows).
- **Integer epilogue.** `r = clamp((acc + bterm[oc]) >> n)`, then ReLU / ReLU6
  (`thr6`). The host computes `bterm = floor(bias·2^b + 2^(n−1))`. This is
  bit-exact with the CPU float epilogue, and `supported()` proves it per layer
  ([../cpu/README.md](../cpu/README.md#epilogue-run_rows-in-opscommonconv3d_fix_kernelhpp)).
- **Host backend: XRT C API.** The kernel shares the DPU's xclbin, so the host
  never loads one. It uses:
  - `xclOpen`, then `xclIPName2Index("conv3d_fix:conv3d_fix_1")`;
  - `xclOpenContext` with the uuid of the xclbin named in `/etc/vart.conf`
    (or `XLNX_VART_FIRMWARE`);
  - `xclAllocBO`, then `ERT_START_CU` via `xclExecBuf` / `xclExecWait`.

## Limits

These come from `conv3d_fix_hls.h`. A layer outside them, or one that fails
the epilogue exactness check, runs on the CPU kernel automatically.

| limit | value |
|---|---|
| kernel extent KH, KW | ≤ `MAX_K` = 7 |
| output channels OC | ≤ `MAX_OC` = 128 (`NOCT` = OC/16 ≤ 8) |
| OH, OW | ≤ `MAX_OHW` = 512 |
| D tiles × OC tiles | ≤ `MAX_TN` = 16 (D tile = 16 positions) |
| D tiles per line | ≤ `MAX_TILES` = 32 |
| staged input line | ≤ `LINE_W` = 256 words of 16 B (4 KB) |
| output line OD · OC | ≤ `LINE_MAX` = 8192 bytes |
| packed weights | ≤ `W_ROWS` = 8192 rows of RED bytes |
| nonlinear | NONE, RELU, RELU6 (LEAKYRELU, PRELU → CPU) |
| group | 1 |
| fix points | `shift_cut_scale = 2^n`, `shift_bias_scale = 2^b` powers of two, `0 ≤ n ≤ 24`, `−30 ≤ b ≤ 22`, float epilogue provably exact (`epilogue_plan()`) |

All 28 layers of PSMNet kernel 2 (22 `conv3d-fix` + 6 `transposed-conv3d-fix`) pass `supported()`. This was checked
with their real shapes, attributes and fix points from the compiled B512 model (`model/`).

## Build

All commands run from this folder. Tools: Vitis / Vitis HLS / Vivado 2022.2
in `/tools/X22`, and the PetaLinux 2022.2 SDK.

### 1. C simulation (bit-exact vs the CPU kernel)

```sh
make check                 # RED=24
make check RED=16
```

Expected: every case `mismatches=0  PASS`, then `ALL TESTS PASSED (0 failing case(s))`.

### 2. Threaded simulation (deadlock check)

```sh
make check-threaded
```

This build uses `-DCONV3D_THREADED_SIM -Isim_threaded`. Each DATAFLOW stage
runs in its own thread, and the FIFOs are bounded at their real depth. A stage
blocked longer than `CONV3D_SIM_STALL_MS` (default 5000 ms) prints `DEADLOCK`.
Expected: `ALL TESTS PASSED (0 failing case(s))`.

### 3. HLS → `conv3d_fix.xo`

```sh
make hls                   # RED=24, ~25 min
make hls RED=16
```

This runs `vitis_hls -f run_hls.tcl`:

- part `xck26-sfvc784-2LV-c`, 5 ns clock;
- `config_interface -m_axi_max_widen_bitwidth 128`;
- exports `conv3d_fix.xo` and runs Vivado synthesis;
- then `python3 check_util.py conv3d_fix_prj/sol1/impl/report/verilog/export_syn.rpt 40`
  fails if any resource is above 40 %.

Check in the HLS report that every `m_axi_gmemN` is `128 -> 128` and that
`mac_engine` has II = 1.

### 4. Link with the DPU → `dpu.xclbin`

The KV260 holds one xclbin, so the kernel is linked next to `DPUCZDX8G` in a
Vitis 2022.2 IDE project. Creating that project, adding the kernel and setting
the link options (the lines of `system.cfg` plus the DPU's) is described in
[../../README.md, Building the firmware](../../README.md#building-the-firmware).
In `system.cfg`, `weight`, `tab` and `cfg` share one AXI bundle, so they must
stay on the same bank.

The current xclbin was linked in the earlier project
`/mnt/ssd2/abdul_raouf_khalid/GitRepose/dpu_sfm` (same layout). The IDE writes
the link command to `dpu_system_hw_link/Hardware/dpu.xclbin.sh`:

```sh
cd /mnt/ssd2/abdul_raouf_khalid/GitRepose/dpu_sfm/dpu_system_hw_link/Hardware
/tools/X22/Vitis/2022.2/bin/v++ --target hw --link --config dpu-link.cfg -odpu.xclbin ../../dpu_kernels/Hardware/DPUCZDX8G.xo ../../dpu_kernels/Hardware/build/conv3d_fix.xo
```

Linking this repo's `conv3d_fix.xo` instead (from step 3) means swapping in
`/mnt/ssd2/abdul_raouf_khalid/GitRepose/PSMNet/kernel/fpga/conv3d_fix.xo` as
the last argument. `TODO: verify` — this link was not run from this repo.
The link takes 1–2 h.

Before using the result, check:

- `All user specified timing constraints are met.` in
  `dpu.build/link/vivado/vpl/prj/prj.runs/impl_1/system_wrapper_timing_summary_routed.rpt`
- kernel utilization in `dpu.build/reports/link/imp/impl_1_kernel_util_routed.rpt`
- the `UUID (xclbin)` line in `dpu.xclbin.info`

**Current xclbin** (not copied here, about 8 MB):

- file: `/mnt/ssd2/abdul_raouf_khalid/GitRepose/dpu_sfm/dpu_system_hw_link/Hardware/dpu.xclbin`
- built Oct 14, uuid `918c6c7f-44a0-0d67-3d88-253476983547`
- DPUCZDX8G + conv3d_fix (RED=24)
- routed WNS +0.027 ns
- conv3d_fix routed utilization: 42,586 LUT, 33,766 FF, 22 BRAM, 6 URAM, 448 DSP

### 5. Bitstream for the board

`v++` does not create the `.bit.bin`. Convert the link's `system.bit` with
`bootgen` as described in
[../../README.md, Step 5: Create the firmware files](../../README.md#step-5-create-the-firmware-files).

### 6. Op libraries (`libvart*.so`)

```sh
./ops/build_ops_fpga.sh
```

- Output: `ops/aarch64/libvart_op_imp_conv3d-fix.so` and
  `ops/aarch64/libvart_op_imp_transposed-conv3d-fix.so`.
- They depend on `libvitis_ai_library-runner_helper.so.3`,
  `libvart-runner.so.3`, `libxir.so.3`, `libglog.so.0` and
  `libxrt_core.so.2`.
- Overrides: `SYSROOT=...`, `XILINX_HLS=...` (for `ap_int.h`), `CXX=...`, `OUT=...`, `EXTRA_DEFS=...`.

The host libraries take `CONV3D_RED` from `conv3d_fix_hls.h` (`#define
CONV3D_RED 24`, used when nothing else is given). It must match the kernel in
the loaded xclbin (448 DSP routed = RED 24, 320 = RED 16). For a RED=16 xclbin:

```sh
EXTRA_DEFS=-DCONV3D_RED=16 ./ops/build_ops_fpga.sh
```

### 7. Copy to the board

This updates an existing `dpu512_c` app (new xclbin and bitstream) and installs
the op libraries. For the first installation of the firmware, see
[../../README.md, Step 6: Install on the board](../../README.md#step-6-install-on-the-board).

From the host:

```sh
scp ops/aarch64/libvart_op_imp_conv3d-fix.so ops/aarch64/libvart_op_imp_transposed-conv3d-fix.so petalinux@10.42.0.27:~
scp /mnt/ssd2/abdul_raouf_khalid/GitRepose/dpu_sfm/dpu_system_hw_link/Hardware/dpu.xclbin /mnt/ssd2/abdul_raouf_khalid/GitRepose/dpu_sfm/dpu_system_hw_link/Hardware/dpu.build/link/int/system.bit.bin petalinux@10.42.0.27:~
```

On the board (the `dpu512_c` app holds `dpu.xclbin`, `dpu.bit.bin`, `dpu512_c.dtbo`, `shell.json`):

```sh
mkdir -p /home/petalinux/fw_backup_dpu512_c && cp -a /lib/firmware/xilinx/dpu512_c/. /home/petalinux/fw_backup_dpu512_c/
sudo cp ~/dpu.xclbin /lib/firmware/xilinx/dpu512_c/dpu.xclbin
sudo cp ~/system.bit.bin /lib/firmware/xilinx/dpu512_c/dpu.bit.bin
sudo cp ~/libvart_op_imp_conv3d-fix.so ~/libvart_op_imp_transposed-conv3d-fix.so /usr/lib/
sudo xmutil unloadapp
sudo xmutil loadapp dpu512_c
xclbinutil --info --input /lib/firmware/xilinx/dpu512_c/dpu.xclbin | grep "UUID (xclbin)"
strings /usr/lib/libvart_op_imp_conv3d-fix.so | grep -c VAI_CONV3D_FPGA_VERIFY     # 1 = FPGA build
```

Keep backups outside `/lib/firmware`: dfx-mgr can match extra files there.

## Verify on the board

From the application folder on the board (see [../../application/README.md](../../application/README.md)):

```sh
DEBUG_CONV3D_FPGA=1 VAI_CONV3D_FPGA_VERIFY=1 ./demo_psmnet demo_psmnet_left.png demo_psmnet_right.png 2>&1 | grep -E ": FPGA|: CPU|VERIFY"
```

Every conv3d layer must log `FPGA` and `VERIFY ... mismatches 0/<n>`. With
`VAI_CONV3D_FPGA_VERIFY=2` the CPU result replaces the FPGA output, which
separates kernel errors from other errors. For timing, run without VERIFY,
using `DEEPHI_PROFILING=1` (`CPU_OP_EXEC : <us>` per op).

## Known issues and fixes

| issue | symptom | cause | fix (in these sources) |
|---|---|---|---|
| dataflow deadlock | host timeout; CU `ap_ctrl 0x1`, never done | `mac_engine` read the next job's header at the start of each job; `reader` writes it only after all lines of the current job, which do not fit in the FIFOs | read the next header at the start of the job's last line; caught by `make check-threaded` (plain C-sim passed) |
| accumulator RAW hazard | `mismatches` on OC ≤ 16 layers only (NOCT = 1), max diff 30–97; disparity slightly distorted | `DEPENDENCE ... inter false` on `accA`/`accB`; the same entry is updated every 8 iterations, load and store were 13 stages apart | `#pragma HLS DEPENDENCE variable = accA type = inter direction = RAW distance = 8 dependent = true` (same for `accB`) |
| AXI widened to 512 bit | link timing failure in the HP interconnect downsizer / clock converter | the IDE `v++ -c` flow let HLS widen gmem2 to 512 bit; HP ports are 128 bit | `max_widen_bitwidth = 128` on every `#pragma HLS INTERFACE m_axi` (and `config_interface -m_axi_max_widen_bitwidth 128` in `run_hls.tcl`) |
| wrong clock id | kernel at 100 MHz, clock converters in the interconnect | platform clock ids are 1 = 100, 2 = 200, 3 = 400 MHz | `[clock] id=2:conv3d_fix_1.ap_clk` (200 MHz, the DPU's aclk) |
| routed timing miss of tens of ps | WNS −0.023 ns on the epilogue output buffer | placement in a crowded device | not in `system.cfg` (the current link met timing without it); if it recurs, add `[vivado]` `prop=run.impl_1.STEPS.POST_ROUTE_PHYS_OPT_DESIGN.IS_ENABLED=true` and `prop=run.impl_1.STEPS.POST_ROUTE_PHYS_OPT_DESIGN.ARGS.DIRECTIVE=AggressiveExplore` to the link config |
| host could not reach the CU | `No xclbin with uuid` (xrt::kernel), `get_num_of_cu() == 0` (xir::XrtCu) | vart owns the device handle; `XrtCu` only registers the DPU's CUs (by name) | XRT C API backend in `conv3d_fix_fpga_host.hpp` |
| host/kernel RED mismatch | wrong weights silently | host libs built with a different `CONV3D_RED` than the `.xo` | keep `conv3d_fix_hls.h` identical for the `.xo` and `build_ops_fpga.sh` |
| DPU `sp=` lines missing from the generated `dpu-link.cfg` | — | the IDE settings box and the generated file diverge | check the generated `dpu-link.cfg` after every IDE change; `TODO: verify` why |
| `dfx-mgrd` crash after `xmutil loadapp` | `malloc(): invalid next size` | unknown, `TODO: verify`; the DPU and kernel still work | reboot to recover from a stale overlay |
| next run fails after a kernel hang | `Bitstream download failed!` / `xclExecBuf` errors | the hung CU stays hung | reboot the board |
