# What went wrong with conv3d_fix (and how to catch it earlier)

Each entry: symptom as seen, root cause, fix, and the check that would have
caught it before the board.

## 1. Dataflow deadlock (kernel hangs forever)
- **Symptom**: host timeout; `xclRegRead` of the CU shows `ap_ctrl 0x1`
  (`ap_start` set, never `ap_done`/`ap_idle`). The ERT command stays at
  state 1 (NEW) because the CU never finishes.
- **Cause**: `mac_engine` read the *next* job's header at the start of each
  job. `reader` only writes that header after pushing *all* lines of the
  current job (9 lines × ~200 words), which cannot fit in `raw_s(64)` +
  `data_s(32)`. Reader blocked on a full FIFO, MAC blocked on the header.
- **Fix**: read the next header at the start of the job's *last* line, the
  first moment it is needed (`if (last_line && has_next_job) { ... hdr_s.read() ... }`).
- **Catch it**: threaded simulation (`-DCONV3D_THREADED_SIM -Isim_threaded`).
  The old code deadlocked on the first test case; plain C-sim passed.
- **Rule**: a consumer must never wait for data that its producer only emits
  after data the consumer has not consumed yet.

## 2. Accumulator read-modify-write hazard (wrong results, OC ≤ 16 layers only)
- **Symptom**: `VAI_CONV3D_FPGA_VERIFY=1` showed `mismatches 0` for every layer
  with OC = 20/26/32 and 12–93 % mismatches (max |diff| 30–97, mostly not
  off-by-1) for OC = 4/6/10; the OC = 1 layer (single reduction block) was
  exact. Final disparity map: "a bit distorted", structure intact.
- **Cause**: `#pragma HLS DEPENDENCE variable=accA inter false`. The same
  entry is updated every `NOCT·8` iterations — only 8 when `NOCT = 1`. HLS
  scheduled the load at an early stage and the store ~13 stages later
  (iteration latency 15–16) → the next update read a stale value.
- **Fix**: `#pragma HLS DEPENDENCE variable=accA type=inter direction=RAW distance=8 dependent=true`
  (same for `accB`). New schedule: load at stage 4–5, add+store at stage 11, II = 1.
- **Catch it**: (a) never declare `inter false` on a read-modify-write array
  unless it holds for the smallest configuration; (b) read the schedule:
  `conv3d_fix_prj/sol1/.autopilot/db/<..>mac_engine_Pipeline_<loop>.verbose.sched.rpt`,
  grep the source line of `acc[..] += ...` and compare the `ST_<n>` of the
  `load` and the `store`; span must be < the true distance; (c) RTL co-sim of
  a NOCT = 1 case (`TODO: verify` — not run for conv3d).

## 3. AXI port widened to 512 bit → interconnect timing failure
- **Symptom**: link fails `[VPL 101-2] design did not meet timing`, 511
  endpoints, WNS −0.356 ns, paths inside
  `axi_ic_..._S_AXI_HP2_FPD/m00_couplers/auto_ds` (downsizer) → `auto_cc`
  (clock converter); `xclbinutil --info` shows `M_AXI_GMEM2 ... 512 bits`.
- **Cause**: the IDE flow (`v++ -c`) let HLS widen `gmem2` to 512 bit (the
  standalone `run_hls.tcl` capped it with `config_interface
  -m_axi_max_widen_bitwidth 128`); HP ports are 128 bit → downsizer.
  Combined with #4 (100 MHz kernels) → clock converter as well.
- **Fix**: `max_widen_bitwidth = 128` on every `#pragma HLS INTERFACE m_axi`
  (in the source, so every flow gets it). Verified: `gmem0/1/2: 128 -> 128`.
- **Catch it**: after every kernel build, check port widths in the HLS report
  ("m_axi_gmemN | 128 -> 128") and after link with `scripts/check_xclbin.sh`.

## 4. Wrong clock id (clock converters, slow kernels)
- **Symptom**: kernels at 100 MHz; clock converter paths in the HP interconnect.
- **Cause**: `kv260_custom_platform` numbers clocks 1 = 100, 2 = 200,
  3 = 400 MHz (`<xd:clock ... xd:id="2" xd:frequency="199.998000">` in the
  .hpfm inside the .xsa). Our docs said "id 1 = 200 MHz" — wrong.
- **Fix**: `id=2:<cu>.ap_clk` for every PL kernel next to the DPU (aclk 200 MHz).
- **Catch it**: read the ids from the platform, never assume:
  `unzip -p hw/<platform>.xsa '*.hpfm' | grep 'xd:clock '`.

## 5. Epilogue output-buffer path missed timing by 23 ps
- **Symptom**: WNS −0.023 ns, 2 endpoints, `epilogue ... icmp_..._reg → lbuf
  ram_reg_bram_0/DINBDIN`, 76 % routing delay, BRAM ~92 % / LUT ~76 % used.
- **Cause**: rotate + valid compare + 16:1 byte mux into a BRAM data pin in
  one cycle, in a crowded device; the previous link of the same RTL passed
  (+0.091 ns) → placement variance.
- **Fix order**: (1) `[vivado] prop=run.impl_1.STEPS.POST_ROUTE_PHYS_OPT_DESIGN.*`
  (later links met timing: +0.015, +0.028 ns); (2) stronger strategy
  `Performance_ExplorePostRoutePhysOpt`; (3) register rotated bytes +
  write-enables, write one cycle later (`TODO: verify` — not needed so far);
  (4) reduce BRAM/LUT crowding.

## 6. Host driver could not reach the CU (three dead ends)
| attempt | failure on the board |
|---|---|
| `xrt::device(0)` + `xrt::kernel(dev, dev.get_xclbin_uuid(), ...)` | `XRT: No xclbin with uuid '...': Invalid argument` |
| + `dev.load_xclbin(<vart.conf firmware>)` first | same error (vart's process-level handle was first) |
| `xir::XrtCu("conv3d_fix")` + `xir::BufferObject` | `get_num_of_cu() == 0`: vart's `XrtDeviceHandleImp` only registers the DPU's CUs (by name) |
| **XRT C API**: `xclOpen`, `xclIPName2Index("conv3d_fix:conv3d_fix_1")`, `xclOpenContext(uuid from the vart.conf xclbin)`, `xclAllocBO`, ERT `ERT_START_CU`, `xclExecBuf/xclExecWait` | works |
Also: `cmd->type` does not matter — the code leaves it 0 (like vart's DPU
controller), and the board's timeout dump still showed header `0x30013001`
(type 3): XRT sets it. A command stuck at state 1 with `ap_ctrl 0x1` is a
kernel hang (#1), not a rejected command.

## 7. Host constants out of sync with the kernel in the xclbin
- **Symptom**: risk of silently wrong weights: the IDE header had
  `#define CONV3D_RED 16`, host libs were built with the default 24.
- **Fix**: build host libs with `-DCONV3D_RED=<kernel's value>`; every
  constant in the shared header that changes data layout is part of the ABI.
- **Catch it**: check DSP count of the kernel (`OC_PAR × RED` + epilogue) and
  diff the header used for the `.xo` against the one used for the host.

## 8. Board / firmware pitfalls
- **xclbin and .bit.bin from different links** → wrong CU addresses / hangs.
  `scripts/check_xclbin.sh dpu.xclbin dpu.bit.bin` compares the bitstream
  inside the xclbin with the .bit.bin byte for byte.
- **`v++` does not make the `.bit.bin`**: `bootgen` on
  `dpu.build/link/int/system.bit` (`echo 'all:{system.bit}' > b.bif;
  bootgen -w -arch zynqmp -process_bitstream bin -image b.bif`).
- **dfx-mgrd** died with `Connection refused`, then crashed with `malloc():
  invalid next size` after `xmutil loadapp` of the conv3d app (k26-starter-kits
  loads fine). A stale overlay (`fpga_region region0: Region already has
  overlay applied`) made it worse → reboot to recover. Trimming the xclbin's
  100 KB HLS debug metadata (`scripts/trim_xclbin_metadata.sh`) did **not**
  fix it. Root cause `TODO: verify`. The DPU + kernel still work (the PL is
  loaded before the daemon dies).
- **After a kernel hang, reboot** before the next run (`xclExecBuf` /
  `Bitstream download failed!` in the next process otherwise).
- **Backups inside `/lib/firmware/xilinx/<app>/`** (e.g. `dpu.xclbin.old`)
  can be matched by dfx-mgr's file patterns — keep them elsewhere.
- **Wrong library deployed**: `strings /usr/lib/libvart_op_imp_<op>.so | grep
  VAI_<OP>_FPGA_VERIFY` and `md5sum` against the host build.
- **`make clean` deleted the exported `.xo`** once — keep build artifacts out
  of `clean`.

## 9. Things that cost time but were not bugs in the kernel
- The 3.5 compiler (`xcompiler 3.5.0`) dropped fused ReLUs in eltwise ops →
  compile the model zoo 3.0 model with the 3.0 docker.
- The quantized graph ended with a CPU `transpose` (187 s/frame) — trim it from
  the quantized xmodel (`code/trim_tail.py`) before compiling. Profile first
  (`DEEPHI_PROFILING=1` → `CPU_OP_EXEC : <us>` per op) to know what to accelerate.
