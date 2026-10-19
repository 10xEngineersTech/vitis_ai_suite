---
name: kv260-custom-dpu-kernel
description: Use when a model compiled for the DPUCZDX8G on a Kria KV260/K26 (Vitis AI 3.0, PetaLinux 2022.2, XRT 2.14) has a layer the DPU cannot run (it falls back to a CPU subgraph), and a custom HLS kernel must be added to the same xclbin as the DPU, driven from a Vitis-AI cpu_task op library with CPU fallback and bit-exact verification. Distilled from building conv3d_fix for PSMNet.
---

# Custom PL kernel next to the DPU on KV260

Reference implementation: `kernel/fpga/` (conv3d_fix) and `kernel/cpu/` (CPU
golden kernel) in this repo; paths below are relative to the repo root.
`Vitis-AI/src/vai_library/cpu_task/...` paths are upstream Vitis-AI sources
(not in this repo). Details live in `references/`, starting points in
`templates/`, tools in `scripts/`.

| file (real conv3d) | role |
|---|---|
| `kernel/fpga/conv3d_fix_hls.h` | shared kernel/host header: limits, `CfgIdx`, `tab[]` + weight layout, `extern "C"` prototype |
| `kernel/fpga/conv3d_fix_hls.cpp` | HLS kernel (5-stage DATAFLOW) |
| `kernel/cpu/ops/common/conv3d_fix_kernel.hpp` | CPU golden kernel + `Param` |
| `kernel/fpga/conv3d_fix_fpga_host.hpp` | planner (`make_plan`), tables, weight packing, XRT C-API backend |
| `kernel/fpga/conv3d_fix_op_fpga.hpp` | xir op class (`OpImpBase`), FPGA/CPU choice, verify mode |
| `kernel/fpga/ops/conv3d-fix_fpga.cpp` | registration (`DEF_XIR_OP_IMP`) |
| `kernel/fpga/test_conv3d_fix.cpp`, `kernel/fpga/sim_threaded/` | golden test, threaded deadlock sim |
| `kernel/fpga/run_hls.tcl`, `kernel/fpga/system.cfg`, `kernel/fpga/check_util.py` | HLS script, link additions, 40 % budget check |

---

## Phase 0 — Find the unsupported layer

### 1. Identify the CPU-fallback ops
- **Goal**: know exactly which op type(s) run on the CPU and how long they take.
- **Actions**: compile with the docker matching the model's Vitis-AI version;
  `grep "assigned to CPU"` the log; run `scripts/list_cpu_ops.py <xmodel>`;
  profile on the board with `DEEPHI_PROFILING=1` (`CPU_OP_EXEC : <us>` per op).
  See `references/finding-unsupported-layers.md`.
- **Exit**: a table of CPU op types with count and measured time per frame.
- **Mistakes**: accelerating before profiling (PSMNet's biggest CPU cost was a
  187 s `transpose` tail, removable by trimming the graph); using a newer
  compiler than the quantizer (xcompiler 3.5.0 dropped fused ReLUs).

### 2. Record every parameter of every instance
- **Goal**: the full parameter space the kernel must cover.
- **Actions**: per op: input/weight/bias/output shapes, `kernel`, `stride`,
  `pad`, `dilation`, `group`, `pad_mode`, `nonlinear`, fix points of
  input/weight/bias/output, transposed or not. Note the attribute ORDER
  (conv3d: `kernel/stride/dilation` are `{W,H,D}`, `pad` is `{l,r,t,b,near,far}`).
- **Exit**: one row per layer instance (PSMNet kernel 2: 28 layers = 22
  `conv3d-fix` + 6 `transposed-conv3d-fix`, IC ∈ {64,32,26,20,10,6,4},
  OC ∈ {32,26,20,10,6,4,1}, stride 1/2).
- **Mistakes**: designing for "typical" channels (16/32/64) — real pruned
  models have 6, 10, 26.

### 3. Decide whether a kernel pays off
- **Goal**: a number, not a feeling.
- **Actions**: real MACs (skip padding taps/stride holes), a cycle model of
  the planned loop nest at 200 MHz, plus host copy time. Compare with the
  measured CPU time.
- **Exit**: predicted speed-up of the frame, and the PL budget you will spend.
- **Mistakes**: forgetting the host `memcpy` into XRT buffers (~tens of ms
  per 100 MB tensor) and the layers that will fall back.

## Phase 1 — Integrate into the DPU runtime (op class)

### 4. Write the op class on `vart::experimental::OpImpBase`
- **Goal**: the CPU runner calls your code for every op of that type.
- **Actions**:
  - Library name **must** be `libvart_op_imp_<op type>.so` (`:`,`/`,`\` → `_`);
    the runner `dlopen(lib, RTLD_LAZY | RTLD_GLOBAL)` and `dlsym("vart_init_op_imp")`
    (`Vitis-AI/src/vai_library/cpu_task/src/op_imp.cpp: find_dl_lib_for_op`, `get_op_imp`).
  - Class contract (`kernel/cpu/include/vart/op_imp.h`): constructor
    `KLASS(const xir::Op* op, xir::Attrs* attrs)` — read op attributes here;
    public `int calculate(output, inputs...)` with
    `vart::simple_tensor_buffer_t<T>` (optional inputs as
    `std::unique_ptr<...>`, nullptr when absent). Argument count/types must
    match the xir `OpDef` or it is a fatal error.
  - conv3d: `int calculate(simple_tensor_buffer_t<int8_t> output, ... input, ... weight, std::unique_ptr<...> bias)`.
  - Register: `DEF_XIR_OP_IMP(KLASS)` at global scope (it defines
    `extern "C" vart_op_imp_t vart_init_op_imp(const xir_op_t)`).
  - Templates: `templates/op_wrapper.cpp`, `templates/op_fpga.hpp`.
- **Exit**: the op runs through your library on the board (CPU path first).
- **Mistakes**: wrong library name (silently loads the stock CPU lib or
  fails `dlopen`); reading attributes in `calculate` every call.

### 5. Anonymous-namespace rule
- **Goal**: no symbol clashes between op libraries loaded `RTLD_GLOBAL` into
  one process (the runner's own comment: "include the realization of op with
  namespace{}").
- **Actions**:
  - Op library `.cpp`: the class alias (and any helper defined in that `.cpp`)
    inside `namespace { }`; `DEF_XIR_OP_IMP(...)` OUTSIDE every namespace:
    ```cpp
    namespace {
    using MyOpImp = vitis::ai::cpu_task::conv3d::Conv3dFixOpImp<false>;
    }  // namespace
    DEF_XIR_OP_IMP(MyOpImp)
    ```
  - Headers shared by several ops (`conv3d_fix_op_fpga.hpp` is used by
    `conv3d-fix` and `transposed-conv3d-fix`) use a named namespace
    (`vitis::ai::cpu_task::conv3d`) and `inline` free functions — identical
    definitions in every library.
  - HLS kernel: every stage, helper and struct in `namespace { }`; only the
    `extern "C"` kernel entry point outside.
- **Exit**: `aarch64-...-nm -DC --defined-only lib.so | grep -v ' W \| V \| u '`
  shows only `T vart_init_op_imp`; everything else is weak (inline/template)
  code from the shared named namespace (conv3d build: 1 strong + 101 weak).
- **Mistakes**: a non-inline function defined in a shared header (ODR / duplicate symbols).

### 6. Project conventions to follow
- `#pragma once` headers; files: `<op>_hls.h/.cpp`, `<op>_fpga_host.hpp`,
  `<op>_op_fpga.hpp`, `ops/<op type>_fpga.cpp`, `<op>_kernel.hpp` (CPU).
- Build entry in Vitis-AI: `add_op_imp(<type> <src> PRIVATE_REQUIRE ...)`
  (`Vitis-AI/src/vai_library/cpu_task/CMakeLists.txt`) — or cross-build with `templates/build_ops_fpga.sh`.
- Fix points come from tensor attrs: `tensor->get_attr<int>("fix_point")`;
  shifts via `Param::set_fix_point(in, w, out)` / `set_bias_fix_point(in, w, bias)`.
- Env switches per op: `VAI_<OP>_FPGA=0` (force CPU), `DEBUG_<OP>_FPGA=1`
  (log path per layer), `VAI_<OP>_FPGA_VERIFY=1|2`, `VAI_<OP>_FPGA_TIMEOUT_MS`.

## Phase 2 — CPU golden model first

### 7. Implement the layer on the CPU, bit-exact
- **Goal**: the oracle everything else is measured against.
- **Actions**: plain C++ (NEON optional) with the DPU's quantized semantics;
  conv3d: `conv3d::Kernel` in `conv3d_fix_kernel.hpp`, epilogue
  `util::fix(nonlinear_fn(float(acc) + bias·2^b) / 2^n)`.
- **Exit**: step 8 passes. Do not start HLS before.
- **Mistakes**: a "reference" that differs from the runtime's CPU op.

### 8. Define and prove "works"
- **Goal**: an objective pass/fail.
- **Actions**:
  - Match the framework/quantizer output (golden data from the quantizer's
    `dump_xmodel(deploy_check=True)`) — `TODO: verify` per new op.
  - Derive the integer epilogue: `bterm = floor(bias·2^b + 2^(n−1))`,
    `out = clamp((acc + bterm) >> n)`, RELU/RELU6 via `thr6` —
    `references/epilogue-exactness.md`.
  - Golden test cases (conv3d `test_conv3d_fix.cpp`): padding, stride 2,
    dilation, odd IC (7, 20, 26), OC 1/3/4/5/6, `OD` not a multiple of 16
    (D = 60 → 4 tiles), transposed s2/s3, IC 128/OC 128, empty-tap layers,
    every real model shape (small H×W, real C/k/s/pad), and 4 fix-point sets
    including b < 0, n = 0 and heavy saturation.
- **Exit**: 0 mismatching bytes on every case.
- **Mistakes**: one fix-point set only; random inputs so small that nothing saturates.

### 9. Make the CPU kernel the fallback and the oracle
- **Goal**: never produce wrong output, always be able to localize errors.
- **Actions**: in the op class, `FpgaKernel::supported(param, &why)` → FPGA
  else `Kernel`; `VAI_<OP>_FPGA_VERIFY=1` runs both and logs
  `VERIFY <layer> out AxBxCxD: mismatches m/n (p%), off-by-1 k, max |diff| d`
  + the first 6 coordinates; `=2` also outputs the CPU result.
- **Exit**: verify mode exists before the first board run of the kernel.
- **Mistakes**: comparing the final image with an old run instead of a
  `VAI_<OP>_FPGA=0` run of the same build.

## Phase 3 — Design the HLS kernel
Details and numbers: `references/design-patterns.md`.

### 10. Host does the thinking, FPGA does the math
- **Goal**: a kernel with no division, no padding/stride logic, no floats.
- **Actions**: host builds `cfg[32]` (int32, `CFG_STATUS` = word 31),
  `tab[]` (index tables, phases, per-channel `bterm`/`thr6`), packed weights.
  Document the layout ONCE in the shared header (see `conv3d_fix_hls.h`).
- **Exit**: kernel inner loops only add offsets read from tables.
- **Mistakes**: layout described in two places that drift.

### 11. Job decomposition
- **Goal**: a work unit whose working set fits on chip and whose output is contiguous.
- **Actions**: conv3d: one output line `(n, oh, ow)`, `njobs = N·OH·OW`;
  inputs = one staged line per valid `(kh, kw)`; all `NT·NOCT ≤ 16`
  accumulator sets stay on chip.
- **Exit**: on-chip sizes bounded by `MAX_*` constants for every model layer.

### 12. Flattened reduction
- **Goal**: odd channel counts waste no lanes.
- **Actions**: reduce over the contiguous `L·IC` bytes in `RED`-byte chunks:
  IC 26 × 3 taps = 78 B → `NRB = 5` chunks of 16 (97.5 % useful) instead of
  3 × 32 padded lanes.
- **Exit**: weight packing = `m = f / IC`, `c = f % IC`, zero past `L·IC`.

### 13. Transposed conv by phases
- **Goal**: no cycles on stride holes.
- **Actions**: per output, tap pattern of length L → phase id; one weight set
  per phase (holes = 0); tiles group outputs of one phase.
- **Exit**: transposed and normal convs share the datapath; tests pass for s2, s3+dil.

### 14. DSP packing
- **Goal**: 2 int8 MACs per DSP48E2.
- **Actions**: `packed = (b << 18) + a; p = packed * w;` sum ≤ GROUP products
  (`GROUP · 2^14 < 2^17`, GROUP = 6 or 4); `lo = ps[17:0]`,
  `hi = (ps >> 18) + (lo < 0)`; `BIND_OP variable=p op=mul impl=dsp`.
  DSPs ≈ `OC_PAR × RED` (+ few): RED 16 → 320 routed, RED 24 → 448.
- **Exit**: C-sim exact on full-range int8 (−128 · −128 included).
- **Mistakes**: `BIND_OP op=add impl=dsp` on the sums (+320 DSP, no fusion).

### 15. Memory plan
- **Goal**: II = 1 with no port conflicts.
- **Actions**: weights URAM even/odd banks (2 rows/cycle); block reused 8
  cycles while the next is shifted into `wnxt` (shift register, no muxes);
  staged input LUTRAM `ram_1wnr`, 4 banks, ping-pong; accumulators LUTRAM;
  output line BRAM byte banks.
- **Exit**: HLS reports II = 1 for the MAC loop, no port warnings.
- **Mistakes**: 2 reads + 1 write on one BRAM bank (HLS II violation);
  dynamic ping/pong selects that HLS cannot prove exclusive.

### 16. Five-stage DATAFLOW
- **Goal**: load, compute and write-back overlap.
- **Actions**: reader → aligner → mac_engine → epilogue → writer with
  `hls::stream<T, DEPTH>`; depth rule in `references/design-patterns.md` §7.
- **Exit**: threaded simulation passes (step 20).

### 17. Resource budget
- **Goal**: fit next to the DPU with timing margin.
- **Actions**: set a per-kernel cap (ours: 40 % of each K26 resource) and
  compute the TOTAL. conv3d RED 16 routed: 40,999 LUT (37 %), 9,845
  LUT-as-mem, 28,935 FF, 22 BRAM, 4 URAM, 320 DSP; whole design with DPU B512:
  LUT 72 %, BRAM 62.5 %, WNS +0.028 ns. Measure with Vivado synthesis
  (`VIVADO_SYN=1` → `export_syn.rpt`), not the HLS estimate (same RTL:
  HLS 73,191 LUT, Vivado 45,711).
- **Exit**: `scripts/check_util.py <export_syn.rpt> 40` passes; total < ~75 % LUT.
- **Mistakes**: budgeting the kernel alone — crowding caused routing-dominated misses.

### 18. Limits and fallback
- **Goal**: never run an unsupported config on the PL.
- **Actions**: `MAX_*` in the shared header; kernel entry check →
  `cfg[CFG_STATUS] = ST_BAD_CFG; return;`; host `make_plan()` returns false
  with a reason for each limit (`"weights do not fit on chip"`, `"D tiles x OC
  tiles > MAX_TN"`, `"float epilogue not provably exact in integers"`, ...).
  Weight limit: `nphase·KH·KW·NRB·NOCT·16 ≤ W_ROWS`.
- **Exit**: every model layer is either `FPGA` or `CPU (<reason>)` on purpose.

## Phase 4 — Verify before hardware

### 19. C simulation vs the golden kernel
- **Actions**: `make check RED=<n>` — `FpgaKernel` in emulation mode calls the
  kernel C++ directly; 0 mismatches on all cases, for every compile-time
  variant you will build.

### 20. Threaded simulation (deadlocks)
- **Why**: C-sim runs the stages one after another with unbounded FIFOs, so a
  FIFO-depth deadlock can never appear there.
- **Actions**: wrap the dataflow calls in `#ifdef <OP>_THREADED_SIM` (one
  `std::thread` per stage) and build with `-D<OP>_THREADED_SIM -Isim_threaded`
  first in the include path (`templates/sim_threaded/hls_stream.h`: bounded
  blocking FIFO with the real depth; a stall > `CONV3D_SIM_STALL_MS` prints
  `DEADLOCK` and exits 3). `templates/Makefile: check-threaded`.
- **Exit**: all cases pass threaded, at every variant. (conv3d: the old code
  deadlocked on case 1; the fixed code passes.)

### 21. RTL co-simulation
- **Actions**: `cosim_design` on 1–2 tiny cases including a `NOCT = 1` layer
  with > 1 reduction block. `TODO: verify` — not run for conv3d.
- **Catches**: pipeline hazards from wrong `DEPENDENCE` pragmas, real FIFO
  behaviour. **Does not catch**: interconnect/timing issues, XRT/host problems.

## Phase 5 — HLS pitfalls (Rule / Why / Check)
Full stories: `references/lessons-conv3d.md`.

22. **Header ordering deadlock.** Rule: a consumer must never block on data
    its producer only emits after data the consumer has not consumed yet.
    Why: `mac_engine` waited for job j+1's header; the reader could only send
    it after all of job j's lines. Check: threaded sim. Fix used: read the
    next header at the start of the job's last line.
23. **Accumulator RAW hazard.** Rule: no `DEPENDENCE ... inter false` on a
    read-modify-write array unless true for the smallest config; declare
    `type=inter direction=RAW distance=<NOCT_min·8> dependent=true`.
    Why: with NOCT = 1 the same entry recurs every 8 iterations, HLS put
    load at stage ~2 and store at ~15. Symptom: OC ≤ 16 layers wrong, wide
    layers exact, C-sim passes. Check: in
    `<prj>/sol1/.autopilot/db/<..>_Pipeline_<loop>.verbose.sched.rpt` grep the
    accumulate's source line; `ST_` of `store` − `ST_` of `load` < distance.
24. **AXI widening.** Rule: `max_widen_bitwidth = 128` on every `m_axi`
    pragma. Why: IDE flow widened `gmem2` to 512 bit → downsizer to the
    128-bit HP port → 511 failing endpoints. Check: HLS report
    `m_axi_gmemN | 128 -> 128`; `scripts/check_xclbin.sh`.
25. **Clocks.** Rule: kernel on the DPU's 200 MHz clock. On
    `kv260_custom_platform` the ids are 1 = 100, 2 = 200, 3 = 400 MHz →
    `id=2:<cu>.ap_clk`. Why: id 1 put a clock
    converter in the HP interconnect. Check: read
    `unzip -p hw/<platform>.xsa '*.hpfm' | grep 'xd:clock '`.
26. **Timing closure.** The `lbuf` write (rotate + valid compare + 16:1 mux →
    BRAM DIN) missed by 23 ps (76 % routing). Order: (1) `[vivado]
    prop=run.impl_1.STEPS.POST_ROUTE_PHYS_OPT_DESIGN.*` (+ route
    `AggressiveExplore`); (2) `prop=run.impl_1.STRATEGY=Performance_ExplorePostRoutePhysOpt`;
    (3) register the rotated bytes/enables and write a cycle later
    (`TODO: verify`, not needed so far); (4) move memories BRAM↔LUTRAM↔URAM.
    Spot a failing build: `[VPL 101-2] design did not meet timing`, WNS < 0 in
    `impl_1/system_wrapper_timing_summary_routed.rpt` / `dr_timing_summary.rpt`.
27. **Bit-exactness.** Rule: the host proves the integer epilogue equals the
    CPU float maths (`epilogue_plan`: powers of two, `lim1/lim2 ≥ 256`) or the
    layer runs on the CPU. Check: multi-fix-point golden test.
28. **Also cost time**: host libs built with a different `RED` than the
    kernel; IDE header edited separately from the source tree; `make clean`
    deleting the `.xo`; HLS LUT estimates 2–3× too high; xclbin and
    `.bit.bin` from different links.

## Phase 6 — XRT host driver
Template: `templates/xrt_c_api_backend.hpp` (from `conv3d_fix_fpga_host.hpp`).

29. **Driver flow (XRT C API — the one that works next to the DPU runtime).**
    `xclOpen(0)` → read the xclbin path from `XLNX_VART_FIRMWARE` or
    `/etc/vart.conf`, its uuid from the `axlf` header → `xclIPName2Index(h,
    "<kernel>:<cu>")` → `xclOpenContext(h, uuid, cu_index, true)` →
    `xclAllocBO` (cmd: `XCL_BO_FLAGS_EXECBUF`; data: `XCL_BO_FLAGS_CACHEABLE`)
    → `xclMapBO` / `xclGetBOProperties().paddr` → memcpy + `xclSyncBO(TO_DEVICE)`
    → ERT `ert_start_kernel_cmd{state=NEW, opcode=ERT_START_CU, count=1+REG_END/4,
    cu_mask=1<<cu_index}` with 64-bit addresses at the register offsets →
    `xclExecBuf` → poll `xclExecWait` until `state >= COMPLETED` →
    `xclSyncBO(FROM_DEVICE)` → check `cfg[CFG_STATUS] == ST_OK`.
    **Do not** use `xrt::kernel`/`xrt::bo` (`No xclbin with uuid` in a vart
    process), `xir::XrtCu`/`xir::BufferObject` (vart only registers the DPU's
    CUs, by name), or `load_xclbin` of any other file (reprograms
    the PL under the DPU).
30. **Buffer lifetime**: one process-wide context + cmd BO; input/output BOs
    shared by all layers and only grown (PSMNet tensors ≈ 106 MB — one pair per
    layer would exhaust CMA); weights/tab/cfg BOs per layer, uploaded only when
    the weight/bias pointer changes (`packed_for_`, `bias_for_`).
31. **Alignment/sizes**: 16-byte words (`w128`); `words(bytes) = (bytes+15)/16 + 4`
    (slack for the reader's last word and the writer's partial word);
    weight rows stored in 32 B slots (2 × w128).
32. **Errors**: `ST_BAD_CFG` → CHECK with message; timeout
    (`VAI_<OP>_FPGA_TIMEOUT_MS`, default 30000) → dump `ap_ctrl` via
    `xclRegRead` (`0x1` = started, never done = kernel hang); any exception at
    construction → CPU fallback with the reason logged.
33. **Coexistence with the DPU runner**: same device index 0, own handle and
    shared CU context; one `std::mutex` per CU around run(); never load an
    xclbin; the kernel must be in the xclbin vart loads.

## Phase 7 — Build, link, deploy
34. **Commands**: kernel `make hls` (HLS + `.xo` + Vivado synth) or IDE
    `dpu_kernels` (put `max_widen_bitwidth` in the source); link with
    `v++ --link --config dpu-link.cfg --config system.cfg ... <op>.xo`
    (`templates/system.cfg`: `nk`, `sp=` for every master, `id=2`,
    `[vivado]` props); `bootgen` the link's `system.bit` → `.bit.bin`.
    Host lib: `templates/build_ops_fpga.sh` with the kernel's compile-time
    constants (`EXTRA_DEFS=-DCONV3D_RED=16`). See `references/board-bringup.md`.
35. **Post-build checks**: `scripts/check_xclbin.sh <xclbin> <bit.bin> <link dir>`
    — WNS ≥ 0, 128-bit masters, `id=2`, `sp=` for every master, register
    offsets = host `enum Reg`, bitstream matches.
36. **Deploy**: copy `dpu.xclbin` + `.bit.bin` into the app folder that
    `/etc/vart.conf` points to (backups outside `/lib/firmware`), reload with
    `xmutil` or reboot, confirm the uuid with `xbutil examine` and
    `xclbinutil --info`; install `libvart_op_imp_<type>.so` in `/usr/lib`.

## Phase 8 — Board validation
37. **Run with verify**: `DEBUG_<OP>_FPGA=1 VAI_<OP>_FPGA_VERIFY=1 ./app`.
    Every layer: `FPGA` and `mismatches 0`.
38. **Debug a mismatch**: take the FIRST failing layer; classify by
    `(oh, ow, od, oc, tile, NOCT, transposed)` with the table in
    `references/board-bringup.md` §6 (NOCT = 1 only → RAW hazard; last tile
    only → partial tile; edges only → padding/guards; scattered/random →
    timing or cache sync).
39. **End-to-end**: compare the model output with the same build at
    `VAI_<OP>_FPGA=0`. "Small, localized noise, structure intact" = a few
    narrow layers wrong while the wide ones are exact (conv3d: the NOCT = 1
    hazard).

---

## Master checklist
- [ ] CPU ops profiled; target op's instances tabulated (shapes, attrs, fix points)
- [ ] Op library name = `libvart_op_imp_<type>.so`; `OpImpBase` class; `DEF_XIR_OP_IMP` at global scope; alias in `namespace {}`
- [ ] CPU golden kernel exact; epilogue derived; golden test incl. 4 fix-point sets and all edge cases
- [ ] Shared header holds limits, `cfg`/`tab`/weight layout, prototype
- [ ] C-sim exact for every compile-time variant
- [ ] Threaded sim passes (no `DEADLOCK`) for every variant
- [ ] No `DEPENDENCE inter false` on read-modify-write arrays; schedule span < real distance
- [ ] `max_widen_bitwidth = 128` on every `m_axi`; HLS report shows 128 → 128
- [ ] Vivado-synth resources within budget; total with DPU checked
- [ ] Link: `id=2` clock, `sp=` for every master, `[vivado]` post-route props; WNS ≥ 0
- [ ] `.bit.bin` bootgen'd from THIS link (`check_xclbin.sh` says OK)
- [ ] Host lib built with the kernel's constants; XRT C-API backend; mutex; timeout + ap_ctrl dump
- [ ] Board: uuid matches `/etc/vart.conf`; every layer `FPGA`; verify `mismatches 0`; final output = CPU-path output
- [ ] Timing measured (`DEEPHI_PROFILING=1`) and compared with the step-3 estimate

## What went wrong with conv3d
| symptom | root cause | fix | should have been caught by |
|---|---|---|---|
| CU started (`ap_ctrl 0x1`), never finishes | next-job header read before the producer could send it | read it at the last line | threaded sim |
| OC ≤ 16 layers 12–93 % wrong, wide layers exact | `DEPENDENCE accA inter false`, true distance 8 < load→store span | `distance=8 dependent=true` | schedule report / co-sim of a NOCT=1 case |
| link timing fail, 511 endpoints in HP2 interconnect | `gmem2` widened to 512 bit + kernels on 100 MHz | `max_widen_bitwidth=128`, `id=2` | port-width + clock check after kernel build |
| 23 ps miss on `lbuf` write | rotate/compare/mux → BRAM DIN in a crowded device | post-route phys-opt | budget the total utilization; register the write |
| `No xclbin with uuid` / `0 CUs` | native XRT C++ API and vart's CU-name filter | XRT C API with own context | test the host path on the board early with a trivial kernel |
| risk of wrong weights | host built with RED 24, kernel RED 16 | `-DCONV3D_RED=16` | ABI check of shared constants |
| `dfx-mgrd` crash after `loadapp` | unknown (`TODO: verify`); stale overlay made it worse | reboot; avoid restarting the daemon on a loaded overlay | — |

## Template layout for a new kernel
```
<op>/                         (like kernel/fpga/ in this repo)
├── <op>_hls.h                shared header (limits, CfgIdx, tab/weight layout, prototype)
├── <op>_hls.cpp              HLS kernel            ← templates/kernel_hls_skeleton.cpp
├── <op>_fpga_host.hpp        planner + tables + XRT C-API backend ← templates/xrt_c_api_backend.hpp
├── <op>_op_fpga.hpp          op class (OpImpBase), fallback, verify ← templates/op_fpga.hpp
├── ops/<type>_fpga.cpp       DEF_XIR_OP_IMP wrapper ← templates/op_wrapper.cpp
├── ops/build_ops_fpga.sh     aarch64 build of libvart_op_imp_<type>.so
├── test_<op>.cpp             golden test (CPU kernel vs FpgaKernel emulation)
├── sim_threaded/hls_stream.h bounded-FIFO test stream
├── run_hls.tcl, Makefile     HLS / check / check-threaded / hls targets
├── system.cfg                link additions (nk, sp, clock id, vivado props)
└── check_util.py             40 % budget check
CPU golden kernel: kernel/cpu/ops/common/<op>_kernel.hpp (like conv3d_fix_kernel.hpp)
```

## Quick reference
| quantity | formula (conv3d) |
|---|---|
| jobs | `njobs = N · OH · OW` |
| OC groups | `NOCT = ceil(OC / 16)` (OC_PAR = 16) |
| D window per output | `L` = span of input positions it reads (conv: `(KD−1)·dil+1`; transposed: from the phase patterns) |
| reduction blocks | `NRB = ceil(L · IC / RED)` |
| D tiles | `NT = Σ_phases ceil(outputs_in_phase / 16)`, need `NT·NOCT ≤ MAX_TN = 16` |
| staged line words | `LW = ceil((GF + D + GB)·IC / 16) + 6 ≤ LINE_W = 256` |
| weight rows | `nphase·KH·KW·NRB·NOCT·16 ≤ W_ROWS = 8192` (blocks ≤ W_ROWS/16 = 512) |
| MAC cycles per staged line | `NT · NRB · NOCT · 8`; per job ≈ `pairs · max(that, LW) + drain` |
| DSPs | `≈ OC_PAR · RED` (+ ~4–64 epilogue/address) |
| MAC/cycle | `2 · OC_PAR · RED` |
| packing bound | `GROUP · 2^14 < 2^17` → GROUP ≤ 7 |
| accumulator reuse distance | `NOCT · 8` iterations, minimum **8** |
| epilogue | `bterm = floor(bias·2^b + 2^(n−1))`, `out = clamp((acc + bterm) >> n)` |
| platform clocks | id 1 = 100, **2 = 200**, 3 = 400 MHz |
