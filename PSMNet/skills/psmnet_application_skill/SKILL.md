---
name: psmnet-application
description: Use when writing or changing a Vitis AI 3.0 C++ application for the Kria KV260 (DPUCZDX8G, PetaLinux 2022.2) that runs a network split over several xmodels and CPU ops, in the style of the PSMNet demo — one class owns the models, tensors, CPU glue and post-processing, and thin demos call create/run/get_result. Covers the recipe for a new network, the cross build, running on the board, env vars and the mistakes made on PSMNet.
---

# Vitis AI application in the PsmNet style (KV260)

Real code: `application/` in this repo:

- `psmnet.hpp` / `psmnet.cpp`: the class.
- `demo_psmnet.cpp`, `demo_psmnet_live.cpp`: the demos.
- `resize.*`, `cost_volume.*`, `lut_sfm.*`: CPU stages, one class per file
  (constructor binds the tensors, `run()` does the work).

Templates that cross-compile as they are: `references/` (`mynet.hpp`,
`mynet.cpp`, `demo_mynet.cpp`, `Makefile`). Custom PL kernels for CPU ops are
a separate skill: `../conv3d_kernel_skill/SKILL.md`.

## 1. Shape: the class owns everything, the demos stay thin

```cpp
class PsmNet {                          // psmnet.hpp: the only header a demo includes
 public:
  static std::unique_ptr<PsmNet> create();
  virtual void run(const std::vector<std::pair<cv::Mat, cv::Mat>>& imgs) = 0;
  virtual std::vector<cv::Mat> get_result() = 0;
  virtual size_t get_input_batch() = 0;
  virtual int get_input_width() const = 0;
  virtual int get_input_height() const = 0;
};
```

- **`PsmNetImp` constructor** does all setup, once:
  - creates every `DpuTask` / `GraphRunner`;
  - wraps every tensor (`to_tensor`, `sort_tensors`);
  - builds the CPU stages (`Resize`, `CostVolume`, `LutSfm`);
  - reads the fix points.
- **`run()`** only moves data: resize → DPU → CPU glue → DPU → post-processing.
  It allocates nothing per frame.
- **`get_result()`** turns the output buffer into `cv::Mat`.
- **Demos** contain only I/O, threads and display:
  - `demo_psmnet.cpp`: about 30 lines;
  - `demo_psmnet_live.cpp`: camera, inference and display threads around one
    `PsmNet` instance.

## 2. Recipe for a new network

1. **List the subgraphs and CPU ops of each xmodel.** Run
   `python ../conv3d_kernel_skill/scripts/list_cpu_ops.py <model>.xmodel`, or
   `xir dump_txt` in the docker. Profile on the board with
   `DEEPHI_PROFILING=1` before optimising anything.
2. **Pick the runner per model:**
   - **DPU subgraphs with your own CPU code in between** → `vitis::ai::DpuTask`.
     It does mean/scale and gives `getInputTensor(i)` / `getOutputTensor(i)`
     per subgraph:

     ```cpp
     tasks_[0]->setImageRGB(left);
     tasks_[0]->run(0u);
     for (auto i : {0, 1, 2, 3}) cpu_res_[i]->run();     // Resize on the CPU
     copy_into_tensor(outputs_l_[0], inputs_l_[0]);      // fix-point requant
     tasks_[0]->run(1u);
     ```

   - **Graph with CPU ops that have a `libvart_op_imp_<type>.so`** →
     `vitis::ai::GraphRunner`, which runs the DPU subgraphs and the cpu_task
     subgraphs. `CHECK` that the format is NHWC, and call `sync_for_write`
     before and `sync_for_read` after `execute_async` / `wait`
     (`psmnet.cpp`, kernel 2).
3. **Wrap the GraphRunner buffers** as `vitis::ai::library::InputTensor` /
   `OutputTensor` with `to_tensor<T>()`. It records the fix point (or float
   dtype) and the per-batch pointers.
4. **Match tensors by layer-name substring** (`sort_tensors`). Never use the
   index: the order changes between compiles.
5. **Handle fix points between models.** Requantise with the DPU's rounding
   (round half up on a right shift). Examples:
   - `cost_volume.cpp`: `vrshlq_s8` (rounding shift right) / `vqshlq_s8`
     (saturating shift left);
   - `copy_into_tensor` in `psmnet.cpp`: `vrshlq_s8` / `vshlq_s8`. Its left
     shift wraps instead of saturating, kept bit-identical to the original
     scalar code.
6. **Read post-processing parameters from the graph, not from constants.**
   `logits_fix_point(graph2_)` reads the `fix_point` of the tail
   `fix2float` input, and `LutSfm` uses it for an exact 256-entry exp table.
   An env var (`PSMNET_SFM_FIX`) overrides it.
7. **Handle both layouts if the model may be recompiled.** `PsmNetImp` checks
   for `[H][W][D]` vs `[D][H][W]` (`disp_first`) and adapts.
8. **CPU stages in NEON, one class per file**, bit-identical to a scalar
   reference. `Resize` and `CostVolume` replace scalar code:
   - keep the float steps and their order;
   - use separate `vmulq_f32` / `vaddq_f32`, never a fused multiply-add;
   - for a tail shorter than 16 lanes, run the same kernel on a zero-padded
     copy instead of keeping a scalar loop;
   - prove it by building the old scalar code next to the NEON code with the
     same compiler and flags, then comparing the outputs under
     `qemu-aarch64 -L <sysroot>`.
9. **Model paths** use `DEF_ENV_PARAM_2(<NAME>_MODEL_<i>, "dir/dir.xmodel", std::string)`.
   The paths are relative to the working directory and can be overridden per run.

## 3. Build, run, env

- **Build** (host, cross): `make` in `application/` (or `references/Makefile`).
  Default SDK `/mnt/ssd2/abdul_raouf_khalid/petalinux_sdk_2022.2`; override
  with `make SDK=...`.
- **Libraries**: `glog vitis_ai_library-{model_config,graph_runner,dpu_task}
  protobuf vart-runner xir opencv_*`.
  - Add `-lvart-dpu-controller` or XRT only if the application itself drives a PL core.
- **Board**: copy the binaries and the model folders into one directory and
  run from there:
  - `./demo_psmnet demo_psmnet_left.png demo_psmnet_right.png` → `result_psmnet_0.jpg`;
  - `./demo_psmnet_live` (cameras) or `DEBUG=1 DEBUG_DIR=data ./demo_psmnet_live`.
- **Env**: every variable, with its default, is listed in
  `application/README.md`. Add new ones with `DEF_ENV_PARAM` next to the code
  that reads them, and list them there.

## 4. Checklist

- [ ] One `create()`; all allocation in the constructor; `run()` allocates nothing.
- [ ] Demo includes only the class header; no tensor or fix-point code in it.
- [ ] Tensors matched by name; `CHECK_EQ` on the count.
- [ ] Every new NEON stage compared bit-for-bit with its scalar reference (qemu or board).
- [ ] GraphRunner: NHWC checked, `sync_for_write` / `sync_for_read` around the job.
- [ ] Fix points read from the model; right-shift requantisation rounds half up; decide (and document) saturate vs wrap on left shifts.
- [ ] Every CPU op type of the GraphRunner model has its `libvart_op_imp_<type>.so`
      in `/usr/lib` on the board, built against the same Vitis AI version (`.so.3` = 3.0).
- [ ] `readelf -d <binary> | grep NEEDED` lists only libraries the board image has.
- [ ] Models compiled for the DPU in the loaded xclbin (B512 here: `DPUCZDX8G_ISA1_B512`).
- [ ] Output compared against a known-good run (CPU-only op libraries) before timing.

## 5. Mistakes made on PSMNet

| mistake | effect | fix |
|---|---|---|
| Keeping the VCK190 PL cost-volume header (`CostVolumeAccel`, needs XRT) in the KV260 build | extra dependency, XRT headers needed for nothing | dropped (the new `cost_volume.*` is the CPU NEON `CostVolume`); link only what is used, check `NEEDED` |
| Graph ending in a CPU `transpose` | 187 s per frame in one op | trim the tail from the quantized model before compiling (`model/trim_tail.py`, see `model/README.md`) |
| Compiling the 3.0 model with a newer compiler (xcompiler 3.5) | fused ReLUs dropped | compile with the matching 3.0 docker |
| Models compiled for another DPU size (the old app folder had B4096 models; the board runs B512) | `TODO: verify` exact error | compile for B512 (`model/README.md`); match the arch fingerprint of `/etc/vart.conf`'s xclbin |
| Running an old binary after a rebuild | old code path ran | copy the new build to the board, check with `strings` / `md5sum` |
| Root runs leave `/tmp/DPU_*` locks | next non-root run fails | `sudo rm -f /tmp/DPU_*` |
| Timing the first run after `xmutil loadapp` | 14 s instead of 12 s | warm-up run before measuring |
