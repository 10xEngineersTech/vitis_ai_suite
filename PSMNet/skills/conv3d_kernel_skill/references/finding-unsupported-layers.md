# Finding the layer the DPU cannot run

## Compiler log (vai_c_xir, Vitis AI 3.0 docker)
```sh
vai_c_xir -x quantized/MODEL_int.xmodel -a arch.json -o compiled/X -n X 2>&1 | tee compile.log
grep -E "assigned to CPU|Total device subgraph number" compile.log
#   [UNILOG][WARNING] xir::Op{name = ..., type = const-fix} has been assigned to CPU: [...]
#   [UNILOG][INFO] Total device subgraph number 24, DPU subgraph number 10
```
Use the compiler that matches the quantizer / model zoo version (3.0 model →
3.0 docker); xcompiler 3.5.0 dropped fused ReLUs on PSMNet's eltwise ops.

## Graph dump
```sh
xir dump_txt compiled/X/X.xmodel X.txt          # inside the docker
python scripts/list_cpu_ops.py compiled/X/X.xmodel            # CPU op types + counts
python scripts/list_cpu_ops.py compiled/X/X.xmodel conv3d-fix # every op of one type
```
For PSMNet kernel 2 the CPU op types were `conv3d-fix`,
`transposed-conv3d-fix`, `fix2float`, `const-fix` (+ a `transpose`/`reshape`
tail removed by trimming the quantized graph).

## Record per op (feeds `Param` and the planner)
| item | where |
|---|---|
| op type (→ library name `libvart_op_imp_<type>.so`) | `op.get_type()` |
| input / weight / bias / output shapes | `get_input_ops()`, `get_output_tensor().dims` |
| fix points | tensor attr `fix_point` of input, weight, bias, output |
| kernel / stride / pad / dilation / group / pad_mode | op attrs (conv3d: `kernel`,`stride` `{W,H,D}`; `pad` `{l,r,t,b,near,far}`) |
| activation | op attr `nonlinear` (`NONE/RELU/RELU6/LEAKYRELU/PRELU/...`) |
| transposed? | op type `transposed-*` |

## Is a kernel worth it?
1. Profile on the board first: `DEEPHI_PROFILING=1 ./app ...` prints
   `CPU_OP_EXEC : <us>` per CPU op and `XRT_RUN : <us>` per DPU run.
   PSMNet: the 28 conv3d ops (22 `conv3d-fix` + 6 `transposed-conv3d-fix`) =
   59.6 s of a 258 s frame; a `transpose` tail = 187 s
   (removing it from the graph was worth more than any kernel).
2. Count real MACs (skip padding taps / stride holes): PSMNet conv3d ≈ 164 GMAC
   (≈ 268 GMAC counting every tap).
3. Estimate kernel time = Σ over layers of cycles / f_clk (model the loop
   nest; conv3d RED=24 model: ~2.2 s at 200 MHz) + host copies
   (input/output memcpy into XRT buffers, ~tens of ms per 100 MB).
4. Check DDR traffic per job ≤ what one HP port gives (~16 B/cycle at 200 MHz).
