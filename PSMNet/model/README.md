# PSMNet models for the KV260 (DPUCZDX8G B512)

The application needs three compiled xmodels:

| xmodel | part of PSMNet | runner in `application/psmnet.cpp` |
|---|---|---|
| `PSMNet_pruned_0_pt` | feature extraction, left image | `DpuTask` (2 DPU subgraphs, Resize on the CPU in between) |
| `PSMNet_pruned_1_pt` | feature extraction, right image | `DpuTask` |
| `PSMNet_pruned_2_pt` | 3D hourglass (cost-volume filtering, upsampling) | `GraphRunner` (DPU + `conv3d-fix` / `transposed-conv3d-fix` CPU ops) |

They are not stored in this repo. This folder holds what is needed to make them:

| file | role |
|---|---|
| `trim_tail.py` | cuts the CPU `transpose` + `reshape` tail off kernel 2  |
| `arch_kv260_b512.json` | DPU arch for the compiler: `{"fingerprint": "0x101000056010200"}` = DPUCZDX8G B512, the DPU in the `dpu512_c` xclbin |

## Source model (Vitis AI 3.0 model zoo)

`pt_psmnet_sceneflow_576_960_0.68_696G_3.0`: pruned PSMNet (0.68 pruning,
696 GOPs), trained on Scene Flow, input 576×960, PyTorch.
Model card: `Vitis-AI/model_zoo/model-list/pt_psmnet_sceneflow_576_960_0.68_696G_3.0/model.yaml`.

| package | board | link |
|---|---|---|
| float & quantized (GPU / CPU) | GPU | https://www.xilinx.com/bin/public/openDownload?filename=pt_psmnet_sceneflow_576_960_0.68_696G_3.0.zip |
| compiled `PSMNet_pruned_0_pt` | VCK190 only | https://www.xilinx.com/bin/public/openDownload?filename=PSMNet_pruned_0_pt-vck190-r3.0.0.tar.gz |
| compiled `PSMNet_pruned_1_pt` | VCK190 only | https://www.xilinx.com/bin/public/openDownload?filename=PSMNet_pruned_1_pt-vck190-r3.0.0.tar.gz |
| compiled `PSMNet_pruned_2_pt` | VCK190 only | https://www.xilinx.com/bin/public/openDownload?filename=PSMNet_pruned_2_pt-vck190-r3.0.0.tar.gz |

There is no KV260 build in the model zoo, so the models are compiled here from
the GPU/CPU package. That zip (60,323,238 bytes) contains:

- `float/model_best_single.pth`: pruned float model;
- `float_unpruned/`: unpruned float model;
- `quantized/PSMNet_{0,1,2}_int.xmodel`: the quantized models used below;
- `quantized/PSMNet.py`, `quantized/model_best_deployable.pth`: the quantized model in PyTorch;
- `code/`: training, quantization and test scripts;
- `readme.md`: the model zoo instructions.

## Flow

```
download zip  ->  quantized/PSMNet_{0,1,2}_int.xmodel   (already quantized by AMD/Xilinx)
                       |
                       |  kernel 2 only: trim_tail.py  (drop the CPU transpose + reshape)
                       v
                  vai_c_xir -a arch_kv260_b512.json      (Vitis AI 3.0 docker)
                       |
                       v
                  compiled/PSMNet_pruned_{0,1,2}_pt/PSMNet_pruned_{0,1,2}_pt.xmodel  ->  board
```

Tools: Docker image `xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.0.0.106`
(Vitis AI 3.0).

All commands run from this folder (`PSMNet/model`).

### 1. Download and unpack

```sh
wget "https://www.xilinx.com/bin/public/openDownload?filename=pt_psmnet_sceneflow_576_960_0.68_696G_3.0.zip" -O pt_psmnet_sceneflow_576_960_0.68_696G_3.0.zip
unzip pt_psmnet_sceneflow_576_960_0.68_696G_3.0.zip
```

### 2. Start the Vitis AI 3.0 docker

```sh
docker run -it --rm -u $(id -u):$(id -g) -e HOME=/tmp -v $PWD/..:/workspace -w /workspace/model xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.0.0.106 bash
```

The whole repo is mounted (so `../skills` is visible); the shell starts in `model/`. Inside the container:

```sh
source /opt/vitis_ai/conda/etc/profile.d/conda.sh
conda activate vitis-ai-pytorch
```

### 3. Cut the tail of kernel 2

The compiler leaves the transpose and reshape on the CPU, where they took
187 s per frame (out of 258 s) and about 1 GB of float32 copies. `trim_tail.py`
makes the `fix` op the graph output. Kernel 2 then outputs `[H][W][D]`, which
`psmnet.cpp` reads as-is.

```sh
python trim_tail.py pt_psmnet_sceneflow_576_960_0.68_696G_3.0/quantized/PSMNet_2_int.xmodel PSMNet_2_int_notail.xmodel
```

Expected output:

```
cut after: PSMNet__PSMNet_Upsample_upsample3_2__42353_fix  [1, 576, 960, 192, 1] fix_point 1
  remove reshape    PSMNet__PSMNet_inputs
  remove stack      PSMNet__PSMNet_inputs_shape_i0
  remove const      PSMNet__PSMNet_inputs_shape_attr_3_
  remove const      PSMNet__PSMNet_inputs_shape_attr_2_
  remove const      PSMNet__PSMNet_inputs_shape_attr_1_
  remove const      PSMNet__PSMNet_inputs_shape_attr_0_
  remove transpose  PSMNet__PSMNet_Upsample_upsample3_2__42353_sink_transpose_96
wrote PSMNet_2_int_notail.xmodel: 219 ops (was 226), output [1, 576, 960, 192, 1]
```

`fix_point 1` is the fix point `LutSfm` reads from the compiled model's tail
`fix2float` op. To cut at another op, use `--cut <op name>` (the op must be a
`fix` op).

Kernels 0 and 1 need no trimming.

### 4. Compile for the B512 DPU

```sh
vai_c_xir -x pt_psmnet_sceneflow_576_960_0.68_696G_3.0/quantized/PSMNet_0_int.xmodel -a arch_kv260_b512.json -o compiled/PSMNet_pruned_0_pt -n PSMNet_pruned_0_pt
vai_c_xir -x pt_psmnet_sceneflow_576_960_0.68_696G_3.0/quantized/PSMNet_1_int.xmodel -a arch_kv260_b512.json -o compiled/PSMNet_pruned_1_pt -n PSMNet_pruned_1_pt
vai_c_xir -x PSMNet_2_int_notail.xmodel -a arch_kv260_b512.json -o compiled/PSMNet_pruned_2_pt -n PSMNet_pruned_2_pt
```

Expected (kernels 0/1 take a few minutes each, kernel 2 about 27 min):

| model | compiler summary | size |
|---|---|---|
| `PSMNet_pruned_0_pt` | `Total device subgraph number 8, DPU subgraph number 2` | 3.5 MB |
| `PSMNet_pruned_1_pt` | `Total device subgraph number 8, DPU subgraph number 2` | 3.5 MB |
| `PSMNet_pruned_2_pt` | `Total device subgraph number 24, DPU subgraph number 10` | 7.0 MB |

Kernel 2 also prints 56 warnings `xir::Op{... type = const-fix} has been
assigned to CPU`: the weights and biases of the 3D convolutions that run on
the CPU. They are expected.

Each folder contains `<name>.xmodel`, `meta.json` and `md5sum.txt`. The
`meta.json` must say `"target": "DPUCZDX8G_ISA1_B512"`.

List the CPU ops of kernel 2 (`conv3d-fix`, `transposed-conv3d-fix`, and the
tail `fix2float` with `input[1, 576, 960, 192, 1]@fix1`):

```sh
python ../skills/conv3d_kernel_skill/scripts/list_cpu_ops.py compiled/PSMNet_pruned_2_pt/PSMNet_pruned_2_pt.xmodel
```

Every CPU op type needs its `libvart_op_imp_<type>.so` on the board; for
`conv3d-fix` / `transposed-conv3d-fix` see [../kernel/cpu/README.md](../kernel/cpu/README.md)
and [../kernel/fpga/README.md](../kernel/fpga/README.md).

Leave the container (`exit`).

### 5. Copy to the board

The three `compiled/PSMNet_pruned_{0,1,2}_pt/` folders are copied to the board
together with the application, see
[../application/README.md, Board setup](../application/README.md#board-setup).

## Other DPU sizes

Use another arch file with `-a`, for example
`/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json` in the docker (the
standard KV260 B4096 DPU). The fingerprint must match the DPU in the xclbin
the board loads (`/etc/vart.conf`).

## Re-quantizing (optional)

The zip's quantized models are used as they are. To quantize again
(PTQ/QAT, needs the Scene Flow dataset in `data/`, follow
`pt_psmnet_sceneflow_576_960_0.68_696G_3.0/readme.md` (`code/test_psmnet_ptq.sh`,
`code/test_psmnet_qat.sh`). Then trim and compile the new `PSMNet_{0,1,2}_int.xmodel`
as above. Not done for this project.
