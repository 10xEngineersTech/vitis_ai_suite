#!/bin/bash
# Cross-compile the op library with the FPGA path (real:
# kernel/fpga/ops/build_ops_fpga.sh). Output name = the CPU-only library's,
# so it replaces it on the board. Compile-time constants shared with the
# kernel (conv3d: CONV3D_RED) MUST equal the ones of the kernel in the xclbin.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYSROOT="${SYSROOT:-/mnt/ssd2/abdul_raouf_khalid/petalinux_sdk_2022.2/sysroots/cortexa72-cortexa53-xilinx-linux}"
XILINX_HLS="${XILINX_HLS:-/tools/X22/Vitis_HLS/2022.2}"   # ap_int.h, header only
CXX="${CXX:-/tools/X22/Vitis/2022.2/gnu/aarch64/lin/aarch64-linux/bin/aarch64-linux-gnu-g++}"
CPU_TASK="${CPU_TASK:?path to Vitis-AI/src/vai_library/cpu_task}"
OP_TYPE="${OP_TYPE:?xir op type, e.g. conv3d-fix}"
mkdir -p "$HERE/aarch64"
"$CXX" --sysroot="$SYSROOT" -O2 -std=c++17 -fPIC -shared -DUSE_XRT ${EXTRA_DEFS:-} \
  -Wall -Wno-unused-parameter -Wno-unknown-pragmas -Wno-comment \
  -I "$HERE/.." -I "$CPU_TASK/ops/common" -I "$CPU_TASK/include" \
  -I "$SYSROOT/usr/include/xrt" -I "$SYSROOT/usr/include" -I "$XILINX_HLS/include" \
  -Wl,--no-undefined -L "$SYSROOT/usr/lib" \
  -o "$HERE/aarch64/libvart_op_imp_${OP_TYPE}.so" "$HERE/${OP_TYPE}_fpga.cpp" \
  -lvitis_ai_library-runner_helper -lvart-runner -lxir -lglog -lxrt_core -lpthread
"${CXX%g++}readelf" -d "$HERE/aarch64/libvart_op_imp_${OP_TYPE}.so" | sed -n 's/.*NEEDED.*\[\(.*\)\]/  needs \1/p'
