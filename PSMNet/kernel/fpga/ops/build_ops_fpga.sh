#!/bin/bash
# Cross-compile libvart_op_imp_conv3d-fix.so and
# libvart_op_imp_transposed-conv3d-fix.so with the conv3d_fix PL kernel path
# (-DUSE_XRT). Same output names as the CPU-only build
# (../cpu: make lib), so they replace those files on
# the board. Layers the kernel cannot run still use the CPU kernel.
#
#   ./build_ops_fpga.sh                                  RED from conv3d_fix_hls.h (24)
#   EXTRA_DEFS=-DCONV3D_RED=16 ./build_ops_fpga.sh       for a RED=16 kernel
#   SYSROOT=... XILINX_HLS=... ./build_ops_fpga.sh
# The host's CONV3D_RED must match the kernel in the loaded xclbin.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONV3D="$(cd "$HERE/.." && pwd)"
CPU_TASK="$(cd "$CONV3D/../cpu" && pwd)"
SYSROOT="${SYSROOT:-/mnt/ssd2/abdul_raouf_khalid/petalinux_sdk_2022.2/sysroots/cortexa72-cortexa53-xilinx-linux}"
XILINX_HLS="${XILINX_HLS:-/tools/X22/Vitis_HLS/2022.2}"   # ap_int.h (header only)
CXX="${CXX:-/tools/X22/Vitis/2022.2/gnu/aarch64/lin/aarch64-linux/bin/aarch64-linux-gnu-g++}"
OUT="${OUT:-$HERE/aarch64}"

[ -x "$CXX" ] || { echo "no cross compiler at $CXX" >&2; exit 1; }
[ -f "$SYSROOT/usr/include/xrt/xrt/xrt_kernel.h" ] || {
  echo "no XRT headers in $SYSROOT/usr/include/xrt" >&2; exit 1; }
[ -f "$XILINX_HLS/include/ap_int.h" ] || { echo "no ap_int.h in $XILINX_HLS" >&2; exit 1; }
mkdir -p "$OUT"

FLAGS=(
  --sysroot="$SYSROOT"
  -O2 -std=c++17 -fPIC -shared -DUSE_XRT ${EXTRA_DEFS:-}
  -Wall -Wno-unused-parameter -Wno-unknown-pragmas -Wno-comment
  -I "$CONV3D" -I "$CPU_TASK/ops/common" -I "$CPU_TASK/include"
  -I "$SYSROOT/usr/include/xrt" -I "$SYSROOT/usr/include"
  -I "$XILINX_HLS/include"
  -Wl,--no-undefined
  -L "$SYSROOT/usr/lib"
  -lvitis_ai_library-runner_helper -lvart-runner -lxir -lglog
  -lxrt_core -lpthread
)

build() {          # build <op-type> <source>
  local so="$OUT/libvart_op_imp_$1.so"
  echo "==> $so"
  "$CXX" "${FLAGS[@]}" -o "$so" "$2"
  "${CXX%g++}readelf" -d "$so" | sed -n 's/.*NEEDED.*\[\(.*\)\]/      needs \1/p'
}

build conv3d-fix            "$HERE/conv3d-fix_fpga.cpp"
build transposed-conv3d-fix "$HERE/transposed-conv3d-fix_fpga.cpp"
