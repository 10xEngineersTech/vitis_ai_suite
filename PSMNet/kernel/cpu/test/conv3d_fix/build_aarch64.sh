#!/bin/bash
# Cross-compile the conv3d-fix / transposed-conv3d-fix cpu_task plugins for
# aarch64 (ZCU102 / ZCU104 / KV260) without configuring the whole Vitis-AI
# CMake tree. Mirrors what add_op_imp() in ../../CMakeLists.txt produces:
# a shared object named libvart_op_imp_<op_type>.so, which cpu_task dlopen()s
# by op type at graph-runner init.
#
#   ./build_aarch64.sh                       # uses the sysroot below
#   SYSROOT=/path/to/3.0/sysroot ./build_aarch64.sh
#
# NOTE on runtime ABI: the DT_NEEDED soname recorded in the output comes from
# whatever sysroot is used, and it must match the release on the board.
#   petalinux_sdk_2022.2  ships Vitis-AI 3.0 -> libvart-runner.so.3, libxir.so.3
#   petalinux_sdk_2022.1  ships Vitis-AI 2.5 -> .so.2 (will not dlopen on a 3.0 board)
# The default below is the 3.0 sysroot, which is what the KV260 PetaLinux
# 2022.2 / Vitis AI 3.0 image (DPUCZDX8G_ISA1_B512) expects. Point SYSROOT elsewhere to target a different release;
# nothing in the source changes, since the in-tree 3.0 vart/op_imp.h is
# byte-identical to the copies in both sysroots.
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPU_TASK="$(cd "$HERE/../.." && pwd)"
__ARM_NEON=1
SYSROOT="${SYSROOT:-/mnt/ssd2/abdul_raouf_khalid/petalinux_sdk_2022.2/sysroots/cortexa72-cortexa53-xilinx-linux}"
echo $SYSROOT
unset CXX
CXX="${CXX:-/tools/X22/Vitis/2022.2/gnu/aarch64/lin/aarch64-linux/bin/aarch64-linux-gnu-g++}"
echo $CXX
OUT="${OUT:-$HERE/aarch64}"

[ -x "$CXX" ] || { echo "no cross compiler at $CXX" >&2; exit 1; }
[ -d "$SYSROOT" ] || { echo "no sysroot at $SYSROOT" >&2; exit 1; }
[ -f "$SYSROOT/usr/include/vart/op_imp.h" ] || {
  echo "sysroot has no vart/op_imp.h -- is it a Vitis-AI sysroot?" >&2; exit 1; }

mkdir -p "$OUT"

# -I the in-tree include first so the 3.0 vart/op_imp.h wins over the sysroot's.
# --no-undefined turns a missing symbol into a link error here rather than a
# dlopen failure on the board.
FLAGS=(
  --sysroot="$SYSROOT"
  -O2 -std=c++17 -fPIC -shared
  -Wall -Wextra -Wno-unused-parameter
  -I "$CPU_TASK/include"
  -I "$SYSROOT/usr/include"
  -Wl,--no-undefined
  -L "$SYSROOT/usr/lib"
  -lvitis_ai_library-runner_helper -lvart-runner -lxir -lglog -lpthread
)

build() {          # build <op-type> <source>
  local op="$1" src="$2" so="$OUT/libvart_op_imp_$1.so"
  echo "==> $so"
  "$CXX" "${FLAGS[@]}" -o "$so" "$src"
  "${CXX%g++}readelf" -d "$so" | sed -n 's/.*NEEDED.*\[\(.*\)\]/      needs \1/p'
}

build conv3d-fix            "$CPU_TASK/ops/conv3d-fix/conv3d-fix.cpp"
build transposed-conv3d-fix "$CPU_TASK/ops/transposed-conv3d-fix/transposed-conv3d-fix.cpp"

echo
echo "built in $OUT ; the entry point cpu_task looks up is vart_init_op_imp:"
for so in "$OUT"/libvart_op_imp_*conv3d-fix.so; do
  printf '  %-40s ' "$(basename "$so")"
  "${CXX%g++}nm" -D --defined-only "$so" | grep -c ' T vart_init_op_imp' \
    | sed 's/^/vart_init_op_imp x/'
done
echo
echo "install on the board (petalinux@10.42.0.27) with:"
echo "  scp $OUT/libvart_op_imp_*conv3d-fix.so petalinux@10.42.0.27:~"
echo "  ssh -t petalinux@10.42.0.27 sudo cp libvart_op_imp_conv3d-fix.so libvart_op_imp_transposed-conv3d-fix.so /usr/lib/"
