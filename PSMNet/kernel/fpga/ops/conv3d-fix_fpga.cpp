// xir::Op implementation of `conv3d-fix` with the conv3d_fix PL kernel and
// CPU fallback. Builds libvart_op_imp_conv3d-fix.so (see build_ops_fpga.sh),
// a drop-in replacement for the CPU-only library of the same name.
#include "conv3d_fix_op_fpga.hpp"

namespace {
using MyOpImp = vitis::ai::cpu_task::conv3d::Conv3dFixOpImp<false>;
}  // namespace

DEF_XIR_OP_IMP(MyOpImp)
