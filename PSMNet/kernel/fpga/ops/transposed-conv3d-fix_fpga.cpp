// xir::Op implementation of `transposed-conv3d-fix` with the conv3d_fix PL
// kernel and CPU fallback. Builds libvart_op_imp_transposed-conv3d-fix.so.
#include "conv3d_fix_op_fpga.hpp"

namespace {
using MyOpImp = vitis::ai::cpu_task::conv3d::Conv3dFixOpImp<true>;
}  // namespace

DEF_XIR_OP_IMP(MyOpImp)
