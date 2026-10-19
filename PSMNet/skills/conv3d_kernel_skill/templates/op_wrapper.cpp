// Template: the shared library the CPU runner dlopen()s for one xir op type.
// Real example: kernel/fpga/ops/conv3d-fix_fpga.cpp
//
// File name / build output MUST be libvart_op_imp_<op type>.so where <op type>
// is xir::Op::get_type() with ':', '/', '\\' replaced by '_'
// (Vitis-AI/src/vai_library/cpu_task/src/op_imp.cpp: find_dl_lib_for_op). The runner dlopen()s it with
// RTLD_LAZY | RTLD_GLOBAL and calls the C symbol `vart_init_op_imp`.
#include "MYOP_op_fpga.hpp"  // defines MyOpFpgaImp (inherits OpImpBase)

// Anonymous namespace: the alias (and anything else defined in this .cpp) gets
// internal linkage. All op libraries are loaded RTLD_GLOBAL into one process,
// so a non-unique external name here could bind to another op's definition.
namespace {
using MyOpImp = vitis::ai::cpu_task::MYOP::MyOpFpgaImp;
}  // namespace

// Must be at global scope (outside every namespace): it defines
// extern "C" vart_op_imp_t vart_init_op_imp(const xir_op_t op).
DEF_XIR_OP_IMP(MyOpImp)
