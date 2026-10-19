// Template: xir::Op glue with FPGA path, CPU fallback and verify mode.
// Real example: kernel/fpga/conv3d_fix_op_fpga.hpp (read it; this is a
// trimmed copy with the conv3d specifics replaced by MYOP placeholders).
#pragma once
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "MYOP_fpga_host.hpp"  // FpgaKernel (XRT driver) + supported()
#include "MYOP_kernel.hpp"     // Param + CPU Kernel (golden / fallback)
#include "vart/op_imp.h"
#include "vart/runner_helper.hpp"

namespace vitis { namespace ai { namespace cpu_task { namespace MYOP {

inline int env_int(const char* n, int dflt) {
  auto e = std::getenv(n);
  return e ? std::atoi(e) : dflt;
}

// Base class contract (kernel/cpu/include/vart/op_imp.h):
//  * constructor KLASS(const xir::Op*, xir::Attrs*) -> read op attributes here
//  * public calculate(output, inputs...) with vart::simple_tensor_buffer_t<T>;
//    optional inputs are std::unique_ptr<...> (nullptr when absent). The
//    number/types must match the xir::OpDef, or it is a fatal error.
struct MyOpFpgaImp : public vart::experimental::OpImpBase {
  MyOpFpgaImp(const xir::Op* op, xir::Attrs* attrs)
      : vart::experimental::OpImpBase{op, attrs}, name_{op->get_name()} {
    // Required attributes: get_attr<T>("..."). Optional ones: has_attr first.
    param_.kernel = op->get_attr<std::vector<int>>("kernel");
    if (op->has_attr("nonlinear"))
      param_.nonlinear = util::get_nonlinear(op->get_attr<std::string>("nonlinear"));
    // TODO: the rest of MYOP's attributes
  }

  int calculate(vart::simple_tensor_buffer_t<int8_t> output,
                vart::simple_tensor_buffer_t<int8_t> input,
                vart::simple_tensor_buffer_t<int8_t> weight,
                std::unique_ptr<vart::simple_tensor_buffer_t<int8_t>> bias) {
    if (!fpga_ && !cpu_) {  // shapes and fix points are fixed per op: plan once
      param_.input_shape = input.tensor->get_shape();
      param_.weight_shape = weight.tensor->get_shape();
      param_.output_shape = output.tensor->get_shape();
      param_.set_fix_point(input.tensor->get_attr<int>("fix_point"),
                           weight.tensor->get_attr<int>("fix_point"),
                           output.tensor->get_attr<int>("fix_point"));
      if (bias)
        param_.set_bias_fix_point(input.tensor->get_attr<int>("fix_point"),
                                  weight.tensor->get_attr<int>("fix_point"),
                                  bias->tensor->get_attr<int>("fix_point"));
      std::string why = "disabled by VAI_MYOP_FPGA=0";
      if (env_int("VAI_MYOP_FPGA", 1) && FpgaKernel::supported(param_, &why)) {
        try { fpga_ = std::make_unique<FpgaKernel>(param_); }
        catch (const std::exception& e) { why = std::string("XRT: ") + e.what(); }
      }
      if (!fpga_) cpu_ = std::make_unique<Kernel>(param_);
      LOG_IF(INFO, env_int("DEBUG_MYOP_FPGA", 0))
          << name_ << ": " << (fpga_ ? "FPGA" : "CPU (" + why + ")");
    }
    int8_t* b = bias ? bias->data : nullptr;
    if (fpga_) {
      fpga_->run(input.data, weight.data, b, output.data);
      if (env_int("VAI_MYOP_FPGA_VERIFY", 0) > 0) verify(input.data, weight.data, b, output.data);
    } else {
      cpu_->run(input.data, weight.data, b, output.data);
    }
    return 0;
  }

 private:
  // Run the CPU golden kernel on the same inputs and report the difference.
  // =2 also replaces the output with the CPU result (known-good final image).
  void verify(const int8_t* in, const int8_t* w, const int8_t* b, int8_t* out) {
    if (!ref_kernel_) ref_kernel_ = std::make_unique<Kernel>(param_);
    const auto& os = param_.output_shape;
    size_t n = 1;
    for (auto d : os) n *= d;
    ref_.resize(n);
    ref_kernel_->run(in, w, b, ref_.data());
    size_t bad = 0, off1 = 0; int maxd = 0; std::vector<size_t> first;
    for (size_t i = 0; i < n; ++i) {
      const int d = std::abs(int(out[i]) - int(ref_[i]));
      if (!d) continue;
      ++bad; off1 += (d == 1); maxd = std::max(maxd, d);
      if (first.size() < 6) first.push_back(i);
    }
    std::ostringstream ss;
    ss << name_ << ": mismatches " << bad << "/" << n << ", off-by-1 " << off1
       << ", max |diff| " << maxd;
    for (auto i : first) ss << "\n    [flat " << i << "] fpga " << int(out[i])
                             << " cpu " << int(ref_[i]);  // TODO: decode coords
    LOG(INFO) << "VERIFY " << ss.str();
    if (env_int("VAI_MYOP_FPGA_VERIFY", 0) >= 2) std::memcpy(out, ref_.data(), n);
  }

  std::string name_;
  Param param_;
  std::unique_ptr<FpgaKernel> fpga_;
  std::unique_ptr<Kernel> cpu_, ref_kernel_;
  std::vector<int8_t> ref_;
};

}}}}  // namespace vitis::ai::cpu_task::MYOP
