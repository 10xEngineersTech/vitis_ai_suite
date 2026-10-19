/*
 * conv3d_fix_op_fpga.hpp
 *
 * Drop-in replacement for ../cpu/ops/common/conv3d_fix_op.hpp: the same
 * xir::Op glue, but each layer runs on the `conv3d_fix` PL kernel when the
 * kernel supports it and falls back to the CPU (NEON) conv3d::Kernel
 * otherwise. Used by both conv3d-fix and transposed-conv3d-fix.
 *
 *   VAI_CONV3D_FPGA=0            force the CPU path
 *   VAI_CONV3D_FIX_NUM_THREADS   CPU path threads (as before)
 *   DEBUG_CONV3D_FPGA=1          log which path every layer takes
 *   VAI_CONV3D_FPGA_VERIFY=1     also run the CPU kernel on every FPGA layer
 *                                and log mismatches (=2: also output the CPU
 *                                result, so the app's final image is the CPU one)
 */
#pragma once
#include <cstdint>
#include <cstring>
#include <sstream>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "conv3d_fix_fpga_host.hpp"
#include "conv3d_fix_kernel.hpp"
#include "vart/op_imp.h"
#include "vart/runner_helper.hpp"

namespace vitis {
namespace ai {
namespace cpu_task {
namespace conv3d {

inline int get_num_threads() {
  if (auto env = std::getenv("VAI_CONV3D_FIX_NUM_THREADS")) {
    auto n = std::atoi(env);
    if (n > 0) return n;
  }
  auto hc = static_cast<int>(std::thread::hardware_concurrency());
  return hc > 0 ? hc : 1;
}

inline bool fpga_enabled() {
  auto env = std::getenv("VAI_CONV3D_FPGA");
  return !(env && std::atoi(env) == 0);
}

inline bool fpga_debug() {
  auto env = std::getenv("DEBUG_CONV3D_FPGA");
  return env && std::atoi(env) != 0;
}

template <bool TRANSPOSED>
struct Conv3dFixOpImp : public vart::experimental::OpImpBase {
  Conv3dFixOpImp(const xir::Op* op, xir::Attrs* attrs)
      : vart::experimental::OpImpBase{op, attrs}, name_{op->get_name()} {
    param_.transposed = TRANSPOSED;
    param_.kernel = op->get_attr<std::vector<int>>("kernel");
    param_.stride = op->get_attr<std::vector<int>>("stride");
    CHECK_EQ(param_.kernel.size(), 3u);
    CHECK_EQ(param_.stride.size(), 3u);
    param_.dilation = std::vector<int>{1, 1, 1};
    if (op->has_attr("dilation")) {
      param_.dilation = op->get_attr<std::vector<int>>("dilation");
      CHECK_EQ(param_.dilation.size(), 3u);
    }
    param_.pad = std::vector<int>{0, 0, 0, 0, 0, 0};
    if (op->has_attr("pad")) {
      param_.pad = op->get_attr<std::vector<int>>("pad");
      CHECK_EQ(param_.pad.size(), 6u) << "3D pad is {l,r,t,b,near,far}";
    }
    if (op->has_attr("nonlinear")) {
      param_.nonlinear =
          util::get_nonlinear(op->get_attr<std::string>("nonlinear"));
    }
    if (op->has_attr("pad_mode")) {
      param_.pad_mode = op->get_attr<std::string>("pad_mode");
    }
    if (op->has_attr("group")) {
      param_.group = op->get_attr<std::int32_t>("group");
    }
    param_.num_threads = get_num_threads();
  }

  int calculate(vart::simple_tensor_buffer_t<int8_t> output,
                vart::simple_tensor_buffer_t<int8_t> input,
                vart::simple_tensor_buffer_t<int8_t> weight,
                std::unique_ptr<vart::simple_tensor_buffer_t<int8_t>> bias) {
    // Shapes and fix points are fixed for the op, so the backend (and the
    // FPGA tables / packed weights) is built once, on the first call.
    if (!fpga_ && !cpu_) {
      param_.input_shape = input.tensor->get_shape();
      param_.weight_shape = weight.tensor->get_shape();
      param_.output_shape = output.tensor->get_shape();
      auto input_fp = input.tensor->get_attr<int>("fix_point");
      auto weight_fp = weight.tensor->get_attr<int>("fix_point");
      auto output_fp = output.tensor->get_attr<int>("fix_point");
      param_.set_fix_point(input_fp, weight_fp, output_fp);
      if (bias != nullptr) {
        CHECK_EQ(bias->tensor->get_shape()[0], param_.weight_shape[0]);
        param_.set_bias_fix_point(input_fp, weight_fp,
                                  bias->tensor->get_attr<int>("fix_point"));
      }
      std::string why = "disabled by VAI_CONV3D_FPGA=0";
      if (fpga_enabled() && FpgaKernel::supported(param_, &why)) {
        try {
          fpga_ = std::make_unique<FpgaKernel>(param_);
        } catch (const std::exception& e) {
          why = std::string("XRT: ") + e.what();
        }
      }
      if (!fpga_) cpu_ = std::make_unique<Kernel>(param_);
      LOG_IF(INFO, fpga_debug())
          << name_ << ": " << (fpga_ ? "FPGA" : "CPU (" + why + ")");
    }
    int8_t* b = bias ? bias->data : nullptr;
    if (fpga_) {
      fpga_->run(input.data, weight.data, b, output.data);
      if (verify_level() > 0) verify(input.data, weight.data, b, output.data);
    } else {
      cpu_->run(input.data, weight.data, b, output.data);
    }
    return 0;
  }

 private:
  static int verify_level() {
    auto env = std::getenv("VAI_CONV3D_FPGA_VERIFY");
    return env ? std::atoi(env) : 0;
  }

  // Compare the FPGA output against the CPU kernel, byte by byte.
  void verify(const int8_t* in, const int8_t* w, const int8_t* b, int8_t* out) {
    if (!ref_kernel_) ref_kernel_ = std::make_unique<Kernel>(param_);
    const auto& os = param_.output_shape;  // N, OH, OW, OD, OC
    const size_t n = (size_t)os[0] * os[1] * os[2] * os[3] * os[4];
    ref_.resize(n);
    ref_kernel_->run(in, w, b, ref_.data());
    size_t bad = 0, off1 = 0;
    int maxd = 0;
    std::vector<size_t> first;
    for (size_t i = 0; i < n; ++i) {
      const int d = std::abs(int(out[i]) - int(ref_[i]));
      if (!d) continue;
      ++bad;
      off1 += (d == 1);
      maxd = std::max(maxd, d);
      if (first.size() < 6) first.push_back(i);
    }
    std::ostringstream ss;
    ss << name_ << (param_.transposed ? " [T]" : "") << " out " << os[1] << "x"
       << os[2] << "x" << os[3] << "x" << os[4] << ": mismatches " << bad << "/"
       << n << " (" << (100.0 * bad / n) << "%), off-by-1 " << off1
       << ", max |diff| " << maxd;
    for (auto i : first) {
      size_t r = i;
      const int oc = r % os[4]; r /= os[4];
      const int od = r % os[3]; r /= os[3];
      const int ow = r % os[2]; r /= os[2];
      const int oh = r % os[1];
      ss << "\n    [oh " << oh << " ow " << ow << " od " << od << " oc " << oc
         << "] fpga " << int(out[i]) << " cpu " << int(ref_[i]);
    }
    LOG(INFO) << "VERIFY " << ss.str();
    if (verify_level() >= 2) std::memcpy(out, ref_.data(), n);
  }

 public:

 private:
  std::string name_;
  Param param_;
  std::unique_ptr<FpgaKernel> fpga_;
  std::unique_ptr<Kernel> cpu_;
  std::unique_ptr<Kernel> ref_kernel_;  // VAI_CONV3D_FPGA_VERIFY
  std::vector<int8_t> ref_;
};

}  // namespace conv3d
}  // namespace cpu_task
}  // namespace ai
}  // namespace vitis
