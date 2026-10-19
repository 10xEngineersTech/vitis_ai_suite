/*
 * Copyright 2019 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Filename: conv3d_fix_op.hpp
 *
 * Description: xir::Op glue shared by the conv3d-fix and
 * transposed-conv3d-fix implementations. All arithmetic lives in
 * conv3d_fix_kernel.hpp; this file only reads attributes and tensors.
 */

#pragma once
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "conv3d_fix_kernel.hpp"
#include "vart/op_imp.h"
#include "vart/runner_helper.hpp"

namespace vitis {
namespace ai {
namespace cpu_task {
namespace conv3d {

/// 3D convolution on the APU is expensive enough (PSMNet's 28 conv3d layers are
/// ~268 GMAC per frame counting padding taps, ~164 GMAC without) that running one output row per core is worth the few
/// lines. Workers write disjoint rows, so the result does not depend on the
/// thread count. Override with VAI_CONV3D_FIX_NUM_THREADS, 1 disables it.
inline int get_num_threads() {
  if (auto env = std::getenv("VAI_CONV3D_FIX_NUM_THREADS")) {
    auto n = std::atoi(env);
    if (n > 0) return n;
  }
  auto hc = static_cast<int>(std::thread::hardware_concurrency());
  return hc > 0 ? hc : 1;
}

template <bool TRANSPOSED>
struct Conv3dFixOpImp : public vart::experimental::OpImpBase {
  Conv3dFixOpImp(const xir::Op* op, xir::Attrs* attrs)
      : vart::experimental::OpImpBase{op, attrs} {
    param_.transposed = TRANSPOSED;

    // REQUIRED by FixedConv3dOpDefGenerator, both {W, H, D} ordered.
    param_.kernel = op->get_attr<std::vector<int>>("kernel");
    param_.stride = op->get_attr<std::vector<int>>("stride");
    CHECK_EQ(param_.kernel.size(), 3u);
    CHECK_EQ(param_.stride.size(), 3u);

    // OPTIONAL with defaults.
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
      param_.nonlinear = util::get_nonlinear(op->get_attr<std::string>("nonlinear"));
    }

    // pad_mode and group are NOT declared by FixedConv3dOpDefGenerator, yet
    // DPUCZDX8G-compiled xmodels carry both on these ops. Read defensively so
    // the plugin works either way.
    if (op->has_attr("pad_mode")) {
      param_.pad_mode = op->get_attr<std::string>("pad_mode");
    }
    if (op->has_attr("group")) {
      param_.group = op->get_attr<std::int32_t>("group");
    }

    param_.num_threads = get_num_threads();
  };

  int calculate(vart::simple_tensor_buffer_t<int8_t> output,
                vart::simple_tensor_buffer_t<int8_t> input,
                vart::simple_tensor_buffer_t<int8_t> weight,
                std::unique_ptr<vart::simple_tensor_buffer_t<int8_t>> bias) {
    param_.input_shape = input.tensor->get_shape();
    param_.weight_shape = weight.tensor->get_shape();
    param_.output_shape = output.tensor->get_shape();

    // fix points, turned into scales by Param so that the host-side test
    // drives the very same expressions.
    auto input_fp = input.tensor->get_attr<int>("fix_point");
    auto weight_fp = weight.tensor->get_attr<int>("fix_point");
    auto output_fp = output.tensor->get_attr<int>("fix_point");
    param_.set_fix_point(input_fp, weight_fp, output_fp);

    int8_t* bias_data_ptr = nullptr;
    if (bias != nullptr) {
      auto bias_shape = bias->tensor->get_shape();
      CHECK_EQ(bias_shape[0], param_.weight_shape[0]);
      bias_data_ptr = bias->data;
      auto bias_fp = bias->tensor->get_attr<int>("fix_point");
      param_.set_bias_fix_point(input_fp, weight_fp, bias_fp);
    }

    Kernel(param_).run(input.data, weight.data, bias_data_ptr, output.data);
    return 0;
  }

 private:
  Param param_;
};

}  // namespace conv3d
}  // namespace cpu_task
}  // namespace ai
}  // namespace vitis
 