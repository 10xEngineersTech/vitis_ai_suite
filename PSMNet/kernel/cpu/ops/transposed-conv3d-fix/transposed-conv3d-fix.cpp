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
 * xir::Op implementation of `transposed-conv3d-fix`.
 *
 * Same tensor layouts as conv3d-fix. The operation is evaluated as a
 * stride-1 convolution over the input dilated by `stride`, with the padding
 * rewritten per side:
 *
 *   tconv3d_input = i + (stride - 1) * (i - 1)      per spatial axis
 *   tconv3d_pad   = dilation * (kernel - 1) - pad   per side
 *   tconv3d_stride= 1
 *
 * The general (dilated) form of the padding is required to agree with the
 * output extent below; see conv3d_fix_kernel.hpp.
 *
 * Output extent, matching xir::shape_infer_transposed_conv3d:
 *   oh = (H-1)*stride[1] + dilation[1]*(kernel[1]-1) + 1 - pad[2] - pad[3]
 *   ow = (W-1)*stride[0] + dilation[0]*(kernel[0]-1) + 1 - pad[0] - pad[1]
 *   od = (D-1)*stride[2] + dilation[2]*(kernel[2]-1) + 1 - pad[4] - pad[5]
 */

#include "../common/conv3d_fix_op.hpp"

namespace {
using MyOpImp = vitis::ai::cpu_task::conv3d::Conv3dFixOpImp<true>;
}  // namespace

DEF_XIR_OP_IMP(MyOpImp)
