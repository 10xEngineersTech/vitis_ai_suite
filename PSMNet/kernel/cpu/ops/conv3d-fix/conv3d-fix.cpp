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
 * xir::Op implementation of `conv3d-fix`.
 *
 *   input   [N, H, W, D, IC]
 *   weights [OC, KH, KW, KD, IC]
 *   bias    [OC]  (optional)
 *   output  [N, OH, OW, OD, OC]
 *
 * DPUCZDX8G has no 3D convolution engine, so xcompiler leaves every 3D
 * convolution on a `device == "CPU"` subgraph and the CPU task needs this
 * implementation to execute e.g. the PSMNet hourglass stage.
 */

#include "../common/conv3d_fix_op.hpp"

namespace {
using MyOpImp = vitis::ai::cpu_task::conv3d::Conv3dFixOpImp<false>;
}  // namespace

DEF_XIR_OP_IMP(MyOpImp)
