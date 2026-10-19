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
#pragma once
#include <cstdint>
#include <vector>
#include <vitis/ai/library/tensor.hpp>

// PSMNet cost volume on the CPU (AArch64 NEON), the input of kernel 2:
//
//   out[h][w][m][0:C]  = left (h, w)          m = 0 .. 47 (192 / 4)
//   out[h][w][m][C:2C] = right(h, w - m)      zero where w < m
//
// Both feature maps are requantised once per run from the features' fix point
// to the cost volume's (rounding right shift or saturating left shift, the
// DPU's rounding), so the inner loop is only memcpy / memset.
class CostVolume {
 public:
  CostVolume(const vitis::ai::library::OutputTensor& left,
             const vitis::ai::library::OutputTensor& right,
             const vitis::ai::library::InputTensor& out);
  void run();

 private:
  static constexpr size_t kDisp = 48;
  vitis::ai::library::OutputTensor l_, r_;
  vitis::ai::library::InputTensor out_;
  size_t H_, W_, C_;
  int shift_;                       // left fix point - cost volume fix point
  std::vector<int8_t> req_l_, req_r_;  // requantised maps (shift_ != 0)
};
