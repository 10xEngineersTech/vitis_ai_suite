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

// Bilinear upsampling (half-pixel centres, align_corners = false) of an int8
// NHWC DPU output into an int8 DPU input, requantised from the input's fix
// point to the output's with round-half-up, as the DPU does. Used between the
// two subgraphs of kernels 0/1 (branchN -> upsampleN).
//
// AArch64 NEON, 16 channels per step. Every lane does the float steps of the
// reference formulation in the same order (no fused multiply-add), so the
// result is bit-identical to it.
class Resize {
 public:
  Resize(vitis::ai::library::OutputTensor& input,
         vitis::ai::library::InputTensor& output);
  void run();

 private:
  struct Tap {
    int lower;   // element offset (position * C) of the lower neighbour
    int upper;   // element offset of the upper neighbour
    float lerp;  // weight of the upper neighbour
  };
  size_t B_, IH_, IW_, OH_, OW_, C_;
  float fix_scale_;           // 2^(out_fixpos - in_fixpos)
  std::vector<Tap> xs_, ys_;  // per output column / row, built once
  std::vector<const int8_t*> in_;
  std::vector<int8_t*> out_;
};
