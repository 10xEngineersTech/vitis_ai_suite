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
#include <vector>
#include <vitis/ai/library/tensor.hpp>

// Disparity regression (softmax over D, then sum(d * p_d)) on the CPU with an
// exp lookup table.
//
// Kernel 2's logits are int8 at fix point `fix` (value = q / 2^fix), handed
// over either as int8 or as the float32 output of the model's fix2float op,
// which is exactly q / 2^fix. Softmax depends only on q - max(q), an integer
// in [-255, 0], so every exp comes from a 256-entry table: exact w.r.t. the
// float softmax of those values.
// Rows are split across threads (PSMNET_SFM_THREADS, default: all cores).
//
// fix < 0 (unknown): falls back to expf per element, same maths.
class LutSfm {
 public:
  LutSfm(vitis::ai::library::OutputTensor& input, bool disp_first, int fix);
  void run();
  float* get_output() { return out_.data(); }

 private:
  void run_rows(size_t b, size_t h0, size_t h1);
  size_t B_, H_, W_, D_;
  bool is_float_;
  bool disp_first_;
  int fix_;
  float qscale_;              // 2^fix: float logit -> integer q
  float in_scale_;            // 2^-fix (expf fallback)
  int nthreads_;
  std::vector<const void*> inputs_;
  std::vector<float> lut_;    // lut_[k] = exp(-k / 2^fix), k = 0..255
  std::vector<float> out_;    // disparity, [B][H][W]
};
