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
#include "./lut_sfm.hpp"

#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <thread>

LutSfm::LutSfm(vitis::ai::library::OutputTensor& input, bool disp_first,
               int fix)
    : B_(input.batch),
      H_(input.height),
      W_(input.width),
      D_(input.channel),
      is_float_(input.dtype == vitis::ai::library::DT_FLOAT),
      disp_first_(disp_first),
      fix_(is_float_ ? fix : input.fixpos) {
  CHECK(!disp_first_ || is_float_) << "disparity-first int8 not supported";
  qscale_ = std::ldexp(1.0f, fix_ < 0 ? 0 : fix_);
  in_scale_ = is_float_ ? 1.0f : std::ldexp(1.0f, -input.fixpos);
  lut_.resize(256);
  for (int k = 0; k < 256; ++k)
    lut_[k] = std::exp(-static_cast<float>(k) / qscale_);
  const char* t = std::getenv("PSMNET_SFM_THREADS");
  nthreads_ = t ? std::atoi(t) : static_cast<int>(std::thread::hardware_concurrency());
  nthreads_ = std::max(1, nthreads_);
  for (size_t b = 0; b < B_; ++b) inputs_.push_back(input.get_data(b));
  out_.resize(B_ * H_ * W_);
}

void LutSfm::run() {
  for (size_t b = 0; b < B_; ++b) {
    std::vector<std::thread> th;
    const size_t per = (H_ + nthreads_ - 1) / nthreads_;
    for (int i = 0; i < nthreads_; ++i) {
      const size_t h0 = i * per, h1 = std::min(H_, h0 + per);
      if (h0 >= h1) break;
      th.emplace_back([=] { run_rows(b, h0, h1); });
    }
    for (auto& x : th) x.join();
  }
}

void LutSfm::run_rows(size_t b, size_t h0, size_t h1) {
  std::vector<int> q(W_ * D_);    // one row of integer logits, [W][D]
  std::vector<float> xf(W_ * D_); // expf fallback only
  const bool use_lut = fix_ >= 0;
  for (size_t h = h0; h < h1; ++h) {
    // ---- gather row h as [W][D] --------------------------------------------
    if (is_float_) {
      const float* in = static_cast<const float*>(inputs_[b]);
      for (size_t w = 0; w < W_; ++w)
        for (size_t d = 0; d < D_; ++d) {
          const float x = disp_first_ ? in[(d * H_ + h) * W_ + w]
                                      : in[(h * W_ + w) * D_ + d];
          if (use_lut)
            q[w * D_ + d] = static_cast<int>(std::lrint(x * qscale_));
          else
            xf[w * D_ + d] = x;
        }
    } else {
      const int8_t* in = static_cast<const int8_t*>(inputs_[b]) + h * W_ * D_;
      for (size_t i = 0; i < W_ * D_; ++i) q[i] = in[i];
    }

    // ---- softmax over D and sum(d * p_d) per pixel ---------------------------
    float* o = &out_[(b * H_ + h) * W_];
    for (size_t w = 0; w < W_; ++w) {
      float sum = 0.f, dsum = 0.f;
      if (use_lut) {
        const int* r = &q[w * D_];
        const int m = *std::max_element(r, r + D_);
        for (size_t d = 0; d < D_; ++d) {
          const int k = m - r[d];
          const float e = k < 256 ? lut_[k] : 0.f;
          sum += e;
          dsum += e * static_cast<float>(d);
        }
      } else {
        const float* r = &xf[w * D_];
        const float m = *std::max_element(r, r + D_);
        for (size_t d = 0; d < D_; ++d) {
          const float e = std::exp((r[d] - m) * in_scale_);
          sum += e;
          dsum += e * static_cast<float>(d);
        }
      }
      o[w] = dsum / sum;
    }
  }
}
