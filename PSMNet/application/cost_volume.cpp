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
#include "./cost_volume.hpp"

#include <arm_neon.h>
#include <glog/logging.h>

#include <cstring>

CostVolume::CostVolume(const vitis::ai::library::OutputTensor& left,
                       const vitis::ai::library::OutputTensor& right,
                       const vitis::ai::library::InputTensor& out)
    : l_(left),
      r_(right),
      out_(out),
      H_(left.height),
      W_(left.width),
      C_(left.channel),
      shift_(left.fixpos - out.fixpos) {
  CHECK_EQ(left.size, right.size);
  CHECK_EQ(out.size / out.batch, H_ * W_ * kDisp * 2 * C_);
  if (shift_ != 0) {
    req_l_.resize(H_ * W_ * C_);
    req_r_.resize(H_ * W_ * C_);
  }
}

// 16 lanes of the requantisation:
//   shift > 0: (x >> s) + ((x >> (s - 1)) & 1)   = rounding shift right (SRSHL)
//   shift < 0: saturate(x << -s)                  = saturating shift left (SQSHL)
static inline int8x16_t requant16(int8x16_t v, int8x16_t cnt, bool right) {
  return right ? vrshlq_s8(v, cnt) : vqshlq_s8(v, cnt);
}

static void requant(int8_t* dst, const int8_t* src, size_t n, int shift) {
  const int8x16_t cnt = vdupq_n_s8(static_cast<int8_t>(-shift));
  const bool right = shift > 0;
  size_t i = 0;
  for (; i + 16 <= n; i += 16)
    vst1q_s8(dst + i, requant16(vld1q_s8(src + i), cnt, right));
  if (i < n) {  // n not a multiple of 16: same instruction on a padded copy
    int8_t t[16] = {};
    std::memcpy(t, src + i, n - i);
    vst1q_s8(t, requant16(vld1q_s8(t), cnt, right));
    std::memcpy(dst + i, t, n - i);
  }
}

void CostVolume::run() {
  const size_t pair = 2 * C_, n = H_ * W_ * C_;
  for (size_t b = 0; b < l_.batch; ++b) {
    auto lp = static_cast<const int8_t*>(l_.get_data(b));
    auto rp = static_cast<const int8_t*>(r_.get_data(b));
    auto o = static_cast<int8_t*>(out_.get_data(b));
    if (shift_ != 0) {
      requant(req_l_.data(), lp, n, shift_);
      requant(req_r_.data(), rp, n, shift_);
      lp = req_l_.data();
      rp = req_r_.data();
    }
    for (size_t h = 0; h < H_; ++h)
      for (size_t w = 0; w < W_; ++w)
        for (size_t m = 0; m < kDisp; ++m, o += pair) {
          if (w < m) {
            std::memset(o, 0, pair);
            continue;
          }
          std::memcpy(o, lp + (h * W_ + w) * C_, C_);
          std::memcpy(o + C_, rp + (h * W_ + w - m) * C_, C_);
        }
  }
}
