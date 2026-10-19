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
#include "./resize.hpp"

#include <arm_neon.h>
#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vitis/ai/env_config.hpp>

DEF_ENV_PARAM(DEBUG_RESIZE, "0");

Resize::Resize(vitis::ai::library::OutputTensor& input,
               vitis::ai::library::InputTensor& output)
    : B_(input.batch),
      IH_(input.height),
      IW_(input.width),
      OH_(output.height),
      OW_(output.width),
      C_(input.channel) {
  CHECK_EQ(input.channel, output.channel) << "Resize keeps the channel count";
  CHECK_EQ(input.batch, output.batch);
  fix_scale_ = vitis::ai::library::tensor_scale(input) *
               vitis::ai::library::tensor_scale(output);
  LOG_IF(INFO, ENV_PARAM(DEBUG_RESIZE))
      << input.name << " " << IH_ << "x" << IW_ << " -> " << output.name << " "
      << OH_ << "x" << OW_ << ", C " << C_ << ", scale " << fix_scale_;

  // Source position of each output row / column (half-pixel centres).
  auto taps = [](size_t out_size, size_t in_size, int stride) {
    const float scale = in_size / static_cast<float>(out_size);
    std::vector<Tap> t(out_size);
    for (size_t i = 0; i < out_size; ++i) {
      const float in = (static_cast<float>(i) + 0.5f) * scale - 0.5f;
      const int lower = std::max(static_cast<int>(std::floor(in)), 0);
      const int upper = std::min(static_cast<int>(std::ceil(in)),
                                 static_cast<int>(in_size) - 1);
      t[i] = {lower * stride, upper * stride, in - lower};
    }
    return t;
  };
  ys_ = taps(OH_, IH_, static_cast<int>(IW_ * C_));
  xs_ = taps(OW_, IW_, static_cast<int>(C_));

  for (size_t b = 0; b < B_; ++b) {
    in_.push_back(static_cast<const int8_t*>(input.get_data(b)));
    out_.push_back(static_cast<int8_t*>(output.get_data(b)));
  }
}

// 4 int8x16 -> float32x4 quarters.
static inline void widen(int8x16_t v, float32x4_t f[4]) {
  const int16x8_t lo = vmovl_s8(vget_low_s8(v)), hi = vmovl_s8(vget_high_s8(v));
  f[0] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo)));
  f[1] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo)));
  f[2] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi)));
  f[3] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi)));
}

// 16 channels of one output position:
//   top = tl + (tr - tl) * xl,  bottom = bl + (br - bl) * xl,
//   v = top + (bottom - top) * yl,  out = round_half_up(v * scale)
// The narrowing keeps the low 8 bits, like the float -> int8_t conversion of
// the reference.
static inline int8x16_t lerp16(const int8_t* tl, const int8_t* tr,
                               const int8_t* bl, const int8_t* br,
                               float32x4_t xl, float32x4_t yl,
                               float32x4_t scale) {
  float32x4_t a[4], b[4], c[4], d[4];
  widen(vld1q_s8(tl), a);
  widen(vld1q_s8(tr), b);
  widen(vld1q_s8(bl), c);
  widen(vld1q_s8(br), d);
  const float32x4_t half = vdupq_n_f32(0.5f), one = vdupq_n_f32(1.0f);
  int32x4_t r[4];
  for (int q = 0; q < 4; ++q) {
    const float32x4_t top = vaddq_f32(a[q], vmulq_f32(vsubq_f32(b[q], a[q]), xl));
    const float32x4_t bot = vaddq_f32(c[q], vmulq_f32(vsubq_f32(d[q], c[q]), xl));
    const float32x4_t v =
        vmulq_f32(vaddq_f32(top, vmulq_f32(vsubq_f32(bot, top), yl)), scale);
    const float32x4_t fl = vrndmq_f32(v);  // floor
    const uint32x4_t up = vcgeq_f32(vsubq_f32(v, fl), half);
    r[q] = vcvtq_s32_f32(vbslq_f32(up, vaddq_f32(fl, one), fl));
  }
  const int16x8_t lo = vcombine_s16(vmovn_s32(r[0]), vmovn_s32(r[1]));
  const int16x8_t hi = vcombine_s16(vmovn_s32(r[2]), vmovn_s32(r[3]));
  return vcombine_s8(vmovn_s16(lo), vmovn_s16(hi));
}

void Resize::run() {
  const float32x4_t scale = vdupq_n_f32(fix_scale_);
  const size_t full = C_ / 16 * 16, rest = C_ - full;
  for (size_t b = 0; b < B_; ++b) {
    const int8_t* in = in_[b];
    int8_t* o = out_[b];
    for (size_t y = 0; y < OH_; ++y) {
      const int8_t* row_lo = in + ys_[y].lower;
      const int8_t* row_hi = in + ys_[y].upper;
      const float32x4_t yl = vdupq_n_f32(ys_[y].lerp);
      for (size_t x = 0; x < OW_; ++x, o += C_) {
        const Tap& t = xs_[x];
        const float32x4_t xl = vdupq_n_f32(t.lerp);
        const int8_t *tl = row_lo + t.lower, *tr = row_lo + t.upper;
        const int8_t *bl = row_hi + t.lower, *br = row_hi + t.upper;
        for (size_t c = 0; c < full; c += 16)
          vst1q_s8(o + c, lerp16(tl + c, tr + c, bl + c, br + c, xl, yl, scale));
        if (rest) {  // C not a multiple of 16: same kernel on padded copies
          int8_t p[4][16] = {}, r[16];
          std::memcpy(p[0], tl + full, rest);
          std::memcpy(p[1], tr + full, rest);
          std::memcpy(p[2], bl + full, rest);
          std::memcpy(p[3], br + full, rest);
          vst1q_s8(r, lerp16(p[0], p[1], p[2], p[3], xl, yl, scale));
          std::memcpy(o + full, r, rest);
        }
      }
    }
  }
}
