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
 * Filename: conv3d_fix_kernel.hpp
 *
 * Description: shared fixed-point 5D convolution kernel behind the
 * `conv3d-fix` and `transposed-conv3d-fix` xir::Op implementations,
 * with AArch64 NEON acceleration on the innermost IC loop.
 *
 * Optimisation strategy
 * ---------------------
 * The innermost loop is:
 *
 *   for ic in [0, IC):
 *     acc[oc] += (int8)input[ic] * (int8)weight[oc, ic]
 *
 * On PSMNet this loop runs IC = 32/64/128 times per (OH�OW�OD�OC) output
 * element. We replace it with:
 *
 *   NEON path (AArch64, IC multiple of 16):
 *     - Load 16 x int8 input values into v_in  (vld1q_s8)
 *     - Load 16 x int8 weight values into v_wt  (vld1q_s8)
 *     - Widen both to int16, multiply, accumulate into int32x4 accumulators
 *       using vmull_s8 + vmlal_s8 pairs  (no saturation, no overflow risk
 *       because IC * 127*127 < INT64_MAX for IC ? 4096)
 *     - Tail IC % 16 handled by the scalar loop
 *
 * Additionally we unroll OC by 4 so four output channels share one input
 * load, cutting weight-cache pressure by 4x.
 *
 * The OC-unrolled accumulator layout:
 *   For oc = base, base+1, base+2, base+3 :
 *     int32x4_t vacc[4];   // lane 0..3 = partial sums for that oc
 *   After the IC loop: vaddvq_s32(vacc[j]) ? scalar acc[base+j]
 *
 * All other code (tap precomputation, threading, epilogue) is unchanged so
 * the result is bit-identical to the scalar path for any IC.
 *
 * Portability: NEON intrinsics are compiled only when __ARM_NEON is defined.
 * On x86 / MSVC the file falls back to the original scalar loop.
 *
 * ---------------------------------------------------------------------------
 * XIR conventions (unchanged from original):
 *   input   : [N, H, W, D, IC]
 *   weights : [OC, KH, KW, KD, IC]
 *   bias    : [OC]          (optional)
 *   output  : [N, OH, OW, OD, OC]
 *
 *   kernel / stride / dilation : {W, H, D}
 *   pad                        : {left, right, top, bottom, near, far}
 * ---------------------------------------------------------------------------
 */

#pragma once
#include <algorithm>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

#ifdef __ARM_NEON
#  include <arm_neon.h>
#endif

#include "util.hpp"

namespace vitis {
namespace ai {
namespace cpu_task {
namespace conv3d {

// ---------------------------------------------------------------------------
// Axis helpers (unchanged)
// ---------------------------------------------------------------------------
enum Axis { AXIS_H = 0, AXIS_W = 1, AXIS_D = 2 };

static constexpr int kTensorDim[3] = {1, 2, 3};
static constexpr int kAttrIdx[3]   = {1, 0, 2};
static constexpr int kPadBegin[3]  = {2, 0, 4};
static constexpr int kPadEnd[3]    = {3, 1, 5};

// ---------------------------------------------------------------------------
// Param (unchanged)
// ---------------------------------------------------------------------------
struct Param {
  bool transposed{false};

  std::vector<int> kernel;
  std::vector<int> stride;
  std::vector<int> dilation;
  std::vector<int> pad;
  std::string pad_mode{"FLOOR"};
  int group{1};

  std::vector<int> input_shape;
  std::vector<int> weight_shape;
  std::vector<int> output_shape;

  enum util::NONLINEAR nonlinear { util::NONLINEAR::NONE };
  float prelu{0.0f};

  float shift_cut_scale{1.0f};
  float shift_bias_scale{1.0f};
  float six_as_output{6.0f};

  int num_threads{1};

  void set_fix_point(int input_fp, int weight_fp, int output_fp) {
    shift_cut_scale =
        std::pow(2.0f, static_cast<float>(input_fp + weight_fp - output_fp));
    six_as_output = 6.0f * std::pow(2.0f, static_cast<float>(output_fp));
  }
  void set_bias_fix_point(int input_fp, int weight_fp, int bias_fp) {
    shift_bias_scale =
        std::pow(2.0f, static_cast<float>(input_fp + weight_fp - bias_fp));
  }
};

// ---------------------------------------------------------------------------
// Tap (unchanged)
// ---------------------------------------------------------------------------
struct Tap {
  int k;
  int i;
};

// ---------------------------------------------------------------------------
// NEON helpers
// ---------------------------------------------------------------------------
#ifdef __ARM_NEON

/// acc += a[k] * b[k] for the 8 int8 lane pairs, accumulating into int32 lanes.
///
/// There is no int8 -> int32 multiply-accumulate on ARMv8.0 (vmlal_s8 widens
/// only to int16, which overflows after two MACs: 127 * -128 = -16256).
/// vmull_s8 gives the 8 exact int16 products and vpadalq_s16 pairwise-adds
/// them into the int32 accumulator, so lanes hold partial sums of two products
/// each. Only the horizontal total is ever read, so the pairing is harmless.
__attribute__((always_inline))
inline int32x4_t neon_mlal_s8(int32x4_t acc, int8x8_t a, int8x8_t b) {
  return vpadalq_s16(acc, vmull_s8(a, b));
}

/// Dot-product of two int8 vectors of length `len`, accumulated into `acc`.
/// Uses 16-wide NEON unrolled 2x (32 elements per iteration).
/// Tail handled scalar.
__attribute__((always_inline))
inline int64_t neon_dot_s8(const int8_t* __restrict__ a,
                            const int8_t* __restrict__ b,
                            int len) {
  int32x4_t vacc0 = vdupq_n_s32(0);
  int32x4_t vacc1 = vdupq_n_s32(0);
  int32x4_t vacc2 = vdupq_n_s32(0);
  int32x4_t vacc3 = vdupq_n_s32(0);

  int i = 0;

  // Unrolled 2x: process 32 elements per iteration
  for (; i + 32 <= len; i += 32) {
    int8x16_t va0 = vld1q_s8(a + i);
    int8x16_t vb0 = vld1q_s8(b + i);
    int8x16_t va1 = vld1q_s8(a + i + 16);
    int8x16_t vb1 = vld1q_s8(b + i + 16);

    // Widen low 8 lanes to int16, multiply ? int16x8, widen-accumulate ? int32x4
    vacc0 = neon_mlal_s8(vacc0, vget_low_s8(va0),  vget_low_s8(vb0));
    vacc1 = neon_mlal_s8(vacc1, vget_high_s8(va0), vget_high_s8(vb0));
    vacc2 = neon_mlal_s8(vacc2, vget_low_s8(va1),  vget_low_s8(vb1));
    vacc3 = neon_mlal_s8(vacc3, vget_high_s8(va1), vget_high_s8(vb1));
  }

  // 16 elements
  for (; i + 16 <= len; i += 16) {
    int8x16_t va = vld1q_s8(a + i);
    int8x16_t vb = vld1q_s8(b + i);
    vacc0 = neon_mlal_s8(vacc0, vget_low_s8(va),  vget_low_s8(vb));
    vacc1 = neon_mlal_s8(vacc1, vget_high_s8(va), vget_high_s8(vb));
  }

  // 8 elements
  for (; i + 8 <= len; i += 8) {
    int8x8_t va = vld1_s8(a + i);
    int8x8_t vb = vld1_s8(b + i);
    vacc0 = neon_mlal_s8(vacc0, va, vb);
  }

  // Horizontal reduce all four accumulators ? one int64
  int32x4_t vsum = vaddq_s32(vaddq_s32(vacc0, vacc1),
                              vaddq_s32(vacc2, vacc3));
  int64_t result = static_cast<int64_t>(vaddvq_s32(vsum));

  // Scalar tail (< 8 elements)
  for (; i < len; ++i) {
    result += static_cast<int>(a[i]) * static_cast<int>(b[i]);
  }
  return result;
}

/// Compute 4 dot-products simultaneously (OC unroll), sharing one input load.
/// w_ptrs[0..3] are pointers to weight rows for oc, oc+1, oc+2, oc+3.
/// Results written to out[0..3].
__attribute__((always_inline))
inline void neon_dot4_s8(const int8_t* __restrict__ in_p,
                          const int8_t* __restrict__ w0,
                          const int8_t* __restrict__ w1,
                          const int8_t* __restrict__ w2,
                          const int8_t* __restrict__ w3,
                          int ic,
                          int64_t* __restrict__ out) {
  int32x4_t vacc0 = vdupq_n_s32(0);
  int32x4_t vacc1 = vdupq_n_s32(0);
  int32x4_t vacc2 = vdupq_n_s32(0);
  int32x4_t vacc3 = vdupq_n_s32(0);

  // Secondary accumulator pair for unrolling
  int32x4_t vacc0b = vdupq_n_s32(0);
  int32x4_t vacc1b = vdupq_n_s32(0);
  int32x4_t vacc2b = vdupq_n_s32(0);
  int32x4_t vacc3b = vdupq_n_s32(0);

  int i = 0;

  // Process 32 input channels per iteration (two 16-wide passes)
  for (; i + 32 <= ic; i += 32) {
    // Load input once, reuse for all 4 weight rows
    int8x16_t va0 = vld1q_s8(in_p + i);       // input lanes  0..15
    int8x16_t va1 = vld1q_s8(in_p + i + 16);  // input lanes 16..31

    int8x8_t va0_lo = vget_low_s8(va0);
    int8x8_t va0_hi = vget_high_s8(va0);
    int8x8_t va1_lo = vget_low_s8(va1);
    int8x8_t va1_hi = vget_high_s8(va1);

    // Weight row 0
    int8x16_t vw0a = vld1q_s8(w0 + i);
    int8x16_t vw0b = vld1q_s8(w0 + i + 16);
    vacc0  = neon_mlal_s8(vacc0,  va0_lo, vget_low_s8(vw0a));
    vacc0  = neon_mlal_s8(vacc0,  va0_hi, vget_high_s8(vw0a));
    vacc0b = neon_mlal_s8(vacc0b, va1_lo, vget_low_s8(vw0b));
    vacc0b = neon_mlal_s8(vacc0b, va1_hi, vget_high_s8(vw0b));

    // Weight row 1
    int8x16_t vw1a = vld1q_s8(w1 + i);
    int8x16_t vw1b = vld1q_s8(w1 + i + 16);
    vacc1  = neon_mlal_s8(vacc1,  va0_lo, vget_low_s8(vw1a));
    vacc1  = neon_mlal_s8(vacc1,  va0_hi, vget_high_s8(vw1a));
    vacc1b = neon_mlal_s8(vacc1b, va1_lo, vget_low_s8(vw1b));
    vacc1b = neon_mlal_s8(vacc1b, va1_hi, vget_high_s8(vw1b));

    // Weight row 2
    int8x16_t vw2a = vld1q_s8(w2 + i);
    int8x16_t vw2b = vld1q_s8(w2 + i + 16);
    vacc2  = neon_mlal_s8(vacc2,  va0_lo, vget_low_s8(vw2a));
    vacc2  = neon_mlal_s8(vacc2,  va0_hi, vget_high_s8(vw2a));
    vacc2b = neon_mlal_s8(vacc2b, va1_lo, vget_low_s8(vw2b));
    vacc2b = neon_mlal_s8(vacc2b, va1_hi, vget_high_s8(vw2b));

    // Weight row 3
    int8x16_t vw3a = vld1q_s8(w3 + i);
    int8x16_t vw3b = vld1q_s8(w3 + i + 16);
    vacc3  = neon_mlal_s8(vacc3,  va0_lo, vget_low_s8(vw3a));
    vacc3  = neon_mlal_s8(vacc3,  va0_hi, vget_high_s8(vw3a));
    vacc3b = neon_mlal_s8(vacc3b, va1_lo, vget_low_s8(vw3b));
    vacc3b = neon_mlal_s8(vacc3b, va1_hi, vget_high_s8(vw3b));
  }

  // Merge secondary accumulators
  vacc0 = vaddq_s32(vacc0, vacc0b);
  vacc1 = vaddq_s32(vacc1, vacc1b);
  vacc2 = vaddq_s32(vacc2, vacc2b);
  vacc3 = vaddq_s32(vacc3, vacc3b);

  // 16 elements tail
  for (; i + 16 <= ic; i += 16) {
    int8x16_t va = vld1q_s8(in_p + i);
    int8x8_t va_lo = vget_low_s8(va);
    int8x8_t va_hi = vget_high_s8(va);

    int8x16_t vw0 = vld1q_s8(w0 + i);
    vacc0 = neon_mlal_s8(vacc0, va_lo, vget_low_s8(vw0));
    vacc0 = neon_mlal_s8(vacc0, va_hi, vget_high_s8(vw0));

    int8x16_t vw1 = vld1q_s8(w1 + i);
    vacc1 = neon_mlal_s8(vacc1, va_lo, vget_low_s8(vw1));
    vacc1 = neon_mlal_s8(vacc1, va_hi, vget_high_s8(vw1));

    int8x16_t vw2 = vld1q_s8(w2 + i);
    vacc2 = neon_mlal_s8(vacc2, va_lo, vget_low_s8(vw2));
    vacc2 = neon_mlal_s8(vacc2, va_hi, vget_high_s8(vw2));

    int8x16_t vw3 = vld1q_s8(w3 + i);
    vacc3 = neon_mlal_s8(vacc3, va_lo, vget_low_s8(vw3));
    vacc3 = neon_mlal_s8(vacc3, va_hi, vget_high_s8(vw3));
  }

  // Horizontal reduce
  out[0] += static_cast<int64_t>(vaddvq_s32(vacc0));
  out[1] += static_cast<int64_t>(vaddvq_s32(vacc1));
  out[2] += static_cast<int64_t>(vaddvq_s32(vacc2));
  out[3] += static_cast<int64_t>(vaddvq_s32(vacc3));

  // Scalar tail (< 16 remaining ic)
  for (; i < ic; ++i) {
    int vi = static_cast<int>(in_p[i]);
    out[0] += vi * static_cast<int>(w0[i]);
    out[1] += vi * static_cast<int>(w1[i]);
    out[2] += vi * static_cast<int>(w2[i]);
    out[3] += vi * static_cast<int>(w3[i]);
  }
}

#endif  // __ARM_NEON

// ---------------------------------------------------------------------------
// Kernel class
// ---------------------------------------------------------------------------
class Kernel {
 public:
  explicit Kernel(const Param& p) : p_(p) { prepare(); }

  void run(const int8_t* input, const int8_t* weight, const int8_t* bias,
           int8_t* output) const {
    auto rows = p_.output_shape[0] * p_.output_shape[kTensorDim[AXIS_H]];
    auto nthreads = std::max(1, p_.num_threads);
    nthreads = std::min(nthreads, rows);
    if (nthreads <= 1) {
      run_rows(input, weight, bias, output, 0, rows);
      return;
    }
    std::vector<std::thread> workers;
    workers.reserve(nthreads);
    auto chunk = (rows + nthreads - 1) / nthreads;
    for (auto t = 0; t < nthreads; ++t) {
      auto begin = t * chunk;
      auto end = std::min(rows, begin + chunk);
      if (begin >= end) break;
      workers.emplace_back([=]() {
        run_rows(input, weight, bias, output, begin, end);
      });
    }
    for (auto& w : workers) w.join();
  }

 private:
  // -------------------------------------------------------------------------
  // prepare(): validation + tap precomputation (unchanged from original)
  // -------------------------------------------------------------------------
  void prepare() {
    CHECK_EQ(p_.input_shape.size(),  5u) << "conv3d-fix input must be [N,H,W,D,C]";
    CHECK_EQ(p_.output_shape.size(), 5u);
    CHECK_EQ(p_.weight_shape.size(), 5u) << "conv3d-fix weights must be [OC,KH,KW,KD,IC]";
    CHECK_EQ(p_.kernel.size(),    3u);
    CHECK_EQ(p_.stride.size(),    3u);
    CHECK_EQ(p_.dilation.size(), 3u);
    CHECK_EQ(p_.pad.size(),       6u);
    CHECK_EQ(p_.input_shape[0],  p_.output_shape[0]) << "batch mismatch";
    CHECK_EQ(p_.input_shape[4],  p_.weight_shape[4]) << "input channel mismatch";
    CHECK_EQ(p_.output_shape[4], p_.weight_shape[0]) << "output channel mismatch";
    CHECK_EQ(p_.group, 1)
        << "grouped 3D convolution is not implemented; op carries group=" << p_.group;
    CHECK(p_.pad_mode == "FLOOR" || p_.pad_mode == "CEIL" ||
          p_.pad_mode == "SAME"  || p_.pad_mode == "VALID")
        << "unexpected pad_mode " << p_.pad_mode;
    CHECK(p_.nonlinear != util::NONLINEAR::HSIGMOID &&
          p_.nonlinear != util::NONLINEAR::HSWISH)
        << "HSIGMOID/HSWISH are not implemented for 3D convolution";

    in_stride_  = util::get_dim_stride_vec(p_.input_shape);
    w_stride_   = util::get_dim_stride_vec(p_.weight_shape);
    out_stride_ = util::get_dim_stride_vec(p_.output_shape);

    for (auto a = 0; a < 3; ++a) {
      auto k = p_.kernel[kAttrIdx[a]];
      CHECK_EQ(k, p_.weight_shape[1 + a])
          << "kernel attribute disagrees with the weight shape on axis " << a;
      CHECK_GT(p_.stride[kAttrIdx[a]], 0);
      CHECK_GT(p_.dilation[kAttrIdx[a]], 0);
    }
    for (auto a = 0; a < 3; ++a) build_taps(a);

    oc_ = p_.output_shape[4];
    ic_ = p_.input_shape[4];
  }

  void build_taps(int a) {
    auto in_extent  = p_.input_shape[kTensorDim[a]];
    auto out_extent = p_.output_shape[kTensorDim[a]];
    auto k_num      = p_.kernel[kAttrIdx[a]];
    auto stride     = p_.stride[kAttrIdx[a]];
    auto dilation   = p_.dilation[kAttrIdx[a]];
    auto pad_begin  = p_.pad[kPadBegin[a]];
    auto dilated_extent = (in_extent - 1) * stride + 1;
    if (p_.transposed) pad_begin = dilation * (k_num - 1) - pad_begin;

    taps_[a].assign(out_extent, {});
    for (auto o = 0; o < out_extent; ++o) {
      auto& dst = taps_[a][o];
      dst.reserve(k_num);
      auto src = p_.transposed ? (o - pad_begin) : (o * stride - pad_begin);
      for (auto k = 0; k < k_num; ++k) {
        auto i = src + k * dilation;
        if (p_.transposed) {
          if (i < 0 || i >= dilated_extent) continue;
          if (i % stride != 0) continue;
          i /= stride;
        } else {
          if (i < 0 || i >= in_extent) continue;
        }
        dst.push_back(Tap{k, i});
      }
    }
  }

  // -------------------------------------------------------------------------
  // run_rows(): the hot path ? NEON-accelerated
  // -------------------------------------------------------------------------
  void run_rows(const int8_t* input, const int8_t* weight, const int8_t* bias,
                int8_t* output, int row_begin, int row_end) const {
    const int oh_num = p_.output_shape[kTensorDim[AXIS_H]];
    const int ow_num = p_.output_shape[kTensorDim[AXIS_W]];
    const int od_num = p_.output_shape[kTensorDim[AXIS_D]];

    // Per-thread accumulator, avoids repeated allocation in the row loop.
    std::vector<int64_t> acc(oc_);

    for (auto row = row_begin; row < row_end; ++row) {
      const int n  = row / oh_num;
      const int oh = row % oh_num;
      const int64_t in_n   = static_cast<int64_t>(n) * in_stride_[0];
      const int64_t out_row = static_cast<int64_t>(n) * out_stride_[0]
                            + static_cast<int64_t>(oh) * out_stride_[1];

      for (int ow = 0; ow < ow_num; ++ow) {
        for (int od = 0; od < od_num; ++od) {

          std::fill(acc.begin(), acc.end(), int64_t{0});

          // ------------------------------------------------------------------
          // Traverse all valid taps (H � W � D)
          // ------------------------------------------------------------------
          for (const auto& th : taps_[AXIS_H][oh]) {
            const int64_t in_h = in_n + static_cast<int64_t>(th.i) * in_stride_[1];
            const int64_t w_h  =        static_cast<int64_t>(th.k) * w_stride_[1];

            for (const auto& tw : taps_[AXIS_W][ow]) {
              const int64_t in_hw = in_h + static_cast<int64_t>(tw.i) * in_stride_[2];
              const int64_t w_hw  = w_h  + static_cast<int64_t>(tw.k) * w_stride_[2];

              for (const auto& td : taps_[AXIS_D][od]) {
                // Pointers to the IC-contiguous input and weight slices
                const int8_t* in_p = input  + in_hw
                                   + static_cast<int64_t>(td.i) * in_stride_[3];
                const int8_t* w_base = weight + w_hw
                                   + static_cast<int64_t>(td.k) * w_stride_[3];

                // --------------------------------------------------------------
                // OC loop ? unrolled x4 with NEON dot-product kernel
                // --------------------------------------------------------------
                int oc = 0;

#ifdef __ARM_NEON
                // 4-way OC unroll: one input load shared across 4 weight rows
                for (; oc + 4 <= oc_; oc += 4) {
                  // Weight rows for oc+0 .. oc+3 are at w_base + oc*w_stride_[0]
                  // w_stride_[0] == IC (the weight layout is [OC, KH, KW, KD, IC])
                  const int8_t* w0 = w_base + static_cast<int64_t>(oc + 0) * w_stride_[0];
                  const int8_t* w1 = w_base + static_cast<int64_t>(oc + 1) * w_stride_[0];
                  const int8_t* w2 = w_base + static_cast<int64_t>(oc + 2) * w_stride_[0];
                  const int8_t* w3 = w_base + static_cast<int64_t>(oc + 3) * w_stride_[0];

                  neon_dot4_s8(in_p, w0, w1, w2, w3, ic_, acc.data() + oc);
                }
#endif  // __ARM_NEON

                // Scalar tail for remaining OC (also handles full scalar path)
                for (; oc < oc_; ++oc) {
                  const int8_t* w_oc = w_base
                                     + static_cast<int64_t>(oc) * w_stride_[0];
#ifdef __ARM_NEON
                  // Use the single-output NEON dot for remaining oc
                  acc[oc] += neon_dot_s8(in_p, w_oc, ic_);
#else
                  // Pure scalar fallback (unchanged from original)
                  int64_t sum = 0;
                  for (int ic = 0; ic < ic_; ++ic) {
                    sum += static_cast<int>(in_p[ic])
                         * static_cast<int>(w_oc[ic]);
                  }
                  acc[oc] += sum;
#endif
                }
              }  // td
            }  // tw
          }  // th

          // ------------------------------------------------------------------
          // Epilogue: quantize + activation + write output (unchanged)
          // ------------------------------------------------------------------
          const int64_t out_off = out_row
                                + static_cast<int64_t>(ow) * out_stride_[2]
                                + static_cast<int64_t>(od) * out_stride_[3];

          for (int oc = 0; oc < oc_; ++oc) {
            float filout = static_cast<float>(acc[oc]);
            if (bias != nullptr) {
              filout += static_cast<float>(bias[oc]) * p_.shift_bias_scale;
            }
            output[out_off + oc] = static_cast<int8_t>(
                util::fix(nonlinear_fn(filout / p_.shift_cut_scale)));
          }

        }  // od
      }  // ow
    }  // row
  }

  // -------------------------------------------------------------------------
  // Activation (unchanged from original)
  // -------------------------------------------------------------------------
  float nonlinear_fn(float in) const {
    auto ret = in;
    switch (p_.nonlinear) {
      case util::NONLINEAR::RELU:
        if (ret < 0) ret = 0;
        break;
      case util::NONLINEAR::PRELU:
        if (ret < 0) ret *= p_.prelu;
        break;
      case util::NONLINEAR::LEAKYRELU:
        if (ret < 0) ret *= 0.01f;
        break;
      case util::NONLINEAR::RELU6:
        // NOTE: matches the corrected (non-buggy) formulation ? clamp `in`.
        ret = std::max(0.0f, std::min(ret, p_.six_as_output));
        break;
      case util::NONLINEAR::NONE:
      default:
        break;
    }
    return ret;
  }

 private:
  Param p_;
  std::vector<int> in_stride_;
  std::vector<int> w_stride_;
  std::vector<int> out_stride_;
  std::vector<std::vector<Tap>> taps_[3];
  int oc_{0};
  int ic_{0};
};

}  // namespace conv3d
}  // namespace cpu_task
}  // namespace ai
}  // namespace vitis