// conv3d_fix_hls.h
// Shared between the HLS kernel (conv3d_fix_hls.cpp) and the host driver
// (conv3d_fix_fpga_host.hpp). Only ap_int.h / hls_stream.h from Vitis HLS are
// needed, so the kernel also builds with plain g++ for C simulation.
#pragma once
#include <cstdint>

namespace conv3d_hls {

// ----------------------------------------------------------------------------
// Architecture (decides the resource usage)
// ----------------------------------------------------------------------------
// MAC array = OC_PAR output channels x RED reduction lanes x 2 output
// positions. Two int8 products share one DSP48E2 (27x18 multiplier, the two
// positions packed into the 27-bit port), so DSPs = OC_PAR * RED.
//   RED 24 -> 384 DSP (31 % of the K26), 768 MAC/cycle
//   RED 16 -> 256 DSP (21 %),            512 MAC/cycle
#ifndef CONV3D_RED
#define CONV3D_RED 24
#endif
constexpr int OC_PAR = 16;
constexpr int RED = CONV3D_RED;
constexpr int TD = 16;        // output positions along D per tile (2 x 8)
constexpr int TD2 = TD / 2;   // cycles per weight block
static_assert(RED % 8 == 0 && RED <= 32, "RED must be 8..32, multiple of 8");

// Limits (the host checks them and falls back to the CPU otherwise)
constexpr int MAX_OC = 128;                  // NOCT <= 8
constexpr int MAX_NOCT = MAX_OC / OC_PAR;
constexpr int MAX_K = 7;                     // kernel extent per axis (H, W)
constexpr int MAX_PAIRS = MAX_K * MAX_K;     // valid (kh, kw) per output
constexpr int MAX_OHW = 512;                 // output extent on H and W
constexpr int MAX_TILES = 32;                // D tiles per output line
constexpr int MAX_TN = 16;                   // tiles * NOCT (accumulator sets)
constexpr int LINE_W = 256;                  // 16 B words per staged line
constexpr int W_ROWS = 8192;                 // weight rows (RED bytes each)
constexpr int LINE_MAX = 8192;               // OD * OC bytes per output line
// packed products summed before unpacking; needs GROUP * 2^14 < 2^17
constexpr int GROUP = (RED % 6 == 0) ? 6 : 4;
static_assert(RED % GROUP == 0 && GROUP <= 7, "RED / GROUP");

// ----------------------------------------------------------------------------
// cfg[] (int32). Floats are stored as raw bit patterns.
// ----------------------------------------------------------------------------
enum CfgIdx {
  CFG_N = 0, CFG_H, CFG_W, CFG_D, CFG_IC,
  CFG_OH, CFG_OW, CFG_OD, CFG_OC,
  CFG_KH, CFG_KW,
  CFG_NONLINEAR,      // Nonlinear
  // Integer epilogue, exact w.r.t. the CPU's float maths (host-checked):
  //   out = clamp((acc + bterm[oc]) >> EPI_N),  RELU6: acc >= thr6[oc] -> SIX_Q
  CFG_EPI_N,
  CFG_SIX_Q,
  CFG_NOCT,           // ceil(OC / 16)
  CFG_NRB,            // ceil(L * IC / RED) reduction blocks
  CFG_NTILES,         // D tiles per output line
  CFG_LW,             // 16 B words per staged input line
  CFG_GF_BYTES,       // leading zero guard of a staged line, bytes
  CFG_W_ROWS,         // packed weight rows
  CFG_OUT_ROWS,       // ceil(OD * OC / 16)
  CFG_NPHASE,
  CFG_STATUS = 31     // written by the kernel
};
constexpr int CFG_WORDS = 32;

enum Nonlinear { NL_NONE = 0, NL_RELU = 1, NL_RELU6 = 2 };

enum Status { ST_OK = 0, ST_BAD_CFG = 1 };

// ----------------------------------------------------------------------------
// tab[] (int32), built by the host:
//   tapH  [OH][KH]      input row for (oh, kh), -1 = outside
//   tapW  [OW][KW]      input column for (ow, kw), -1 = outside
//   phase [NTILES]      D phase of each tile (selects the weight set)
//   pos   [NTILES][16]  byte offset of the position's D window in a staged
//                       line ((d_start + GF) * IC)
//   obyte [NTILES][16]  od * OC of the position in the output line, -1 = none
//   bterm [OC]          floor(bias * 2^b + 2^(n-1)): bias + rounding, n, b =
//                       log2 of shift_cut_scale, shift_bias_scale
//   thr6  [OC]          RELU6 only: smallest acc with value >= six
//
// Along D every output position o reads L consecutive input positions
// d_start(o) .. d_start(o)+L-1. The reduction runs over that flattened
// (L x IC) byte range, so IC needs no padding. The kd tap of each of the L
// slots depends only on the position's "phase" (1 phase for a conv, `stride`
// phases for a transposed conv); the host packs one weight set per phase.
//
// Weights (w128 words, 2 per row), row = block * 16 + j:
//   block = ((phase * KH*KW + kh*KW + kw) * NRB + rb) * NOCT + oct
//   row byte r = W[oct*16 + j][kh][kw][kd][c]  for flat = rb*RED + r,
//                m = flat / IC, c = flat % IC, kd = pattern[phase][m]
//   (0 where oc >= OC, m >= L or the slot has no tap)
// ----------------------------------------------------------------------------

}  // namespace conv3d_hls

#include <ap_int.h>
typedef ap_uint<128> w128;

// Kernel entry point (Vitis kernel name == "conv3d_fix").
extern "C" void conv3d_fix(const w128* input,    // [N,H,W,D,IC] int8
                           const w128* weight,   // packed rows
                           const int32_t* tab,   // tables above (incl. bias)
                           w128* output,         // [N,OH,OW,OD,OC] int8
                           int32_t* cfg);        // CFG_WORDS x int32
