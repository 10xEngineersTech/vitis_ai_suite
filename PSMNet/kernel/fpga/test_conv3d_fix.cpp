// Golden-model test: the CPU Kernel (conv3d_fix_kernel.hpp) vs the HLS kernel
// source run through the host driver in software-emulation mode. Every
// output byte must match.
#include <cstdio>
#include <random>

#include "conv3d_fix_fpga_host.hpp"

using namespace vitis::ai::cpu_task;
using conv3d::Param;
using NL = util::NONLINEAR;

struct Case {
  const char* name;
  bool transposed;
  int N, H, W, D, IC, OC;
  int kH, kW, kD, sH, sW, sD, dH, dW, dD;
  int pad[6];  // {left, right, top, bottom, near, far}
  NL nl;
  bool bias;
};

static int out_ext(const Case& c, int in, int k, int s, int d, int pb, int pe) {
  if (c.transposed) return (in - 1) * s + d * (k - 1) + 1 - pb - pe;
  return (in + pb + pe - d * (k - 1) - 1) / s + 1;
}

struct Fix {
  int in, w, out, bias, xr, wr;  // fix points, input / weight value ranges
};

static int run_case(const Case& c, std::mt19937& rng,
                    Fix f = {4, 4, 1, 4, 40, 12}) {
  Param p;
  p.transposed = c.transposed;
  p.kernel = {c.kW, c.kH, c.kD};  // {W, H, D}
  p.stride = {c.sW, c.sH, c.sD};
  p.dilation = {c.dW, c.dH, c.dD};
  p.pad = {c.pad[0], c.pad[1], c.pad[2], c.pad[3], c.pad[4], c.pad[5]};
  const int oH = out_ext(c, c.H, c.kH, c.sH, c.dH, c.pad[2], c.pad[3]);
  const int oW = out_ext(c, c.W, c.kW, c.sW, c.dW, c.pad[0], c.pad[1]);
  const int oD = out_ext(c, c.D, c.kD, c.sD, c.dD, c.pad[4], c.pad[5]);
  p.input_shape = {c.N, c.H, c.W, c.D, c.IC};
  p.weight_shape = {c.OC, c.kH, c.kW, c.kD, c.IC};
  p.output_shape = {c.N, oH, oW, oD, c.OC};
  p.nonlinear = c.nl;
  p.prelu = 0.25f;
  p.set_fix_point(f.in, f.w, c.nl == NL::RELU6 ? f.out - 1 : f.out);
  p.set_bias_fix_point(f.in, f.w, f.bias);

  std::string why;
  if (!conv3d::FpgaKernel::supported(p, &why)) {
    std::printf("%-38s CPU fallback (%s)\n", c.name, why.c_str());
    return 0;
  }

  std::uniform_int_distribution<int> di(-f.xr, f.xr), dw(-f.wr, f.wr),
      db(-128, 127);
  const size_t in_n = (size_t)c.N * c.H * c.W * c.D * c.IC;
  const size_t w_n = (size_t)c.OC * c.kH * c.kW * c.kD * c.IC;
  const size_t out_n = (size_t)c.N * oH * oW * oD * c.OC;
  std::vector<int8_t> in(in_n), w(w_n), b(c.OC), ref(out_n), got(out_n, 55);
  for (auto& v : in) v = di(rng);
  for (auto& v : w) v = dw(rng);
  for (auto& v : b) v = db(rng);

  conv3d::Kernel golden(p);
  golden.run(in.data(), w.data(), c.bias ? b.data() : nullptr, ref.data());

  conv3d::FpgaKernel fpga(p);
  fpga.run(in.data(), w.data(), c.bias ? b.data() : nullptr, got.data());

  size_t bad = 0, nz = 0, sat = 0;
  for (size_t i = 0; i < out_n; ++i) {
    bad += (ref[i] != got[i]);
    nz += (ref[i] != 0);
    sat += (ref[i] == 127 || ref[i] == -128);
  }
  std::printf("%-38s out=%8zu nz=%5.1f%% sat=%5.1f%% mismatches=%zu  %s\n",
              c.name, out_n, 100.0 * nz / out_n, 100.0 * sat / out_n, bad,
              bad ? "FAIL" : "PASS");
  return bad ? 1 : 0;
}

int main() {
  std::mt19937 rng(1234);
  // PSMNet stride-1 pad {1,1,1,1,1,1}; stride-2 pad {1,0,1,0,1,0}
#define P1 {1, 1, 1, 1, 1, 1}
#define P2 {1, 0, 1, 0, 1, 0}
  const Case cases[] = {
    // name, T, N,H,W,D, IC,OC, k(H,W,D), s(H,W,D), dil(H,W,D), pad, nl, bias
    // ---- generic -------------------------------------------------------
    {"conv 3x3x3 s1 IC32 OC32",            false,1,6,7,20, 32,32, 3,3,3, 1,1,1, 1,1,1, P1, NL::RELU, true},
    {"conv 3x3x3 s2 IC16 OC24",            false,1,9,8,21, 16,24, 3,3,3, 2,2,2, 1,1,1, P1, NL::NONE, true},
    {"conv IC20 OC40 dil2",                false,2,7,7,18, 20,40, 3,3,3, 1,1,1, 2,2,2, {2,2,2,2,2,2}, NL::RELU6, true},
    {"conv 1x1x1 IC64 OC128 nobias",       false,1,4,5,9,  64,128,1,1,1, 1,1,1, 1,1,1, {0,0,0,0,0,0}, NL::NONE, false},
    {"conv IC128 OC128",                   false,1,4,4,12,128,128,3,3,3, 1,1,1, 1,1,1, P1, NL::RELU, true},
    {"conv stride-2 big pad (empty taps)", false,1,3,3,5,  32,16, 3,3,3, 2,2,2, 1,1,1, {4,4,4,4,4,4}, NL::RELU, true},
    {"convT 3x3x3 s2 IC32 OC16",           true, 1,5,6,9,  32,16, 3,3,3, 2,2,2, 1,1,1, P1, NL::RELU, true},
    {"convT 4x4x4 s2 IC48 OC8",            true, 1,5,5,8,  48,8,  4,4,4, 2,2,2, 1,1,1, P1, NL::NONE, false},
    {"convT 3x3x3 s1 IC16 OC16",           true, 1,5,5,5,  16,16, 3,3,3, 1,1,1, 1,1,1, P1, NL::RELU, true},
    {"convT s3 dil2 IC10 OC5",             true, 1,4,4,7,  10,5,  3,3,3, 3,3,3, 2,2,2, {1,2,1,2,2,1}, NL::NONE, true},
    {"conv aniso k(3,5,1) s(1,2,1)",       false,1,6,9,35, 32,48, 3,5,1, 1,2,1, 1,1,1, {2,2,1,1,0,0}, NL::RELU, true},
    {"conv D=60 (4 tiles) IC7 OC3",        false,1,3,3,60, 7,3,   3,3,3, 1,1,1, 1,1,1, P1, NL::NONE, true},
    {"leaky -> CPU fallback",              false,1,4,4,4,  16,16, 3,3,3, 1,1,1, 1,1,1, P1, NL::LEAKYRELU, true},
    // ---- PSMNet kernel-2 layers (real C, k, stride, pad; small H x W) ----
    {"psm dres0.0  IC64 OC26",             false,1,5,6,48, 64,26, 3,3,3, 1,1,1, 1,1,1, P1, NL::RELU, true},
    {"psm dres0.1  IC26 OC26",             false,1,5,6,48, 26,26, 3,3,3, 1,1,1, 1,1,1, P1, NL::RELU, true},
    {"psm          IC26 OC10",             false,1,5,6,48, 26,10, 3,3,3, 1,1,1, 1,1,1, P1, NL::RELU, true},
    {"psm          IC10 OC26",             false,1,5,6,48, 10,26, 3,3,3, 1,1,1, 1,1,1, P1, NL::NONE, true},
    {"psm conv1 s2 IC26 OC20",             false,1,6,8,48, 26,20, 3,3,3, 2,2,2, 1,1,1, P2, NL::RELU, true},
    {"psm conv2    IC20 OC32",             false,1,3,4,24, 20,32, 3,3,3, 1,1,1, 1,1,1, P1, NL::RELU, true},
    {"psm conv3 s2 IC32 OC6",              false,1,6,8,24, 32,6,  3,3,3, 2,2,2, 1,1,1, P2, NL::RELU, true},
    {"psm conv5 T  IC6 OC32",              true, 1,3,4,12, 6,32,  3,3,3, 2,2,2, 1,1,1, P2, NL::NONE, true},
    {"psm conv6 T  IC32 OC26",             true, 1,3,4,24, 32,26, 3,3,3, 2,2,2, 1,1,1, P2, NL::NONE, true},
    {"psm s2       IC26 OC6",              false,1,6,8,48, 26,6,  3,3,3, 2,2,2, 1,1,1, P2, NL::RELU, true},
    {"psm          IC6 OC32",              false,1,3,4,24, 6,32,  3,3,3, 1,1,1, 1,1,1, P1, NL::RELU, true},
    {"psm classif  IC26 OC4",              false,1,5,6,48, 26,4,  3,3,3, 1,1,1, 1,1,1, P1, NL::NONE, true},
  };
  int fails = 0;
  for (const auto& c : cases) fails += run_case(c, rng);
  // other fix points: fractional bias (b < 0), n = 0, heavy saturation
  const Fix fixes[] = {
      {5, 6, 2, 14, 60, 30},     // n = 9, b = -3
      {3, 3, 6, 9, 20, 6},       // n = 0, b = -3
      {6, 7, 3, 6, 127, 127},    // n = 10, b = 7, mostly saturated
  };
  for (const auto& f : fixes) {
    std::printf("\n-- fix in=%d w=%d out=%d bias=%d\n", f.in, f.w, f.out, f.bias);
    for (const auto& c : cases)
      if (c.name[0] == 'p' || c.nl == NL::RELU6) fails += run_case(c, rng, f);
  }
  std::printf("\n%s (%d failing case(s))\n",
              fails ? "SOME TESTS FAILED" : "ALL TESTS PASSED", fails);
  return fails != 0;
}
