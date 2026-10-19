/*
 * conv3d_fix_fpga_host.hpp
 *
 * Host driver for the `conv3d_fix` HLS kernel; same run() contract as
 * conv3d::Kernel in conv3d_fix_kernel.hpp, for both conv3d-fix and
 * transposed-conv3d-fix (Param::transposed):
 *
 *     if (conv3d::FpgaKernel::supported(param)) {
 *       conv3d::FpgaKernel k(param);          // once per layer
 *       k.run(input, weight, bias, output);   // every frame
 *     }
 *
 * The constructor builds everything the kernel needs from Param: H/W tap
 * tables, the D-axis phases and tiles, and the packed weights (re-packed only
 * if the weight pointer changes).
 *
 * Back-ends
 *   -DUSE_XRT : runs on the KV260 through XRT. The kernel must be linked into
 *               the xclbin that is ALREADY loaded for the DPU (one xclbin per
 *               device); this driver never loads an xclbin itself.
 *   default   : software emulation, calls conv3d_fix() from conv3d_fix_hls.cpp
 *               (C simulation and the unit test).
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "conv3d_fix_hls.h"
#include "conv3d_fix_kernel.hpp"  // Param, axis tables, CHECK macros

#ifdef USE_XRT
#include <ert.h>
#include <experimental/xrt-next.h>  // xclRegRead (diagnostics)
#include <xclbin.h>
#include <xrt.h>

#include <chrono>
#endif

namespace vitis {
namespace ai {
namespace cpu_task {
namespace conv3d {

class FpgaKernel {
 public:
  /// True if the kernel can run this layer; otherwise use conv3d::Kernel.
  /// `why` (optional) receives the reason.
  static bool supported(const Param& p, std::string* why = nullptr) {
    Plan plan;
    return make_plan(p, plan, why);
  }

  explicit FpgaKernel(const Param& p) : p_(p) {
    std::string why;
    CHECK(make_plan(p_, plan_, &why)) << "conv3d FPGA: " << why;
    build_tables();
#ifdef USE_XRT
    init_xrt();
#endif
  }

  /// NHWDC int8 in/out, weights [OC,KH,KW,KD,IC], bias [OC] or nullptr.
  void run(const int8_t* input, const int8_t* weight, const int8_t* bias,
           int8_t* output) {
    using namespace conv3d_hls;
    if (weight != packed_for_) {
      pack_weights(weight);
      packed_for_ = weight;
      weights_dirty_ = true;
    }
    if (!bias_set_ || bias != bias_for_) {
      set_bias(bias);
      bias_for_ = bias;
      bias_set_ = true;
      tab_dirty_ = true;
    }
    cfg_[CFG_STATUS] = -1;
#ifdef USE_XRT
    run_xrt(input, output);
#else
    run_emulated(input, output);
#endif
    CHECK_EQ(cfg_[CFG_STATUS], static_cast<int32_t>(ST_OK))
        << "conv3d_fix kernel rejected the configuration";
  }

 private:
  // --------------------------------------------------------------------------
  // Plan: everything derived from Param
  // --------------------------------------------------------------------------
  struct Plan {
    int n, ic, oc, noct;
    int in_ext[3], out_ext[3], k[3];  // H, W, D
    int L, nrb, gf, gb, lw, nphase, nl;
    int epi_n, epi_b, six_q;  // integer epilogue
    double six;
    std::vector<std::vector<int>> patterns;  // [phase][L] kd or -1
    std::vector<int> dstart, phase_of;       // per od
    struct Tile {
      int phase;
      std::vector<int> ods;
    };
    std::vector<Tile> tiles;
    int w_rows;
  };

  static int floordiv(int a, int b) {
    return (a >= 0) ? a / b : -((-a + b - 1) / b);
  }

  static bool fail(std::string* why, const std::string& m) {
    if (why) *why = m;
    return false;
  }

  static bool make_plan(const Param& p, Plan& pl, std::string* why) {
    using namespace conv3d_hls;
    if (p.input_shape.size() != 5 || p.output_shape.size() != 5 ||
        p.weight_shape.size() != 5 || p.kernel.size() != 3 ||
        p.stride.size() != 3 || p.dilation.size() != 3 || p.pad.size() != 6)
      return fail(why, "bad rank");
    if (p.group != 1) return fail(why, "group != 1");
    switch (p.nonlinear) {
      case util::NONLINEAR::NONE: pl.nl = NL_NONE; break;
      case util::NONLINEAR::RELU: pl.nl = NL_RELU; break;
      case util::NONLINEAR::RELU6: pl.nl = NL_RELU6; break;
      default: return fail(why, "nonlinear not supported by the kernel");
    }
    if (!epilogue_plan(p, pl, why)) return false;

    pl.n = p.input_shape[0];
    pl.ic = p.input_shape[4];
    pl.oc = p.output_shape[4];
    pl.noct = (pl.oc + OC_PAR - 1) / OC_PAR;
    if (pl.oc > MAX_OC) return fail(why, "OC > MAX_OC");
    for (int a = 0; a < 3; ++a) {
      pl.in_ext[a] = p.input_shape[kTensorDim[a]];
      pl.out_ext[a] = p.output_shape[kTensorDim[a]];
      pl.k[a] = p.kernel[kAttrIdx[a]];
      if (pl.k[a] != p.weight_shape[1 + a]) return fail(why, "kernel/weight");
      if (p.stride[kAttrIdx[a]] <= 0 || p.dilation[kAttrIdx[a]] <= 0)
        return fail(why, "stride/dilation");
    }
    if (pl.k[0] > MAX_K || pl.k[1] > MAX_K) return fail(why, "KH/KW > MAX_K");
    if (pl.out_ext[0] > MAX_OHW || pl.out_ext[1] > MAX_OHW)
      return fail(why, "OH/OW > MAX_OHW");

    // ---- D axis: per output, the L consecutive input slots it reads ------
    const int D = pl.in_ext[2], OD = pl.out_ext[2], K = pl.k[2];
    const int s = p.stride[kAttrIdx[2]], dil = p.dilation[kAttrIdx[2]];
    int pb = p.pad[kPadBegin[2]];
    if (p.transposed) pb = dil * (K - 1) - pb;
    std::vector<std::vector<std::pair<int, int>>> taps(OD);  // (k, i)
    pl.L = 1;
    for (int o = 0; o < OD; ++o) {
      for (int k = 0; k < K; ++k) {
        int i;
        if (p.transposed) {
          const int id = o - pb + k * dil;
          if (id - floordiv(id, s) * s != 0) continue;  // stride hole
          i = floordiv(id, s);
        } else {
          i = o * s - pb + k * dil;
        }
        taps[o].push_back({k, i});
      }
      if (!taps[o].empty()) {
        int lo = taps[o][0].second, hi = lo;
        for (auto& t : taps[o]) lo = std::min(lo, t.second), hi = std::max(hi, t.second);
        pl.L = std::max(pl.L, hi - lo + 1);
      }
    }
    pl.dstart.assign(OD, 0);
    pl.phase_of.assign(OD, 0);
    std::map<std::vector<int>, int> pat_id;
    int dmin = 0, dmax = D - 1;
    bool any = false;
    for (int o = 0; o < OD; ++o) {
      std::vector<int> pat(pl.L, -1);
      if (!taps[o].empty()) {
        int lo = taps[o][0].second;
        for (auto& t : taps[o]) lo = std::min(lo, t.second);
        for (auto& t : taps[o]) pat[t.second - lo] = t.first;
        pl.dstart[o] = lo;
        dmin = any ? std::min(dmin, lo) : lo;
        dmax = any ? std::max(dmax, lo + pl.L - 1) : lo + pl.L - 1;
        any = true;
      }
      auto it = pat_id.find(pat);
      if (it == pat_id.end()) {
        it = pat_id.emplace(pat, static_cast<int>(pl.patterns.size())).first;
        pl.patterns.push_back(pat);
      }
      pl.phase_of[o] = it->second;
    }
    pl.gf = std::max(0, -dmin);
    pl.gb = std::max(0, dmax - (D - 1));
    for (int o = 0; o < OD; ++o)
      if (taps[o].empty()) pl.dstart[o] = -pl.gf;  // weights are all 0
    pl.nphase = static_cast<int>(pl.patterns.size());
    pl.nrb = (pl.L * pl.ic + RED - 1) / RED;
    const long staged = static_cast<long>(pl.gf + D + pl.gb) * pl.ic;
    pl.lw = static_cast<int>((staged + 15) / 16) + 6;  // slice over-read slack

    // tiles: positions of one phase, ascending, 16 at a time
    pl.tiles.clear();
    for (int ph = 0; ph < pl.nphase; ++ph) {
      Plan::Tile t{ph, {}};
      for (int o = 0; o < OD; ++o) {
        if (pl.phase_of[o] != ph) continue;
        t.ods.push_back(o);
        if (static_cast<int>(t.ods.size()) == TD) {
          pl.tiles.push_back(t);
          t.ods.clear();
        }
      }
      if (!t.ods.empty()) pl.tiles.push_back(t);
    }
    pl.w_rows = pl.nphase * pl.k[0] * pl.k[1] * pl.nrb * pl.noct * OC_PAR;

    if (static_cast<int>(pl.tiles.size()) > MAX_TILES)
      return fail(why, "too many D tiles");
    if (static_cast<int>(pl.tiles.size()) * pl.noct > MAX_TN)
      return fail(why, "D tiles x OC tiles > MAX_TN");
    if (pl.w_rows > W_ROWS) return fail(why, "weights do not fit on chip");
    if (pl.lw > LINE_W) return fail(why, "staged input line > LINE_W");
    if (static_cast<long>(OD) * pl.oc > LINE_MAX)
      return fail(why, "output line > LINE_MAX");
    return true;
  }

  static bool pow2_exp(float v, int& e2) {
    int e = 0;
    if (!(v > 0.0f) || std::frexp(v, &e) != 0.5f) return false;
    e2 = e - 1;
    return true;
  }

  // The kernel's epilogue is integer, per output channel:
  //   out = clamp((acc + bterm[oc]) >> n),  bterm = floor(bias * 2^b + 2^(n-1))
  // with n = log2(shift_cut_scale), b = log2(shift_bias_scale). The CPU does
  // round_half_up((float(acc) + bias * 2^b) / 2^n) in float. For integer acc,
  // real f and n >= 0, floor((acc + f) / 2^n) == floor((acc + floor(f)) / 2^n),
  // so the two agree whenever the float maths is exact. It is exact unless the
  // sum needs > 24 significant bits (|sum| >= 2^24 * min(1, 2^b), or
  // |acc| >= 2^24); the checks below make every such value >= 256 after the
  // / 2^n, so both sides saturate to the same int8 (rounding never flips the
  // sign). Layers that fail the checks run on the CPU.
  static bool epilogue_plan(const Param& p, Plan& pl, std::string* why) {
    int n = 0, b = 0;
    if (!pow2_exp(p.shift_cut_scale, n))
      return fail(why, "shift_cut_scale not a power of two");
    if (!pow2_exp(p.shift_bias_scale, b))
      return fail(why, "shift_bias_scale not a power of two");
    if (n < 0 || n > 24 || b < -30 || b > 22)
      return fail(why, "fix points out of the epilogue's range");
    const double lim1 = std::ldexp(1.0, 24 + std::min(0, b) - n);
    const double lim2 =
        (std::ldexp(1.0, 24) - std::ldexp(1.0, 7 + std::max(0, b))) /
        std::ldexp(1.0, n);
    if (lim1 < 256.0 || lim2 < 256.0)
      return fail(why, "float epilogue not provably exact in integers");
    pl.epi_n = n;
    pl.epi_b = b;
    pl.six = p.six_as_output;
    pl.six_q = (pl.nl == conv3d_hls::NL_RELU6) ? util::fix(p.six_as_output) : 0;
    return true;
  }

  // tab_ tail: bterm[OC], thr6[OC] (depend on the bias values)
  void set_bias(const int8_t* bias) {
    const Plan& pl = plan_;
    tab_.resize(tab_base_);
    const double half = std::ldexp(1.0, pl.epi_n) * 0.5;
    for (int oc = 0; oc < pl.oc; ++oc) {
      const double bf = bias ? std::ldexp(static_cast<double>(bias[oc]), pl.epi_b) : 0.0;
      const double t = std::floor(bf + half);
      CHECK(std::fabs(t) < 2147483647.0) << "bias term out of range";
      tab_.push_back(static_cast<int32_t>(t));
    }
    for (int oc = 0; oc < pl.oc; ++oc) {
      const double bf = bias ? std::ldexp(static_cast<double>(bias[oc]), pl.epi_b) : 0.0;
      double t = std::ceil(std::ldexp(pl.six, pl.epi_n) - bf);  // acc >= t
      t = std::max(-2147483648.0, std::min(2147483647.0, t));
      tab_.push_back(static_cast<int32_t>(t));
    }
  }

  // --------------------------------------------------------------------------
  void build_tables() {
    using namespace conv3d_hls;
    const Plan& pl = plan_;
    tab_.clear();
    for (int a = 0; a < 2; ++a) {  // H, W tap tables, as Kernel::build_taps
      const int in_extent = pl.in_ext[a], k_num = pl.k[a];
      const int stride = p_.stride[kAttrIdx[a]];
      const int dilation = p_.dilation[kAttrIdx[a]];
      int pad_begin = p_.pad[kPadBegin[a]];
      const int dilated_extent = (in_extent - 1) * stride + 1;
      if (p_.transposed) pad_begin = dilation * (k_num - 1) - pad_begin;
      for (int o = 0; o < pl.out_ext[a]; ++o) {
        const int src = p_.transposed ? (o - pad_begin) : (o * stride - pad_begin);
        for (int k = 0; k < k_num; ++k) {
          int i = src + k * dilation;
          bool ok;
          if (p_.transposed) {
            ok = !(i < 0 || i >= dilated_extent || i % stride != 0);
            if (ok) i /= stride;
          } else {
            ok = !(i < 0 || i >= in_extent);
          }
          tab_.push_back(ok ? i : -1);
        }
      }
    }
    for (auto& t : pl.tiles) tab_.push_back(t.phase);
    for (auto& t : pl.tiles)
      for (int i = 0; i < TD; ++i)
        tab_.push_back(i < static_cast<int>(t.ods.size())
                           ? (pl.dstart[t.ods[i]] + pl.gf) * pl.ic
                           : 0);
    for (auto& t : pl.tiles)
      for (int i = 0; i < TD; ++i)
        tab_.push_back(i < static_cast<int>(t.ods.size()) ? t.ods[i] * pl.oc
                                                           : -1);

    std::memset(cfg_, 0, sizeof(cfg_));
    cfg_[CFG_N] = pl.n;
    cfg_[CFG_H] = pl.in_ext[0];
    cfg_[CFG_W] = pl.in_ext[1];
    cfg_[CFG_D] = pl.in_ext[2];
    cfg_[CFG_IC] = pl.ic;
    cfg_[CFG_OH] = pl.out_ext[0];
    cfg_[CFG_OW] = pl.out_ext[1];
    cfg_[CFG_OD] = pl.out_ext[2];
    cfg_[CFG_OC] = pl.oc;
    cfg_[CFG_KH] = pl.k[0];
    cfg_[CFG_KW] = pl.k[1];
    cfg_[CFG_NONLINEAR] = pl.nl;
    cfg_[CFG_EPI_N] = pl.epi_n;
    cfg_[CFG_SIX_Q] = pl.six_q;
    cfg_[CFG_NOCT] = pl.noct;
    cfg_[CFG_NRB] = pl.nrb;
    cfg_[CFG_NTILES] = static_cast<int>(pl.tiles.size());
    cfg_[CFG_LW] = pl.lw;
    cfg_[CFG_GF_BYTES] = pl.gf * pl.ic;
    cfg_[CFG_W_ROWS] = pl.w_rows;
    cfg_[CFG_OUT_ROWS] = (pl.out_ext[2] * pl.oc + 15) / 16;
    cfg_[CFG_NPHASE] = pl.nphase;
    tab_base_ = tab_.size();
    set_bias(nullptr);
  }

  // rows of RED bytes, each stored in 32 B (two w128 words)
  void pack_weights(const int8_t* w) {
    using namespace conv3d_hls;
    const Plan& pl = plan_;
    const int KH = pl.k[0], KW = pl.k[1], KD = pl.k[2], IC = pl.ic;
    packed_w_.assign(static_cast<size_t>(pl.w_rows) * 32, 0);
    size_t row = 0;
    for (int ph = 0; ph < pl.nphase; ++ph)
      for (int kh = 0; kh < KH; ++kh)
        for (int kw = 0; kw < KW; ++kw)
          for (int rb = 0; rb < pl.nrb; ++rb)
            for (int oct = 0; oct < pl.noct; ++oct)
              for (int j = 0; j < OC_PAR; ++j, ++row) {
                const int oc = oct * OC_PAR + j;
                if (oc >= pl.oc) continue;
                for (int r = 0; r < RED; ++r) {
                  const int flat = rb * RED + r;
                  const int m = flat / IC, c = flat % IC;
                  if (m >= pl.L) continue;
                  const int kd = pl.patterns[ph][m];
                  if (kd < 0) continue;
                  packed_w_[row * 32 + r] =
                      w[((((static_cast<size_t>(oc) * KH + kh) * KW + kw) * KD +
                          kd) * IC) + c];
                }
              }
  }

  size_t in_bytes() const {
    return static_cast<size_t>(plan_.n) * plan_.in_ext[0] * plan_.in_ext[1] *
           plan_.in_ext[2] * plan_.ic;
  }
  size_t out_bytes() const {
    return static_cast<size_t>(plan_.n) * plan_.out_ext[0] * plan_.out_ext[1] *
           plan_.out_ext[2] * plan_.oc;
  }
  static size_t words(size_t bytes) { return (bytes + 15) / 16 + 4; }

  // --------------------------------------------------------------------------
  // Back-end 1: software emulation
  // --------------------------------------------------------------------------
#ifndef USE_XRT
  static void to_words(const int8_t* src, size_t n, std::vector<w128>& dst) {
    dst.assign(words(n), w128(0));
    for (size_t i = 0; i < n; ++i)
      dst[i / 16].range(8 * (i % 16) + 7, 8 * (i % 16)) =
          static_cast<uint8_t>(src[i]);
  }

  void run_emulated(const int8_t* input, int8_t* output) {
    std::vector<w128> in_w, w_w, out_w(words(out_bytes()), w128(0));
    to_words(input, in_bytes(), in_w);
    to_words(packed_w_.data(), packed_w_.size(), w_w);
    conv3d_fix(in_w.data(), w_w.data(), tab_.data(),
               out_w.data(), cfg_);
    for (size_t i = 0; i < out_bytes(); ++i)
      output[i] = static_cast<int8_t>(
          out_w[i / 16].range(8 * (i % 16) + 7, 8 * (i % 16)).to_uint());
  }
#endif

  // --------------------------------------------------------------------------
  // Back-end 2: the KV260 PL, through the XRT C API
  // --------------------------------------------------------------------------
  // The same calls Vitis-AI's xir::XrtCu makes for the DPU
  // (xclOpenContext / xclExecBuf / xclExecWait on an ERT start-CU command).
  // XrtCu itself cannot be used: Vitis-AI's device handle only registers the
  // DPU's CUs (matched by name). The xclbin is the one the DPU
  // runtime loads (/etc/vart.conf or XLNX_VART_FIRMWARE); its uuid comes from
  // that file. (The native xrt::kernel API cannot see that xclbin in a
  // process where vart opened the device first.)
#ifdef USE_XRT
  // conv3d_fix s_axilite register map (from the exported kernel.xml)
  enum Reg : uint32_t {
    REG_GIE = 0x04, REG_IER = 0x08, REG_INPUT = 0x10, REG_WEIGHT = 0x1c,
    REG_TAB = 0x28, REG_OUTPUT = 0x34, REG_CFG = 0x40, REG_END = 0x48
  };

  // A device buffer: cacheable, mapped, with its physical address.
  struct XBo {
    xclDeviceHandle h;
    xclBufferHandle bo;
    void* p;
    uint64_t pa;
    size_t size;
    XBo(xclDeviceHandle handle, size_t sz, unsigned flags)
        : h(handle), size(sz) {
      bo = xclAllocBO(h, sz, 0, flags);
      CHECK(bo != NULLBO) << "xclAllocBO(" << sz << ") failed (CMA?)";
      p = xclMapBO(h, bo, true);
      CHECK(p != nullptr) << "xclMapBO failed";
      xclBOProperties prop{};
      CHECK_EQ(xclGetBOProperties(h, bo, &prop), 0);
      pa = prop.paddr;
    }
    ~XBo() {
      xclUnmapBO(h, bo, p);
      xclFreeBO(h, bo);
    }
    void to_device(size_t n) {
      xclSyncBO(h, bo, XCL_BO_SYNC_BO_TO_DEVICE, n, 0);
    }
    void from_device(size_t n) {
      xclSyncBO(h, bo, XCL_BO_SYNC_BO_FROM_DEVICE, n, 0);
    }
  };

  static std::string vart_firmware() {
    if (auto env = std::getenv("XLNX_VART_FIRMWARE")) return env;
    std::ifstream conf("/etc/vart.conf");
    std::string line;
    while (std::getline(conf, line)) {
      auto k = line.find("firmware:");
      if (k == std::string::npos) continue;
      auto v = line.substr(k + 9);
      v.erase(0, v.find_first_not_of(" \t\""));
      v.erase(v.find_last_not_of(" \t\"\r") + 1);
      return v;
    }
    return "";
  }

  // One CU context per process; the big input/output buffers are shared by
  // all layers (PSMNet's are ~106 MB each, too much CMA per layer).
  struct Shared {
    std::mutex mtx;
    xclDeviceHandle h = nullptr;
    int cu_index = -1;
    std::unique_ptr<XBo> cmd, in, out;
  };
  static Shared& shared() {
    static Shared* s = [] {
      auto* p = new Shared;
      const auto fw = vart_firmware();
      CHECK(!fw.empty()) << "no xclbin: set XLNX_VART_FIRMWARE or /etc/vart.conf";
      axlf top{};
      std::ifstream f(fw, std::ios::binary);
      CHECK(f.read(reinterpret_cast<char*>(&top), sizeof(top)).good())
          << "cannot read " << fw;
      p->h = xclOpen(0, nullptr, XCL_QUIET);
      CHECK(p->h != nullptr) << "xclOpen(0) failed";
      p->cu_index = xclIPName2Index(p->h, "conv3d_fix:conv3d_fix_1");
      CHECK_GE(p->cu_index, 0)
          << "no conv3d_fix:conv3d_fix_1 CU in the loaded xclbin (" << fw << ")";
      CHECK_EQ(xclOpenContext(p->h, top.m_header.uuid, p->cu_index, true), 0)
          << "xclOpenContext failed: is " << fw << " the loaded xclbin?";
      p->cmd = std::make_unique<XBo>(p->h, 4096, XCL_BO_FLAGS_EXECBUF);
      // allow reading the CU's registers for timeout diagnostics
      xclIPSetReadRange(p->h, p->cu_index, 0x0, REG_END);
      if (auto d = std::getenv("DEBUG_CONV3D_FPGA"); d && std::atoi(d))
        LOG(INFO) << "conv3d_fix: xclbin " << fw << ", cu_index "
                  << p->cu_index;
      return p;
    }();
    return *s;
  }
  static std::unique_ptr<XBo> make_bo(size_t size) {
    return std::make_unique<XBo>(shared().h, size, XCL_BO_FLAGS_CACHEABLE);
  }

  void init_xrt() {
    shared();
    bo_w_ = make_bo(static_cast<size_t>(plan_.w_rows) * 32 + 64);
    bo_tab_ = make_bo(tab_.size() * sizeof(int32_t) + 64);
    bo_cfg_ = make_bo(4096);
  }

  static void set64(ert_start_kernel_cmd* c, uint32_t reg, uint64_t v) {
    c->data[reg / 4] = static_cast<uint32_t>(v & 0xffffffff);
    c->data[reg / 4 + 1] = static_cast<uint32_t>(v >> 32);
  }

  void run_xrt(const int8_t* input, int8_t* output) {
    auto& s = shared();
    std::lock_guard<std::mutex> lock(s.mtx);  // one CU
    const size_t ib = words(in_bytes()) * 16, ob = words(out_bytes()) * 16;
    if (!s.in || s.in->size < ib) s.in = make_bo(ib);
    if (!s.out || s.out->size < ob) s.out = make_bo(ob);

    std::memcpy(s.in->p, input, in_bytes());
    s.in->to_device(ib);
    if (weights_dirty_) {
      std::memcpy(bo_w_->p, packed_w_.data(), packed_w_.size());
      bo_w_->to_device(packed_w_.size());
      weights_dirty_ = false;
    }
    if (tab_dirty_) {
      std::memcpy(bo_tab_->p, tab_.data(), tab_.size() * sizeof(int32_t));
      bo_tab_->to_device(tab_.size() * sizeof(int32_t));
      tab_dirty_ = false;
    }
    std::memcpy(bo_cfg_->p, cfg_, sizeof(cfg_));
    bo_cfg_->to_device(sizeof(cfg_));

    auto* c = static_cast<ert_start_kernel_cmd*>(s.cmd->p);
    std::memset(c, 0, 4096);
    c->state = ERT_CMD_STATE_NEW;
    c->opcode = ERT_START_CU;
    // type left at 0 (ERT_DEFAULT), as Vitis-AI's DPU controller does
    c->count = 1 + REG_END / 4;  // cu_mask word + register words
    c->cu_mask = 1u << s.cu_index;
    c->data[REG_GIE / 4] = 1;
    c->data[REG_IER / 4] = 1;    // ap_done interrupt
    set64(c, REG_INPUT, s.in->pa);
    set64(c, REG_WEIGHT, bo_w_->pa);
    set64(c, REG_TAB, bo_tab_->pa);
    set64(c, REG_OUTPUT, s.out->pa);
    set64(c, REG_CFG, bo_cfg_->pa);

    CHECK_EQ(xclExecBuf(s.h, s.cmd->bo), 0) << "xclExecBuf failed";
    const auto t0 = std::chrono::steady_clock::now();
    const char* tmo = std::getenv("VAI_CONV3D_FPGA_TIMEOUT_MS");
    const long timeout_ms = tmo ? std::atol(tmo) : 30000;
    while (c->state < ERT_CMD_STATE_COMPLETED) {
      xclExecWait(s.h, 1000);
      const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now() - t0).count();
      if (ms >= timeout_ms) {
        uint32_t ctrl = 0xffffffff, gie = 0, ier = 0, isr = 0;
        xclRegRead(s.h, s.cu_index, 0x00, &ctrl);
        xclRegRead(s.h, s.cu_index, REG_GIE, &gie);
        xclRegRead(s.h, s.cu_index, REG_IER, &ier);
        xclRegRead(s.h, s.cu_index, 0x0c, &isr);
        LOG(FATAL) << "conv3d_fix timeout: cmd state " << c->state
                   << " (1=new 2=queued 3=running), cu_index " << s.cu_index
                   << " cu_mask 0x" << std::hex << c->cu_mask << " header 0x"
                   << c->header << " ap_ctrl 0x" << ctrl << " gie 0x" << gie
                   << " ier 0x" << ier << " isr 0x" << isr << std::dec;
      }
    }
    CHECK_EQ(static_cast<int>(c->state), static_cast<int>(ERT_CMD_STATE_COMPLETED))
        << "conv3d_fix command failed";

    s.out->from_device(ob);
    std::memcpy(output, s.out->p, out_bytes());
    bo_cfg_->from_device(sizeof(cfg_));
    std::memcpy(cfg_, bo_cfg_->p, sizeof(cfg_));
  }
#endif

 private:
  Param p_;
  Plan plan_;
  std::vector<int32_t> tab_;
  size_t tab_base_{0};  // tab_ size before the bias-dependent tail
  const int8_t* bias_for_{nullptr};
  bool bias_set_{false};
  bool tab_dirty_{true};
  std::vector<int8_t> packed_w_;
  const int8_t* packed_for_{nullptr};
  bool weights_dirty_{true};
  int32_t cfg_[conv3d_hls::CFG_WORDS]{};

#ifdef USE_XRT
  std::unique_ptr<XBo> bo_w_, bo_tab_, bo_cfg_;
#endif
};

}  // namespace conv3d
}  // namespace cpu_task
}  // namespace ai
}  // namespace vitis
