// conv3d_fix_hls.cpp
//
// Vitis HLS kernel for int8 Conv3D and Transposed-Conv3D on the Kria KV260,
// bit-exact with the CPU kernel (conv3d_fix_kernel.hpp). Sized to stay under
// 40 % of the K26 at 200 MHz.
//
// ============================== ARCHITECTURE ================================
//
// One "job" = one output line (n, oh, ow): all OD x OC outputs. The kernel is
// a 5-stage DATAFLOW pipeline, so loading, compute and write-back overlap:
//
//   reader   : for every valid (kh, kw) of the job, burst-reads the input
//              line [n, ih, iw, 0..D-1, 0..IC-1] (contiguous in DDR).
//   aligner  : re-aligns each line to 16 B and adds the zero guards along D
//              (padding) -> "staged lines".
//   mac      : (kh, kw) pairs outermost: while it runs every D tile of the
//              job against staged line q, it stages line q+1 into the other
//              half of a small ping-pong buffer. Accumulators for all tiles
//              of the job stay on chip. MAC array: 16 oc x RED lanes x 2
//              positions, two int8 products per DSP48E2. A 16 x RED weight
//              block is used for 8 cycles (16 positions) while the next one
//              is prefetched from URAM.
//   epilogue : bias, power-of-two scale, activation, DPU rounding, in exact
//              integer arithmetic, 2 x 16 channels per cycle.
//   writer   : packs the output byte stream into full 16 B bursts.
//
// The reduction runs over the flattened (kd, ic) range of each output
// position (see conv3d_fix_hls.h), so odd channel counts (6, 10, 26 ...) do
// not waste lanes, and transposed convs skip the stride holes (per-phase
// weight sets).
// ============================================================================
#include "conv3d_fix_hls.h"

#include <hls_stream.h>
#ifdef CONV3D_THREADED_SIM
#include <thread>
#endif

using namespace conv3d_hls;

namespace {

typedef ap_uint<RED * 8> wrow_t;

struct AccW {
  int32_t v[OC_PAR];
};

struct Chunk {
  w128 data;
  ap_uint<5> cnt;  // valid bytes, 1..16
};

struct LineDesc {
  int64_t a0;  // byte address of the line in the input tensor
  bool dummy;  // no valid (kh, kw): stage an all-zero line
};

struct EpiCfg {
  int n, nl, six_q;
};

// ---------------------------------------------------------------------------
// Stage 1: reader
// ---------------------------------------------------------------------------
void reader(const w128* input, const int16_t tapH[MAX_OHW * MAX_K],
            const int16_t tapW[MAX_OHW * MAX_K], hls::stream<int32_t>& hdr_s,
            hls::stream<int32_t>& cnt_s, hls::stream<LineDesc>& desc_s,
            hls::stream<w128>& raw_s, int N, int H, int W, int OH, int OW,
            int KH, int KW, int line_bytes) {
  for (int n = 0; n < N; ++n) {
    for (int oh = 0; oh < OH; ++oh) {
      for (int ow = 0; ow < OW; ++ow) {
        int khw[MAX_PAIRS];
        int ihw[MAX_PAIRS];
        int np = 0;
        for (int kh = 0; kh < KH; ++kh) {
          for (int kw = 0; kw < KW; ++kw) {
#pragma HLS PIPELINE II = 1
            const int ih = tapH[oh * KH + kh];
            const int iw = tapW[ow * KW + kw];
            if (ih >= 0 && iw >= 0) {
              khw[np] = kh * KW + kw;
              ihw[np] = ih * W + iw;
              ++np;
            }
          }
        }
        const bool dummy = (np == 0);
        const int nl = dummy ? 1 : np;
        hdr_s.write(nl);
        for (int q = 0; q < nl; ++q) {
#pragma HLS PIPELINE II = 1
          hdr_s.write(dummy ? 0 : khw[q]);
        }
        cnt_s.write(nl);
        for (int q = 0; q < nl; ++q) {
          LineDesc d;
          d.dummy = dummy;
          d.a0 = dummy ? 0
                       : (static_cast<int64_t>(n) * H * W + ihw[q]) *
                             static_cast<int64_t>(line_bytes);
          desc_s.write(d);
          if (!dummy) {
            const int64_t kfirst = d.a0 >> 4;
            const int64_t klast = (d.a0 + line_bytes - 1) >> 4;
            const int nw = static_cast<int>(klast - kfirst + 1);
            for (int k = 0; k < nw; ++k) {
#pragma HLS PIPELINE II = 1
              raw_s.write(input[kfirst + k]);
            }
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Stage 2: aligner. Staged line = [GF*IC zeros][D*IC data][zeros], LW words.
// ---------------------------------------------------------------------------
void aligner(hls::stream<int32_t>& cnt_s, hls::stream<LineDesc>& desc_s,
             hls::stream<w128>& raw_s, hls::stream<w128>& data_s, int njobs,
             int LW, int gf_bytes, int line_bytes) {
  for (int j = 0; j < njobs; ++j) {
    const int nl = cnt_s.read();
    for (int q = 0; q < nl; ++q) {
      const LineDesc d = desc_s.read();
      const int64_t v0 = d.a0 - gf_bytes;  // virtual address of staged byte 0
      const int64_t kb = v0 >> 4;          // floor
      const int s = static_cast<int>(v0 & 15);
      const int64_t kfirst = d.a0 >> 4;
      const int64_t klast = (d.a0 + line_bytes - 1) >> 4;
      w128 cur = 0;
      if (!d.dummy && kb >= kfirst && kb <= klast) cur = raw_s.read();
      for (int o = 0; o < LW; ++o) {
#pragma HLS PIPELINE II = 1
        const int64_t k1 = kb + o + 1;
        w128 nxt = 0;
        if (!d.dummy && k1 >= kfirst && k1 <= klast) nxt = raw_s.read();
        ap_uint<256> two = (static_cast<ap_uint<256>>(nxt) << 128) | cur;
        w128 out = static_cast<w128>(two >> (s * 8));
        const int rel0 = o * 16 - gf_bytes;  // zero bytes outside the line
        for (int i = 0; i < 16; ++i) {
#pragma HLS UNROLL
          const int rel = rel0 + i;
          if (d.dummy || rel < 0 || rel >= line_bytes)
            out.range(8 * i + 7, 8 * i) = 0;
        }
        data_s.write(out);
        cur = nxt;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Stage 3: MAC engine
// ---------------------------------------------------------------------------
inline wrow_t load_row(const w128* weight, int r) {
  ap_uint<256> v = (static_cast<ap_uint<256>>(weight[2 * r + 1]) << 128) |
                   weight[2 * r];
  return static_cast<wrow_t>(v);
}

// RED bytes starting at byte offset `off` of a 4-bank staged line
#define SLICE(BUF, off, dst)                                              \
  {                                                                       \
    const int w0 = (off) >> 4;                                            \
    const int sh = (off) & 15;                                            \
    w128 bw[4];                                                           \
    for (int b = 0; b < 4; ++b) {                                         \
      _Pragma("HLS UNROLL") const int k = (b - w0) & 3;                   \
      bw[b] = (k < 3) ? BUF[b][(w0 + k) >> 2] : w128(0);                  \
    }                                                                     \
    ap_uint<8> by[48];                                                    \
    for (int k = 0; k < 3; ++k) {                                         \
      _Pragma("HLS UNROLL") const w128 wk = bw[(w0 + k) & 3];             \
      for (int i = 0; i < 16; ++i) {                                      \
        _Pragma("HLS UNROLL") by[16 * k + i] = wk.range(8 * i + 7, 8 * i); \
      }                                                                   \
    }                                                                     \
    for (int i = 0; i < RED; ++i) {                                       \
      _Pragma("HLS UNROLL") dst.range(8 * i + 7, 8 * i) = by[sh + i];     \
    }                                                                     \
  }

void mac_engine(const w128* weight, hls::stream<int32_t>& hdr_s,
                hls::stream<w128>& data_s, hls::stream<AccW>& outA,
                hls::stream<AccW>& outB, const int16_t tile_phase[MAX_TILES],
                const int16_t pos[TD][MAX_TILES], int njobs, int KHW,
                int NOCT, int NRB, int NT, int LW, int w_rows) {
  // weights: even rows in wm0, odd rows in wm1 -> 2 rows per cycle
  static wrow_t wm0[W_ROWS / 2];
  static wrow_t wm1[W_ROWS / 2];
#pragma HLS BIND_STORAGE variable = wm0 type = ram_2p impl = uram
#pragma HLS BIND_STORAGE variable = wm1 type = ram_2p impl = uram

  // ping-pong staged line, 4 banks of 16 B words: any 3 consecutive words
  // sit in 3 different banks. 1 write + 2 reads (positions A, B) per cycle.
  static w128 ln0[4][LINE_W / 4];
  static w128 ln1[4][LINE_W / 4];
#pragma HLS ARRAY_PARTITION variable = ln0 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = ln1 complete dim = 1
#pragma HLS BIND_STORAGE variable = ln0 type = ram_1wnr impl = lutram
#pragma HLS BIND_STORAGE variable = ln1 type = ram_1wnr impl = lutram

  // accumulators: [oc lane][(tile * NOCT + oct) * 8 + t]
  // A = positions 0..7 of a tile, B = 8..15
  static int32_t accA[OC_PAR][MAX_TN * TD2];
  static int32_t accB[OC_PAR][MAX_TN * TD2];
#pragma HLS ARRAY_PARTITION variable = accA complete dim = 1
#pragma HLS ARRAY_PARTITION variable = accB complete dim = 1
#pragma HLS BIND_STORAGE variable = accA type = ram_2p impl = lutram
#pragma HLS BIND_STORAGE variable = accB type = ram_2p impl = lutram

  // current weight block and the one being prefetched
  wrow_t wcur[OC_PAR], wnxt[OC_PAR];
#pragma HLS ARRAY_PARTITION variable = wcur complete
#pragma HLS ARRAY_PARTITION variable = wnxt complete

  int khw_l[2][MAX_PAIRS];
#pragma HLS ARRAY_PARTITION variable = khw_l complete dim = 1
  int np_l[2];
#pragma HLS ARRAY_PARTITION variable = np_l complete dim = 0

  for (int r = 0; r < w_rows; ++r) {
#pragma HLS PIPELINE II = 2
    const wrow_t v = load_row(weight, r);
    if (r & 1)
      wm1[r >> 1] = v;
    else
      wm0[r >> 1] = v;
  }
  for (int a = 0; a < MAX_TN * TD2; ++a) {
#pragma HLS PIPELINE II = 1
    for (int j = 0; j < OC_PAR; ++j) {
#pragma HLS UNROLL
      accA[j][a] = 0;
      accB[j][a] = 0;
    }
  }

  // job 0: header + first line into ln0
  np_l[0] = hdr_s.read();
  for (int q = 0; q < np_l[0]; ++q) khw_l[0][q] = hdr_s.read();
  for (int w = 0; w < LW; ++w) {
#pragma HLS PIPELINE II = 1
    ln0[w & 3][w >> 2] = data_s.read();
  }
  {  // prologue: first block -> wcur
    const int blk0 = (tile_phase[0] * KHW + khw_l[0][0]) * NRB * NOCT;
    for (int t = 0; t < TD2; ++t) {
#pragma HLS PIPELINE II = 1
      wcur[2 * t] = wm0[blk0 * TD2 + t];
      wcur[2 * t + 1] = wm1[blk0 * TD2 + t];
    }
  }

  const int mac_steps = NT * NRB * NOCT * TD2;  // per staged line
  const int drain_steps = NT * NOCT * TD2;      // per job
  int lp = 0;                                   // ping-pong of current line

  for (int job = 0; job < njobs; ++job) {
    const int jp = job & 1;
    const bool has_next_job = (job + 1) < njobs;
    const int np = np_l[jp];

    for (int q = 0; q < np; ++q) {
      const bool last_line = (q + 1 == np);
      // The next job's header is first needed now (its first weight block is
      // prefetched during this line, its first line staged). The reader only
      // writes it after all of this job's lines, which are consumed by now;
      // reading it earlier deadlocks once a job's lines exceed the FIFOs.
      if (last_line && has_next_job) {
        np_l[jp ^ 1] = hdr_s.read();
        for (int k = 0; k < np_l[jp ^ 1]; ++k) khw_l[jp ^ 1][k] = hdr_s.read();
      }
      const bool has_next_line = !last_line || has_next_job;
      const int words_next = has_next_line ? LW : 0;
      const int steps = mac_steps + (last_line ? drain_steps : 0);
      const int iters = steps > words_next ? steps : words_next;
      // the block after this line's last one: next pair, or next job
      const int nq_khw = !last_line ? khw_l[jp][q + 1] : khw_l[jp ^ 1][0];

      int tile = 0, rb = 0, oct = 0, t = 0, dr = 0;
      int rb_off = 0;     // rb * RED
      int nxt_blk = 0;    // block prefetched during the current one

      for (int it = 0; it < iters; ++it) {
#pragma HLS PIPELINE II = 1
// The same accumulator entry is read-modified-written again after
// NOCT * 8 >= 8 iterations (next reduction block). Declare that true distance
// (not "inter false"): HLS then keeps the read->write span inside 8 cycles.
// With "false" it read early and wrote 15+ stages later, and NOCT == 1
// layers (OC <= 16) lost partial sums on hardware.
#pragma HLS DEPENDENCE variable = accA type = inter direction = RAW distance = 8 dependent = true
#pragma HLS DEPENDENCE variable = accB type = inter direction = RAW distance = 8 dependent = true
#pragma HLS DEPENDENCE variable = ln0 inter false
#pragma HLS DEPENDENCE variable = ln1 inter false
        // ---- stage the next line ------------------------------------------
        if (it < words_next) {
          const w128 v = data_s.read();
          if (lp == 0)
            ln1[it & 3][it >> 2] = v;
          else
            ln0[it & 3][it >> 2] = v;
        }

        if (it < mac_steps) {
          // ---- prefetch the next block, 2 rows per cycle ------------------
          if (t == 0) {
            int ntile = tile, nrb = rb, noct = oct + 1, kk = khw_l[jp][q];
            bool valid = true;
            if (noct == NOCT) {
              noct = 0;
              if (++nrb == NRB) {
                nrb = 0;
                if (++ntile == NT) {
                  ntile = 0;
                  kk = nq_khw;
                  valid = has_next_line;
                }
              }
            }
            nxt_blk = valid ? ((tile_phase[ntile] * KHW + kk) * NRB + nrb) *
                                      NOCT + noct
                            : -1;
          }
          // shift in rows 2t, 2t+1 of the next block. Unconditional (no
          // muxes): after the last block of a line wnxt is not used again
          // before the next line refills it.
          {
            const int ra = (nxt_blk < 0 ? 0 : nxt_blk) * TD2 + t;
            for (int j = 0; j < OC_PAR - 2; ++j) {
#pragma HLS UNROLL
              wnxt[j] = wnxt[j + 2];
            }
            wnxt[OC_PAR - 2] = wm0[ra];
            wnxt[OC_PAR - 1] = wm1[ra];
          }

          // ---- input slices of positions t (A) and t + 8 (B) -------------
          const int offA = pos[t][tile] + rb_off;
          const int offB = pos[t + TD2][tile] + rb_off;
          wrow_t xa, xb;
          if (lp == 0) {
            SLICE(ln0, offA, xa);
            SLICE(ln0, offB, xb);
          } else {
            SLICE(ln1, offA, xa);
            SLICE(ln1, offB, xb);
          }

          // ---- 16 x RED MACs, 2 positions per DSP ------------------------
          // p = (b * 2^18 + a) * w = b*w * 2^18 + a*w. Up to GROUP packed
          // products are summed before unpacking (|sum a*w| < 2^17).
          const int aidx = (tile * NOCT + oct) * TD2 + t;
          for (int j = 0; j < OC_PAR; ++j) {
#pragma HLS UNROLL
            ap_int<24> suma = 0, sumb = 0;
            for (int g = 0; g < RED / GROUP; ++g) {
#pragma HLS UNROLL
              ap_int<40> ps = 0;  // |sum| < 2^38
              for (int i = g * GROUP; i < (g + 1) * GROUP; ++i) {
#pragma HLS UNROLL
                const ap_int<8> a = xa.range(8 * i + 7, 8 * i);
                const ap_int<8> b = xb.range(8 * i + 7, 8 * i);
                const ap_int<8> w = wcur[j].range(8 * i + 7, 8 * i);
                const ap_int<27> packed =
                    (static_cast<ap_int<27>>(b) << 18) + a;
                ap_int<36> p = packed * w;
#pragma HLS BIND_OP variable = p op = mul impl = dsp
                ps += p;
              }
              const ap_int<18> lo = ps.range(17, 0);
              const ap_int<21> hi = (ps >> 18) + (lo < 0 ? 1 : 0);
              suma += lo;
              sumb += hi;
            }
            accA[j][aidx] += suma;
            accB[j][aidx] += sumb;
          }

          // ---- odometer: t, oct, rb, tile -----------------------------------
          if (++t == TD2) {
            t = 0;
            for (int j = 0; j < OC_PAR; ++j) {
#pragma HLS UNROLL
              wcur[j] = wnxt[j];
            }
            if (++oct == NOCT) {
              oct = 0;
              rb_off += RED;
              if (++rb == NRB) {
                rb = 0;
                rb_off = 0;
                ++tile;
              }
            }
          }
        } else if (it < steps) {
          // ---- last pair of the job: drain all tiles to the epilogue -------
          AccW a, b;
          for (int j = 0; j < OC_PAR; ++j) {
#pragma HLS UNROLL
            a.v[j] = accA[j][dr];
            b.v[j] = accB[j][dr];
            accA[j][dr] = 0;
            accB[j][dr] = 0;
          }
          outA.write(a);
          outB.write(b);
          ++dr;
        }
      }
      lp ^= 1;
    }
  }
}

// ---------------------------------------------------------------------------
// Stage 4: epilogue (integer, see CFG_EPI_*)
// ---------------------------------------------------------------------------
inline int8_t epi_lane(int32_t acc, int32_t bterm, int32_t thr6,
                       const EpiCfg& e) {
  if (e.nl == NL_RELU6 && acc >= thr6) return static_cast<int8_t>(e.six_q);
  const ap_int<34> v = static_cast<ap_int<34>>(acc) + bterm;
  ap_int<34> r = v >> e.n;
  if (e.nl != NL_NONE && r < 0) r = 0;
  if (r > 127) r = 127;
  if (r < -128) r = -128;
  return static_cast<int8_t>(r.to_int());
}

void epilogue(hls::stream<AccW>& inA, hls::stream<AccW>& inB,
              hls::stream<Chunk>& out_s, const int32_t bterm[MAX_OC],
              const int32_t thr6[MAX_OC],
              const int16_t obyte[TD][MAX_TILES], int njobs, int NT,
              int NOCT, int OC, EpiCfg e, int out_rows,
              int line_bytes_out) {
  static int8_t lbuf[16][LINE_MAX / 16];
#pragma HLS ARRAY_PARTITION variable = lbuf complete dim = 1
#pragma HLS BIND_STORAGE variable = lbuf type = ram_s2p impl = bram

  for (int job = 0; job < njobs; ++job) {
    const int n_words = NT * NOCT * TD2 * 2;
    int tile = 0, oct = 0, t = 0, sel = 0;  // same order as the MAC drain
    for (int k = 0; k < n_words; ++k) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = lbuf inter false
      const AccW in = sel ? inB.read() : inA.read();
      const int ob = obyte[t + sel * TD2][tile];
      const int base = ob + oct * OC_PAR;
      const int nvalid = OC - oct * OC_PAR;
      int8_t q8[OC_PAR];
      for (int j = 0; j < OC_PAR; ++j) {
#pragma HLS UNROLL
        const int oc = (oct * OC_PAR + j) & (MAX_OC - 1);
        q8[j] = epi_lane(in.v[j], bterm[oc], thr6[oc], e);
      }
      if (ob >= 0) {
        for (int b = 0; b < 16; ++b) {
#pragma HLS UNROLL
          const int j = (b - base) & 15;
          if (j < nvalid) lbuf[b][(base + j) >> 4] = q8[j];
        }
      }
      sel ^= 1;
      if (sel) continue;
      if (++t == TD2) {
        t = 0;
        if (++oct == NOCT) {
          oct = 0;
          ++tile;
        }
      }
    }
    for (int r = 0; r < out_rows; ++r) {
#pragma HLS PIPELINE II = 1
      const int rem = line_bytes_out - r * 16;
      Chunk c;
      for (int b = 0; b < 16; ++b) {
#pragma HLS UNROLL
        // bytes past the line end must be 0: the writer ORs chunks together
        c.data.range(8 * b + 7, 8 * b) =
            b < rem ? static_cast<ap_uint<8>>(lbuf[b][r]) : ap_uint<8>(0);
      }
      c.cnt = rem >= 16 ? 16 : rem;
      out_s.write(c);
    }
  }
}

// ---------------------------------------------------------------------------
// Stage 5: writer
// ---------------------------------------------------------------------------
void writer(hls::stream<Chunk>& in_s, w128* output, int n_chunks) {
  ap_uint<256> buf = 0;
  int fill = 0;  // bytes in buf, < 16 between chunks
  int widx = 0;
  for (int c = 0; c < n_chunks; ++c) {
#pragma HLS PIPELINE II = 1
    const Chunk ch = in_s.read();
    buf |= static_cast<ap_uint<256>>(ch.data) << (fill * 8);
    fill += ch.cnt;
    if (fill >= 16) {
      output[widx++] = static_cast<w128>(buf);
      buf >>= 128;
      fill -= 16;
    }
  }
  if (fill > 0) output[widx] = static_cast<w128>(buf);
}

void run_dataflow(const w128* input, const w128* weight, w128* output,
                  const int16_t tapH[MAX_OHW * MAX_K],
                  const int16_t tapW[MAX_OHW * MAX_K],
                  const int16_t tile_phase[MAX_TILES],
                  const int16_t pos[TD][MAX_TILES],
                  const int16_t obyte[TD][MAX_TILES],
                  const int32_t bterm[MAX_OC], const int32_t thr6[MAX_OC],
                  int N, int H, int W, int OH,
                  int OW, int KH, int KW, int line_bytes, int njobs, int LW,
                  int gf_bytes, int NOCT, int NRB, int NT, int w_rows, int OC,
                  EpiCfg e, int out_rows, int line_bytes_out,
                  int n_chunks) {
#pragma HLS DATAFLOW
  hls::stream<int32_t, 2 * MAX_PAIRS + 4> hdr_s;
  hls::stream<int32_t, 4> cnt_s;
  hls::stream<LineDesc, 8> desc_s;
  hls::stream<w128, 64> raw_s;
  hls::stream<w128, 32> data_s;
  hls::stream<AccW, 4> accA_s;
  hls::stream<AccW, 4> accB_s;
  hls::stream<Chunk, 4> chunk_s;
#pragma HLS BIND_STORAGE variable = hdr_s type = fifo impl = lutram
#pragma HLS BIND_STORAGE variable = desc_s type = fifo impl = srl
#pragma HLS BIND_STORAGE variable = accA_s type = fifo impl = srl
#pragma HLS BIND_STORAGE variable = accB_s type = fifo impl = srl
#pragma HLS BIND_STORAGE variable = chunk_s type = fifo impl = srl

#ifdef CONV3D_THREADED_SIM
  // test only: run the stages concurrently with bounded FIFOs (sim_threaded/)
  std::thread t0([&] {
    reader(input, tapH, tapW, hdr_s, cnt_s, desc_s, raw_s, N, H, W, OH, OW,
           KH, KW, line_bytes);
  });
  std::thread t1([&] {
    aligner(cnt_s, desc_s, raw_s, data_s, njobs, LW, gf_bytes, line_bytes);
  });
  std::thread t2([&] {
    mac_engine(weight, hdr_s, data_s, accA_s, accB_s, tile_phase, pos, njobs,
               KH * KW, NOCT, NRB, NT, LW, w_rows);
  });
  std::thread t3([&] {
    epilogue(accA_s, accB_s, chunk_s, bterm, thr6, obyte, njobs, NT, NOCT, OC,
             e, out_rows, line_bytes_out);
  });
  writer(chunk_s, output, n_chunks);
  t0.join(); t1.join(); t2.join(); t3.join();
#else
  reader(input, tapH, tapW, hdr_s, cnt_s, desc_s, raw_s, N, H, W, OH, OW, KH,
         KW, line_bytes);
  aligner(cnt_s, desc_s, raw_s, data_s, njobs, LW, gf_bytes, line_bytes);
  mac_engine(weight, hdr_s, data_s, accA_s, accB_s, tile_phase, pos, njobs,
             KH * KW, NOCT, NRB, NT, LW, w_rows);
  epilogue(accA_s, accB_s, chunk_s, bterm, thr6, obyte, njobs, NT, NOCT, OC,
           e, out_rows, line_bytes_out);
  writer(chunk_s, output, n_chunks);
#endif
}

}  // namespace

extern "C" {

void conv3d_fix(const w128* input, const w128* weight, const int32_t* tab,
                w128* output, int32_t* cfg) {
#pragma HLS INTERFACE m_axi port = input bundle = gmem0 offset = slave depth = 65536 max_read_burst_length = 64 num_read_outstanding = 16 max_widen_bitwidth = 128
#pragma HLS INTERFACE m_axi port = output bundle = gmem1 offset = slave depth = 65536 max_write_burst_length = 64 num_write_outstanding = 16 max_widen_bitwidth = 128
#pragma HLS INTERFACE m_axi port = weight bundle = gmem2 offset = slave depth = 16384 max_widen_bitwidth = 128
#pragma HLS INTERFACE m_axi port = tab bundle = gmem2 offset = slave depth = 8192 max_widen_bitwidth = 128
#pragma HLS INTERFACE m_axi port = cfg bundle = gmem2 offset = slave depth = 32 max_widen_bitwidth = 128
#pragma HLS INTERFACE s_axilite port = input bundle = control
#pragma HLS INTERFACE s_axilite port = weight bundle = control
#pragma HLS INTERFACE s_axilite port = tab bundle = control
#pragma HLS INTERFACE s_axilite port = output bundle = control
#pragma HLS INTERFACE s_axilite port = cfg bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

  int32_t c[CFG_WORDS];
  for (int i = 0; i < CFG_WORDS; ++i) {
#pragma HLS PIPELINE II = 1
    c[i] = cfg[i];
  }
  const int N = c[CFG_N], H = c[CFG_H], W = c[CFG_W], D = c[CFG_D];
  const int IC = c[CFG_IC];
  const int OH = c[CFG_OH], OW = c[CFG_OW], OD = c[CFG_OD], OC = c[CFG_OC];
  const int KH = c[CFG_KH], KW = c[CFG_KW];
  EpiCfg e;
  e.n = c[CFG_EPI_N];
  e.nl = c[CFG_NONLINEAR];
  e.six_q = c[CFG_SIX_Q];
  const int NOCT = c[CFG_NOCT], NRB = c[CFG_NRB], NT = c[CFG_NTILES];
  const int LW = c[CFG_LW], gf_bytes = c[CFG_GF_BYTES];
  const int w_rows = c[CFG_W_ROWS], out_rows = c[CFG_OUT_ROWS];

  // cheap sanity checks; the host validates everything in detail
  if (N < 1 || OH < 1 || OW < 1 || OD < 1 || OC < 1 || OC > MAX_OC ||
      KH < 1 || KW < 1 || KH > MAX_K || KW > MAX_K || OH > MAX_OHW ||
      OW > MAX_OHW || NT < 1 || NT > MAX_TILES || NOCT * OC_PAR < OC ||
      NOCT > MAX_NOCT || NT * NOCT > MAX_TN || NRB < 1 || w_rows > W_ROWS ||
      LW < 1 || LW > LINE_W || OD * OC > LINE_MAX || e.n < 0 || e.n > 31 ||
      out_rows * 16 < OD * OC) {
    cfg[CFG_STATUS] = ST_BAD_CFG;
    return;
  }

  // ---- tables (one private copy per consumer) ------------------------------
  static int16_t tapH[MAX_OHW * MAX_K];
  static int16_t tapW[MAX_OHW * MAX_K];
  static int16_t tile_phase[MAX_TILES];
  static int16_t pos[TD][MAX_TILES];
  static int16_t obyte[TD][MAX_TILES];
  static int32_t bterm[MAX_OC];
  static int32_t thr6[MAX_OC];
#pragma HLS ARRAY_PARTITION variable = pos complete dim = 1
#pragma HLS ARRAY_PARTITION variable = obyte complete dim = 1
#pragma HLS ARRAY_PARTITION variable = bterm cyclic factor = 16
#pragma HLS ARRAY_PARTITION variable = thr6 cyclic factor = 16
#pragma HLS BIND_STORAGE variable = tile_phase type = ram_2p impl = lutram
#pragma HLS BIND_STORAGE variable = pos type = ram_2p impl = lutram
#pragma HLS BIND_STORAGE variable = obyte type = ram_2p impl = lutram
#pragma HLS BIND_STORAGE variable = bterm type = ram_2p impl = lutram
#pragma HLS BIND_STORAGE variable = thr6 type = ram_2p impl = lutram

  // tab = [tapH | tapW | tile_phase | pos | obyte | bterm | thr6]
  const int e0 = OH * KH, e1 = e0 + OW * KW, e2 = e1 + NT, e3 = e2 + NT * TD,
            e4 = e3 + NT * TD, e5 = e4 + OC, e6 = e5 + OC;
  for (int i = 0; i < e6; ++i) {
#pragma HLS PIPELINE II = 1
    const int32_t v = tab[i];
    if (i < e0) {
      tapH[i] = v;
    } else if (i < e1) {
      tapW[i - e0] = v;
    } else if (i < e2) {
      tile_phase[i - e1] = v;
    } else if (i < e3) {
      pos[(i - e2) % TD][(i - e2) / TD] = v;
    } else if (i < e4) {
      obyte[(i - e3) % TD][(i - e3) / TD] = v;
    } else if (i < e5) {
      bterm[i - e4] = v;
    } else {
      thr6[i - e5] = v;
    }
  }

  const int njobs = N * OH * OW;
  run_dataflow(input, weight, output, tapH, tapW, tile_phase, pos, obyte,
               bterm, thr6, N, H, W, OH, OW, KH, KW, D * IC, njobs, LW,
               gf_bytes, NOCT, NRB, NT, w_rows, OC, e, out_rows, OD * OC,
               njobs * out_rows);
  cfg[CFG_STATUS] = ST_OK;
}

}  // extern "C"
