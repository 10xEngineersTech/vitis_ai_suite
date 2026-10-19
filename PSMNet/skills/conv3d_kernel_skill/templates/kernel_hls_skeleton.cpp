// Template: structure of a DPU-companion HLS kernel. NOT compilable as is.
// Real, complete example: kernel/fpga/conv3d_fix_hls.cpp (+ conv3d_fix_hls.h)
#include "MYOP_hls.h"   // shared with the host: limits, CfgIdx, table layout,
                        // extern "C" prototype, typedef ap_uint<128> w128
#include <hls_stream.h> // -Isim_threaded first -> bounded test FIFOs
#ifdef MYOP_THREADED_SIM
#include <thread>
#endif

using namespace myop_hls;

// Everything except the extern "C" entry point lives in an anonymous
// namespace: internal linkage, no clash with other kernels / host code.
namespace {

void reader(const w128* input, /* tables */ hls::stream<int32_t>& hdr_s,
            hls::stream<w128>& raw_s /* ... */);      // burst reads
void aligner(/* ... */ hls::stream<w128>& raw_s, hls::stream<w128>& data_s);
void mac_engine(const w128* weight, hls::stream<int32_t>& hdr_s,
                hls::stream<w128>& data_s /* ... */) {
  // ...
  for (int it = 0; it < iters; ++it) {
#pragma HLS PIPELINE II = 1
    // Read-modify-write accumulators: declare the REAL minimum distance
    // between two updates of the same entry (conv3d: NOCT*8 >= 8). Never
    // "inter false" unless it is true for every configuration.
#pragma HLS DEPENDENCE variable = accA type = inter direction = RAW distance = 8 dependent = true
    // ...
  }
}
void epilogue(/* ... */);
void writer(/* ... */ w128* output, int n_chunks);

void run_dataflow(/* ports, tables, scalars */) {
#pragma HLS DATAFLOW
  // Depths are part of correctness: run the threaded sim with these values.
  hls::stream<int32_t, 2 * MAX_PAIRS + 4> hdr_s;
  hls::stream<w128, 64> raw_s;
  hls::stream<w128, 32> data_s;
#ifdef MYOP_THREADED_SIM
  std::thread t0([&] { reader(/* ... */); });
  std::thread t1([&] { aligner(/* ... */); });
  std::thread t2([&] { mac_engine(/* ... */); });
  std::thread t3([&] { epilogue(/* ... */); });
  writer(/* ... */);
  t0.join(); t1.join(); t2.join(); t3.join();
#else
  reader(/* ... */);
  aligner(/* ... */);
  mac_engine(/* ... */);
  epilogue(/* ... */);
  writer(/* ... */);
#endif
}

}  // namespace

extern "C" {
void myop_fix(const w128* input, const w128* weight, const int32_t* tab,
              w128* output, int32_t* cfg) {
  // 128-bit ports to match the KV260 HP ports. Without max_widen_bitwidth the
  // IDE (v++ -c) flow widened a bundle to 512 bit -> downsizer in front of HP2
  // -> 511 failing timing endpoints.
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
  // Cheap up-front sanity check: never start the dataflow on a config that
  // would index past an on-chip array. The host validates in detail.
  if (/* any c[...] outside MAX_* limits */ false) {
    cfg[CFG_STATUS] = ST_BAD_CFG;
    return;
  }
  // Copy the small tables from tab[] into on-chip arrays (one copy per
  // consuming dataflow stage), then:
  run_dataflow(/* ... */);
  cfg[CFG_STATUS] = ST_OK;
}
}  // extern "C"
