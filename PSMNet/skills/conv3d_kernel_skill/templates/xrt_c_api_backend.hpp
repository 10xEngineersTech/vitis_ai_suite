// Template: the host-side XRT backend that WORKS next to the Vitis-AI DPU
// runtime on KV260 / XRT 2.14. Lifted from the USE_XRT section of
// kernel/fpga/conv3d_fix_fpga_host.hpp -- copy from there, not from here,
// if the two ever differ.
//
// Why the XRT C API (xcl*) and not xrt::kernel / xrt::bo / xir::XrtCu:
//  * xrt::kernel(device, uuid, ...) -> "No xclbin with uuid ..." in a process
//    where vart opened the device first, even after device.load_xclbin().
//  * xir::XrtCu / xir::BufferObject: vart's XrtDeviceHandle only registers the
//    DPU's CUs (matched by name) -> 0 CUs found for our kernel.
//  * device.load_xclbin(other file) would reprogram the PL under the DPU.
// The xcl* calls below are the same ones vart uses for the DPU.
#pragma once
#include <ert.h>
#include <experimental/xrt-next.h>  // xclRegRead (timeout diagnostics)
#include <xclbin.h>
#include <xrt.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>

namespace myop_xrt {

// s_axilite register map: take the offsets from `xclbinutil --info`
// (Register Offset per argument) or the exported kernel.xml.
enum Reg : uint32_t {
  REG_GIE = 0x04, REG_IER = 0x08,
  REG_INPUT = 0x10, REG_WEIGHT = 0x1c, REG_TAB = 0x28, REG_OUTPUT = 0x34,
  REG_CFG = 0x40, REG_END = 0x48   // TODO: your kernel's offsets
};

struct XBo {  // cacheable, mapped buffer with its physical address
  xclDeviceHandle h; xclBufferHandle bo; void* p; uint64_t pa; size_t size;
  XBo(xclDeviceHandle handle, size_t sz, unsigned flags) : h(handle), size(sz) {
    bo = xclAllocBO(h, sz, 0, flags);
    CHECK(bo != NULLBO) << "xclAllocBO(" << sz << ") failed (CMA?)";
    p = xclMapBO(h, bo, true);
    CHECK(p != nullptr);
    xclBOProperties prop{};
    CHECK_EQ(xclGetBOProperties(h, bo, &prop), 0);
    pa = prop.paddr;
  }
  ~XBo() { xclUnmapBO(h, bo, p); xclFreeBO(h, bo); }
  void to_device(size_t n) { xclSyncBO(h, bo, XCL_BO_SYNC_BO_TO_DEVICE, n, 0); }
  void from_device(size_t n) { xclSyncBO(h, bo, XCL_BO_SYNC_BO_FROM_DEVICE, n, 0); }
};

// The xclbin the DPU runtime uses: $XLNX_VART_FIRMWARE, else /etc/vart.conf.
inline std::string vart_firmware() {
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

struct Shared {          // one CU context per process, shared by all layers
  std::mutex mtx;        // one CU -> serialize runs
  xclDeviceHandle h = nullptr;
  int cu_index = -1;
  std::unique_ptr<XBo> cmd, in, out;   // in/out grow-only, reused by layers
};

inline Shared& shared(const char* cu_name /* "myop_fix:myop_fix_1" */) {
  static Shared* s = [&] {
    auto* p = new Shared;
    const auto fw = vart_firmware();
    CHECK(!fw.empty()) << "no xclbin: XLNX_VART_FIRMWARE or /etc/vart.conf";
    axlf top{};  // uuid comes from the file the DPU runtime loaded
    std::ifstream f(fw, std::ios::binary);
    CHECK(f.read(reinterpret_cast<char*>(&top), sizeof(top)).good());
    p->h = xclOpen(0, nullptr, XCL_QUIET);
    CHECK(p->h != nullptr);
    p->cu_index = xclIPName2Index(p->h, cu_name);
    CHECK_GE(p->cu_index, 0) << "no " << cu_name << " CU in " << fw;
    CHECK_EQ(xclOpenContext(p->h, top.m_header.uuid, p->cu_index, true), 0)
        << "xclOpenContext failed: is " << fw << " the loaded xclbin?";
    p->cmd = std::make_unique<XBo>(p->h, 4096, XCL_BO_FLAGS_EXECBUF);
    xclIPSetReadRange(p->h, p->cu_index, 0x0, REG_END);  // for diagnostics
    return p;
  }();
  return *s;
}

inline void set64(ert_start_kernel_cmd* c, uint32_t reg, uint64_t v) {
  c->data[reg / 4] = static_cast<uint32_t>(v & 0xffffffff);
  c->data[reg / 4 + 1] = static_cast<uint32_t>(v >> 32);
}

// Start the CU and wait. Caller holds s.mtx and has synced all inputs.
inline void start_and_wait(Shared& s, uint64_t a_in, uint64_t a_w,
                           uint64_t a_tab, uint64_t a_out, uint64_t a_cfg) {
  auto* c = static_cast<ert_start_kernel_cmd*>(s.cmd->p);
  std::memset(c, 0, 4096);
  c->state = ERT_CMD_STATE_NEW;
  c->opcode = ERT_START_CU;           // type left 0, as vart's DPU controller
  c->count = 1 + REG_END / 4;         // cu_mask word + register words
  c->cu_mask = 1u << s.cu_index;
  c->data[REG_GIE / 4] = 1;
  c->data[REG_IER / 4] = 1;
  set64(c, REG_INPUT, a_in);  set64(c, REG_WEIGHT, a_w);  set64(c, REG_TAB, a_tab);
  set64(c, REG_OUTPUT, a_out); set64(c, REG_CFG, a_cfg);
  CHECK_EQ(xclExecBuf(s.h, s.cmd->bo), 0);
  const auto t0 = std::chrono::steady_clock::now();
  const char* tmo = std::getenv("VAI_MYOP_FPGA_TIMEOUT_MS");
  const long timeout_ms = tmo ? std::atol(tmo) : 30000;
  while (c->state < ERT_CMD_STATE_COMPLETED) {
    xclExecWait(s.h, 1000);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0).count();
    if (ms >= timeout_ms) {  // ap_ctrl 0x1 = started, never done -> kernel hang
      uint32_t ctrl = 0; xclRegRead(s.h, s.cu_index, 0x00, &ctrl);
      LOG(FATAL) << "timeout: cmd state " << c->state << " cu_index "
                 << s.cu_index << " ap_ctrl 0x" << std::hex << ctrl;
    }
  }
  CHECK_EQ(int(c->state), int(ERT_CMD_STATE_COMPLETED));
}

}  // namespace myop_xrt
