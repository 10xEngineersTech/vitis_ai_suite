// Test-only replacement for <hls_stream.h>: a bounded, blocking FIFO with the
// real DEPTH, used with -DCONV3D_THREADED_SIM so the DATAFLOW stages run in
// parallel threads like the hardware. A stage blocked for longer than
// CONV3D_SIM_STALL_MS (default 5000) is reported as a deadlock -- the plain
// C simulation (sequential stages, unbounded FIFOs) cannot see those.
#pragma once
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <mutex>

namespace hls {
// stream<T> (depth 0 = base, as in Vitis HLS) is what functions take by
// reference; stream<T, DEPTH> derives from it and only sets the depth.
template <typename T, int DEPTH = 0>
class stream;

template <typename T>
class stream<T, 0> {
 public:
  stream() = default;
  explicit stream(const char*) {}
  void write(const T& v) {
    std::unique_lock<std::mutex> l(m_);
    if (!cv_.wait_for(l, stall(), [&] { return (int)q_.size() < depth_; }))
      deadlock("write (FIFO full)");
    q_.push_back(v);
    cv_.notify_all();
  }
  T read() {
    std::unique_lock<std::mutex> l(m_);
    if (!cv_.wait_for(l, stall(), [&] { return !q_.empty(); }))
      deadlock("read (FIFO empty)");
    T v = q_.front();
    q_.pop_front();
    cv_.notify_all();
    return v;
  }
  bool empty() {
    std::lock_guard<std::mutex> l(m_);
    return q_.empty();
  }

 protected:
  int depth_ = 2;  // Vitis HLS default FIFO depth

 private:
  static std::chrono::milliseconds stall() {
    const char* e = std::getenv("CONV3D_SIM_STALL_MS");
    return std::chrono::milliseconds(e ? std::atol(e) : 5000);
  }
  [[noreturn]] void deadlock(const char* what) {
    std::fprintf(stderr, "DEADLOCK: a dataflow stage is stuck on %s of a "
                         "stream (depth %d)\n", what, depth_);
    std::fflush(stderr);
    std::_Exit(3);
  }
  std::deque<T> q_;
  std::mutex m_;
  std::condition_variable cv_;
};

template <typename T, int DEPTH>
class stream : public stream<T, 0> {
 public:
  stream() { this->depth_ = DEPTH; }
  explicit stream(const char*) { this->depth_ = DEPTH; }
};
}  // namespace hls
