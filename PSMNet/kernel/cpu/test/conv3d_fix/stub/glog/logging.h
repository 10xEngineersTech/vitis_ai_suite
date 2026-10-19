/*
 * Minimal glog replacement so ops/common/{util,conv3d_fix_kernel}.hpp can be
 * compiled on a build host that has no glog installed. Test-only.
 */
#pragma once
#include <cstdlib>
#include <iostream>
#include <sstream>

namespace vai_test_stub {
class Check {
 public:
  explicit Check(bool failed) : failed_(failed) {}
  ~Check() {
    if (failed_) {
      std::cerr << "CHECK failed: " << os_.str() << std::endl;
      std::abort();
    }
  }
  template <typename T>
  Check& operator<<(const T& v) {
    if (failed_) os_ << v;
    return *this;
  }

 private:
  bool failed_;
  std::ostringstream os_;
};
}  // namespace vai_test_stub

#define LOG(severity) ::vai_test_stub::Check(true)
#define CHECK(cond) ::vai_test_stub::Check(!(cond)) << #cond << " "
#define CHECK_EQ(a, b) ::vai_test_stub::Check(!((a) == (b))) << #a " == " #b " "
#define CHECK_NE(a, b) ::vai_test_stub::Check(!((a) != (b))) << #a " != " #b " "
#define CHECK_GT(a, b) ::vai_test_stub::Check(!((a) > (b))) << #a " > " #b " "
#define CHECK_GE(a, b) ::vai_test_stub::Check(!((a) >= (b))) << #a " >= " #b " "
#define CHECK_LT(a, b) ::vai_test_stub::Check(!((a) < (b))) << #a " < " #b " "
#define CHECK_LE(a, b) ::vai_test_stub::Check(!((a) <= (b))) << #a " <= " #b " "
