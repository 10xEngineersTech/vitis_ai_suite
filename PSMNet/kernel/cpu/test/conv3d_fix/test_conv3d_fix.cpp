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
 * Host-side driver for ops/common/conv3d_fix_kernel.hpp.
 *
 * It links no VART/XIR, so the exact arithmetic that the conv3d-fix and
 * transposed-conv3d-fix plugins run can be diffed against a PyTorch reference
 * on the build machine. Driven by gen_and_check.py.
 *
 *   test_conv3d_fix conv <case_dir>            run one convolution case
 *   test_conv3d_fix fix  <in.f32> <out.i32>    run util::fix over raw floats
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "../../ops/common/conv3d_fix_kernel.hpp"

using namespace vitis::ai::cpu_task;

namespace {

std::vector<char> read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::cerr << "cannot open " << path << std::endl;
    std::exit(2);
  }
  return std::vector<char>((std::istreambuf_iterator<char>(f)),
                           std::istreambuf_iterator<char>());
}

void write_file(const std::string& path, const void* data, size_t n) {
  std::ofstream f(path, std::ios::binary);
  f.write(static_cast<const char*>(data), n);
}

/// case.txt is "<key> <int|string>..." one per line.
std::map<std::string, std::vector<std::string>> read_case(
    const std::string& path) {
  std::map<std::string, std::vector<std::string>> kv;
  std::ifstream f(path);
  if (!f) {
    std::cerr << "cannot open " << path << std::endl;
    std::exit(2);
  }
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream is(line);
    std::string key, tok;
    is >> key;
    while (is >> tok) kv[key].push_back(tok);
  }
  return kv;
}

std::vector<int> ints(const std::map<std::string, std::vector<std::string>>& kv,
                      const std::string& key) {
  auto it = kv.find(key);
  if (it == kv.end()) {
    std::cerr << "missing key " << key << std::endl;
    std::exit(2);
  }
  std::vector<int> out;
  for (auto& s : it->second) out.push_back(std::stoi(s));
  return out;
}

std::string str(const std::map<std::string, std::vector<std::string>>& kv,
                const std::string& key, const std::string& dflt) {
  auto it = kv.find(key);
  if (it == kv.end() || it->second.empty()) return dflt;
  return it->second[0];
}

int run_conv(const std::string& dir) {
  auto kv = read_case(dir + "/case.txt");
  conv3d::Param p;
  p.transposed = ints(kv, "transposed")[0] != 0;
  p.kernel = ints(kv, "kernel");
  p.stride = ints(kv, "stride");
  p.dilation = ints(kv, "dilation");
  p.pad = ints(kv, "pad");
  p.pad_mode = str(kv, "pad_mode", "FLOOR");
  p.group = ints(kv, "group")[0];
  p.input_shape = ints(kv, "input_shape");
  p.weight_shape = ints(kv, "weight_shape");
  p.output_shape = ints(kv, "output_shape");
  p.nonlinear = util::get_nonlinear(str(kv, "nonlinear", "NONE"));
  p.num_threads = ints(kv, "threads")[0];

  auto input_fp = ints(kv, "input_fp")[0];
  auto weight_fp = ints(kv, "weight_fp")[0];
  auto output_fp = ints(kv, "output_fp")[0];
  p.set_fix_point(input_fp, weight_fp, output_fp);

  auto input = read_file(dir + "/input.bin");
  auto weight = read_file(dir + "/weight.bin");
  std::vector<char> bias;
  const int8_t* bias_ptr = nullptr;
  if (ints(kv, "has_bias")[0]) {
    bias = read_file(dir + "/bias.bin");
    bias_ptr = reinterpret_cast<const int8_t*>(bias.data());
    p.set_bias_fix_point(input_fp, weight_fp, ints(kv, "bias_fp")[0]);
  }

  size_t out_n = 1;
  for (auto d : p.output_shape) out_n *= static_cast<size_t>(d);
  std::vector<int8_t> output(out_n, 0);

  conv3d::Kernel(p).run(reinterpret_cast<const int8_t*>(input.data()),
                        reinterpret_cast<const int8_t*>(weight.data()),
                        bias_ptr, output.data());
  write_file(dir + "/got.bin", output.data(), output.size());
  return 0;
}

int run_fix(const std::string& in_path, const std::string& out_path) {
  auto raw = read_file(in_path);
  auto n = raw.size() / sizeof(float);
  const float* in = reinterpret_cast<const float*>(raw.data());
  std::vector<int32_t> out(n);
  for (size_t i = 0; i < n; ++i) out[i] = util::fix(in[i]);
  write_file(out_path, out.data(), out.size() * sizeof(int32_t));
  return 0;
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc >= 3 && std::strcmp(argv[1], "conv") == 0) {
    return run_conv(argv[2]);
  }
  if (argc >= 4 && std::strcmp(argv[1], "fix") == 0) {
    return run_fix(argv[2], argv[3]);
  }
  std::cerr << "usage: " << argv[0] << " conv <case_dir>\n"
            << "       " << argv[0] << " fix <in.f32> <out.i32>" << std::endl;
  return 1;
}
