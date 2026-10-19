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
#include "./psmnet.hpp"
#include "./cost_volume.hpp"
#include "./lut_sfm.hpp"
#include "./resize.hpp"
#include <arm_neon.h>

#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/dpu_task.hpp>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/graph_runner.hpp>
#include <xir/attrs/attrs.hpp>

DEF_ENV_PARAM_2(PSMNET_MODEL_0, "PSMNet_pruned_0_pt/PSMNet_pruned_0_pt.xmodel", std::string);
DEF_ENV_PARAM_2(PSMNET_MODEL_1, "PSMNet_pruned_1_pt/PSMNet_pruned_1_pt.xmodel", std::string);
DEF_ENV_PARAM_2(PSMNET_MODEL_2, "PSMNet_pruned_2_pt/PSMNet_pruned_2_pt.xmodel", std::string);
// Fix point of kernel 2's logits for LutSfm; -1 = read it from the model.
DEF_ENV_PARAM(PSMNET_SFM_FIX, "-1");

using namespace std;

namespace vitis {
namespace ai {

// Wraps a GraphRunner (NHWC) tensor buffer as a library Input/OutputTensor.
template <typename T>
static T to_tensor(vart::TensorBuffer* tb) {
  T ret{};
  auto tensor = tb->get_tensor();
  auto dims = tensor->get_shape();
  ret.batch = dims[0];
  ret.size = tensor->get_element_num() *
             std::ceil(tensor->get_data_type().bit_width / 8.f);
  ret.height = dims.size() > 1 ? dims[1] : 1;
  ret.width = dims.size() > 2 ? dims[2] : 1;
  ret.channel = 1;
  for (auto i = 3u; i < dims.size(); ++i) ret.channel *= dims[i];
  if (tensor->get_data_type().type == xir::DataType::XINT) {
    ret.fixpos = tensor->template get_attr<int>("fix_point");
    ret.dtype = vitis::ai::library::DT_INT8;
  } else {
    ret.fixpos = 0;
    ret.dtype = vitis::ai::library::DT_FLOAT;
  }
  ret.name = tensor->get_name();
  for (auto b = 0; b < dims[0]; ++b) {
    auto idx = std::vector<int32_t>(dims.size(), 0);
    idx[0] = b;
    ret.get_data(b) = (void*)tb->data(idx).first;
  }
  return ret;
}

// Reorders tensors to follow layer_names (matched as substrings).
template <typename T>
static vector<T> sort_tensors(const vector<T>& tensors,
                              const vector<string>& layer_names) {
  vector<T> ordered;
  for (auto& name : layer_names)
    for (auto& t : tensors)
      if (t.name.find(name) != std::string::npos) {
        ordered.push_back(t);
        break;
      }
  CHECK_EQ(ordered.size(), layer_names.size()) << "missing kernel tensors";
  return ordered;
}

// Copies an int8 DPU output into an int8 DPU input, requantised from the
// output's fix point to the input's (shift = in.fixpos - out.fixpos):
//   shift > 0: rounding shift right (round half up, as the DPU does)
//   shift < 0: shift left, keeping the low 8 bits
// 16 bytes per NEON instruction; a tail shorter than 16 runs on a padded copy.
static void copy_into_tensor(const vitis::ai::library::OutputTensor& in,
                             const vitis::ai::library::InputTensor& out) {
  const size_t size = out.height * out.width * out.channel;
  const int shift = in.fixpos - out.fixpos;
  const int8x16_t cnt = vdupq_n_s8(static_cast<int8_t>(-shift));
  auto requant16 = [&](int8x16_t v) {
    return shift > 0 ? vrshlq_s8(v, cnt) : vshlq_s8(v, cnt);
  };
  for (size_t b = 0; b < out.batch; ++b) {
    auto src = static_cast<const int8_t*>(in.get_data(b));
    auto dst = (int8_t*)out.get_data(b);
    if (shift == 0) {
      memcpy(dst, src, size);
      continue;
    }
    size_t i = 0;
    for (; i + 16 <= size; i += 16)
      vst1q_s8(dst + i, requant16(vld1q_s8(src + i)));
    if (i < size) {
      int8_t t[16] = {};
      memcpy(t, src + i, size - i);
      vst1q_s8(t, requant16(vld1q_s8(t)));
      memcpy(dst + i, t, size - i);
    }
  }
}

// Fix point of kernel 2's final logits: the int8 tensor behind the model's
// output (a trailing fix2float reads it; a trailing fix op produces it).
// -1 if the graph ends in something else.
static int logits_fix_point(const xir::Graph* g) {
  for (auto op : g->get_tail_ops()) {
    const xir::Tensor* t = nullptr;
    if (op->get_type() == "fix2float")
      t = op->get_input_tensor("input");
    else if (op->get_type() == "fix")
      t = op->get_output_tensor();
    if (t && t->has_attr("fix_point")) return t->get_attr<int>("fix_point");
    if (op->get_type() == "fix" && op->has_attr("fix_point"))
      return op->get_attr<int>("fix_point");
  }
  return -1;
}

class PsmNetImp : public PsmNet {
 public:
  PsmNetImp();
  void run(const std::vector<std::pair<cv::Mat, cv::Mat>>& imgs) override;
  std::vector<cv::Mat> get_result() override;
  size_t get_input_batch() override { return tasks_[0]->get_input_batch(0, 0); }
  int get_input_width() const override {
    return tasks_[0]->getInputTensor(0u)[0].width;
  }
  int get_input_height() const override {
    return tasks_[0]->getInputTensor(0u)[0].height;
  }

 private:
  std::vector<std::unique_ptr<vitis::ai::DpuTask>> tasks_;
  std::unique_ptr<xir::Graph> graph2_;
  std::unique_ptr<xir::Attrs> attrs2_;
  std::unique_ptr<vart::RunnerExt> graph_runner2_;
  vitis::ai::library::InputTensor input_tensor_k2_;
  vitis::ai::library::OutputTensor final_tensor2_;
  vector<vitis::ai::library::InputTensor> inputs_l_, inputs_r_;
  vector<vitis::ai::library::OutputTensor> outputs_l_, outputs_r_;
  std::vector<std::unique_ptr<Resize>> cpu_res_;
  std::unique_ptr<CostVolume> cost_volume_;
  std::unique_ptr<LutSfm> lut_sfm_;  // softmax + disparity regression
};

PsmNet::PsmNet() {}
PsmNet::~PsmNet() {}
std::unique_ptr<PsmNet> PsmNet::create() {
  return std::make_unique<PsmNetImp>();
}

PsmNetImp::PsmNetImp() {
  for (auto& m : {ENV_PARAM(PSMNET_MODEL_0), ENV_PARAM(PSMNET_MODEL_1)}) {
    tasks_.emplace_back(vitis::ai::DpuTask::create(m));
    tasks_.back()->setMeanScaleBGR({103.53, 116.28, 123.675},
                                   {0.017429, 0.017507, 0.01712475});
  }

  graph2_ = xir::Graph::deserialize(ENV_PARAM(PSMNET_MODEL_2));
  attrs2_ = xir::Attrs::create();
  graph_runner2_ =
      vitis::ai::GraphRunner::create_graph_runner(graph2_.get(), attrs2_.get());
  CHECK(graph_runner2_ != nullptr);
  CHECK(graph_runner2_->get_tensor_format() == vart::Runner::TensorFormat::NHWC);
  input_tensor_k2_ = to_tensor<vitis::ai::library::InputTensor>(
      graph_runner2_->get_inputs()[0]);
  final_tensor2_ = to_tensor<vitis::ai::library::OutputTensor>(
      graph_runner2_->get_outputs()[0]);

  // Kernel 2 output is [H][W][D] (released model) or [D][H][W] (recompiled
  // model ending in a transpose). Describe it as (H, W, D) either way.
  const auto& img = tasks_[0]->getInputTensor(0u)[0];
  bool disp_first = false;
  if (final_tensor2_.height != img.height || final_tensor2_.width != img.width) {
    CHECK(final_tensor2_.width == img.height && final_tensor2_.channel == img.width)
        << "unexpected kernel 2 output layout";
    disp_first = true;
    final_tensor2_.channel = final_tensor2_.height;
    final_tensor2_.height = img.height;
    final_tensor2_.width = img.width;
  }

  // Matched by layer name; branchN feeds upsampleN.
  const vector<string> task0_outs = {
      "layer2__BasicBlock_15__Add", "layer4__BasicBlock_2__Add",
      "branch4__", "branch3__", "branch2__", "branch1__"};
  const vector<string> task1_inps = {
      "layer2__BasicBlock_15__Add", "layer4__BasicBlock_2__Add",
      "upsample4__", "upsample3__", "upsample2__", "upsample1__"};
  outputs_l_ = sort_tensors(tasks_[0]->getOutputTensor(0u), task0_outs);
  inputs_l_ = sort_tensors(tasks_[0]->getInputTensor(1u), task1_inps);
  outputs_r_ = sort_tensors(tasks_[1]->getOutputTensor(0u), task0_outs);
  inputs_r_ = sort_tensors(tasks_[1]->getInputTensor(1u), task1_inps);

  for (auto i : {2, 3, 4, 5})
    cpu_res_.emplace_back(std::make_unique<Resize>(outputs_l_[i], inputs_l_[i]));
  for (auto i : {2, 3, 4, 5})
    cpu_res_.emplace_back(std::make_unique<Resize>(outputs_r_[i], inputs_r_[i]));
  cost_volume_ = std::make_unique<CostVolume>(tasks_[0]->getOutputTensor(1u)[0],
                                              tasks_[1]->getOutputTensor(1u)[0],
                                              input_tensor_k2_);
  int fix = ENV_PARAM(PSMNET_SFM_FIX);
  if (fix < 0) fix = logits_fix_point(graph2_.get());
  LOG_IF(WARNING, fix < 0 && final_tensor2_.dtype == vitis::ai::library::DT_FLOAT)
      << "logits fix point unknown: LutSfm falls back to expf "
         "(set PSMNET_SFM_FIX)";
  lut_sfm_ = std::make_unique<LutSfm>(final_tensor2_, disp_first, fix);
}

void PsmNetImp::run(const vector<pair<cv::Mat, cv::Mat>>& imgs) {
  auto in = tasks_[0]->getInputTensor(0u)[0];
  vector<cv::Mat> left, right;
  for (size_t i = 0; i < in.batch; ++i) {
    cv::Mat l, r;
    cv::resize(imgs[i].first, l, cv::Size(in.width, in.height));
    cv::resize(imgs[i].second, r, cv::Size(in.width, in.height));
    left.push_back(l);
    right.push_back(r);
  }

  // Feature extraction: kernel 0 (left) and kernel 1 (right), each two DPU
  // subgraphs with the upsampling (Resize) on the CPU in between.
  tasks_[0]->setImageRGB(left);
  tasks_[0]->run(0u);
  for (auto i : {0, 1, 2, 3}) cpu_res_[i]->run();
  copy_into_tensor(outputs_l_[0], inputs_l_[0]);
  copy_into_tensor(outputs_l_[1], inputs_l_[1]);
  tasks_[0]->run(1u);

  tasks_[1]->setImageRGB(right);
  tasks_[1]->run(0u);
  for (auto i : {4, 5, 6, 7}) cpu_res_[i]->run();
  copy_into_tensor(outputs_r_[0], inputs_r_[0]);
  copy_into_tensor(outputs_r_[1], inputs_r_[1]);
  tasks_[1]->run(1u);

  cost_volume_->run();

  // Kernel 2: 3D hourglass, DPU + CPU subgraphs.
  auto inputs = graph_runner2_->get_inputs();
  auto outputs = graph_runner2_->get_outputs();
  for (auto t : inputs)
    t->sync_for_write(0, t->get_tensor()->get_data_size() /
                             t->get_tensor()->get_shape()[0]);
  auto job = graph_runner2_->execute_async(inputs, outputs);
  CHECK_EQ(job.second, 0);
  CHECK_EQ(graph_runner2_->wait(job.first, -1), 0);
  for (auto t : outputs)
    t->sync_for_read(0, t->get_tensor()->get_data_size() /
                            t->get_tensor()->get_shape()[0]);

  // Softmax + disparity regression on the CPU (exp lookup table).
  lut_sfm_->run();
}

std::vector<cv::Mat> PsmNetImp::get_result() {
  std::vector<cv::Mat> rets;
  const float* disp = lut_sfm_->get_output();
  const auto H = final_tensor2_.height, W = final_tensor2_.width;
  for (size_t b = 0; b < final_tensor2_.batch; ++b) {
    cv::Mat ret(H, W, CV_8UC3);
    for (auto h = 0u; h < H; ++h)
      for (auto w = 0u; w < W; ++w) {
        uint8_t gray = (int)disp[(b * H + h) * W + w] * 1.7;
        ret.at<cv::Vec3b>(h, w) = cv::Vec3b(gray, gray, gray);
      }
    rets.push_back(ret);
  }
  return rets;
};

} // namespace ai
} // namespace vitis