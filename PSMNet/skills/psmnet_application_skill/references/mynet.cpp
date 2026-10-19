// Template: implementation of a network split into a DpuTask model (DPU only,
// CPU op between its subgraphs) and a GraphRunner model (DPU + cpu_task op
// subgraphs). Pattern of application/psmnet.cpp, trimmed to one of each.
// Replace the layer names, the CPU glue and the post-processing.
#include "./mynet.hpp"

#include <glog/logging.h>

#include <cmath>
#include <cstring>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vitis/ai/dpu_task.hpp>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/graph_runner.hpp>
#include <xir/attrs/attrs.hpp>

// Model paths relative to the working directory; override per run.
DEF_ENV_PARAM_2(MYNET_MODEL_0, "mynet_0/mynet_0.xmodel", std::string);
DEF_ENV_PARAM_2(MYNET_MODEL_1, "mynet_1/mynet_1.xmodel", std::string);

using namespace std;

namespace vitis {
namespace ai {

// GraphRunner (NHWC) tensor buffer -> library Input/OutputTensor.
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

// Tensor order is not stable across compiles: match by layer-name substring.
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

// int8 copy between two tensors with different fix points (round half up,
// saturate). Scalar for readability; application/psmnet.cpp has the NEON
// version (rounding shift right, wrapping shift left).
static void copy_into_tensor(const vitis::ai::library::OutputTensor& in,
                             const vitis::ai::library::InputTensor& out) {
  const size_t size = out.height * out.width * out.channel;
  const float scale = exp2f(out.fixpos - in.fixpos);
  for (size_t b = 0; b < out.batch; ++b) {
    auto src = (const int8_t*)in.get_data(b);
    auto dst = (int8_t*)out.get_data(b);
    if (in.fixpos == out.fixpos) {
      memcpy(dst, src, size);
      continue;
    }
    for (size_t i = 0; i < size; ++i) {
      const float v = src[i] * scale;
      const float r = v - floor(v) == 0.5f ? ceil(v) : round(v);
      dst[i] = (int8_t)std::max(-128.f, std::min(127.f, r));
    }
  }
}

class MyNetImp : public MyNet {
 public:
  MyNetImp();
  void run(const vector<pair<cv::Mat, cv::Mat>>& imgs) override;
  vector<cv::Mat> get_result() override;
  size_t get_input_batch() override { return task0_->get_input_batch(0, 0); }
  int get_input_width() const override {
    return task0_->getInputTensor(0u)[0].width;
  }
  int get_input_height() const override {
    return task0_->getInputTensor(0u)[0].height;
  }

 private:
  // The class owns every model, tensor view and buffer: allocate in the
  // constructor, never per frame.
  std::unique_ptr<vitis::ai::DpuTask> task0_;
  std::unique_ptr<xir::Graph> graph1_;
  std::unique_ptr<xir::Attrs> attrs1_;
  std::unique_ptr<vart::RunnerExt> runner1_;
  vitis::ai::library::InputTensor in1_;
  vitis::ai::library::OutputTensor out1_;
  vector<vitis::ai::library::OutputTensor> sub0_outs_;
  vector<vitis::ai::library::InputTensor> sub1_ins_;
};

MyNet::MyNet() {}
MyNet::~MyNet() {}
std::unique_ptr<MyNet> MyNet::create() { return std::make_unique<MyNetImp>(); }

MyNetImp::MyNetImp() {
  task0_ = vitis::ai::DpuTask::create(ENV_PARAM(MYNET_MODEL_0));
  task0_->setMeanScaleBGR({103.53, 116.28, 123.675},
                          {0.017429, 0.017507, 0.01712475});

  graph1_ = xir::Graph::deserialize(ENV_PARAM(MYNET_MODEL_1));
  attrs1_ = xir::Attrs::create();
  runner1_ = vitis::ai::GraphRunner::create_graph_runner(graph1_.get(),
                                                         attrs1_.get());
  CHECK(runner1_ != nullptr);
  CHECK(runner1_->get_tensor_format() == vart::Runner::TensorFormat::NHWC);
  in1_ = to_tensor<vitis::ai::library::InputTensor>(runner1_->get_inputs()[0]);
  out1_ = to_tensor<vitis::ai::library::OutputTensor>(runner1_->get_outputs()[0]);

  // Subgraph 0 output -> subgraph 1 input of the DpuTask model.
  sub0_outs_ = sort_tensors(task0_->getOutputTensor(0u), {"layer_a__Add"});
  sub1_ins_ = sort_tensors(task0_->getInputTensor(1u), {"layer_a__Add"});
}

void MyNetImp::run(const vector<pair<cv::Mat, cv::Mat>>& imgs) {
  auto in = task0_->getInputTensor(0u)[0];
  vector<cv::Mat> left;
  for (size_t i = 0; i < in.batch; ++i) {
    cv::Mat l;
    cv::resize(imgs[i].first, l, cv::Size(in.width, in.height));
    left.push_back(l);
  }
  task0_->setImageRGB(left);
  task0_->run(0u);
  copy_into_tensor(sub0_outs_[0], sub1_ins_[0]);  // CPU glue between subgraphs
  task0_->run(1u);

  // ... fill in1_ from task0_->getOutputTensor(1u) on the CPU ...

  auto inputs = runner1_->get_inputs();
  auto outputs = runner1_->get_outputs();
  for (auto t : inputs)
    t->sync_for_write(0, t->get_tensor()->get_data_size() /
                             t->get_tensor()->get_shape()[0]);
  auto job = runner1_->execute_async(inputs, outputs);
  CHECK_EQ(job.second, 0);
  CHECK_EQ(runner1_->wait(job.first, -1), 0);
  for (auto t : outputs)
    t->sync_for_read(0, t->get_tensor()->get_data_size() /
                            t->get_tensor()->get_shape()[0]);
}

vector<cv::Mat> MyNetImp::get_result() {
  vector<cv::Mat> rets;
  for (size_t b = 0; b < out1_.batch; ++b) {
    // float output when the graph ends in fix2float, else int8 at out1_.fixpos
    cv::Mat m(out1_.height, out1_.width, CV_32FC(1), out1_.get_data(b));
    rets.push_back(m.clone());
  }
  return rets;
}

}  // namespace ai
}  // namespace vitis
