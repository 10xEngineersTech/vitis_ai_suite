// Template: thin demo (pattern of application/demo_psmnet.cpp). No model,
// tensor or fix-point logic here: create, run, get_result, save.
#include <glog/logging.h>

#include <opencv2/imgcodecs.hpp>
#include <string>

#include "./mynet.hpp"

int main(int argc, char* argv[]) {
  CHECK_GE(argc, 3) << "usage: " << argv[0] << " <left image> <right image>";
  cv::Mat l = cv::imread(argv[1]), r = cv::imread(argv[2]);
  CHECK(!l.empty() && !r.empty()) << "cannot read the input images";

  auto net = vitis::ai::MyNet::create();
  std::vector<std::pair<cv::Mat, cv::Mat>> imgs(net->get_input_batch(), {l, r});
  net->run(imgs);
  auto results = net->get_result();
  for (size_t i = 0; i < results.size(); ++i)
    cv::imwrite("result_mynet_" + std::to_string(i) + ".tiff", results[i]);
  LOG(INFO) << "done";
  return 0;
}
