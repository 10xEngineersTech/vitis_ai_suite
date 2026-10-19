// Template: public interface of a multi-model network class (pattern of
// application/psmnet.hpp). The demo only sees this header.
#pragma once
#include <memory>
#include <opencv2/core.hpp>
#include <utility>
#include <vector>

namespace vitis {
namespace ai {

class MyNet {
 public:
  static std::unique_ptr<MyNet> create();  // loads every model, allocates once

 protected:
  explicit MyNet();
  MyNet(const MyNet&) = delete;
  MyNet& operator=(const MyNet&) = delete;

 public:
  virtual ~MyNet();
  virtual void run(const std::vector<std::pair<cv::Mat, cv::Mat>>& imgs) = 0;
  virtual std::vector<cv::Mat> get_result() = 0;
  virtual size_t get_input_batch() = 0;
  virtual int get_input_width() const = 0;
  virtual int get_input_height() const = 0;
};

}  // namespace ai
}  // namespace vitis
