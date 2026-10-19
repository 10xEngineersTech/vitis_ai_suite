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
// Live stereo demo: two USB cameras -> PSMNet -> display via GStreamer.
//
// Threads:
//   capture x2 : keep the latest frame of each camera (drops stale frames)
//   inference  : take the newest L/R pair, run PSMNet (~8 s), publish result,
//                and immediately take the next pair
//   main/gui   : compose live views + last input + last disparity and push
//                them to a GStreamer appsrc pipeline (fbdevsink by default)
//
// Environment:
//   DEBUG               1 = loop over DEBUG_DIR/left/* and DEBUG_DIR/right/*
//                       (paired by file name) instead of the cameras (0)
//   DEBUG_DIR           image folder for DEBUG=1 (data)
//   IMG_W, IMG_H        capture size (must be an MJPG mode of both cameras)
//   CAM_LEFT, CAM_RIGHT v4l2 devices          (/dev/video0, /dev/video2)
//   CAM_FPS             capture frame rate    (30, must be an MJPG rate of
//                       both cameras at IMG_WxIMG_H)
//   DISP_W, DISP_H      display canvas size   (1280x720)
//   DISP_FPS            display refresh rate  (5)
//   DEMO_VIDEO_WRITER   GStreamer sink pipeline, must start with appsrc
#include <glog/logging.h>
#include <signal.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <thread>
#include <vitis/ai/env_config.hpp>

#include "./psmnet.hpp"

DEF_ENV_PARAM(DEBUG, "0");
DEF_ENV_PARAM_2(DEBUG_DIR, "data", std::string);
DEF_ENV_PARAM(IMG_W, "640");
DEF_ENV_PARAM(IMG_H, "480");
DEF_ENV_PARAM_2(CAM_LEFT, "/dev/video0", std::string);
DEF_ENV_PARAM_2(CAM_RIGHT, "/dev/video2", std::string);
DEF_ENV_PARAM(CAM_FPS, "30");
DEF_ENV_PARAM(DISP_W, "1280");
DEF_ENV_PARAM(DISP_H, "720");
DEF_ENV_PARAM(DISP_FPS, "5");
DEF_ENV_PARAM_2(DEMO_VIDEO_WRITER,
                "appsrc ! queue max-size-buffers=2 leaky=downstream ! "
                "videoconvert n-threads=2 ! video/x-raw,format=RGB16 ! "
                "fbdevsink device=/dev/fb0 sync=false",
                std::string);

using namespace std;
using Clock = chrono::steady_clock;

static atomic<bool> g_stop{false};
static void on_signal(int) { g_stop = true; }

// Latest frame of one camera, overwritten continuously (or set from a file
// in DEBUG mode).
struct Camera {
  string device;
  cv::VideoCapture cap;
  mutex mtx;
  cv::Mat latest;
  string label;
  unsigned long seq = 0;

  bool open() {
    char pipe[512];
    snprintf(pipe, sizeof(pipe),
             "v4l2src device=%s do-timestamp=true ! "
             "image/jpeg,width=%d,height=%d,framerate=%d/1 ! jpegdec ! "
             "videoconvert ! video/x-raw,format=BGR ! "
             "appsink drop=true max-buffers=1 sync=false",
             device.c_str(), ENV_PARAM(IMG_W), ENV_PARAM(IMG_H),
             ENV_PARAM(CAM_FPS));
    LOG(INFO) << "camera pipeline: " << pipe;
    return cap.open(pipe, cv::CAP_GSTREAMER);
  }

  void loop() {
    cv::Mat f;
    while (!g_stop) {
      if (!cap.read(f) || f.empty()) {
        LOG(WARNING) << "no frame from " << device;
        this_thread::sleep_for(chrono::milliseconds(100));
        continue;
      }
      lock_guard<mutex> lock(mtx);
      latest = f.clone();
      ++seq;
    }
  }

  void set(const cv::Mat& f, const string& l) {
    lock_guard<mutex> lock(mtx);
    latest = f;
    label = l;
  }

  cv::Mat get() {
    lock_guard<mutex> lock(mtx);
    return latest.clone();
  }

  string get_label() {
    lock_guard<mutex> lock(mtx);
    return label;
  }
};

// What the inference thread publishes for the display.
struct Shared {
  mutex mtx;
  cv::Mat input_l, disparity;
  string input_name;
  unsigned long count = 0;
  double last_ms = 0;
  Clock::time_point busy_since = Clock::now();
};

using FilePairs = vector<pair<string, string>>;

static string basename_of(const string& path) {
  return path.substr(path.find_last_of('/') + 1);
}

// DEBUG mode: <dir>/left/<name> paired with <dir>/right/<name>.
static FilePairs list_debug_pairs(const string& dir) {
  vector<cv::String> lefts;
  cv::glob(dir + "/left/*", lefts, false);
  FilePairs pairs;
  for (auto& l : lefts) {
    auto r = dir + "/right/" + basename_of(l);
    if (cv::haveImageReader(r))
      pairs.emplace_back(l, r);
    else
      LOG(WARNING) << "no right image for " << l << ", skipped";
  }
  CHECK(!pairs.empty()) << "no image pairs found in " << dir
                        << "/left and " << dir << "/right";
  LOG(INFO) << "DEBUG mode: " << pairs.size() << " image pairs from " << dir;
  return pairs;
}

static void inference_loop(Camera& left, Camera& right, const FilePairs& files,
                           Shared& out) {
  auto psmnet = vitis::ai::PsmNet::create();
  const auto batch = psmnet->get_input_batch();
  LOG(INFO) << "PSMNet ready, input " << psmnet->get_input_width() << "x"
            << psmnet->get_input_height() << " batch " << batch;

  size_t file_idx = 0;
  while (!g_stop) {
    cv::Mat l, r;
    string name;
    if (!files.empty()) {
      // DEBUG: next file pair, looping; shown in the top row.
      const auto& f = files[file_idx++ % files.size()];
      l = cv::imread(f.first);
      r = cv::imread(f.second);
      CHECK(!l.empty()) << "cannot read " << f.first;
      CHECK(!r.empty()) << "cannot read " << f.second;
      name = basename_of(f.first);
      left.set(l, "FILE L " + name);
      right.set(r, "FILE R " + name);
    } else {
      // Grab the newest pair right after the previous run finished.
      l = left.get();
      r = right.get();
      if (l.empty() || r.empty()) {
        this_thread::sleep_for(chrono::milliseconds(50));
        continue;
      }
    }
    {
      lock_guard<mutex> lock(out.mtx);
      out.busy_since = Clock::now();
    }
    {
      lock_guard<mutex> lock(out.mtx);
      out.busy_since = Clock::now();
    }
    vector<pair<cv::Mat, cv::Mat>> imgs(batch, make_pair(l, r));
    auto t0 = Clock::now();
    psmnet->run(imgs);
    auto result = psmnet->get_result();
    auto ms = chrono::duration<double, milli>(Clock::now() - t0).count();
    if (result.empty()) continue;

    cv::Mat gray, color;
    cv::cvtColor(result[0], gray, cv::COLOR_BGR2GRAY);
    cv::applyColorMap(gray, color, cv::COLORMAP_JET);

    lock_guard<mutex> lock(out.mtx);
    out.input_l = l;
    out.input_name = name;
    out.disparity = color;
    out.last_ms = ms;
    out.count++;
    LOG(INFO) << "result #" << out.count << " " << name << " in "
              << ms / 1000.0 << " s";
  }
}

// Resize img to fit in roi of canvas, keeping aspect ratio.
static void draw_fit(cv::Mat& canvas, const cv::Rect& roi, const cv::Mat& img,
                     const string& label) {
  if (!img.empty()) {
    double s = min((double)roi.width / img.cols, (double)roi.height / img.rows);
    cv::Size sz((int)(img.cols * s), (int)(img.rows * s));
    cv::Rect dst(roi.x + (roi.width - sz.width) / 2,
                 roi.y + (roi.height - sz.height) / 2, sz.width, sz.height);
    cv::resize(img, canvas(dst), sz, 0, 0, cv::INTER_NEAREST);
  }
  cv::putText(canvas, label, cv::Point(roi.x + 8, roi.y + 22),
              cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
}

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  signal(SIGINT, on_signal);
  signal(SIGTERM, on_signal);

  const bool debug = ENV_PARAM(DEBUG);
  Camera left, right;
  left.label = "LIVE L";
  right.label = "LIVE R";
  FilePairs files;
  if (debug) {
    files = list_debug_pairs(ENV_PARAM(DEBUG_DIR));
  } else {
    left.device = ENV_PARAM(CAM_LEFT);
    right.device = ENV_PARAM(CAM_RIGHT);
    CHECK(left.open()) << "cannot open left camera " << left.device
                       << " (check IMG_W/IMG_H/CAM_FPS against "
                          "v4l2-ctl --list-formats-ext)";
    CHECK(right.open()) << "cannot open right camera " << right.device
                        << " (check IMG_W/IMG_H/CAM_FPS against "
                           "v4l2-ctl --list-formats-ext)";
  }

  const int W = ENV_PARAM(DISP_W), H = ENV_PARAM(DISP_H);
  cv::VideoWriter writer(ENV_PARAM(DEMO_VIDEO_WRITER), cv::CAP_GSTREAMER, 0,
                         ENV_PARAM(DISP_FPS), cv::Size(W, H), true);
  CHECK(writer.isOpened()) << "cannot open display pipeline "
                           << ENV_PARAM(DEMO_VIDEO_WRITER);

  Shared shared;
  thread th_l, th_r;
  if (!debug) {
    th_l = thread([&] { left.loop(); });
    th_r = thread([&] { right.loop(); });
  }
  thread th_inf([&] { inference_loop(left, right, files, shared); });

  // Layout: top = live L | live R, bottom = input L | disparity.
  const cv::Rect top_l(0, 0, W / 2, H / 2), top_r(W / 2, 0, W / 2, H / 2);
  const cv::Rect bot_l(0, H / 2, W / 2, H / 2), bot_r(W / 2, H / 2, W / 2, H / 2);
  const auto period = chrono::milliseconds(1000 / max(1, ENV_PARAM(DISP_FPS)));
  cv::Mat canvas(H, W, CV_8UC3);

  while (!g_stop) {
    auto next = Clock::now() + period;
    canvas.setTo(cv::Scalar::all(0));
    draw_fit(canvas, top_l, left.get(), left.get_label());
    draw_fit(canvas, top_r, right.get(), right.get_label());
    {
      lock_guard<mutex> lock(shared.mtx);
      char buf[128];
      snprintf(buf, sizeof(buf), "INPUT #%lu %s", shared.count,
               shared.input_name.c_str());
      draw_fit(canvas, bot_l, shared.input_l, buf);
      snprintf(buf, sizeof(buf), "DISPARITY  %.1f s  (running %.0f s)",
               shared.last_ms / 1000.0,
               chrono::duration<double>(Clock::now() - shared.busy_since)
                   .count());
      draw_fit(canvas, bot_r, shared.disparity, buf);
    }
    writer << canvas;
    this_thread::sleep_until(next);
  }

  LOG(INFO) << "stopping (waiting for the current inference to finish)";
  if (th_l.joinable()) th_l.join();
  if (th_r.joinable()) th_r.join();
  th_inf.join();
  writer.release();
  return 0;
}
