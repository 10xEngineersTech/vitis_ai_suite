// Build: see build.sh  (requires -lvart-runner -lvart-mem-manager -lvart-util -lxir)
#include <xir/graph/graph.hpp>
#include <xir/graph/subgraph.hpp>
#include <vart/runner.hpp>
#include <vart/mm/host_flat_tensor_buffer.hpp>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace std;
using namespace cv;
using namespace xir;

static vector<const xir::Subgraph*> get_dpu_subgraphs(const xir::Graph* graph)
{
    auto children = graph->get_root_subgraph()->children_topological_sort();
    vector<const xir::Subgraph*> dpu;
    for (const xir::Subgraph* c : children) {
        if (!c->has_attr("device")) continue;
        string dev = c->get_attr<string>("device");
        transform(dev.begin(), dev.end(), dev.begin(), ::toupper);
        if (dev == "DPU") dpu.push_back(c);
    }
    return dpu;
}

int main(int argc, char* argv[])
{
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <model.xmodel> <image>\n";
        return 1;
    }

    auto graph = xir::Graph::deserialize(argv[1]);
    auto dpu_subgraphs = get_dpu_subgraphs(graph.get());
    if (dpu_subgraphs.empty())
        throw runtime_error("No DPU subgraph found in xmodel.");

    auto runner = vart::Runner::create_runner(dpu_subgraphs[0], "run");
    auto input_tensors  = runner->get_input_tensors();
    auto output_tensors = runner->get_output_tensors();

    const xir::Tensor* in_tensor  = input_tensors[0];
    const xir::Tensor* out_tensor = output_tensors[0];

    auto in_shape = in_tensor->get_shape();  // {1, H, W, C}
    int model_h = in_shape[1], model_w = in_shape[2];

    Mat img = imread(argv[2], IMREAD_GRAYSCALE);
    if (img.empty())
        throw runtime_error(string("Cannot read image: ") + argv[2]);

    if (img.rows != model_h || img.cols != model_w) {
        cerr << "Warning: resizing image from " << img.cols << "x" << img.rows
             << " to " << model_w << "x" << model_h << "\n";
        resize(img, img, Size(model_w, model_h));
    }

    int64_t in_size  = in_tensor->get_element_num();
    int64_t out_size = out_tensor->get_element_num();

    vart::mm::HostFlatTensorBuffer input_tb(in_tensor);
    vart::mm::HostFlatTensorBuffer output_tb(out_tensor);

    auto* in_ptr = reinterpret_cast<int8_t*>(get<0>(input_tb.data({})));
    float scale  = pow(2.0f, (float)in_tensor->get_attr<int>("fix_point"));
    for (int64_t i = 0; i < in_size; ++i) {
        float q = round((img.data[i] / 255.0f) * scale);
        in_ptr[i] = (int8_t)max(-128.0f, min(127.0f, q));
    }

    vector<vart::TensorBuffer*> in_vec  = { &input_tb  };
    vector<vart::TensorBuffer*> out_vec = { &output_tb };

    auto job = runner->execute_async(in_vec, out_vec);
    runner->wait(job.first, -1);

    auto* out_ptr = reinterpret_cast<int8_t*>(get<0>(output_tb.data({})));
    int predicted = (int)(max_element(out_ptr, out_ptr + out_size) - out_ptr);
    cout << "Predicted digit: " << predicted << "\n";

    return 0;
}
