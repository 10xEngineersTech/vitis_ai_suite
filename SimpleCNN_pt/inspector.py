# inspector.py
import torch
from pytorch_nndct.apis import Inspector
from model import SimpleCNN

device = torch.device("cpu")

# 1. Initialize your SimpleCNN architecture
model = SimpleCNN()

# 2. Load the trained weights
model.load_state_dict(torch.load("float_model/simple_cnn.pth", map_location="cpu"))
model.eval().to(device)

# 3. Setup the Vitis-AI target inspector
inspector = Inspector("DPUCZDX8G_ISA1_B4096")

# 4. Input tensor must match SimpleCNN's expected shape:
#    - 1 image in the batch (concrete integer required for graph tracing)
#    - 1 channel (grayscale, as defined by conv1's in_channels=1)
#    - 28x28 pixels (MNIST image size)
input_tensor = torch.randn([1, 1, 28, 28]).to(device)

# 5. Run inspector and generate the SVG graph file
inspector.inspect(model, (input_tensor,), device=device, output_dir="./inspect_output", image_format="svg")

print("Inspection complete! Check the './inspect_output' folder for your SVG graph.")