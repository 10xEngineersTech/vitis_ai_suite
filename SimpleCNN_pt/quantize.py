# quantize.py
import torch
from pytorch_nndct.apis import torch_quantizer
from model import SimpleCNN
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

model = SimpleCNN()
model.load_state_dict(torch.load("float_model/simple_cnn.pth"))
model.eval()

transform = transforms.Compose([transforms.ToTensor()])
calib_set = datasets.MNIST("./data", train=False, transform=transform)
calib_loader = DataLoader(calib_set, batch_size=1, shuffle=True)

dummy_input = torch.randn(1, 1, 28, 28)

# ---- calibration ----
quantizer = torch_quantizer(
    quant_mode="calib",
    module=model,
    input_args=dummy_input,
    output_dir="quantize_result",
    bitwidth=8,   # INT8 — the only width the KV260 DPU can execute (see note below)
)
quant_model = quantizer.quant_model
with torch.no_grad():
    for i, (images, _) in enumerate(calib_loader):
        quant_model(images)
        if i >= 200:
            break
quantizer.export_quant_config()

# ---- test / export ----
quantizer = torch_quantizer(
    quant_mode="test",
    module=model,
    input_args=dummy_input,
    output_dir="quantize_result",
    bitwidth=8,
)
quant_model = quantizer.quant_model
with torch.no_grad():
    for images, _ in calib_loader:
        quant_model(images)
        break

quantizer.export_xmodel(output_dir="quantize_result", deploy_check=True)