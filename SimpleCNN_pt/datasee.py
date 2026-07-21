from torchvision import datasets
import PIL.Image, os

ds = datasets.MNIST("./data", train=False, download=True)
os.makedirs("samples", exist_ok=True)
for i in range(20):
    img, label = ds[i]
    img.save(f"samples/{i}_label{label}.png")
print("Saved 20 samples to ./samples/")