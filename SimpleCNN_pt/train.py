# train.py
import torch, torch.nn as nn, torch.optim as optim, os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import SimpleCNN

transform = transforms.Compose([transforms.ToTensor()])
train_set = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_set  = datasets.MNIST("./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_set, batch_size=128)

model = SimpleCNN()
opt = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(5):
    model.train()
    for x, y in train_loader:
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    print(f"epoch {epoch} loss {loss.item():.4f}")

os.makedirs("float_model", exist_ok=True)
torch.save(model.state_dict(), "float_model/simple_cnn.pth")