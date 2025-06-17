import sys
import os

# ✅ Add project root to sys.path for src module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# ✅ Import your custom ResNet builder
from src.models.resnet import build_resnet

# ✅ Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(use_attention=False, epochs=10):
    # ✅ Prepare dataset and data loader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

    # ✅ Build model
    model = build_resnet(use_attention=use_attention).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ✅ Train the model
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(trainloader):.4f}")

    # ✅ Ensure 'output/' directory exists before saving
    os.makedirs("output", exist_ok=True)
    torch.save(model.state_dict(), f"output/resnet_attention_{use_attention}.pth")
    print(f"✅ Model saved: output/resnet_attention_{use_attention}.pth")

if __name__ == "__main__":
    train_model(use_attention=False)
    train_model(use_attention=True)
