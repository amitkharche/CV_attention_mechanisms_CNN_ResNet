import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from src.models.resnet import build_resnet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model(use_attention=False):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=64, shuffle=False)

    model = build_resnet(use_attention=use_attention).to(device)
    model.load_state_dict(torch.load(f"output/resnet_attention_{use_attention}.pth", map_location=device))
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy with attention={use_attention}: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    evaluate_model(False)
    evaluate_model(True)
