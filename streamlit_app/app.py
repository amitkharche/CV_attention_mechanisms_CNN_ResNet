import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from src.models.resnet import build_resnet

st.title("🔍 ResNet Attention Classifier (CIFAR-10)")

use_attention = st.checkbox("Use Attention (SE Block)", value=True)
uploaded_file = st.file_uploader("Upload a CIFAR-10 like image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB").resize((32, 32))
    st.image(image, caption="Input Image", use_column_width=False)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    input_tensor = transform(image).unsqueeze(0)

    model = build_resnet(use_attention=use_attention)
    model.load_state_dict(torch.load(f"output/resnet_attention_{use_attention}.pth", map_location="cpu"))
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
    classes = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    st.write(f"🔍 Predicted Class: {classes[predicted.item()]}")
