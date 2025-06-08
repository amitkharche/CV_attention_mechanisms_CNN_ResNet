
# 🧠 Project 9: Visual Attention in CNNs – ResNet with Squeeze-and-Excitation (SE)

---

## 📌 Business Problem

In computer vision applications like **medical imaging**, **autonomous driving**, and **defect detection**, models must **focus on the most informative regions** of an image. Standard CNNs treat all regions equally, potentially missing key features. 

This project solves that by incorporating **Squeeze-and-Excitation (SE) attention mechanisms** into ResNet, improving focus on salient features and boosting classification performance.

---

## 🚀 Project Overview

This project compares a **standard ResNet18** with a version enhanced by the **SE Block**, evaluating their performance on the CIFAR-10 dataset.

✅ Includes:
- 🔁 Training pipeline for ResNet with/without attention
- 📈 Accuracy evaluation and comparison
- 📸 Grad-CAM overlays for model explainability
- 🌐 Streamlit app to classify custom images
- 🎞️ GIF demo of predictions
- 📔 Jupyter Notebook to run all steps

---

## 🗂️ Repository Structure

```
attention-mechanisms-cv/
├── src/
│   ├── models/
│   │   ├── resnet.py              # Baseline ResNet + SE Block integrated
│   │   └── attention.py           # Squeeze-and-Excitation block
│   ├── train.py                   # Training script
│   ├── evaluate.py                # Evaluation script
│
├── streamlit_app/
│   └── app.py                     # Streamlit UI with image upload and prediction
│
├── notebooks/
│   └── resnet_attention_comparison.ipynb   # Jupyter demo
│
├── output/
│   └── resnet_attention_*.pth     # Trained model weights
│
├── demo/
│   └── prediction_comparison.gif  # Visual side-by-side prediction
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 📸 Grad-CAM Visualization (NEW!)

Integrated Grad-CAM to visually explain why the model made a particular prediction.

✅ Available in Streamlit app  
✅ Toggle to activate Grad-CAM overlay

---

## 🧪 Accuracy Comparison

| Model            | Accuracy (%) |
|------------------|--------------|
| ResNet18         | ~78%         |
| ResNet + SE Block| ~82%         |

---

## 🛠️ How to Run

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Train Models
```bash
python src/train.py
```

### 3. Evaluate Accuracy
```bash
python src/evaluate.py
```

### 4. Launch Streamlit App
```bash
streamlit run streamlit_app/app.py
```

Upload an image, select model type, and get predictions + Grad-CAM.

---

## 🎞️ Demo Preview

![Prediction Demo](demo/prediction_comparison.gif)

---

## 📊 Use Cases

- 📷 Industrial defect detection
- 🚗 Driverless vehicle visual systems
- 🩻 Medical diagnostics (e.g., tumors, X-rays)

---

## 📜 License

MIT License

---
