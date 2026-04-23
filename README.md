#  Brain Tumour Detection
### Deep Learning Medical Imaging Analysis

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-red)
![Accuracy](https://img.shields.io/badge/Accuracy-85.53%25-brightgreen)
![HuggingFace](https://img.shields.io/badge/🤗-Live%20Demo-yellow)

> AI-powered brain tumour classification from T1-weighted MRI scans using ResNet50 transfer learning. Classifies 4 tumour types with OOD detection, Grad-CAM explainability, and automated PDF diagnostic report generation.

**🔗 Live Demo:** [huggingface.co/spaces/Nainika0205/BrainTumourDetection](https://huggingface.co/spaces/Nainika0205/BrainTumourDetection)

---

## Overview

This project builds an automated brain tumour classification system. Given a T1-weighted brain MRI image, the system:
1. Verifies the image is a valid brain MRI (OOD detection)
2. Classifies into **4 classes**: Glioma, Meningioma, Pituitary Tumour, No Tumour
3. Assigns severity level (Low Confidence / Moderate / High)
4. Detects tumour location using Grad-CAM heatmap analysis
5. Provides tumour-specific clinical recommendations
6. Generates a downloadable PDF diagnostic report

---

## Features

| Feature | Description |
|---------|-------------|
| **4-Class Classification** | Glioma / Meningioma / Pituitary / No Tumour |
| **OOD Detection** | Mahalanobis Distance rejects CT scans, X-rays, non-MRI images |
| **Severity Scoring** | Low Confidence / Moderate / High |
| **Location Detection** | Frontal / Central / Posterior region via Grad-CAM |
| **Clinical Recommendations** | Tumour-type specific next steps |
| **PDF Report** | Professional neurological report with Grad-CAM |
| **Test Time Augmentation** | 5-augmentation TTA for improved accuracy |

---

## Model Architecture

- **Base Model:** ResNet50 (pretrained on ImageNet)
- **Trainable Layers:** Layer2, Layer3, Layer4, FC
- **Custom FC:** Dropout(0.5) -> Linear(2048 -> 4)
- **Optimizer:** Adam (lr=0.00005, weight_decay=1e-4)
- **Loss:** CrossEntropyLoss with class weights [2.0, 1.0, 1.0, 1.5]
- **Early Stopping:** patience=5

---

## Dataset

- **Source:** [Kaggle - Brain Tumor Classification MRI](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)
- **Total Images:** 3260
- **Classes:** Glioma (826), Meningioma (822), Pituitary (827), No Tumour (395)
- **Split:** Train 2441 / Val 429 / Test 394

---

## Results

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Glioma | 0.97 | 0.60 | 0.75 |
| Meningioma | 0.72 | 0.97 | 0.84 |
| No Tumour | 0.87 | 1.00 | 0.93 |
| Pituitary | 0.95 | 0.82 | 0.88 |
| **Overall (TTA)** | | | **85.53%** |

### OOD Detection
- Threshold: 153.25 (99th percentile)
- CT scans, X-rays, random images: REJECTED

---

## Model Iterations

| Version | Accuracy | Notes |
|---------|----------|-------|
| V1 | 75.89% | Baseline |
| V2 | 85.28% | Best base model |
| V2 + TTA | **85.53%** | **Best overall** |
| V3 | 81.22% | Overfitting |
| V4 + TTA | 79.95% | Label smoothing hurt |

---

## Installation
```
git clone https://github.com/nainika-2504/BrainTumourDetection
cd BrainTumourDetection
pip install -r requirements.txt
python app/gradio_app.py
```

---

## Project Structure
```
BrainTumourDetection/
├── app/
│   └── gradio_app.py
├── outputs/
│   ├── brain_confusion_matrix_v2.png
│   ├── brain_confusion_matrix_tta.png
│   ├── brain_gradcam.png
│   └── brain_real_prediction.png
├── requirements.txt
└── README.md
```

Model files hosted on Hugging Face due to GitHub 100MB limit.

---

## Tech Stack

Python | PyTorch | ResNet50 | Grad-CAM | Gradio | ReportLab | Hugging Face Spaces

---

## Disclaimer

For educational purposes only. Not a substitute for professional medical advice.
Always consult a qualified neurologist for medical decisions.

---

## Author
Nainika M 
Nandhini
