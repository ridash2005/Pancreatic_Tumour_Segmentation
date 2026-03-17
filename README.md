# Pancreatic Tumour Segmentation via Hybrid TransAttUNet++

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)

This repository implements **TransAttUNet++**, a state-of-the-art hybrid architecture for medical image segmentation. It combines the local feature extraction of **CNNs**, the global context modeling of **Transformers**, and the spatial refinement of **Attention Gating**, all within a **Nested U-Net (UNet++)** framework.

![Segmentation Hero](pancreas_segmentation_hero.png)

## 🚀 Key Features
- **Hybrid Topology**: Integrates Transformer blocks in the bottleneck for long-range dependency modeling.
- **TransAttUNet++ 3D**: Full volumetric support for NIfTI datasets with Patient-Aware 5-Fold Cross-Validation.
- **nnU-Net Integration**: Automated scripts to prepare and run official nnU-Net v2 pipelines.
- **Awareness Compliance**: Guaranteed patient-level data splitting to prevent data leakage and ensure publication validity.

---

## 📂 Repository Structure
```
Pancreatic_Tumour_Segmentation/
├── src/
│   ├── models/       # TransAttUNet++ 2D & 3D Architectures
│   ├── data/         # Patient-Aware Dataloaders & Volume Processing
│   └── utils/        # Dice Loss, Metrics & Training Logic
├── scripts/          # nnU-Net Setup & Utility Tools
├── notebooks/        # Jupyter/Colab Training Notebooks
├── main.py           # Experiment Entry Point
└── requirements.txt  # Dependencies
```

---

## 🛠️ Usage
### 1. Requirements
Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Training (3D NIfTI with 5-Fold CV)
To run the full Patient-Aware 5-Fold Cross-Validation:
```bash
python transattunet_3d.py
```

### 3. Training (2D Slices)
To train on individual DICOM slices:
```bash
python architecture.py
```

### 4. nnU-Net Pipeline
To set up and run the official nnU-Net v2 baseline:
```bash
python nnUNet.py
```

---

