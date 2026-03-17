# Pancreatic Tumour Segmentation via Hybrid TransAttUNet++

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)

This repository focuses on **TransAttUNet++**, a novel hybrid architecture for high-precision pancreatic tumour segmentation. The project explores the integration of **CNNs** for local features, **Transformers** for global context, and **Attention Gating** in a **Nested U-Net (UNet++)** topology.

![Segmentation Hero](pancreas_segmentation_hero.png)

## 🌟 Original Contribution: TransAttUNet++
The core of this work is a specialized hybrid model designed for volumetric medical data:
- **Nested Skip Connections**: Improved gradient flow and feature reuse via UNet++ backbone.
- **Transformer Bottleneck**: Long-range dependency modeling to capture global tumour morphology.
- **Dynamic Attention**: Spatially-aware attention gates to suppress non-relevant background tissue.

## 📂 Repository Structure
```bash
Pancreatic_Tumour_Segmentation/
├── src/
│   ├── models/       # TransAttUNet++ Core Architectures (2D/3D)
│   ├── data/         # Patient-Aware Dataloaders & Volume Processing
│   └── utils/        # Training Logic, Dice Loss & Specialized Metrics
├── experiments/      # Development & Evaluation Notebooks
├── baselines/        # Comparative Baselines (e.g., nnU-Net v2)
│   └── nnUNet/       # Managed scripts and configs for nnU-Net
├── requirements.txt  # Project Dependencies
└── README.md         # Documentation
```

## 🛠️ Getting Started
1. **Installation**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Train TransAttUNet++**:
   Navigate to `experiments/` to run the primary training notebooks or explore `src/` for custom implementation.

## 📊 Baselines & Evaluation
To ensure rigorous performance verification, we compare our architecture against state-of-the-art baselines.
- **nnU-Net v2**: Automated baseline scripts are located in `baselines/nnUNet/`.

---
