# 🩺 Pancreatic Tumour Segmentation via TransAttUNet++

<div align="center">
  <img src="pancreas_segmentation_hero.png" alt="Pancreatic Tumour Segmentation Hero" width="800">
  <p><em>High-precision hybrid architecture for volumetric medical data segmentation.</em></p>

  [![Project Website](https://img.shields.io/badge/Project-Website-blue?style=for-the-badge&logo=google-chrome&logoColor=white)](https://ridash2005.github.io/Pancreatic_Tumour_Segmentation/)
  [![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ridash2005/Pancreatic_Tumour_Segmentation)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
</div>

---

## 🚀 Overview

**TransAttUNet++** is a cutting-edge hybrid deep learning architecture designed for the challenging task of pancreatic tumour segmentation. By fusing the local feature extraction capabilities of **CNNs** (via a UNet++ backbone) with the global context modeling of **Transformers** and the targeted precision of **Attention Gating**, this project achieves state-of-the-art results on complex volumetric medical datasets.

### 🛠️ Built With

| Category | Technologies |
| :--- | :--- |
| **Core** | [![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/) |
| **Data Processing** | [![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)](https://numpy.org/) [![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/) [![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/) |
| **Augmentation** | [![Albumentations](https://img.shields.io/badge/Albumentations-black?style=flat)](https://albumentations.ai/) [![Pillow](https://img.shields.io/badge/Pillow-blue?style=flat)](https://python-pillow.org/) |
| **Visualization** | [![Matplotlib](https://img.shields.io/badge/Matplotlib-ffffff?style=flat&logo=matplotlib&logoColor=black)](https://matplotlib.org/) |
| **Baselines** | [![nnU-Net](https://img.shields.io/badge/nnU--Net-v2-blue?style=flat)](https://github.com/MIC-DKFZ/nnUNet) |

---

## 💎 Key Features

- **Nested Skip Connections (UNet++)**: Facilitates multi-scale feature fusion and smoother gradient flow for better convergence.
- **Transformer Bottleneck**: Captures long-range spatial dependencies to model global tumour morphology accurately.
- **Dynamic Attention Gating**: Suppresses non-relevant background signals, focusing the model on the intricate boundaries of pancreatic tissue.
- **Automated Pipeline**: End-to-end support for NIfTI/DICOM processing, 5-fold cross-validation, and performance reporting.

---

## 📂 Repository Structure

```bash
Pancreatic_Tumour_Segmentation/
├── src/
│   ├── models/           # Core TransAttUNet++ Architectures (2D/3D)
│   ├── data/             # Advanced Dataloaders & Preprocessing
│   └── utils/            # Custom Loss Functions (Dice/IoU) & Metrics
├── experiments/          # Model Training & Validation Notebooks
├── baselines/            # Comparative Analysis (nnU-Net v2 implementation)
├── configs/              # Hyperparameter & Experiment Configurations
├── docs/                 # Detailed Documentation & Colab Guides
├── scripts/              # Utility scripts for data conversion and cleanup
└── README.md             # Project Documentation
```

---

## ⚙️ Getting Started

### 1. Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/ridash2005/Pancreatic_Tumour_Segmentation.git
cd Pancreatic_Tumour_Segmentation
pip install -r requirements.txt
```

### 2. Dataset Preparation
Ensure your data is organized in the `Dataset/` directory. The project supports `.nii.gz` and DICOM formats. For automated nnU-Net baseline preparation, refer to `baselines/nnUNet/`. Detailed instructions for training in Google Colab can be found in the [Colab Guide](docs/COLAB_GUIDE.md).

### 3. Training
To start training the TransAttUNet++ model with 5-fold cross-validation:
```bash
python main.py --config configs/train_config.yaml
```

---

## 👤 Author
**Rickarya Das**
- GitHub: [@ridash2005](https://github.com/ridash2005)
- Project: [Pancreatic Tumour Segmentation](https://github.com/ridash2005/Pancreatic_Tumour_Segmentation)

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

<div align="center">
  <sub>Built with ❤️ for Medical AI Research</sub>
</div>
