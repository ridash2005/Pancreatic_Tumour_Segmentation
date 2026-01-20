# Pancreatic Tumour Segmentation via Hybrid Trans-Att-UNet++

![Pancreas Segmentation Hero](pancreas_segmentation_hero.png)

## 📌 Overview
This repository contains a state-of-the-art implementation for **Pancreatic Tumour Segmentation** using a novel hybrid architecture, **Trans-Att-UNet++**. This model integrates the power of **Transformers**, **Attention Gating**, and **Nested U-Net (U-Net++)** structures to achieve highly precise segmentation on medical imaging data (CT slices).

Specifically designed for the **Panther Challange Dataset**, this pipeline handles DICOM slices, performs advanced augmentations, and achieves superior localization of both the pancreas and associated tumours.

---

## 🚀 Key Features
- **Hybrid Architecture**: Combines local feature extraction (CNNs), nested skip connections (U-Net++), spatial attention gates, and global context modeling (Transformers).
- **Transformer Bottleneck**: Captures long-range dependencies in anatomical structures using a Transformer Encoder block at the latent bottleneck.
- **Attention-Gated Decoders**: Filters features passed through skip connections to focus on relevant regions of interest (ROI).
- **Mixed-Precision Training**: Leverages PyTorch AMP for efficient GPU memory usage and faster training.
- **Comprehensive Metrics**: Evaluation includes Dice Coefficient, Accuracy, Precision, Sensitivity (Recall), Specificity, and F1-Score.

---

## 🏗️ Model Architecture: Trans-Att-UNet++
The architecture follows a sophisticated multi-stage approach:
1. **Encoder**: Sequential convolution blocks with downsampling.
2. **Bottleneck**: Flattened spatial features are projected into a Transformer Encoder with 2D sine-cosine positional embeddings to understand global organ topology.
3. **Nested skip connections (U-Net++)**: Bridges the gap between encoder and decoder with intermediate dense blocks to reduce the semantic gap.
4. **Attention Gates**: Specifically attends to the tumour pixels by using gating signals from the decoder to weigh the skip-connection features.

---

## 📊 Experimental Results (500 Epochs)
After training for **500 epochs** on the Pancreas dataset, the model achieved the following performance on the validation set:

| Metric | Score |
| :--- | :--- |
| **Best Dice Coefficient** | **0.9265** |
| Average Dice (Final Epoch) | 0.9181 |
| **Accuracy** | **99.92%** |
| Precision | 0.9129 |
| Sensitivity (Recall) | 0.9416 |
| Specificity | 0.9997 |
| F1-Score | 0.8974 |
| Training Loss | 0.6241 |
| Validation Loss | 0.6988 |

*Training Time: ~1.07 min per epoch on standard GPU resources.*

---

## 📁 Dataset Structure
The project is configured to work with DICOM-formatted slices.
```text
DCM/
├── Task1/
│   ├── ImagesTr/    # Training Images (.dcm)
│   └── LabelsTr/    # Mask Labels (.dcm)
```
*Note: The dataset is expected to be provided in a zip file or mounted drive for the training script.*

---

## 🛠️ Getting Started

### Prerequisites
- Python 3.8+
- PyTorch (with CUDA support)
- pydicom
- Albumentations
- Matplotlib, NumPy, SciKit-Learn

### Installation
```bash
git clone https://github.com/ridash2005/Pancreatic_Tumour_Segmentation.git
cd Pancreatic_Tumour_Segmentation
pip install -r requirements.txt  # If applicable, or manual install
```

### Usage
Run the training and evaluation directly from the provided notebook:
1. Open `Hybrid Architecture.ipynb`.
2. Configure your dataset path.
3. Run all cells to begin training and generate segmented outputs.

---

## 🖼️ Sample Predictions
The model generates side-by-side visualizations of the **Input CT Slice**, **Ground Truth Mask**, and **Predicted Segmentation**, allowing for immediate qualitative assessment of performance.

---

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.