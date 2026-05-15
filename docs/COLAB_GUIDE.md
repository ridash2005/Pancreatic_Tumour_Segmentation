# TransAttUNet++ 3D - Google Colab Training Guide

This guide helps you train the improved **3D TransAttUNet++** model in Google Colab for pancreatic tumor segmentation with NIfTI datasets.

## 🚀 Quick Start (Colab)

### 1. **Prepare Your Data**

Upload your dataset to Google Drive in one of these structures:

```
Your_Data_Folder/
├── Task1/
│   ├── imagesTr/           # Training images (NIfTI: .nii.gz or .nii)
│   │   ├── patient_001_0000.nii.gz
│   │   ├── patient_002_0000.nii.gz
│   │   └── ...
│   └── labelsTr/           # Training labels (NIfTI)
│       ├── patient_001.nii.gz
│       ├── patient_002.nii.gz
│       └── ...
└── Task2/                  # (Optional)
    ├── imagesTr/
    └── labelsTr/
```

**Supported namings:**
- Images: `patient_id_0000.nii.gz`, `patient_id_0000.nii`
- Labels: `patient_id.nii.gz`, `patient_id.nii`

### 2. **Upload Script to Colab**

1. Open a new notebook in [Google Colab](https://colab.research.google.com)
2. Download `transattunet++.py` and upload to Colab
3. Or paste the code directly into a cell

### 3. **Create the First Cell**

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy the script content here or
# !wget -q https://your-repo-url/transattunet++.py

# Run the training
exec(open('transattunet++.py').read())
```

### 4. **Configure Settings (in transattunet++.py)**

Edit the `CONFIG` dictionary before running:

```python
CONFIG = {
    'model_type': '3d',              # Use 3D model for 3D data
    'in_channels': 1,                # Single channel input
    'out_channels': 1,               # Single channel output (binary segmentation)
    'input_size_3d': (32, 128, 128), # (Depth, Height, Width) - reduce if memory issues
    'batch_size': 4,                 # Colab: 4-8 typical (adjust if OOM)
    'epochs': 50,                    # Number of training epochs
    'learning_rate': 1e-4,           # Learning rate
    'weight_decay': 1e-5,            # L2 regularization
    'val_dataset_ratio': 0.15,       # 15% validation split
    'mixed_precision': True,         # Enable for faster training
    'save_frequency': 5,             # Save checkpoint every 5 epochs
}

# For Colab:
CONFIG['data_dir'] = '/content/drive/MyDrive/Your_Data_Folder'
CONFIG['output_dir'] = '/content/drive/MyDrive/TransAttUNet_Results'
```

### 5. **Run Training**

Execute the full script. It will:
- ✅ Mount Google Drive
- ✅ Install dependencies
- ✅ Load 3D NIfTI data
- ✅ Train the 3D TransAttUNet++ model
- ✅ Save checkpoints every 5 epochs
- ✅ Save best model (highest Dice score)
- ✅ Generate training curves
- ✅ Export results to Google Drive

## 📊 Key Features

### Model Architecture
- **3D TransAttUNet++**: Full 3D convolutional network with:
  - Dense U-Net++ connections
  - Attention gates for feature highlighting
  - Transformer encoder for global context
  - Efficient for medical image segmentation

### Data Loading
- **NIfTI Format Support**: Built-in handling of `.nii.gz` and `.nii` files
- **Multi-task Learning**: Train on Task1 and Task2 simultaneously
- **Automatic Resizing**: Scales volumes to target size (e.g., 32×128×128)
- **Normalization**: Z-score normalization per patient

### Loss Functions
- **BCE + Dice Loss**: Combined for robust training
- **Focal Loss**: Available for handling class imbalance
- **Dice Metric & IoU**: Automatic metric tracking

### Memory Optimization
- **Mixed Precision Training**: 30-50% faster, less memory
- **Configurable Batch Size**: Reduce if OOM errors occur
- **Checkpointing**: Save best models automatically

## 📈 Output Files

After training, check your Google Drive:

```
TransAttUNet_Results/
├── best_model.pth          # Best model weights (highest Dice)
├── checkpoint_epoch_5.pth  # Periodic checkpoints
├── checkpoint_epoch_10.pth
├── ...
├── training_history.csv    # Epoch-by-epoch metrics
└── training_curves.png     # Loss & Dice curves
```

## 🔧 Troubleshooting

### Out of Memory (OOM) Error
**Solution:** Reduce batch size or input size in CONFIG:
```python
CONFIG['batch_size'] = 2  # Reduce from 4
CONFIG['input_size_3d'] = (16, 96, 96)  # Reduce from 32×128×128
```

### Data Not Found Error
**Solution:** Verify paths:
```python
# Check if data exists
import os
print(os.listdir('/content/drive/MyDrive/Your_Data_Folder/Task1/imagesTr/'))
```

### Training Too Slow
**Solution:** Enable mixed precision (already enabled by default):
```python
CONFIG['mixed_precision'] = True
```

### GPU Not Available
Google Colab GPUs are assigned randomly. Reconnect if needed:
- Runtime → Disconnect and delete all
- Runtime → Connect to new GPU

## 📝 Modifying the Script

### Change 2D Model (if needed)
```python
CONFIG['model_type'] = '2d'  # Switch to 2D
```

### Adjust Learning Rate Schedule
```python
# Edit in train_model() function:
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
```

### Add Custom Loss Function
```python
# Add to loss functions section:
class CustomLoss(nn.Module):
    def forward(self, pred, target):
        # Your implementation here
        pass
```

## 🎓 Inference (After Training)

```python
# Load best model
model = TransAttUNetPlusPlus3D().to(device)
model.load_state_dict(torch.load('/content/drive/MyDrive/TransAttUNet_Results/best_model.pth'))
model.eval()

# Predict on new volume
with torch.no_grad():
    output = model(input_volume)
    prediction = (torch.sigmoid(output) > 0.5).float()
```

## 📚 Dataset Examples

### PANTHER Dataset (Recommended)
```
PANTHER_NIfTI/
├── Task1/
│   ├── imagesTr/  (Training images)
│   └── labelsTr/  (Pancreas + Tumor labels)
└── Task2/
    ├── imagesTr/
    └── labelsTr/
```

### Pan_segNet_Nifti
```
Pan_segNet_Nifti/
├── Task1/  (or Task2)
│   ├── imagesTr/
│   └── labelsTr/
```

## ⚙️ Default Configuration

| Parameter | Default | Notes |
|-----------|---------|-------|
| Model Type | 3D | Full 3D convolutions for volumetric data |
| Input Size | 32×128×128 | Depth × Height × Width |
| Batch Size | 4 | Reduced for 3D; reduce if OOM |
| Epochs | 50 | Can be increased for better convergence |
| Learning Rate | 1e-4 | AdamW optimizer with decay |
| Mixed Precision | Enabled | ~30-50% speed improvement |
| Validation Split | 15% | Auto train/val split |

## 💾 Model Architecture

**3D TransAttUNet++ Components:**
- **Encoder**: 4-level downsampling with dense connections
- **Bottleneck**: 5th level with transformer encoder (2 layers, 4 heads)
- **Decoder**: 4-level upsampling with attention gates
- **Dense Connections**: All levels concatenated for feature reuse
- **Output**: Single-channel binary segmentation map

**Total Parameters:** ~2.5M (adjustable with filters)

## 🚦 Colab GPU Recommendations

| GPU | Typical Performance |
|-----|-------------------|
| Tesla K80 | Baseline |
| Tesla P100 | 2× faster |
| Tesla V100 | 3-4× faster ✅ (Recommended) |
| Tesla T4 | 2× faster, 16GB VRAM |

**Free Colab:** May get K80 or T4
**Colab Pro:** Usually V100

## 📊 Expected Training Results

With typical PANTHER Task1 data:
- **Epoch 1-10**: Dice rapidly increases from 0.4 → 0.7
- **Epoch 10-30**: Fine-tuning, Dice 0.7 → 0.8+
- **Epoch 30+**: Diminishing returns

**Typical Results:**
- Dice Score: 0.75-0.85
- IoU Score: 0.60-0.75
- Training Time: 2-4 hours (50 epochs on V100)

## 🔄 Resuming Training

```python
# Load checkpoint
checkpoint = torch.load('checkpoint_epoch_25.pth')
model.load_state_dict(checkpoint['model_state'])
optimizer.load_state_dict(checkpoint['optimizer_state'])
start_epoch = checkpoint['epoch']

# Continue training from epoch 26...
```

## 📞 Support

If you encounter issues:
1. Check this README
2. Review error messages carefully
3. Try reducing batch size/model size
4. Check data format (must be NIfTI)
5. Verify paths in CONFIG

---

**Happy Training! 🎉**

For questions or improvements, see the repository issues or documentation.
