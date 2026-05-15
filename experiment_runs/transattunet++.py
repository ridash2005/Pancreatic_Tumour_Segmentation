"""
TransAttUNet++ 3D - Colab Ready Training Script
Designed for 3D Pancreatic Tumor Segmentation with NIfTI datasets
Compatible with: PANTHER Task1/Task2, Pan_segNet_Nifti
"""

# ============================================================
# 1. COLAB SETUP & GOOGLE DRIVE MOUNT
# ============================================================
print("=" * 60)
print("TransAttUNet++ 3D - Colab Training Script")
print("=" * 60)

try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
    print("⚠️  Not running in Colab. Local paths will be used.")

# ============================================================
# 2. INSTALL DEPENDENCIES
# ============================================================
import subprocess
import sys

def install_package(package_name):
    """Install package safely"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package_name])
        return True
    except:
        return False

print("\n📦 Installing dependencies...")
packages = ["torch", "torchvision", "SimpleITK", "nibabel", "pandas", "tqdm", "matplotlib", "scikit-learn"]
for pkg in packages:
    install_package(pkg)
print("✅ Dependencies installed")


# ============================================================
# 3. IMPORTS
# ============================================================
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import nibabel as nib
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
import time
import pandas as pd
from pathlib import Path
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from sklearn.model_selection import KFold
import zipfile
import shutil

print("\n✅ All imports successful")

# ============================================================
# 4. CONFIGURATION & DEVICE SETUP
# ============================================================

# ---- Device Setup ----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🖥️  Device: {device}")
if device.type == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    torch.cuda.empty_cache()

# ---- Hyperparameters ----
CONFIG = {
    'model_type': '3d',  # '2d' or '3d'
    'in_channels': 1,
    'out_channels': 3,  # 3-class: background(0), pancreas(1), tumor(2)
    'input_size_3d': (32, 128, 128),  # (D, H, W)
    'batch_size': 4,  # Reduce for 3D (4-8 is typical for 3D)
    'epochs': 1,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'num_workers': 0 if IN_COLAB else 2,
    'mixed_precision': True,
    'save_frequency': 5,  # Save model every N epochs
    'n_splits': 5,  # 5-fold cross-validation
    'segmentation_labels': [1, 2],  # Labels to segment: 1=pancreas, 2=tumor. User can also specify [1] or [2]
}

if IN_COLAB:
    CONFIG['batch_size'] = 4  # Colab typically has 16GB VRAM
    CONFIG['drive_data_dir'] = '/content/drive/MyDrive'
    CONFIG['zip_name'] = 'nnUNet_raw.zip'  # Zip file in Google Drive
    CONFIG['output_dir'] = '/content/drive/MyDrive/TransAttUNet_Results'
else:
    CONFIG['batch_size'] = 8
    CONFIG['data_dir'] = r'D:\GitHub\my_repo\Pancreatic_Tumour_Segmentation\nnUNet_raw'
    CONFIG['output_dir'] = r'.\TransAttUNet_Results'

# Create output directory
os.makedirs(CONFIG['output_dir'], exist_ok=True)

print("\n⚙️  Configuration:")
for key, value in CONFIG.items():
    print(f"   {key}: {value}")


# ============================================================
# 5. MODEL ARCHITECTURE - 3D TRANSATTUNET++
# ============================================================

class DoubleConv3D(nn.Module):
    """Double convolution block for 3D"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class AttentionBlock3D(nn.Module):
    """Attention mechanism for 3D"""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, 1, bias=True),
            nn.BatchNorm3d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, 1, bias=True),
            nn.BatchNorm3d(F_int)
        )
        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv3d(F_int, 1, 1, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        psi = self.psi(self.W_g(g) + self.W_x(x))
        return x * psi


class TransAttUNetPlusPlus3D(nn.Module):
    """Full 3D TransAttUNet++ model with dense skip connections"""
    def __init__(self, in_channels=1, out_channels=1, filters=None):
        super().__init__()
        if filters is None:
            filters = [32, 64, 128, 256, 512]
        
        f1, f2, f3, f4, f5 = filters
        
        # Encoder
        self.enc1 = DoubleConv3D(in_channels, f1)
        self.enc2 = DoubleConv3D(f1, f2)
        self.enc3 = DoubleConv3D(f2, f3)
        self.enc4 = DoubleConv3D(f3, f4)
        self.bottleneck = DoubleConv3D(f4, f5)
        
        # Transformer encoder (reduced layers for 3D)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=f5, nhead=4, dim_feedforward=512, 
            batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Decoder upsampling
        self.up4 = nn.ConvTranspose3d(f5, f4, 2, 2)
        self.up3 = nn.ConvTranspose3d(f4, f3, 2, 2)
        self.up2 = nn.ConvTranspose3d(f3, f2, 2, 2)
        self.up1 = nn.ConvTranspose3d(f2, f1, 2, 2)
        
        # Attention gates
        self.att4 = AttentionBlock3D(f4, f4, f4 // 2)
        self.att3 = AttentionBlock3D(f3, f3, f3 // 2)
        self.att2 = AttentionBlock3D(f2, f2, f2 // 2)
        self.att1 = AttentionBlock3D(f1, f1, f1 // 2)
        
        # Dense connections - fixed to match concatenation channel counts
        # Each concatenation outputs 2 or 3 feature maps: up + att(up) + skip (optional)
        self.x3_1 = DoubleConv3D(f4 * 2, f4)
        self.x2_1 = DoubleConv3D(f3 * 2, f3)
        self.x2_2 = DoubleConv3D(f3 * 3, f3)
        self.x1_1 = DoubleConv3D(f2 * 2, f2)
        self.x1_2 = DoubleConv3D(f2 * 3, f2)
        self.x1_3 = DoubleConv3D(f2 * 3, f2)  # Concatenates 3 f2 features
        self.x0_1 = DoubleConv3D(f1 * 2, f1)  # Concatenates 2 f1 features (up + att(up))
        self.x0_2 = DoubleConv3D(f1 * 3, f1)  # Concatenates 3 f1 features (up + att(up) + e1)
        self.x0_3 = DoubleConv3D(f1 * 3, f1)  # Concatenates 3 f1 features (up + att(up) + e1)
        self.x0_4 = DoubleConv3D(f1 * 3, f1)  # Concatenates 3 f1 features (up + att(up) + e1)
        
        # Output
        self.final = nn.Conv3d(f1, out_channels, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool3d(e1, 2))
        e3 = self.enc3(F.max_pool3d(e2, 2))
        e4 = self.enc4(F.max_pool3d(e3, 2))
        b = self.bottleneck(F.max_pool3d(e4, 2))
        
        # Transformer (apply on flattened spatial dims)
        B, C, D, H, W = b.shape
        tokens = b.flatten(2).transpose(1, 2)
        tokens = self.transformer(tokens)
        b = tokens.transpose(1, 2).view(B, C, D, H, W)
        
        # Decoder with dense connections
        x3_1 = self.x3_1(torch.cat([self.up4(b), self.att4(self.up4(b), e4)], 1))
        
        x2_1 = self.x2_1(torch.cat([self.up3(x3_1), self.att3(self.up3(x3_1), e3)], 1))
        x2_2 = self.x2_2(torch.cat([self.up3(x3_1), self.att3(self.up3(x3_1), e3), e3], 1))
        
        x1_1 = self.x1_1(torch.cat([self.up2(x2_1), self.att2(self.up2(x2_1), e2)], 1))
        x1_2 = self.x1_2(torch.cat([self.up2(x2_1), self.att2(self.up2(x2_1), e2), e2], 1))
        x1_3 = self.x1_3(torch.cat([self.up2(x2_2), self.att2(self.up2(x2_2), e2), e2], 1))
        
        x0_1 = self.x0_1(torch.cat([self.up1(x1_1), self.att1(self.up1(x1_1), e1)], 1))
        x0_2 = self.x0_2(torch.cat([self.up1(x1_1), self.att1(self.up1(x1_1), e1), e1], 1))
        x0_3 = self.x0_3(torch.cat([self.up1(x1_2), self.att1(self.up1(x1_2), e1), e1], 1))
        x0_4 = self.x0_4(torch.cat([self.up1(x1_3), self.att1(self.up1(x1_3), e1), e1], 1))
        
        # Final output (use x0_4 for best quality)
        out = self.final(x0_4)
        return out


# ============================================================
# 6. LOSS FUNCTIONS & METRICS
# ============================================================

class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation"""
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        target = target.view(target.size(0), -1)
        intersection = (probs * target).sum(1)
        union = probs.sum(1) + target.sum(1)
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class FocalLoss(nn.Module):
    """Focal Loss for class imbalance (multi-class)"""
    def __init__(self, gamma=2, alpha=None, num_classes=3):
        super().__init__()
        self.gamma = gamma
        self.num_classes = num_classes
        if alpha is None:
            self.alpha = [1.0] * num_classes  # Equal weight for each class
        else:
            self.alpha = alpha

    def forward(self, logits, target):
        # logits: (B, C, D, H, W), target: (B, D, H, W) with class indices
        ce_loss = F.cross_entropy(logits, target, reduction='none')
        p = F.softmax(logits, dim=1)  # (B, C, D, H, W)
        p_t = p.gather(1, target.unsqueeze(1)).squeeze(1)  # (B, D, H, W)
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss
        return focal_loss.mean()


def dice_score(pred, target):
    """Calculate Dice Score (multi-class)
    pred: (B, C, D, H, W) logits
    target: (B, D, H, W) class indices
    Returns average Dice across all non-background classes
    """
    pred_classes = torch.argmax(pred, dim=1)  # (B, D, H, W)
    dice_scores = []
    for class_id in [1, 2]:  # pancreas and tumor
        pred_mask = (pred_classes == class_id).float()
        target_mask = (target == class_id).float()
        intersection = (pred_mask * target_mask).sum()
        union = pred_mask.sum() + target_mask.sum()
        if union == 0:
            dice_scores.append(1.0 if intersection == 0 else 0.0)
        else:
            dice = (2 * intersection + 1e-6) / (union + 1e-6)
            dice_scores.append(dice.item())
    return np.mean(dice_scores)


def iou_score(pred, target):
    """Calculate IoU (Intersection over Union) - multi-class"""
    pred_classes = torch.argmax(pred, dim=1)  # (B, D, H, W)
    iou_scores = []
    for class_id in [1, 2]:  # pancreas and tumor
        pred_mask = (pred_classes == class_id).float()
        target_mask = (target == class_id).float()
        intersection = (pred_mask * target_mask).sum()
        union = (pred_mask + target_mask - pred_mask * target_mask).sum()
        if union == 0:
            iou_scores.append(1.0 if intersection == 0 else 0.0)
        else:
            iou = (intersection + 1e-6) / (union + 1e-6)
            iou_scores.append(iou.item())
    return np.mean(iou_scores)


def sensitivity_specificity(pred, target):
    """Calculate Sensitivity (Recall) and Specificity - multi-class"""
    pred_classes = torch.argmax(pred, dim=1)  # (B, D, H, W)
    sens_scores = []
    spec_scores = []
    for class_id in [1, 2]:
        pred_mask = (pred_classes == class_id).float()
        target_mask = (target == class_id).float()
        tp = (pred_mask * target_mask).sum()
        fp = (pred_mask * (1 - target_mask)).sum()
        fn = ((1 - pred_mask) * target_mask).sum()
        tn = ((1 - pred_mask) * (1 - target_mask)).sum()
        
        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        sens_scores.append(sensitivity.item())
        spec_scores.append(specificity.item())
    
    return np.mean(sens_scores), np.mean(spec_scores)


def precision_recall_f1(pred, target):
    """Calculate Precision, Recall, and F1-Score - multi-class"""
    pred_classes = torch.argmax(pred, dim=1)  # (B, D, H, W)
    prec_scores = []
    rec_scores = []
    f1_scores = []
    for class_id in [1, 2]:
        pred_mask = (pred_classes == class_id).float()
        target_mask = (target == class_id).float()
        tp = (pred_mask * target_mask).sum()
        fp = (pred_mask * (1 - target_mask)).sum()
        fn = ((1 - pred_mask) * target_mask).sum()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        prec_scores.append(precision.item())
        rec_scores.append(recall.item())
        f1_scores.append(f1.item())
    
    return np.mean(prec_scores), np.mean(rec_scores), np.mean(f1_scores)


# ============================================================
# 7. DATA LOADING - 3D NIFTI DATASET
# ============================================================

class NiftiDataset3D(Dataset):
    """3D NIfTI Dataset for volumetric segmentation"""
    def __init__(self, image_dir, label_dir, patient_ids, target_size=(32, 128, 128), normalize=True):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.patient_ids = patient_ids
        self.target_size = target_size
        self.normalize = normalize

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        
        # Search for image files matching this patient ID
        img_path = None
        try:
            for filename in os.listdir(self.image_dir):
                # Remove extension to get base name
                base_name = filename.replace('.nii.gz', '').replace('.nii', '')
                # Check if this file matches the patient ID
                # Match if base_name starts with patient_id (e.g., 10310 matches 10310_0000)
                if base_name == patient_id or base_name.startswith(patient_id + '_'):
                    img_path = os.path.join(self.image_dir, filename)
                    break
        except OSError:
            pass
        
        # Search for label files - labels are typically just the base case ID
        lbl_path = None
        try:
            for filename in os.listdir(self.label_dir):
                # Remove extension to get base name
                base_name = filename.replace('.nii.gz', '').replace('.nii', '')
                # For labels, match exact case or base case + "_" (in case of suffixes)
                if base_name == patient_id or base_name.startswith(patient_id + '_'):
                    lbl_path = os.path.join(self.label_dir, filename)
                    break
        except OSError:
            pass
        
        if img_path is None or lbl_path is None:
            # List available files for debugging
            available_imgs = []
            available_lbls = []
            try:
                available_imgs = os.listdir(self.image_dir)[:5]
            except:
                pass
            try:
                available_lbls = os.listdir(self.label_dir)[:5]
            except:
                pass
            raise FileNotFoundError(
                f"Missing files for patient {patient_id}\n"
                f"  Image path: {img_path}\n"
                f"  Label path: {lbl_path}\n"
                f"  Sample images in dir: {available_imgs}\n"
                f"  Sample labels in dir: {available_lbls}"
            )
        
        # Load NIfTI files
        image = nib.load(img_path).get_fdata(dtype=np.float32)
        mask = nib.load(lbl_path).get_fdata(dtype=np.float32)
        
        # Normalize image
        if self.normalize:
            mean, std = np.mean(image), np.std(image) + 1e-8
            image = (image - mean) / std
        
        # Keep mask as multi-class (0=background, 1=pancreas, 2=tumor)
        mask_multiclass = mask.astype(np.int64)  # Keep original labels
        
        # Permute from (H, W, D) to (D, H, W)
        if image.ndim == 3:
            image = np.transpose(image, (2, 0, 1))
            mask_multiclass = np.transpose(mask_multiclass, (2, 0, 1))
        
        # Convert to tensors
        image_tensor = torch.from_numpy(image).unsqueeze(0).float()  # (1, D, H, W)
        mask_tensor = torch.from_numpy(mask_multiclass).long()  # (D, H, W) with class indices
        
        # Resize image using interpolation
        image_tensor = F.interpolate(
            image_tensor.unsqueeze(0),
            size=self.target_size,
            mode='trilinear',
            align_corners=False
        ).squeeze(0)
        
        # Resize mask using nearest neighbor (preserve class labels)
        mask_tensor = F.interpolate(
            mask_tensor.unsqueeze(0).unsqueeze(0).float(),
            size=self.target_size,
            mode='nearest'
        ).squeeze(0).squeeze(0).long()
        
        return image_tensor, mask_tensor


def get_patient_ids_from_dir(directory):
    """Extract patient IDs from nnUNet format, handling various naming patterns
    
    Handles patterns like:
    - 10310_0000.nii.gz -> 10310
    - 10310_0001_0000.nii.gz -> 10310
    - 10310.nii.gz -> 10310
    """
    patient_ids = set()
    try:
        for filename in os.listdir(directory):
            if not (filename.endswith('.nii.gz') or filename.endswith('.nii')):
                continue
            
            # Remove file extension
            base_name = filename.replace('.nii.gz', '').replace('.nii', '')
            
            # Try multiple patterns to extract patient ID
            # Pattern 1: Standard nnUNet format {case}_{channel}_{modality} (e.g., 10000_0001_0000)
            if '_0001_0000' in base_name:
                patient_id = base_name.split('_0001_0000')[0]
            # Pattern 2: Format {case}_{modality} where modality is numeric (e.g., 10310_0000)
            elif '_' in base_name:
                parts = base_name.split('_')
                # Check if the last part(s) are all numeric (likely modality suffixes)
                last_part = parts[-1]
                if last_part.isdigit() and len(last_part) == 4:
                    # Likely a modality suffix, extract everything before it
                    patient_id = '_'.join(parts[:-1])
                else:
                    # Not clearly a modality suffix, try extracting first part
                    patient_id = parts[0]
            # Pattern 3: Simple case ID with no suffix
            else:
                patient_id = base_name
            
            if patient_id:
                patient_ids.add(patient_id)
    except Exception as e:
        print(f"Warning: Error reading directory {directory}: {e}")
    
    return sorted(list(patient_ids))


def extract_zip_if_needed(config):
    """Extract nnUNet_raw.zip from Google Drive if not already extracted"""
    if IN_COLAB:
        print("\n📦 Handling nnUNet_raw.zip from Google Drive...")
        drive_dir = config['drive_data_dir']
        zip_path = os.path.join(drive_dir, config['zip_name'])
        extract_dir = os.path.join(drive_dir, 'nnUNet_raw')
        
        # Check if already extracted (look for Dataset directories)
        def has_dataset_dirs(path):
            """Recursively check if directory contains Dataset folders"""
            if not os.path.isdir(path):
                return False
            for item in os.listdir(path):
                if item.startswith('Dataset'):
                    return True
                subdir = os.path.join(path, item)
                if os.path.isdir(subdir) and has_dataset_dirs(subdir):
                    return True
            return False
        
        if os.path.exists(extract_dir) and has_dataset_dirs(extract_dir):
            print(f"✅ {extract_dir} already extracted with Dataset directories found")
            config['data_dir'] = extract_dir
            return extract_dir
        
        # Check if zip file exists
        if not os.path.exists(zip_path):
            raise FileNotFoundError(
                f"❌ {zip_path} not found in Google Drive!\n"
                f"   Please upload nnUNet_raw.zip to {drive_dir}"
            )
        
        print(f"📥 Extracting {zip_path} to {extract_dir}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            print(f"✅ Successfully extracted to {extract_dir}")
            config['data_dir'] = extract_dir
            return extract_dir
        except Exception as e:
            raise RuntimeError(f"❌ Failed to extract zip: {str(e)}")
    else:
        # Local path already configured
        config['data_dir'] = config['data_dir']
        return config['data_dir']


def prepare_data_loaders(config):
    """Prepare all patient data from nnUNet format (Dataset501, Dataset502) for 5-fold cross-validation"""
    print("\n📂 Preparing data for cross-validation...")
    
    # Determine dataset structure (Dataset501_*, Dataset502_*, etc.)
    data_dir = config['data_dir']
    
    # Find all Dataset directories (recursively, in case of nested structure)
    def find_dataset_dirs(path, found_dirs=None):
        """Recursively find all Dataset directories"""
        if found_dirs is None:
            found_dirs = []
        
        try:
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path):
                    if item.startswith('Dataset'):
                        found_dirs.append(item_path)
                    else:
                        # Recursively search subdirectories
                        find_dataset_dirs(item_path, found_dirs)
        except PermissionError:
            pass
        
        return found_dirs
    
    dataset_dirs = find_dataset_dirs(data_dir)
    dataset_dirs = sorted(dataset_dirs)
    
    if not dataset_dirs:
        print(f"  ⚠️  Directory structure at {data_dir}:")
        try:
            for item in os.listdir(data_dir)[:10]:
                print(f"    - {item}")
        except:
            pass
        raise ValueError(f"No Dataset directories found in {data_dir} or subdirectories")
    
    all_patient_data = []  # List of (image_dir, label_dir, patient_ids)
    
    for dataset_path in dataset_dirs:
        print(f"\n  Processing {os.path.basename(dataset_path)}...")
        
        image_dir = os.path.join(dataset_path, 'imagesTr')
        label_dir = os.path.join(dataset_path, 'labelsTr')
        
        if not os.path.exists(image_dir) or not os.path.exists(label_dir):
            print(f"    ⚠️  Skipping {os.path.basename(dataset_path)}: missing imagesTr or labelsTr")
            continue
        
        patient_ids = get_patient_ids_from_dir(image_dir)
        print(f"    Found {len(patient_ids)} image patients")
        
        # Validate that each patient ID has both image and label files
        valid_patient_ids = []
        for pid in patient_ids:
            # Check if image file exists for this patient
            img_exists = False
            for fname in os.listdir(image_dir):
                base_name = fname.replace('.nii.gz', '').replace('.nii', '')
                if base_name == pid or base_name.startswith(pid + '_'):
                    img_exists = True
                    break
            
            # Check if label file exists for this patient
            lbl_exists = False
            for fname in os.listdir(label_dir):
                base_name = fname.replace('.nii.gz', '').replace('.nii', '')
                if base_name == pid or base_name.startswith(pid + '_'):
                    lbl_exists = True
                    break
            
            # Only include if both exist
            if img_exists and lbl_exists:
                valid_patient_ids.append(pid)
        
        print(f"    Valid patients (with both image & label): {len(valid_patient_ids)}/{len(patient_ids)}")
        if len(valid_patient_ids) < len(patient_ids):
            skipped = set(patient_ids) - set(valid_patient_ids)
            print(f"    ⚠️  Skipped {len(skipped)} patients without both image and label files")
        
        all_patient_data.append({
            'image_dir': image_dir,
            'label_dir': label_dir,
            'patient_ids': valid_patient_ids  # Use validated patient IDs only
        })
    
    if not all_patient_data:
        raise ValueError("No patient data found!")
    
    return all_patient_data


def create_cv_dataloaders(patient_data_list, fold_indices, config, is_train=True):
    """
    Create data loaders for a specific fold
    patient_data_list: List of dict with image_dir, label_dir, patient_ids
    fold_indices: Tuple of (train_indices, val_indices) or just train_indices for test
    is_train: Boolean indicating if this is for training or testing
    """
    datasets = []
    all_patients = []
    
    # Collect all patients
    for task_data in patient_data_list:
        all_patients.extend(task_data['patient_ids'])
    
    # Get indices for this fold
    if is_train:
        indices = fold_indices[0]  # train indices
    else:
        indices = fold_indices[1]  # val indices
    
    fold_patients = [all_patients[i] for i in indices]
    
    # Create datasets for each task with fold patients
    for task_data in patient_data_list:
        # Filter patients for this fold that exist in this task
        task_fold_patients = [p for p in fold_patients if p in task_data['patient_ids']]
        
        if task_fold_patients:
            dataset = NiftiDataset3D(
                image_dir=task_data['image_dir'],
                label_dir=task_data['label_dir'],
                patient_ids=task_fold_patients,
                target_size=config['input_size_3d'],
                normalize=True
            )
            datasets.append(dataset)
    
    if not datasets:
        print(f"    ⚠️  No patients for this fold")
        return None
    
    # Combine datasets
    from torch.utils.data import ConcatDataset
    combined_dataset = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    
    # Create data loader
    loader = DataLoader(
        combined_dataset,
        batch_size=config['batch_size'],
        shuffle=is_train,
        num_workers=config['num_workers'],
        pin_memory=True if device.type == 'cuda' else False
    )
    
    return loader


# ============================================================
# 8. TRAINING & VALIDATION FUNCTIONS
# ============================================================

def train_epoch(model, train_loader, optimizer, criterion, scaler, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_dice = 0
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        # Use new torch.amp API with device_type
        device_type = 'cuda' if device.type == 'cuda' else 'cpu'
        with autocast(device_type=device_type, enabled=CONFIG['mixed_precision']):
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        dice = dice_score(outputs.detach(), masks)
        total_dice += dice
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dice': f'{dice:.4f}'})
    
    avg_loss = total_loss / len(train_loader)
    avg_dice = total_dice / len(train_loader)
    return avg_loss, avg_dice


def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    total_dice = 0
    total_iou = 0
    total_sens = 0
    total_spec = 0
    total_prec = 0
    total_rec = 0
    total_f1 = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating", leave=False)
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            total_loss += loss.item()
            dice = dice_score(outputs, masks)
            iou = iou_score(outputs, masks)
            sens, spec = sensitivity_specificity(outputs, masks)
            prec, rec, f1 = precision_recall_f1(outputs, masks)
            
            total_dice += dice
            total_iou += iou
            total_sens += sens
            total_spec += spec
            total_prec += prec
            total_rec += rec
            total_f1 += f1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dice': f'{dice:.4f}'})
    
    n_batches = len(val_loader)
    metrics = {
        'loss': total_loss / n_batches,
        'dice': total_dice / n_batches,
        'iou': total_iou / n_batches,
        'sensitivity': total_sens / n_batches,
        'specificity': total_spec / n_batches,
        'precision': total_prec / n_batches,
        'recall': total_rec / n_batches,
        'f1': total_f1 / n_batches,
    }
    
    return metrics


def train_fold(model, train_loader, val_loader, config, device, fold_num, cv_results):
    """Train model for one fold"""
    print(f"\n{'='*60}")
    print(f"FOLD {fold_num + 1}/{config['n_splits']}")
    print(f"{'='*60}")
    
    if train_loader is None or val_loader is None:
        print(f"⚠️  Skipping fold {fold_num + 1}: insufficient data")
        return cv_results
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Model setup
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Loss function: Cross-Entropy for multi-class segmentation
    criterion = nn.CrossEntropyLoss(reduction='mean')
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    scaler = GradScaler()
    
    best_val_loss = float('inf')
    best_epoch = 0
    fold_history = []
    
    for epoch in range(config['epochs']):
        print(f"\n  Epoch {epoch + 1}/{config['epochs']}")
        
        train_loss, train_dice = train_epoch(model, train_loader, optimizer, criterion, scaler, device)
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        scheduler.step(val_metrics['loss'])
        
        # Log metrics
        log_dict = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_dice': train_dice,
            **val_metrics
        }
        fold_history.append(log_dict)
        
        print(f"    Train Loss: {train_loss:.6f} | Train Dice: {train_dice:.4f}")
        print(f"    Val Loss: {val_metrics['loss']:.6f} | Val Dice: {val_metrics['dice']:.4f}")
        print(f"    Val IoU: {val_metrics['iou']:.4f} | Val F1: {val_metrics['f1']:.4f}")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_epoch = epoch + 1
            model_path = os.path.join(
                config['output_dir'],
                f'fold_{fold_num + 1}_best_model.pth'
            )
            torch.save(model.state_dict(), model_path)
            print(f"    ✅ Best model saved to {model_path}")
        
        # Periodic checkpoint
        if (epoch + 1) % config['save_frequency'] == 0:
            ckpt_path = os.path.join(
                config['output_dir'],
                f'fold_{fold_num + 1}_epoch_{epoch + 1}.pth'
            )
            torch.save(model.state_dict(), ckpt_path)
    
    # Save fold history
    history_df = pd.DataFrame(fold_history)
    history_path = os.path.join(
        config['output_dir'],
        f'fold_{fold_num + 1}_history.csv'
    )
    history_df.to_csv(history_path, index=False)
    
    cv_results[f'fold_{fold_num + 1}_best_epoch'] = best_epoch
    cv_results[f'fold_{fold_num + 1}_best_loss'] = best_val_loss
    
    return cv_results


# ============================================================
# 9. VISUALIZATION & REPORTING
# ============================================================

def plot_cv_results(cv_results, output_dir):
    """Plot cross-validation results"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    fold_nums = list(range(1, cv_results['n_splits'] + 1))
    best_epochs = [cv_results[f'fold_{i}_best_epoch'] for i in fold_nums]
    best_losses = [cv_results[f'fold_{i}_best_loss'] for i in fold_nums]
    
    axes[0].bar(fold_nums, best_epochs, color='steelblue')
    axes[0].set_xlabel('Fold')
    axes[0].set_ylabel('Best Epoch')
    axes[0].set_title('Best Epoch per Fold')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].bar(fold_nums, best_losses, color='coral')
    axes[1].set_xlabel('Fold')
    axes[1].set_ylabel('Best Loss')
    axes[1].set_title('Best Loss per Fold')
    axes[1].grid(True, alpha=0.3)
    
    fig.tight_layout()
    output_path = os.path.join(output_dir, 'cv_results.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 CV results plot saved to {output_path}")
    plt.close()


# ============================================================
# 10. MAIN EXECUTION - 5-FOLD CROSS-VALIDATION
# ============================================================

def main():
    """Main execution with 5-fold cross-validation"""
    print("\n" + "=" * 60)
    print("🚀 Starting 5-Fold Cross-Validation Training")
    print("=" * 60)
    
    try:
        # Extract zip from Drive if in Colab
        extract_zip_if_needed(CONFIG)
        
        # Prepare patient data
        patient_data_list = prepare_data_loaders(CONFIG)
        
        # Collect all patient IDs for CV folding
        all_patients = []
        for task_data in patient_data_list:
            all_patients.extend(task_data['patient_ids'])
        
        n_patients = len(all_patients)
        print(f"\n📊 Total patients: {n_patients}")
        
        # Setup 5-fold cross-validation
        kfold = KFold(n_splits=CONFIG['n_splits'], shuffle=True, random_state=42)
        fold_generator = kfold.split(np.arange(n_patients))
        
        cv_results = {'n_splits': CONFIG['n_splits']}
        all_fold_histories = []
        
        # Training loop for each fold
        for fold_num, (train_idx, val_idx) in enumerate(fold_generator):
            print(f"\n{'='*60}")
            print(f"Preparing Fold {fold_num + 1}/{CONFIG['n_splits']}")
            print(f"{'='*60}")
            print(f"  Train samples: {len(train_idx)}")
            print(f"  Val samples: {len(val_idx)}")
            
            # Create data loaders for this fold
            train_loader = create_cv_dataloaders(
                patient_data_list,
                (train_idx.tolist(), val_idx.tolist()),
                CONFIG,
                is_train=True
            )
            val_loader = create_cv_dataloaders(
                patient_data_list,
                (train_idx.tolist(), val_idx.tolist()),
                CONFIG,
                is_train=False
            )
            
            # Initialize model for this fold
            model = TransAttUNetPlusPlus3D(
                in_channels=CONFIG['in_channels'],
                out_channels=CONFIG['out_channels']
            )
            
            # Train this fold
            cv_results = train_fold(
                model, train_loader, val_loader,
                CONFIG, device, fold_num, cv_results
            )
            
            # Clear GPU cache
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Save CV results summary
        cv_summary_path = os.path.join(CONFIG['output_dir'], 'cv_summary.json')
        with open(cv_summary_path, 'w') as f:
            json.dump(cv_results, f, indent=4)
        print(f"\n📄 CV summary saved to {cv_summary_path}")
        
        # Plot CV results
        plot_cv_results(cv_results, CONFIG['output_dir'])
        
        print("\n" + "=" * 60)
        print("✅ 5-Fold Cross-Validation Training Complete!")
        print("=" * 60)
        print(f"\n📁 Results saved to: {CONFIG['output_dir']}")
        
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
