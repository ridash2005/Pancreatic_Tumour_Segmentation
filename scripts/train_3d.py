"""
TransAttUNet++ 3D - Patient-Aware & 5-Fold Cross Validation
-----------------------------------------------------------
This script extends the TransAttUNet++ architecture to handle 3D Volumetric Data (NIfTI).
Crucially, it guarantees "Patient Awareness" by splitting the data into 5 Folds based
on the PATIENT ID rather than individual 2D slices, completely preventing Data Leakage.
"""

import os
import glob
import time
import math
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

# ==========================================================
# 1. 3D ARCHITECTURE COMPONENTS (TransAttUNet++ 3D)
# ==========================================================

class DoubleConv3D(nn.Module):
    """(Conv3d -> BN -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

def get_1d_sincos_pos_embed_from_grid(embed_dim, N, device=None, dtype=torch.float32):
    """
    Since we flatten the 3D grid (D, H, W) into a 1D sequence of length N,
    we can use a 1D sequence positional embedding for the Transformer tokens.
    """
    position = torch.arange(N, dtype=dtype, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=dtype, device=device) * (-math.log(10000.0) / embed_dim))
    emb = torch.zeros((N, embed_dim), dtype=dtype, device=device)
    emb[:, 0::2] = torch.sin(position * div_term)
    emb[:, 1::2] = torch.cos(position * div_term)
    return emb

class AttentionBlock3D(nn.Module):
    """3D Attention gating"""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.psi(g1 + x1)
        return x * psi

class TransAttUNetPlusPlus3D(nn.Module):
    """
    TransAttUNetPlusPlus adapted for 3D Volumetric Medical Images.
    Takes 5D tensors: (Batch, Channels, Depth, Height, Width)
    """
    def __init__(self, in_channels=1, out_channels=1, filters=[32, 64, 128, 256, 512],
                 trans_depth=2, trans_heads=4, trans_mlp_ratio=4.0, dropout=0.1):
        # NOTE: Filter sizes are halved compared to 2D to fit in Colab GPU Memory (3D is heavy!)
        super().__init__()
        f1, f2, f3, f4, f5 = filters

        # Encoder
        self.enc1 = DoubleConv3D(in_channels, f1)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = DoubleConv3D(f1, f2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = DoubleConv3D(f2, f3)
        self.pool3 = nn.MaxPool3d(2)
        self.enc4 = DoubleConv3D(f3, f4)
        self.pool4 = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck_conv = DoubleConv3D(f4, f5)
        self.embed_dim = f5

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim,
                                                   nhead=trans_heads,
                                                   dim_feedforward=int(self.embed_dim * trans_mlp_ratio),
                                                   dropout=dropout,
                                                   activation='gelu',
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=trans_depth)

        # Upsampling
        self.up4 = nn.ConvTranspose3d(f5, f4, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose3d(f4, f3, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose3d(f3, f2, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose3d(f2, f1, kernel_size=2, stride=2)

        # Attention blocks
        self.att4 = AttentionBlock3D(F_g=f4, F_l=f4, F_int=f4 // 2)
        self.att3 = AttentionBlock3D(F_g=f3, F_l=f3, F_int=f3 // 2)
        self.att2 = AttentionBlock3D(F_g=f2, F_l=f2, F_int=f2 // 2)
        self.att1 = AttentionBlock3D(F_g=f1, F_l=f1, F_int=f1 // 2)

        # Decoder (UNet++ dense connections)
        self.x3_1 = DoubleConv3D(f4 + f4, f4)
        
        self.x2_1 = DoubleConv3D(f3 + f3, f3)
        self.x2_2 = DoubleConv3D(f3 * 3, f3)
        
        self.x1_1 = DoubleConv3D(f2 + f2, f2)
        self.x1_2 = DoubleConv3D(f2 * 3, f2)
        self.x1_3 = DoubleConv3D(f2 * 4, f2)
        
        self.x0_1 = DoubleConv3D(f1 + f1, f1)
        self.x0_2 = DoubleConv3D(f1 * 3, f1)
        self.x0_3 = DoubleConv3D(f1 * 4, f1)
        self.x0_4 = DoubleConv3D(f1 * 5, f1)

        self.final = nn.Conv3d(f1, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)                                 # (B, f1, D, H, W)
        e2 = self.enc2(self.pool1(e1))                    # (B, f2, D/2, H/2, W/2)
        e3 = self.enc3(self.pool2(e2))                    # (B, f3, D/4, H/4, W/4)
        e4 = self.enc4(self.pool3(e3))                    # (B, f4, D/8, H/8, W/8)
        b  = self.bottleneck_conv(self.pool4(e4))         # (B, f5, D/16, H/16, W/16)

        B, C, D, H, W = b.shape
        N = D * H * W
        
        # Flatten spatial dims to tokens: (B, N, C)
        tokens = b.view(B, C, N).transpose(1, 2)
        
        # Add Positional Embedding
        pos_emb = get_1d_sincos_pos_embed_from_grid(self.embed_dim, N, device=tokens.device, dtype=tokens.dtype)
        pos_emb = pos_emb.unsqueeze(0).expand(B, -1, -1)
        tokens = tokens + pos_emb
        
        # Global Context via Transformer
        tokens = self.transformer(tokens)
        
        # Reshape back to 3D volume
        tokens = tokens.transpose(1, 2).view(B, C, D, H, W)

        # UNet++ Decoder + Attention
        up4_b = self.up4(tokens)
        att_e4 = self.att4(up4_b, e4)
        x3_1 = self.x3_1(torch.cat([up4_b, att_e4], dim=1))

        up3_x3_1 = self.up3(x3_1)
        att_e3 = self.att3(up3_x3_1, e3)
        x2_1 = self.x2_1(torch.cat([up3_x3_1, att_e3], dim=1))

        up3_x3_1_again = self.up3(x3_1)
        x2_2 = self.x2_2(torch.cat([e3, x2_1, up3_x3_1_again], dim=1))

        up2_x2_1 = self.up2(x2_1)
        att_e2 = self.att2(up2_x2_1, e2)
        x1_1 = self.x1_1(torch.cat([up2_x2_1, att_e2], dim=1))

        up2_x2_2 = self.up2(x2_2)
        x1_2 = self.x1_2(torch.cat([e2, x1_1, up2_x2_2], dim=1))
        x1_3 = self.x1_3(torch.cat([e2, x1_1, x1_2, up2_x2_2], dim=1))

        up1_x1_1 = self.up1(x1_1)
        att_e1 = self.att1(up1_x1_1, e1)
        x0_1 = self.x0_1(torch.cat([up1_x1_1, att_e1], dim=1))

        up1_x1_2 = self.up1(x1_2)
        x0_2 = self.x0_2(torch.cat([e1, x0_1, up1_x1_2], dim=1))

        up1_x1_3 = self.up1(x1_3)
        x0_3 = self.x0_3(torch.cat([e1, x0_1, x0_2, up1_x1_3], dim=1))
        x0_4 = self.x0_4(torch.cat([e1, x0_1, x0_2, x0_3, up1_x1_3], dim=1))

        out = self.final(x0_4)
        return out


# ==========================================================
# 2. PATIENT-AWARE 3D DATASET
# ==========================================================

class PatientAwareNIfTIDataset(Dataset):
    def __init__(self, image_dir, label_dir, patient_ids, target_size=(32, 128, 128)):
        """
        patient_ids: List of valid patient IDs (e.g. ['pancreas_001', 'pancreas_005'])
                     This ensures strict Patient-Level splitting.
        target_size: To fit a 3D volume into Colab GPUs, resizing or 3D patch cropping is required.
                     (Depth, Height, Width).
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.patient_ids = patient_ids
        self.target_size = target_size

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pat_id = self.patient_ids[idx]
        
        # Assuming nnU-Net format: images have _0000.nii.gz, labels have .nii.gz
        img_path = os.path.join(self.image_dir, f"{pat_id}_0000.nii.gz")
        lbl_path = os.path.join(self.label_dir, f"{pat_id}.nii.gz")

        # Load NIfTI
        img_nii = nib.load(img_path)
        lbl_nii = nib.load(lbl_path)
        
        image = img_nii.get_fdata(dtype=np.float32)
        mask_arr = lbl_nii.get_fdata(dtype=np.float32)

        # Normalize Image (Z-score normalization is common for MRI/CT)
        mean, std = np.mean(image), np.std(image) + 1e-8
        image = (image - mean) / std

        # Ensure mask is binary
        mask_bin = ((mask_arr == 1.0) | (mask_arr == 2.0)).astype(np.float32)

        # Convert to Tensors: Shape (Channels, Depth, Height, Width)
        # NIfTI is usually (H, W, D), transpose to (D, H, W) for PyTorch
        image = np.transpose(image, (2, 0, 1))
        mask_bin = np.transpose(mask_bin, (2, 0, 1))
        
        img_tensor = torch.tensor(image).unsqueeze(0)  # (1, D, H, W)
        mask_tensor = torch.tensor(mask_bin).unsqueeze(0) # (1, D, H, W)

        # Interpolate to target size to prevent GPU OOM
        # mode='trilinear' for images, 'nearest' for masks
        img_tensor = F.interpolate(img_tensor.unsqueeze(0), size=self.target_size, mode='trilinear', align_corners=False).squeeze(0)
        mask_tensor = F.interpolate(mask_tensor.unsqueeze(0), size=self.target_size, mode='nearest').squeeze(0)

        return img_tensor, mask_tensor

# ==========================================================
# 3. METRICS AND LOSS
# ==========================================================

class DiceLoss3D(nn.Module):
    def forward(self, logits, target):
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        target = target.view(target.size(0), -1)
        inter = (probs * target).sum(1)
        union = probs.sum(1) + target.sum(1)
        dice = (2 * inter + 1e-6) / (union + 1e-6)
        return 1 - dice.mean()

def dice_metric_3d(pred, target, eps=1e-6):
    pred = (torch.sigmoid(pred) > 0.5).float()
    intersection = (pred * target).sum((1,2,3,4))
    union = pred.sum((1,2,3,4)) + target.sum((1,2,3,4))
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean().item()

# ==========================================================
# 4. 5-FOLD CROSS-VALIDATION PIPELINE
# ==========================================================

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Configure these paths to your extracted 3D dataset
    IMAGES_DIR = "Dataset/PANTHER_NIfTI/Task1/imagesTr"
    LABELS_DIR = "Dataset/PANTHER_NIfTI/Task1/labelsTr"
    
    if not os.path.exists(IMAGES_DIR) or not os.path.exists(LABELS_DIR):
        print("Error: 3D Image and Label directories not found. Please place NIfTI data in nnUNet_raw format.")
        exit(1)

    # 1. EXTRACT UNIQUE PATIENT IDs
    all_files = os.listdir(IMAGES_DIR)
    patient_ids = sorted([f.replace("_0000.nii.gz", "") for f in all_files if f.endswith("_0000.nii.gz")])
    
    print(f"Total Unique Patients Found: {len(patient_ids)}")
    if len(patient_ids) < 5:
        print("Need at least 5 patients to perform 5-fold cross validation!")
        exit(1)

    # 2. INITIALIZE 5-FOLD CROSS VALIDATION
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    
    EPOCHS = 1
    BATCH_SIZE = 1 # 3D needs very small batches

    for fold, (train_idx, val_idx) in enumerate(kf.split(patient_ids)):
        print(f"\n======================================")
        print(f"       STARTING FOLD {fold + 1} / 5")
        print(f"======================================")
        
        train_patients = [patient_ids[i] for i in train_idx]
        val_patients = [patient_ids[i] for i in val_idx]
        
        print(f"Train Patients: {len(train_patients)} | Val Patients: {len(val_patients)}")

        # Create Patient-Aware Datasets
        train_dataset = PatientAwareNIfTIDataset(IMAGES_DIR, LABELS_DIR, train_patients)
        val_dataset = PatientAwareNIfTIDataset(IMAGES_DIR, LABELS_DIR, val_patients)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Initialize Model, Loss, Optimizer required specifically for THIS fold
        model = TransAttUNetPlusPlus3D(in_channels=1, out_channels=1).to(device)
        criterion_bce = nn.BCEWithLogitsLoss()
        criterion_dice = DiceLoss3D()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scaler = GradScaler()
        
        best_val_dice = -1
        
        # Training Loop
        for epoch in range(EPOCHS):
            model.train()
            train_loss_total = 0
            
            pbar = tqdm(train_loader, desc=f"F{fold+1} Ep {epoch+1}/{EPOCHS}", leave=False)
            for imgs, masks in pbar:
                imgs, masks = imgs.to(device), masks.to(device)
                optimizer.zero_grad()
                
                with autocast():
                    preds = model(imgs)
                    loss = criterion_bce(preds, masks) + criterion_dice(preds, masks)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                train_loss_total += loss.item()
                pbar.set_postfix({'loss': loss.item()})
                
            train_loss_avg = train_loss_total / len(train_loader)
            
            # Validation Loop
            model.eval()
            val_loss_total = 0
            val_dice_total = 0
            
            with torch.no_grad():
                for imgs, masks in val_loader:
                    imgs, masks = imgs.to(device), masks.to(device)
                    preds = model(imgs)
                    
                    loss = criterion_bce(preds, masks) + criterion_dice(preds, masks)
                    val_loss_total += loss.item()
                    val_dice_total += dice_metric_3d(preds, masks)
                    
            val_loss_avg = val_loss_total / len(val_loader)
            val_dice_avg = val_dice_total / len(val_loader)
            
            print(f"Fold {fold+1} | Ep {epoch+1}/{EPOCHS} | Train Loss: {train_loss_avg:.4f} | Val Loss: {val_loss_avg:.4f} | Val Dice: {val_dice_avg:.4f}")
            
            if val_dice_avg > best_val_dice:
                best_val_dice = val_dice_avg
                torch.save(model.state_dict(), f"transattunet_3d_fold{fold}.pth")
                
        print(f"Fold {fold+1} Complete. Best Validation Dice: {best_val_dice:.4f}")
        fold_results.append(best_val_dice)
        
    # FINAL RESULTS
    print("\n======================================")
    print("   5-FOLD CROSS VALIDATION RESULTS")
    print("======================================")
    for i, res in enumerate(fold_results):
        print(f"Fold {i+1} Best Dice: {res:.4f}")
    
    print(f"\nAverage 5-Fold Cross-Validation Dice: {np.mean(fold_results):.4f} +/- {np.std(fold_results):.4f}")
    
    # Save to Excel
    df = pd.DataFrame({"Fold": [1,2,3,4,5], "Best Val Dice": fold_results})
    df.loc[len(df)] = ["Average", np.mean(fold_results)]
    df.to_csv("transattunet_3d_5fold_results.csv", index=False)
    print("✅ Results saved to transattunet_3d_5fold_results.csv")
