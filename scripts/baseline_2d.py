import zipfile, pydicom, io, numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import os

# Configuration (replace with local paths as needed)
zip_path = 'DCM.zip'

if not os.path.exists(zip_path):
    print(f"Warning: {zip_path} not found. Please ensure it exists in the current directory.")

if os.path.exists(zip_path):
    z = zipfile.ZipFile(zip_path)

    img_slice_paths = [f for f in z.namelist()
                       if f.startswith('DCM/Task1/ImagesTr/') and f.lower().endswith('.dcm')]

    folders = sorted(set(f.split('/')[3] for f in img_slice_paths))
    print(f'Folders found: {folders[:10]}... Total: {len(folders)}')

    img_files, lbl_files = [], []
    for folder in folders:
        imgs = [f for f in img_slice_paths if f.startswith(f'DCM/Task1/ImagesTr/{folder}/IMG')]
        lbls = [f for f in z.namelist() if f.startswith(f'DCM/Task1/LabelsTr/{folder}/IMG')]
        img_dict = {os.path.basename(f): f for f in imgs}
        lbl_dict = {os.path.basename(f): f for f in lbls}
        for fname in img_dict:
            if fname in lbl_dict:
                img_files.append(img_dict[fname])
                lbl_files.append(lbl_dict[fname])

    print(f'Total paired slices: {len(img_files)}')

    train_imgs, val_imgs, train_lbls, val_lbls = train_test_split(
        img_files, lbl_files, test_size=0.2, random_state=42
    )

    print(f'Train: {len(train_imgs)}, Val: {len(val_imgs)}')

class ZipDicomSliceDataset(Dataset):
    def __init__(self, zip_path, img_files, lbl_files, target_size=(256,256)):
        self.zip_path = zip_path
        self.img_files = img_files
        self.lbl_files = lbl_files
        self.target_size = target_size
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize(self.target_size, interpolation=Image.NEAREST)
        self.z = zipfile.ZipFile(zip_path)
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_bytes = self.z.read(self.img_files[idx])
        mask_bytes = self.z.read(self.lbl_files[idx])

        img_dcm = pydicom.dcmread(io.BytesIO(img_bytes))
        mask_dcm = pydicom.dcmread(io.BytesIO(mask_bytes))

        image = img_dcm.pixel_array.astype(np.float32)
        image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-7)

        mask_arr = mask_dcm.pixel_array.astype(np.float32)
        mask_bin = ((mask_arr == 1.0) | (mask_arr == 2.0)).astype(np.float32)

        image = self.resize(Image.fromarray((image * 255).astype(np.uint8)))
        mask = self.resize(Image.fromarray((mask_bin * 255).astype(np.uint8)))

        image = self.to_tensor(image)
        mask = self.to_tensor(mask)
        mask = (mask > 0.5).float()

        if idx < 3:
            print(f"Sample {idx}: image shape {image.shape}, mask unique {mask.unique().tolist()}")

        return image, mask

if os.path.exists(zip_path):
    train_loader = DataLoader(
        ZipDicomSliceDataset(zip_path, train_imgs, train_lbls),
        batch_size=16, shuffle=True, num_workers=1, drop_last=False
    )
    val_loader = DataLoader(
        ZipDicomSliceDataset(zip_path, val_imgs, val_lbls),
        batch_size=16, shuffle=False, num_workers=1, drop_last=False
    )

    imgs, masks = next(iter(train_loader))
    print(f"Batch shape: {imgs.shape}, {masks.shape}")

import torch
import torch.nn as nn
import math

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

def get_2d_sincos_pos_embed(embed_dim, grid_h, grid_w, device=None, dtype=torch.float32):
    if embed_dim % 2 != 0:
        raise ValueError("Embed dim must be even for sincos pos embedding.")
    grid_y = torch.arange(grid_h, dtype=dtype, device=device)
    grid_x = torch.arange(grid_w, dtype=dtype, device=device)
    yy, xx = torch.meshgrid(grid_y, grid_x, indexing='ij')
    grid = torch.stack([yy, xx], dim=-1).reshape(-1, 2)

    N = grid.shape[0]
    emb = torch.zeros((N, embed_dim), dtype=dtype, device=device)
    half = embed_dim // 2
    
    def angle_rates(dim):
        idx = torch.arange(dim, dtype=dtype, device=device)
        return 1.0 / (10000 ** (2 * (idx // 2) / dim))

    dim_y = half
    rates_y = angle_rates(dim_y)
    angles_y = grid[:, 0:1] * rates_y.unsqueeze(0)
    emb[:, :dim_y:2] = torch.sin(angles_y[:, ::2])
    emb[:, 1:dim_y:2] = torch.cos(angles_y[:, ::2])

    dim_x = half
    rates_x = angle_rates(dim_x)
    angles_x = grid[:, 1:2] * rates_x.unsqueeze(0)
    emb[:, dim_y:dim_y+dim_x:2] = torch.sin(angles_x[:, ::2])
    emb[:, dim_y+1:dim_y+dim_x:2] = torch.cos(angles_x[:, ::2])
    return emb

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.psi(g1 + x1)
        return x * psi

class TransAttUNetPlusPlus(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, filters=[64,128,256,512,1024],
                 trans_depth=4, trans_heads=8, trans_mlp_ratio=4.0, trans_embed_dim=None, dropout=0.0):
        super().__init__()
        f1, f2, f3, f4, f5 = filters
        self.filters = filters

        self.enc1 = DoubleConv(in_channels, f1)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(f1, f2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(f2, f3)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(f3, f4)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck_conv = DoubleConv(f4, f5)

        if trans_embed_dim is None:
            self.embed_dim = f5
        else:
            self.embed_dim = trans_embed_dim

        if self.embed_dim == f5:
            self.proj_to_embed = nn.Identity()
            self.proj_back = nn.Identity()
        else:
            self.proj_to_embed = nn.Linear(f5, self.embed_dim)
            self.proj_back = nn.Linear(self.embed_dim, f5)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim,
                                                   nhead=trans_heads,
                                                   dim_feedforward=int(self.embed_dim * trans_mlp_ratio),
                                                   dropout=dropout,
                                                   activation='gelu',
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=trans_depth)

        self.up4 = nn.ConvTranspose2d(f5, f4, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(f4, f3, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(f3, f2, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(f2, f1, kernel_size=2, stride=2)

        self.att4 = AttentionBlock(F_g=f4, F_l=f4, F_int=f4 // 2)
        self.att3 = AttentionBlock(F_g=f3, F_l=f3, F_int=f3 // 2)
        self.att2 = AttentionBlock(F_g=f2, F_l=f2, F_int=f2 // 2)
        self.att1 = AttentionBlock(F_g=f1, F_l=f1, F_int=f1 // 2)

        self.x3_1 = DoubleConv(f4 + f4, f4)
        self.x2_1 = DoubleConv(f3 + f3, f3)
        self.x2_2 = DoubleConv(f3 * 3, f3)
        self.x1_1 = DoubleConv(f2 + f2, f2)
        self.x1_2 = DoubleConv(f2 * 3, f2)
        self.x1_3 = DoubleConv(f2 * 4, f2)
        self.x0_1 = DoubleConv(f1 + f1, f1)
        self.x0_2 = DoubleConv(f1 * 3, f1)
        self.x0_3 = DoubleConv(f1 * 4, f1)
        self.x0_4 = DoubleConv(f1 * 5, f1)
        self.final = nn.Conv2d(f1, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b  = self.bottleneck_conv(self.pool4(e4))

        B, C, H, W = b.shape
        tokens = b.flatten(2).transpose(1, 2)

        if isinstance(self.proj_to_embed, nn.Linear):
            tokens = self.proj_to_embed(tokens)
        else:
            tokens = self.proj_to_embed(tokens)

        pos_emb = get_2d_sincos_pos_embed(self.embed_dim, H, W, device=tokens.device, dtype=tokens.dtype)
        pos_emb = pos_emb.unsqueeze(0).expand(B, -1, -1)
        tokens = tokens + pos_emb

        tokens = self.transformer(tokens)

        if isinstance(self.proj_back, nn.Linear):
            tokens = self.proj_back(tokens)
        else:
            tokens = self.proj_back(tokens)

        tokens = tokens.transpose(1, 2).view(B, C, H, W)

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

def _threshold(pred, thr=0.5):
    pred = torch.sigmoid(pred)
    return (pred > thr).float()

def dice_metric(pred, target, eps=1e-6):
    pred = _threshold(pred)
    intersection = (pred * target).sum((1,2,3))
    union = pred.sum((1,2,3)) + target.sum((1,2,3))
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean()

def accuracy_metric(pred, target, eps=1e-6):
    pred = _threshold(pred)
    correct = (pred == target).float().sum((1,2,3))
    total = torch.tensor(target[0].numel(), device=target.device)
    acc = correct / (total + eps)
    return acc.mean()

def precision_metric(pred, target, eps=1e-6):
    pred = _threshold(pred)
    TP = (pred * target).sum((1,2,3))
    FP = (pred * (1 - target)).sum((1,2,3))
    precision = (TP + eps) / (TP + FP + eps)
    return precision.mean()

def sensitivity_metric(pred, target, eps=1e-6):
    pred = _threshold(pred)
    TP = (pred * target).sum((1,2,3))
    FN = ((1 - pred) * target).sum((1,2,3))
    sensitivity = (TP + eps) / (TP + FN + eps)
    return sensitivity.mean()

def specificity_metric(pred, target, eps=1e-6):
    pred = _threshold(pred)
    TN = ((1 - pred) * (1 - target)).sum((1,2,3))
    FP = (pred * (1 - target)).sum((1,2,3))
    specificity = (TN + eps) / (TN + FP + eps)
    return specificity.mean()

def f1_score_metric(pred, target, eps=1e-6):
    pred = _threshold(pred)
    TP = (pred * target).sum((1,2,3))
    FP = (pred * (1 - target)).sum((1,2,3))
    FN = ((1 - pred) * target).sum((1,2,3))
    precision = (TP + eps) / (TP + FP + eps)
    recall = (TP + eps) / (TP + FN + eps)
    f1 = (2 * precision * recall + eps) / (precision + recall + eps)
    return f1.mean()

class DiceLoss(nn.Module):
    def forward(self, logits, target):
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        target = target.view(target.size(0), -1)
        inter = (probs * target).sum(1)
        union = probs.sum(1) + target.sum(1)
        dice = (2 * inter + 1e-6) / (union + 1e-6)
        return 1 - dice.mean()

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALB_AVAILABLE = True
except:
    ALB_AVAILABLE = False

if ALB_AVAILABLE:
    train_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Affine(scale=(0.9, 1.1), rotate=(-15, 15), translate_percent=(0, 0), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
        A.OneOf([
            A.Blur(blur_limit=3),
            A.MedianBlur(blur_limit=3),
            A.MotionBlur(blur_limit=3),
        ], p=0.3),
        A.Normalize(mean=0.0, std=1.0),
        ToTensorV2()
    ])
    val_transforms = A.Compose([
        A.Normalize(mean=0.0, std=1.0),
        ToTensorV2()
    ])
else:
    train_transforms = None
    val_transforms = None

def show_sample_prediction(model, val_loader, epoch, device):
    model.eval()
    with torch.no_grad():
        imgs, masks = next(iter(val_loader))
        imgs, masks = imgs.to(device), masks.to(device)
        preds = torch.sigmoid(model(imgs))
        preds = (preds > 0.5).float()

    img = imgs[0].cpu().squeeze()
    mask = masks[0].cpu().squeeze()
    pred = preds[0].cpu().squeeze()

    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.imshow(img, cmap="gray"); plt.title("Input"); plt.axis("off")
    plt.subplot(1,3,2); plt.imshow(mask, cmap="gray"); plt.title("Ground Truth"); plt.axis("off")
    plt.subplot(1,3,3); plt.imshow(pred, cmap="gray"); plt.title(f"Pred (Epoch {epoch+1})"); plt.axis("off")
    plt.show()

if __name__ == "__main__":
    import time
    from tqdm import tqdm
    from torch.cuda.amp import autocast, GradScaler
    import pandas as pd

    log_data = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = TransAttUNetPlusPlus(in_channels=1, out_channels=1).to(device)
    
    # Hyperparameters
    lr = 1e-4
    epochs = 10 
    
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_dice = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    scaler = GradScaler()

    if os.path.exists(zip_path):
        train_losses, val_losses, dice_scores = [], [], []
        best_val_dice = -1

        for epoch in range(epochs):
            start = time.time()
            model.train()
            total_loss = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            for imgs, masks in pbar:
                imgs, masks = imgs.to(device), masks.to(device).float()
                optimizer.zero_grad()
                with autocast():
                    preds = model(imgs)
                    loss = criterion_bce(preds, masks) + criterion_dice(preds, masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item() * imgs.size(0)
                pbar.set_postfix({'loss': loss.item()})

            train_loss = total_loss / len(train_loader.dataset)
            train_losses.append(train_loss)

            model.eval()
            val_loss = 0
            dice_total = acc_total = prec_total = sens_total = spec_total = f1_total = 0

            with torch.no_grad():
                for imgs, masks in val_loader:
                    imgs, masks = imgs.to(device), masks.to(device).float()
                    preds = model(imgs)
                    loss = criterion_bce(preds, masks) + criterion_dice(preds, masks)
                    val_loss += loss.item() * imgs.size(0)
                    dice_total += dice_metric(preds, masks).item() * imgs.size(0)
                    acc_total += accuracy_metric(preds, masks).item() * imgs.size(0)
                    prec_total += precision_metric(preds, masks).item() * imgs.size(0)
                    sens_total += sensitivity_metric(preds, masks).item() * imgs.size(0)
                    spec_total += specificity_metric(preds, masks).item() * imgs.size(0)
                    f1_total += f1_score_metric(preds, masks).item() * imgs.size(0)

            N = len(val_loader.dataset)
            val_loss /= N
            dice_epoch, acc_epoch, prec_epoch = dice_total/N, acc_total/N, prec_total/N
            sens_epoch, spec_epoch, f1_epoch = sens_total/N, spec_total/N, f1_total/N

            val_losses.append(val_loss)
            dice_scores.append(dice_epoch)

            print(f"\nEpoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | Dice={dice_epoch:.4f}")

            log_data.append([epoch+1, train_loss, val_loss, dice_epoch, acc_epoch, prec_epoch, sens_epoch, spec_epoch, f1_epoch])
            
            if dice_epoch > best_val_dice:
                best_val_dice = dice_epoch
                torch.save(model.state_dict(), "best_model.pth")

        df_log = pd.DataFrame(log_data, columns=["epoch", "train_loss", "val_loss", "dice", "accuracy", "precision", "sensitivity", "specificity", "f1_score"])
        df_log.to_csv("training_log.csv", index=False)
        print("✅ Training complete.")
    else:
        print("Skipping training because zip_path was not found.")
