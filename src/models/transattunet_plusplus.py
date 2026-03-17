import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# 2D TransAttUNet++
# ==========================================

class DoubleConv2D(nn.Module):
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
    def forward(self, x): return self.double_conv(x)

class AttentionBlock2D(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1, bias=True), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1, bias=True), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(F_int, 1, 1, bias=True), nn.BatchNorm2d(1), nn.Sigmoid())
    def forward(self, g, x):
        psi = self.psi(self.W_g(g) + self.W_x(x))
        return x * psi

class TransAttUNetPlusPlus2D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, filters=[64,128,256,512,1024]):
        super().__init__()
        f1, f2, f3, f4, f5 = filters
        self.enc1 = DoubleConv2D(in_channels, f1)
        self.enc2 = DoubleConv2D(f1, f2)
        self.enc3 = DoubleConv2D(f2, f3)
        self.enc4 = DoubleConv2D(f3, f4)
        self.bottleneck = DoubleConv2D(f4, f5)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=f5, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        self.up4, self.up3, self.up2, self.up1 = [nn.ConvTranspose2d(filters[i], filters[i-1], 2, 2) for i in range(4, 0, -1)]
        self.att4, self.att3, self.att2, self.att1 = [AttentionBlock2D(filters[i-1], filters[i-1], filters[i-1]//2) for i in range(4, 0, -1)]
        
        self.x3_1 = DoubleConv2D(f4*2, f4)
        self.x2_1, self.x2_2 = DoubleConv2D(f3*2, f3), DoubleConv2D(f3*3, f3)
        self.x1_1, self.x1_2, self.x1_3 = DoubleConv2D(f2*2, f2), DoubleConv2D(f2*3, f2), DoubleConv2D(f2*4, f2)
        self.x0_1, self.x0_2, self.x0_3, self.x0_4 = [DoubleConv2D(f1*(i+2), f1) for i in range(4)]
        self.final = nn.Conv2d(f1, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        b = self.bottleneck(F.max_pool2d(e4, 2))
        
        B, C, H, W = b.shape
        tokens = b.flatten(2).transpose(1, 2)
        tokens = self.transformer(tokens)
        b = tokens.transpose(1, 2).view(B, C, H, W)
        
        x3_1 = self.x3_1(torch.cat([self.up4(b), self.att4(self.up4(b), e4)], 1))
        # ... simplified for brevity in this conceptual structure ...
        # (Full logic from architecture.py would go here)
        return self.final(e1) # Placeholder for the full nested logic

# ==========================================
# 3D TransAttUNet++
# ==========================================

class DoubleConv3D(nn.Module):
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
    def forward(self, x): return self.double_conv(x)

class AttentionBlock3D(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv3d(F_g, F_int, 1, bias=True), nn.BatchNorm3d(F_int))
        self.W_x = nn.Sequential(nn.Conv3d(F_l, F_int, 1, bias=True), nn.BatchNorm3d(F_int))
        self.psi = nn.Sequential(nn.ReLU(inplace=True), nn.Conv3d(F_int, 1, 1, bias=True), nn.BatchNorm3d(1), nn.Sigmoid())
    def forward(self, g, x):
        psi = self.psi(self.W_g(g) + self.W_x(x))
        return x * psi

class TransAttUNetPlusPlus3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, filters=[32, 64, 128, 256, 512]):
        super().__init__()
        f1, f2, f3, f4, f5 = filters
        self.enc1 = DoubleConv3D(in_channels, f1)
        self.enc2 = DoubleConv3D(f1, f2)
        self.enc3 = DoubleConv3D(f2, f3)
        self.enc4 = DoubleConv3D(f3, f4)
        self.bottleneck = DoubleConv3D(f4, f5)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=f5, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.up4, self.up3, self.up2, self.up1 = [nn.ConvTranspose3d(filters[i], filters[i-1], 2, 2) for i in range(4, 0, -1)]
        self.att4, self.att3, self.att2, self.att1 = [AttentionBlock3D(filters[i-1], filters[i-1], filters[i-1]//2) for i in range(4, 0, -1)]
        
        # Dense connections
        self.x3_1 = DoubleConv3D(f4*2, f4)
        self.x2_1, self.x2_2 = DoubleConv3D(f3*2, f3), DoubleConv3D(f3*3, f3)
        self.x1_1, self.x1_2, self.x1_3 = DoubleConv3D(f2*2, f2), DoubleConv2D(f2*3, f2), DoubleConv2D(f2*4, f2)
        self.x0_1, self.x0_2, self.x0_3, self.x0_4 = [DoubleConv3D(f1*(i+2), f1) for i in range(4)]
        self.final = nn.Conv3d(f1, out_channels, 1)

    def forward(self, x):
        # Full 3D logic implementation...
        return self.final(x)
