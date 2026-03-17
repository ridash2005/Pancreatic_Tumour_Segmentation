import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.models.transattunet_plusplus import transattunet_3d
from src.data.datasets import PatientAwareNIfTIDataset
from src.utils.metrics import DiceLoss, dice_metric

def train_transattunet():
    # 1. Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Paths (adjust based on your local setup)
    IMAGE_DIR = "nnUNet_raw/Dataset501_PantherTask1/imagesTr"
    LABEL_DIR = "nnUNet_raw/Dataset501_PantherTask1/labelsTr"
    
    # Mock patient IDs for demonstration
    # In a real run, you'd split these based on a CSV or folder list
    all_patients = [f.split('_')[0] for f in os.listdir(IMAGE_DIR) if f.endswith('.nii')]
    train_ids = all_patients[:10]  # Just 10 patients for demo
    
    # 2. Dataset and Loader
    dataset = PatientAwareNIfTIDataset(IMAGE_DIR, LABEL_DIR, train_ids)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # 3. Model, Loss, Optimizer
    model = transattunet_3d(in_ch=1, out_ch=1).to(device)
    criterion = DiceLoss(is_3d=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # 4. Training Loop
    model.train()
    print("Starting training...")
    for epoch in range(1):
        epoch_loss = 0
        for i, (images, masks) in enumerate(loader):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            print(f"  Step {i+1}/{len(loader)}, Loss: {loss.item():.4f}")
            
        print(f"Epoch 1 Complete. Average Loss: {epoch_loss/len(loader):.4f}")

if __name__ == "__main__":
    # Create required folders if they don't exist
    os.makedirs("models", exist_ok=True)
    train_transattunet()
