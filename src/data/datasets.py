import os
import io
import zipfile
import numpy as np
import pydicom
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class ZipDicomSliceDataset(Dataset):
    """2D DICOM Slice Dataset from ZIP"""
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
        
        return self.to_tensor(image), (self.to_tensor(mask) > 0.5).float()

class PatientAwareNIfTIDataset(Dataset):
    """3D Volume Dataset with Patient Awareness"""
    def __init__(self, image_dir, label_dir, patient_ids, target_size=(32, 128, 128)):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.patient_ids = patient_ids
        self.target_size = target_size

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pat_id = self.patient_ids[idx]
        img_path = os.path.join(self.image_dir, f"{pat_id}_0000.nii.gz")
        lbl_path = os.path.join(self.label_dir, f"{pat_id}.nii.gz")
        
        image = nib.load(img_path).get_fdata(dtype=np.float32)
        mask_arr = nib.load(lbl_path).get_fdata(dtype=np.float32)
        
        image = (image - np.mean(image)) / (np.std(image) + 1e-8)
        mask_bin = ((mask_arr == 1.0) | (mask_arr == 2.0)).astype(np.float32)
        
        image = np.transpose(image, (2, 0, 1))
        mask_bin = np.transpose(mask_bin, (2, 0, 1))
        
        img_tensor = torch.tensor(image).unsqueeze(0)
        mask_tensor = torch.tensor(mask_bin).unsqueeze(0)
        
        img_tensor = F.interpolate(img_tensor.unsqueeze(0), size=self.target_size, mode='trilinear', align_corners=False).squeeze(0)
        mask_tensor = F.interpolate(mask_tensor.unsqueeze(0), size=self.target_size, mode='nearest').squeeze(0)
        
        return img_tensor, mask_tensor
