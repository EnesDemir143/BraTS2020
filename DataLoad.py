import torch
import torch.nn as nn
from torch.utils.data import Dataset
import glob 
import nibabel as nib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class BraTSDataLoader(Dataset):
    def __init__(self,path, transform=None, scaler=MinMaxScaler()):
        self.path = path
        self.t1_list = sorted(glob.glob(self.path + '/*/*t1.nii'))
        self.t2_list = sorted(glob.glob(self.path + '/*/*t2.nii'))
        self.t1ce_list = sorted(glob.glob(self.path + '/*/*t1ce.nii'))
        self.flair_list = sorted(glob.glob(self.path + '/*/*flair.nii'))
        self.mask_list = sorted(glob.glob(self.path + '/*/*seg.nii'))
        
        self.scaler = scaler
        self.transform = transform
        
        sample_t1 = nib.load(self.t1_list[0]).get_fdata()
        self.scaler.fit(sample_t1.reshape(-1, sample_t1.shape[-1]))

    def __len__(self):
        return len(self.t2_list)

    def __getitem__(self, idx):
        t1 = nib.load(self.t1_list[idx]).get_fdata()
        t1 = self.scaler.transform(t1.reshape(-1, t1.shape[-1])).reshape(t1.shape)
        
        t1ce = nib.load(self.t1ce_list[idx]).get_fdata()
        t1ce = self.scaler.transform(t1ce.reshape(-1, t1ce.shape[-1])).reshape(t1ce.shape)
        
        t2 = nib.load(self.t2_list[idx]).get_fdata()
        t2 = self.scaler.transform(t2.reshape(-1, t2.shape[-1])).reshape(t2.shape)
        
        flair = nib.load(self.flair_list[idx]).get_fdata()
        flair = self.scaler.transform(flair.reshape(-1, flair.shape[-1])).reshape(flair.shape)
        
        mask = nib.load(self.mask_list[idx]).get_fdata().astype(np.uint8)
        mask[mask == 4] = 3
        
        combined_data = torch.tensor(np.stack([t1, t1ce, t2, flair], axis=3), dtype=torch.float32)
        mask_tensor = nn.functional.one_hot(torch.tensor(mask, dtype=torch.long), num_classes=4)
        
        if self.transform:
            combined_data = self.transform(combined_data)
            mask_tensor = self.transform(mask_tensor)
        
        return combined_data, mask_tensor
