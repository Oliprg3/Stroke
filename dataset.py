import numpy as np
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
from typing import List, Dict
from .slice_utils import make_2p5d_stacks

class CTStudyDataset(Dataset):
    def __init__(self, meta_rows: List[Dict], stack_k=5, slice_stride=1, transforms=None):
        self.meta = meta_rows
        self.k = stack_k
        self.stride = slice_stride
        self.transforms = transforms

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta[idx]
        path = row["nifti_path"]
        label = row["label"]  
        vol = sitk.ReadImage(path)
        arr = sitk.GetArrayFromImage(vol).astype(np.float32)  
        stacks, slice_idxs = make_2p5d_stacks(arr, k=self.k, stride=self.stride) 

       
        x = torch.from_numpy(stacks).unsqueeze(2) 
        y = torch.tensor(label).float()
        sample = {"x": x, "y": y, "slice_idxs": slice_idxs, "study_id": row.get("study_id","")}
        if self.transforms:
            sample = self.transforms(sample)
        return sample
