import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset


class BrainSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.image_files = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_filename = self.image_files[index]
        mask_filename = image_filename.replace("case_", "seg_")

        image_path = os.path.join(self.image_dir, image_filename)
        mask_path = os.path.join(self.mask_dir, mask_filename)

        # Load image and mask
        image = np.array(Image.open(image_path), dtype=np.float32)
        mask = np.array(Image.open(mask_path), dtype=np.uint8)

        # Normalize MRI from 0-255 to 0-1
        image = image / 255.0

        # Convert mask values: 0,85,170,255 -> 0,1,2,3
        mask_class = np.zeros_like(mask, dtype=np.int64)

        mask_class[mask == 0] = 0
        mask_class[mask == 85] = 1
        mask_class[mask == 170] = 2
        mask_class[mask == 255] = 3

        # Convert to PyTorch tensors
        image = torch.tensor(image, dtype=torch.float32)
        mask_class = torch.tensor(mask_class, dtype=torch.long)

        # Add channel dimension to image
        image = image.unsqueeze(0)

        return image, mask_class