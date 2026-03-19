import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# class index → color mapping for visualization (Cityscapes-like)
CLASS_COLORS = {
    0: (0, 0, 0),         # background
    1: (128, 64, 128),    # road
    2: (244, 35, 232),    # sidewalk
    3: (255, 255, 0),     # lane marking
    4: (196, 196, 196),   # curb
    5: (0, 255, 128),     # crosswalk
}


def get_train_transforms(h, w):
    return A.Compose([
        A.RandomCrop(h, w),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussianBlur(p=0.1),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms(h, w):
    return A.Compose([
        A.Resize(h, w),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


class RoadwayDataset(Dataset):
    """
    expects data_dir with structure:
        data_dir/
            images/  *.png or *.jpg
            masks/   *.png  (single-channel, pixel value = class index)
    """

    def __init__(self, data_dir, transforms=None):
        self.img_dir = os.path.join(data_dir, "images")
        self.mask_dir = os.path.join(data_dir, "masks")
        self.transforms = transforms

        self.samples = sorted([
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        if len(self.samples) == 0:
            raise RuntimeError(f"no images found in {self.img_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name = self.samples[idx]
        stem = os.path.splitext(name)[0]

        img = cv2.imread(os.path.join(self.img_dir, name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask_path = os.path.join(self.mask_dir, stem + ".png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"mask not found: {mask_path}")

        if self.transforms:
            out = self.transforms(image=img, mask=mask)
            img, mask = out["image"], out["mask"].long()
        else:
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).long()

        return img, mask

    @staticmethod
    def mask_to_rgb(mask_np):
        """convert single-channel class mask to RGB for visualization"""
        h, w = mask_np.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for cls, color in CLASS_COLORS.items():
            rgb[mask_np == cls] = color
        return rgb
