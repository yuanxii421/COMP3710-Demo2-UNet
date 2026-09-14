import os
from torch.utils.data import DataLoader

from dataset import BrainSegmentationDataset


BASE_PATH = r"D:\Xu yuanxii\UQ_2025.7-2027.6\2026 Semester2\COMP3710\Demo2\keras_png_slices_data\keras_png_slices_data"

IMAGE_DIR = os.path.join(BASE_PATH, "keras_png_slices_train")
MASK_DIR = os.path.join(BASE_PATH, "keras_png_slices_seg_train")


dataset = BrainSegmentationDataset(
    image_dir=IMAGE_DIR,
    mask_dir=MASK_DIR
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)

images, masks = next(iter(loader))

print("Number of samples:", len(dataset))
print("Images shape:", images.shape)
print("Masks shape:", masks.shape)
print("Mask unique values:", masks.unique())