import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

BASE_PATH = r"D:\Xu yuanxii\UQ_2025.7-2027.6\2026 Semester2\COMP3710\Demo2\keras_png_slices_data\keras_png_slices_data"

IMAGE_DIR = os.path.join(BASE_PATH, "keras_png_slices_train")
MASK_DIR = os.path.join(BASE_PATH, "keras_png_slices_seg_train")

image_filename = sorted(os.listdir(IMAGE_DIR))[0]

mask_filename = image_filename.replace("case_", "seg_")

image_path = os.path.join(IMAGE_DIR, image_filename)
mask_path = os.path.join(MASK_DIR, mask_filename)

print("Image file:", image_filename)
print("Mask file:", mask_filename)

image = np.array(Image.open(image_path))
mask = np.array(Image.open(mask_path))


print("Image shape:", image.shape)
print("Image min/max:", image.min(), image.max())

print("Mask shape:", mask.shape)
print("Mask unique values:", np.unique(mask))

mask_class = np.zeros_like(mask, dtype=np.uint8)
mask_class[mask == 0] = 0
mask_class[mask == 85] = 1
mask_class[mask == 170] = 2
mask_class[mask == 255] = 3

print("Converted mask labels:", np.unique(mask_class))

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("MRI")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(mask, cmap="gray")
plt.title("Original Mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(mask_class)
plt.title("Class Mask")
plt.axis("off")

plt.show()