import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import BrainSegmentationDataset
from vae_model import VAE


BASE_PATH = r"D:\Xu yuanxii\UQ_2025.7-2027.6\2026 Semester2\COMP3710\Demo2\keras_png_slices_data\keras_png_slices_data"

IMAGE_DIR = os.path.join(BASE_PATH, "keras_png_slices_train")
MASK_DIR = os.path.join(BASE_PATH, "keras_png_slices_seg_train")


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


dataset = BrainSegmentationDataset(
    image_dir=IMAGE_DIR,
    mask_dir=MASK_DIR
)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)


model = VAE(latent_dim=2).to(device)

model.load_state_dict(
    torch.load(
        "vae_model.pth",
        map_location=device
    )
)

model.eval()


# --------------------------------
# Reconstruction
# --------------------------------

images, _ = next(iter(loader))
images = images.to(device)

with torch.no_grad():
    reconstruction, mu, logvar = model(images)


original = images[0, 0].cpu().numpy()
reconstructed = reconstruction[0, 0].cpu().numpy()


plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(original, cmap="gray")
plt.title("Original MRI")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(reconstructed, cmap="gray")
plt.title("Reconstructed MRI")
plt.axis("off")

plt.tight_layout()
plt.show()


# --------------------------------
# 2D latent manifold
# --------------------------------

grid_size = 8

latent_range = np.linspace(
    -2,
    2,
    grid_size
)

figure = np.zeros(
    (
        256 * grid_size,
        256 * grid_size
    )
)


with torch.no_grad():

    for i, z1 in enumerate(latent_range):

        for j, z2 in enumerate(latent_range):

            z = torch.tensor(
                [[z1, z2]],
                dtype=torch.float32,
                device=device
            )

            generated = model.decode(z)

            generated = (
                generated[0, 0]
                .cpu()
                .numpy()
            )

            figure[
                i * 256:(i + 1) * 256,
                j * 256:(j + 1) * 256
            ] = generated


plt.figure(figsize=(12, 12))

plt.imshow(
    figure,
    cmap="gray"
)

plt.title(
    "VAE 2D Latent Manifold"
)

plt.axis("off")

plt.show()