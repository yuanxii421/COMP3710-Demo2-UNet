import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import BrainSegmentationDataset
from vae_model import VAE


WINDOWS_PATH = r"D:\Xu yuanxii\UQ_2025.7-2027.6\2026 Semester2\COMP3710\Demo2\keras_png_slices_data\keras_png_slices_data"

RANGPUR_PATH = "/home/groups/comp3710/OASIS"

if os.path.exists(RANGPUR_PATH):
    BASE_PATH = RANGPUR_PATH
else:
    BASE_PATH = WINDOWS_PATH

print("Using dataset:", BASE_PATH)

IMAGE_DIR = os.path.join(BASE_PATH, "keras_png_slices_train")
MASK_DIR = os.path.join(BASE_PATH, "keras_png_slices_seg_train")


dataset = BrainSegmentationDataset(
    image_dir=IMAGE_DIR,
    mask_dir=MASK_DIR
)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


model = VAE(latent_dim=2).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)


def vae_loss(reconstruction, x, mu, logvar):

    reconstruction_loss = F.mse_loss(
        reconstruction,
        x,
        reduction="sum"
    )

    kl_loss = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    total_loss = reconstruction_loss + kl_loss

    return total_loss, reconstruction_loss, kl_loss

train_total_losses = []
train_recon_losses = []
train_kl_losses = []

num_epochs = 3

for epoch in range(num_epochs):

    model.train()

    epoch_loss = 0.0
    epoch_recon = 0.0
    epoch_kl = 0.0

    for images, _ in loader:

        images = images.to(device)

        reconstruction, mu, logvar = model(images)

        loss, recon_loss, kl_loss = vae_loss(
            reconstruction,
            images,
            mu,
            logvar
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_recon += recon_loss.item()
        epoch_kl += kl_loss.item()

    print(f"\nEpoch {epoch + 1}/{num_epochs}")

    print(
        f"Total Loss: "
        f"{epoch_loss / len(loader):.2f}"
    )

    print(
        f"Reconstruction Loss: "
        f"{epoch_recon / len(loader):.2f}"
    )

    print(
        f"KL Loss: "
        f"{epoch_kl / len(loader):.2f}"
    )



avg_total_loss = epoch_loss / len(loader)
avg_recon_loss = epoch_recon / len(loader)
avg_kl_loss = epoch_kl / len(loader)

train_total_losses.append(avg_total_loss)
train_recon_losses.append(avg_recon_loss)
train_kl_losses.append(avg_kl_loss)

print(f"\nEpoch {epoch + 1}/{num_epochs}")
print(f"Total Loss: {avg_total_loss:.2f}")
print(f"Reconstruction Loss: {avg_recon_loss:.2f}")
print(f"KL Loss: {avg_kl_loss:.2f}")


torch.save(
    model.state_dict(),
    "vae_model.pth"
)

print("\nVAE model saved.")

import matplotlib.pyplot as plt

epochs = range(1, num_epochs + 1)

plt.figure(figsize=(8, 5))

plt.plot(epochs, train_total_losses, marker="o", label="Total Loss")
plt.plot(epochs, train_recon_losses, marker="o", label="Reconstruction Loss")
plt.plot(epochs, train_kl_losses, marker="o", label="KL Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("VAE Training Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("vae_loss_curve.png")
plt.show()