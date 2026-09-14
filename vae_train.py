import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import BrainSegmentationDataset
from vae_model import VAE


BASE_PATH = r"D:\Xu yuanxii\UQ_2025.7-2027.6\2026 Semester2\COMP3710\Demo2\keras_png_slices_data\keras_png_slices_data"

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


torch.save(
    model.state_dict(),
    "vae_model.pth"
)

print("\nVAE model saved.")