## Task 1 – Variational Autoencoder (VAE)

### Goal

This task implements a convolutional Variational Autoencoder (VAE) for the preprocessed OASIS brain MRI dataset.

The aim is to learn a low-dimensional latent representation of the MRI images and reconstruct the original images from this latent space.

### Dataset

The model uses the preprocessed OASIS MRI dataset provided for COMP3710.

Input images are grayscale MRI slices with shape:

- 1 channel
- 256 × 256 pixels

Pixel values are normalized from the range 0–255 to 0–1.

### Model Architecture

The VAE consists of:

- Convolutional encoder
- Two latent distribution parameters:
  - Mean (`mu`)
  - Log variance (`logvar`)
- Reparameterization step
- Convolutional decoder

The latent dimension is set to 2 so that the learned latent space can be visualised directly.

The overall pipeline is:

MRI image  
→ Encoder  
→ `mu`, `logvar`  
→ latent sample `z`  
→ Decoder  
→ reconstructed MRI

### Reparameterization

The latent vector is sampled using:

`z = mu + std * epsilon`

where:

- `std = exp(0.5 * logvar)`
- `epsilon` is sampled from a standard normal distribution

This allows stochastic sampling while still allowing gradients to propagate during training.

### Loss Function

The total VAE loss consists of two components:

1. Reconstruction loss  
   Measures the difference between the original MRI and the reconstructed MRI.

2. KL divergence  
   Regularises the latent distribution so that it remains close to a standard normal distribution.

Total loss:

`VAE Loss = Reconstruction Loss + KL Divergence`

### Training

The model was trained using the Adam optimiser.

Example training results:

| Epoch | Total Loss | Reconstruction Loss | KL Loss |
|------:|-----------:|--------------------:|--------:|
| 1 | 1310.28 | 985.76 | 324.52 |
| 2 | 869.32 | 857.77 | 11.55 |
| 3 | 853.60 | 841.60 | 12.00 |

The decreasing reconstruction loss indicates that the model improves its ability to reconstruct the MRI images during training.

### Reconstruction

The trained VAE is able to reproduce the major anatomical structure of the input MRI.

The reconstructed image is smoother than the original image because the MRI is compressed into a two-dimensional latent representation.

### 2D Latent Manifold

Because the latent dimension is 2, the decoder can be sampled over a 2D grid.

A grid of latent coordinates in the range:

`[-2, 2] × [-2, 2]`

is passed through the decoder to generate MRI images.

The resulting manifold shows smooth transitions between neighbouring generated brain images, indicating that the VAE has learned a continuous latent representation.

### Files

- `vae_model.py` – VAE architecture
- `vae_train.py` – training and model saving
- `vae_visualize.py` – reconstruction and latent manifold visualisation
- `vae_model.pth` – trained model weights (not committed to GitHub)
- `vae_loss_curve.png` – training loss plot


Project
COMP3710 Demo 2 - UNet Brain MRI Segmentation

Dataset
Preprocessed OASIS dataset on Rangpur

Input
256 × 256 grayscale MRI

Output
4-class segmentation

Model
UNet with encoder, bottleneck, decoder and skip connections

Training
CrossEntropyLoss
Adam optimizer
3 epochs
A100 GPU

Validation Dice
Class 0: 0.9991
Class 1: 0.9416
Class 2: 0.9575
Class 3: 0.9746
Mean: 0.9682

Test Dice
Class 0: 0.9990
Class 1: 0.9456
Class 2: 0.9550
Class 3: 0.9727
Mean: 0.9681