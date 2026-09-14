import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),   # 256 -> 128
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # 128 -> 64
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 64 -> 32
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # 32 -> 16
            nn.ReLU()
        )

        self.flatten_dim = 256 * 16 * 16

        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 16 -> 32
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 32 -> 64
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 64 -> 128
            nn.ReLU(),

            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),    # 128 -> 256
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)

        z = mu + epsilon * std

        return z

    def decode(self, z):
        x = self.fc_decode(z)
        x = x.view(-1, 256, 16, 16)

        return self.decoder(x)

    def forward(self, x):
        mu, logvar = self.encode(x)

        z = self.reparameterize(mu, logvar)

        reconstruction = self.decode(z)

        return reconstruction, mu, logvar

if __name__ == "__main__":
    model = VAE(latent_dim=2)

    x = torch.randn(2, 1, 256, 256)

    reconstruction, mu, logvar = model(x)

    print("Input shape:", x.shape)
    print("Reconstruction shape:", reconstruction.shape)
    print("Mu shape:", mu.shape)
    print("Logvar shape:", logvar.shape)