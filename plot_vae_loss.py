import matplotlib.pyplot as plt

epochs = [1, 2, 3]

total_loss = [1310.28, 869.32, 853.60]
reconstruction_loss = [985.76, 857.77, 841.60]
kl_loss = [324.52, 11.55, 12.00]

plt.figure(figsize=(8, 5))

plt.plot(epochs, total_loss, marker="o", label="Total Loss")
plt.plot(epochs, reconstruction_loss, marker="o", label="Reconstruction Loss")
plt.plot(epochs, kl_loss, marker="o", label="KL Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("VAE Training Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("vae_loss_curve.png")
plt.show()