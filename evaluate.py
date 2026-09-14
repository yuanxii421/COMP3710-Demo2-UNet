import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import BrainSegmentationDataset
from unet_model import UNet


BASE_PATH = "/home/groups/comp3710/OASIS"

TEST_IMAGE_DIR = os.path.join(BASE_PATH, "keras_png_slices_test")
TEST_MASK_DIR = os.path.join(BASE_PATH, "keras_png_slices_seg_test")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def dice_score(pred, target, num_classes=4, epsilon=1e-6):
    pred = torch.argmax(pred, dim=1)

    dice_scores = []

    for class_id in range(num_classes):
        pred_class = (pred == class_id).float()
        target_class = (target == class_id).float()

        intersection = (pred_class * target_class).sum()

        dice = (
            2.0 * intersection + epsilon
        ) / (
            pred_class.sum() + target_class.sum() + epsilon
        )

        dice_scores.append(dice.item())

    return dice_scores


# Load test dataset
test_dataset = BrainSegmentationDataset(
    image_dir=TEST_IMAGE_DIR,
    mask_dir=TEST_MASK_DIR
)

test_loader = DataLoader(
    test_dataset,
    batch_size=2,
    shuffle=False
)


# Load trained model
model = UNet(in_channels=1, num_classes=4).to(device)

model.load_state_dict(
    torch.load("final_unet_model.pth", map_location=device)
)

model.eval()

print("Loaded final_unet_model.pth")


# Evaluate whole test set
dice_sum = [0.0, 0.0, 0.0, 0.0]

with torch.no_grad():
    for images, masks in test_loader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        batch_dice = dice_score(outputs, masks)

        for i in range(4):
            dice_sum[i] += batch_dice[i]


average_dice = [
    score / len(test_loader)
    for score in dice_sum
]

mean_dice = sum(average_dice) / 4

print("\nTest Dice:")
for i, score in enumerate(average_dice):
    print(f"Class {i}: {score:.4f}")

print(f"Mean Dice: {mean_dice:.4f}")


# Visualise a few test examples
os.makedirs("test_results", exist_ok=True)

model.eval()

with torch.no_grad():
    for idx in range(3):

        image, mask = test_dataset[idx]

        input_tensor = image.unsqueeze(0).to(device)

        output = model(input_tensor)

        prediction = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

        image_np = image.squeeze(0).numpy()
        mask_np = mask.numpy()

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(image_np, cmap="gray")
        plt.title("MRI")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(mask_np)
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(prediction)
        plt.title("Prediction")
        plt.axis("off")

        plt.tight_layout()

        plt.savefig(
            f"test_results/result_{idx}.png"
        )

        plt.close()

print("\nSaved visualisation results to test_results/")
