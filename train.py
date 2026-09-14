import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import BrainSegmentationDataset
from unet_model import UNet


#BASE_PATH = r"D:\Xu yuanxii\UQ_2025.7-2027.6\2026 Semester2\COMP3710\Demo2\keras_png_slices_data\keras_png_slices_data"

BASE_PATH = "/home/groups/comp3710/OASIS"

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


model = UNet(in_channels=1, num_classes=4).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

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


# Validation dataset
VAL_IMAGE_DIR = os.path.join(BASE_PATH, "keras_png_slices_validate")
VAL_MASK_DIR = os.path.join(BASE_PATH, "keras_png_slices_seg_validate")

val_dataset = BrainSegmentationDataset(
    image_dir=VAL_IMAGE_DIR,
    mask_dir=VAL_MASK_DIR
)

val_loader = DataLoader(
    val_dataset,
    batch_size=2,
    shuffle=False
)


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


num_epochs = 3
best_val_dice = 0.0


for epoch in range(num_epochs):

    # --------------------
    # Training
    # --------------------
    model.train()

    train_loss = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()


    average_train_loss = train_loss / len(loader)


    # --------------------
    # Validation
    # --------------------
    model.eval()

    val_loss = 0.0
    dice_sum = [0.0, 0.0, 0.0, 0.0]

    with torch.no_grad():

        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            loss = criterion(outputs, masks)

            val_loss += loss.item()

            batch_dice = dice_score(outputs, masks)

            for i in range(4):
                dice_sum[i] += batch_dice[i]


    average_val_loss = val_loss / len(val_loader)

    average_dice = [
        score / len(val_loader)
        for score in dice_sum
    ]

    mean_dice = sum(average_dice) / 4


    print(f"\nEpoch {epoch + 1}/{num_epochs}")

    print(f"Train Loss: {average_train_loss:.4f}")
    print(f"Validation Loss: {average_val_loss:.4f}")

    print("Validation Dice:")

    for i, score in enumerate(average_dice):
        print(f"Class {i}: {score:.4f}")

    print(f"Mean Dice: {mean_dice:.4f}")


    # Save best model
    if mean_dice > best_val_dice:
        best_val_dice = mean_dice

        torch.save(
            model.state_dict(),
            "best_unet_model.pth"
        )

        print("Best model saved.")