#!/usr/bin/env python3
from pathlib import Path
import json
import random
from collections import Counter
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.transforms import functional as TF
import matplotlib.pyplot as plt
from tqdm import tqdm


# --- Project paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

CROPS_INDEX_PATH = CACHE_DIR / "square_crops_index.csv"
MODEL_PATH = MODELS_DIR / "square_classifier.pt"
NORMALIZATION_PATH = MODELS_DIR / "normalization.json"
LABEL_MAP_PATH = MODELS_DIR / "label_map.json"

# --- Constants ---
IMAGE_SIZE = (96, 96)
BATCH_SIZE = 64
EPOCHS = 15
SEED = 42

# --- Reproducibility ---
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# --- Dataset ---
class SquareCropsDataset(Dataset):
    def __init__(self, df, label_to_idx, transform=None):
        self.df = df.reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["crop_path"]).convert("RGB")
        label = self.label_to_idx[row["label"]]

        if self.transform:
            img = self.transform(img)

        return img, label


# --- Augmentations ---
class RandomRotate90:
    def __call__(self, img):
        k = random.randint(0, 3)
        return TF.rotate(img, angle=90 * k)
    
# --- Normalization stats ---
def compute_mean_std(crop_paths):
    to_tensor = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor()
    ])

    channel_sum = torch.zeros(3)
    channel_sq_sum = torch.zeros(3)
    num_pixels = 0

    for p in tqdm(crop_paths, desc="Computing mean/std"):
        img = Image.open(p).convert("RGB")
        x = to_tensor(img)
        channel_sum += x.sum(dim=(1, 2))
        channel_sq_sum += (x ** 2).sum(dim=(1, 2))
        num_pixels += x.shape[1] * x.shape[2]

    mean = channel_sum / num_pixels
    std = torch.sqrt(channel_sq_sum / num_pixels - mean ** 2)
    return mean.tolist(), std.tolist()
    

# --- Main ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load crop index
    df = pd.read_csv(CROPS_INDEX_PATH)
    print(f"[INFO] Loaded {len(df)} square crops")
    assert "split" in df.columns, "Missing 'split' column in square_crops_index.csv"

    # Filter by split
    df_train = df[df["split"] == "train"]
    df_val   = df[df["split"] == "valid"]

    print(f"[INFO] Train crops: {len(df_train)}")
    print(f"[INFO] Val crops:   {len(df_val)}")
    assert len(df_train) > 0, "Training set is empty!"
    assert len(df_val) > 0, "Validation set is empty!"

    # Compute normalization stats
    if not NORMALIZATION_PATH.exists():
        mean, std = compute_mean_std(df_train["crop_path"].tolist())
        with open(NORMALIZATION_PATH, "w") as f:
            json.dump({"mean": mean, "std": std}, f, indent=2)
        print("[INFO] Computed and saved normalization stats")
    else:
        with open(NORMALIZATION_PATH) as f:
            stats = json.load(f)
        mean, std = stats["mean"], stats["std"]
        print("[INFO] Loaded existing normalization stats")

    # Label mapping (train only)
    labels = sorted(df_train["label"].unique())
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    idx_to_label = {i: lbl for lbl, i in label_to_idx.items()}

    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(idx_to_label, f, indent=2)

    num_classes = len(label_to_idx)
    empty_idx = label_to_idx["empty"]

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.15, hue=0.03),
        RandomRotate90(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # DataLoaders
    train_ds = SquareCropsDataset(df_train, label_to_idx, train_transform)
    val_ds = SquareCropsDataset(df_val, label_to_idx, val_transform)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # Model
    model = models.convnext_tiny(pretrained=True)
    model.classifier[2] = nn.Linear(
        model.classifier[2].in_features, num_classes
    )
    model.to(device)

    # Class weights (train only)
    label_counts = Counter(df_train["label"])
    weights = np.zeros(num_classes, dtype=np.float32)

    for label, idx in label_to_idx.items():
        weights[idx] = 1.0 / np.sqrt(label_counts[label])

    weights[empty_idx] *= 0.5
    class_weights = torch.tensor(weights, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=5e-5, weight_decay=1e-4
    )

    # Training loop
    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        # ---- Train ----
        model.train()
        train_loss = 0.0

        for imgs, labels in tqdm(train_loader, desc="Training"):
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        correct = 0
        correct_no_empty = 0
        total_no_empty = 0

        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc="Validation"):
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()

                mask = labels != empty_idx
                correct_no_empty += (preds[mask] == labels[mask]).sum().item()
                total_no_empty += mask.sum().item()

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        val_acc = correct / len(val_loader.dataset)

        print(f"[INFO] Train loss:          {train_loss:.4f}")
        print(f"[INFO] Val loss:            {val_loss:.4f}")
        print(f"[INFO] Val acc (all):       {val_acc:.4f}")
        if total_no_empty > 0:
            print(f"[INFO] Val acc (no empty):  {correct_no_empty / total_no_empty:.4f}")

    # Save artifacts
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[INFO] Saved model to {MODEL_PATH}")

    plt.figure()
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Valid")
    plt.legend()
    plt.grid(True)
    plt.savefig(MODELS_DIR / "train_valid_loss.png")
    plt.close()


if __name__ == "__main__":
    main()
