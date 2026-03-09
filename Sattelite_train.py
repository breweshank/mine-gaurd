import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.models as models
import h5py
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# ================= CONFIG =================
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-4
LANDSLIDE_THRESHOLD = 0.01
PATIENCE = 6

TRAIN_ROOT = r"/Users/eshankryshabh/Documents/mine_safety_dual/satellite_data/TrainData"

BEST_MODEL_PATH = "satellite_resnet50_best.pth"
CHECKPOINT_PATH = "satellite_checkpoint.pth"

print("Using device:", DEVICE)

# ================= DATASET =================

class LandslideDataset(Dataset):
    def __init__(self, root_folder):
        self.img_folder = os.path.join(root_folder, "img")
        self.mask_folder = os.path.join(root_folder, "mask")

        self.files = sorted([
            f for f in os.listdir(self.img_folder)
            if f.endswith(".h5")
        ])

        print(f"Loaded {len(self.files)} total samples")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]

        img_path = os.path.join(self.img_folder, file_name)
        mask_name = file_name.replace("image", "mask")
        mask_path = os.path.join(self.mask_folder, mask_name)

        with h5py.File(img_path, 'r') as f:
            img = f['img'][:]

        with h5py.File(mask_path, 'r') as f:
            mask = f['mask'][:]

        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)

        mask_ratio = np.sum(mask > 0) / mask.size
        label = 1 if mask_ratio > LANDSLIDE_THRESHOLD else 0

        img = torch.tensor(img).permute(2, 0, 1)
        label = torch.tensor(label).float()

        return img, label


# Load full dataset
full_dataset = LandslideDataset(TRAIN_ROOT)

# Split 80/20
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

print(f"Train samples: {train_size}")
print(f"Val samples: {val_size}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ================= MODEL =================

model = models.resnet50(weights=None)

model.conv1 = nn.Conv2d(
    in_channels=14,
    out_channels=64,
    kernel_size=7,
    stride=2,
    padding=3,
    bias=False
)

model.fc = nn.Linear(model.fc.in_features, 1)
model = model.to(DEVICE)

criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.2]).to(DEVICE))
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_auc = 0
early_stop_counter = 0

# ================= TRAINING =================

for epoch in range(EPOCHS):

    model.train()
    train_loss = 0

    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch} Training"):
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    scheduler.step()

    # ===== VALIDATION =====
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch} Validation"):
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1)

            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    auc = roc_auc_score(all_labels, all_preds)

    print("\n==============================")
    print(f"Epoch {epoch}")
    print(f"Train Loss: {train_loss/len(train_loader):.4f}")
    print(f"Val Loss: {val_loss/len(val_loader):.4f}")
    print(f"AUC Score: {auc:.4f}")
    print("==============================\n")

    if auc > best_auc:
        best_auc = auc
        early_stop_counter = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print("🔥 Best satellite model saved!")
    else:
        early_stop_counter += 1

    if early_stop_counter >= PATIENCE:
        print("⛔ Early stopping triggered.")
        break

print("Satellite Training Complete 🚀")
