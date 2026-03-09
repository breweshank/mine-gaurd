import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
import cv2
import os
from tqdm import tqdm

# ================= CONFIG =================
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 2
EPOCHS = 50
LR = 5e-5
IMG_SIZE = 512
PATIENCE = 8

TRAIN_IMG = "datasets/ground_crack_split/train/images"
TRAIN_MASK = "datasets/ground_crack_split/train/masks"
VAL_IMG = "datasets/ground_crack_split/val/images"
VAL_MASK = "datasets/ground_crack_split/val/masks"

BEST_MODEL_PATH = "unetpp_best_model.pth"
CHECKPOINT_PATH = "unetpp_checkpoint.pth"

print("Using device:", DEVICE)

# ================= DATASET =================

class CrackDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir

        self.images = []
        for img_name in os.listdir(img_dir):
            if os.path.exists(os.path.join(mask_dir, img_name)):
                self.images.append(img_name)
            else:
                print(f"⚠ Missing mask for {img_name}, skipping.")

        self.transform = A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.4),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        image = cv2.imread(os.path.join(self.img_dir, img_name))
        mask = cv2.imread(os.path.join(self.mask_dir, img_name), 0)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        augmented = self.transform(image=image, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"]

        image = torch.tensor(image).permute(2,0,1).float()/255.0
        mask = torch.tensor(mask).unsqueeze(0).float()/255.0

        return image, mask


train_dataset = CrackDataset(TRAIN_IMG, TRAIN_MASK)
val_dataset = CrackDataset(VAL_IMG, VAL_MASK)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, drop_last=True)

# ================= MODEL =================

model = smp.UnetPlusPlus(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation=None
).to(DEVICE)

# ================= LOSS =================

bce = nn.BCEWithLogitsLoss()

def dice_loss(pred, target):
    pred = torch.sigmoid(pred)
    smooth = 1e-6
    intersection = (pred * target).sum()
    return 1 - (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def calculate_metrics(pred, target, threshold=0.4):
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection

    iou = (intersection + 1e-6) / (union + 1e-6)
    dice = (2 * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)
    accuracy = (pred == target).float().mean()

    return iou.item(), dice.item(), accuracy.item()

# ================= OPTIMIZER =================

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ================= RESUME SUPPORT =================

start_epoch = 0
best_iou = 0
early_stop_counter = 0

if os.path.exists(CHECKPOINT_PATH):
    print("🔁 Loading checkpoint...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    start_epoch = checkpoint["epoch"] + 1
    best_iou = checkpoint["best_iou"]
    print(f"Resumed from epoch {start_epoch}")

# ================= TRAINING LOOP =================

for epoch in range(start_epoch, EPOCHS):

    model.train()

    for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch} Training"):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = bce(outputs, masks) + dice_loss(outputs, masks)
        loss.backward()
        optimizer.step()

    scheduler.step()

    # ===== Validation =====
    model.eval()
    total_iou, total_dice, total_acc = 0, 0, 0

    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch} Validation"):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            outputs = model(imgs)

            iou, dice, acc = calculate_metrics(outputs, masks)
            total_iou += iou
            total_dice += dice
            total_acc += acc

    avg_iou = total_iou / len(val_loader)
    avg_dice = total_dice / len(val_loader)
    avg_acc = total_acc / len(val_loader)

    print("\n==============================")
    print(f"Epoch {epoch}")
    print(f"IoU: {avg_iou:.4f}")
    print(f"Dice: {avg_dice:.4f}")
    print(f"Pixel Accuracy: {avg_acc:.4f}")
    print("==============================\n")

    # Save best model
    if avg_iou > best_iou:
        best_iou = avg_iou
        early_stop_counter = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print("🔥 Best model saved!")
    else:
        early_stop_counter += 1

    # Save checkpoint
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_iou": best_iou
    }, CHECKPOINT_PATH)

    if early_stop_counter >= PATIENCE:
        print("⛔ Early stopping triggered.")
        break

print("Training Complete 🚀")
