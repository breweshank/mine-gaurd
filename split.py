import os
import shutil
import random

IMG_DIR = r"/Users/eshankryshabh/Documents/mine_safety_dual/datasets/ground_crack"
MASK_DIR = r"/Users/eshankryshabh/Documents/mine_safety_dual/datasets/ground_crack/masks_refined"

BASE_OUT = r"/Users/eshankryshabh/Documents/mine_safety_dual/datasets/ground_crack_split"

train_img = os.path.join(BASE_OUT, "train/images")
train_mask = os.path.join(BASE_OUT, "train/masks")

val_img = os.path.join(BASE_OUT, "val/images")
val_mask = os.path.join(BASE_OUT, "val/masks")

for path in [train_img, train_mask, val_img, val_mask]:
    os.makedirs(path, exist_ok=True)

valid_extensions = (".jpg", ".jpeg", ".png", ".JPG", ".PNG")

images = [f for f in os.listdir(IMG_DIR) if f.endswith(valid_extensions)]

print(f"Total valid images found: {len(images)}")

random.shuffle(images)

split = int(0.8 * len(images))

train_files = images[:split]
val_files = images[split:]

for file in train_files:
    shutil.copy(os.path.join(IMG_DIR, file), train_img)
    shutil.copy(os.path.join(MASK_DIR, file), train_mask)

for file in val_files:
    shutil.copy(os.path.join(IMG_DIR, file), val_img)
    shutil.copy(os.path.join(MASK_DIR, file), val_mask)

print("Dataset split complete.")
