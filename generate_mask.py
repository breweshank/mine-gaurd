import torch
import segmentation_models_pytorch as smp
import cv2
import os
import numpy as np
from tqdm import tqdm
import albumentations as A

# ================= CONFIG =================
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
IMG_SIZE = 512

MODEL_PATH = "crack_best_model.pth"

INPUT_DIR = r"/Users/eshankryshabh/Documents/mine_safety_dual/datasets/ground_crack"
OUTPUT_DIR = "datasets/ground_crack/masks_merged"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Using device:", DEVICE)

# ================= LOAD MODEL =================
model = smp.UnetPlusPlus(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation=None
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Resize transform for model
transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
])

valid_extensions = (".jpg", ".jpeg", ".png", ".JPG", ".PNG")
image_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(valid_extensions)]

print(f"Processing {len(image_files)} images...")

for img_name in tqdm(image_files):

    img_path = os.path.join(INPUT_DIR, img_name)
    img = cv2.imread(img_path)

    if img is None:
        continue

    original_h, original_w = img.shape[:2]

    # ===================== 1️⃣ DEEP MODEL PREDICTION =====================
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    augmented = transform(image=img_rgb)
    img_resized = augmented["image"]

    tensor = torch.tensor(img_resized).permute(2,0,1).float()/255.0
    tensor = tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(tensor)
        mask_dl = torch.sigmoid(output)[0][0].cpu().numpy()

    mask_dl = (mask_dl > 0.35).astype(np.uint8) * 255
    mask_dl = cv2.resize(mask_dl, (original_w, original_h))

    # ===================== 2️⃣ CLASSICAL ENHANCED DETECTION =====================
    img_small = cv2.resize(img, (512,512))
    gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)

    # ---- Thin crack detection (Canny edges)
    edges = cv2.Canny(gray, 50, 150)
    kernel_edge = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel_edge, iterations=1)

    # ---- Deep dark fissure detection
    _, dark_mask = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY_INV)

    # ---- Remove green grass
    hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    non_green = cv2.bitwise_not(green_mask)

    dark_mask = cv2.bitwise_and(dark_mask, non_green)

    # ---- Merge edge + dark detection
    classical_mask = cv2.bitwise_or(edges, dark_mask)
    classical_mask = cv2.resize(classical_mask, (original_w, original_h))

    # ===================== 3️⃣ MERGE DEEP + CLASSICAL =====================
    merged = cv2.bitwise_or(mask_dl, classical_mask)

    # ===================== 4️⃣ MORPHOLOGICAL CLEANING =====================
    kernel = np.ones((3,3), np.uint8)

    merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, kernel)
    merged = cv2.dilate(merged, kernel, iterations=1)

    # ===================== 5️⃣ REMOVE SMALL NOISE =====================
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(merged, 8)

    final_mask = np.zeros_like(merged)
    min_area = 350

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > min_area:
            final_mask[labels == i] = 255

    cv2.imwrite(os.path.join(OUTPUT_DIR, img_name), final_mask)

print("✅ Merged universal masks generated successfully.")
