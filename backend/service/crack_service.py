import cv2
import numpy as np
import base64
import onnxruntime as ort
import os
import io
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "crack_model.onnx")

session = ort.InferenceSession(MODEL_PATH)

THRESHOLD = 0.3


def analyze_crack(file_bytes):

    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = np.array(image)

    h, w, _ = img.shape

    resized = cv2.resize(img, (256,256))

    inp = resized.astype(np.float32) / 255.0
    inp = np.transpose(inp, (2,0,1))
    inp = np.expand_dims(inp,0)

    output = session.run(None, {session.get_inputs()[0].name: inp})[0]

    mask = output[0,0]
    mask = cv2.resize(mask,(w,h))
    mask = (mask > THRESHOLD).astype(np.uint8)


    # -------------------------
    # Crack metrics
    # -------------------------

    crack_pixels = np.sum(mask)
    total_pixels = mask.size

    area_ratio = crack_pixels / total_pixels


    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    length = 0
    for c in contours:
        length += cv2.arcLength(c, False)


    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    avg_width = float(np.mean(dist[dist>0])*2) if np.any(dist>0) else 0


    density = len(contours) / 10


    # -------------------------
    # Severity score
    # -------------------------

    severity = min(area_ratio*5 + density*0.2, 1)


    # -------------------------
    # Disaster Damage Classification
    # -------------------------

    if severity > 0.75:
        damage_class = "SEVERE"

    elif severity > 0.5:
        damage_class = "HIGH"

    elif severity > 0.25:
        damage_class = "MODERATE"

    else:
        damage_class = "LOW"


    # -------------------------
    # Paris Law Crack Propagation
    # -------------------------

    C = 0.0005
    m = 3

    width_score = min(avg_width/40,1)
    length_score = min(length/1000,1)

    deltaK = width_score + length_score

    propagation = C * (deltaK ** m)


    # -------------------------
    # Overlay
    # -------------------------

    overlay = img.copy()
    overlay[mask==1] = [255,0,0]

    _,buffer = cv2.imencode(".png",overlay)

    overlay_base64 = base64.b64encode(buffer).decode("utf-8")


    return {

        "overlay_image": overlay_base64,
        "severity": float(severity),
        "damage_class": damage_class,
        "area_ratio": float(area_ratio),
        "avg_width": float(avg_width),
        "length": float(length),
        "density": float(density),
        "propagation": float(propagation)

    }
