import h5py
import numpy as np
import base64
import cv2
import onnxruntime as ort
import io
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "satellite_model.onnx")

session = ort.InferenceSession(MODEL_PATH)

THRESHOLD = 0.3


def analyze_satellite(file_bytes, borewell_depth):

    f = h5py.File(io.BytesIO(file_bytes), 'r')

    img = f["img"][:]

    img_norm = img.astype(np.float32)
    img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min() + 1e-6)

    inp = np.transpose(img_norm, (2,0,1))
    inp = np.expand_dims(inp,0)

    output = session.run(None, {session.get_inputs()[0].name: inp})[0]

    prob = output[0,0]
    mask = (prob > THRESHOLD).astype(np.uint8)

    landslide_pixels = mask.sum()
    total_pixels = mask.size

    area_ratio = landslide_pixels / total_pixels

    groundwater_factor = min(borewell_depth / 50, 1)

    landslide_risk = (0.7 * area_ratio) + (0.3 * groundwater_factor)

    if landslide_risk > 0.75:
        level = "SEVERE"
    elif landslide_risk > 0.5:
        level = "HIGH"
    elif landslide_risk > 0.25:
        level = "MODERATE"
    else:
        level = "LOW"

    rgb = img_norm[:,:,0:3]

    overlay = rgb.copy()
    overlay[mask==1] = [1,0,0]

    overlay = (overlay*255).astype(np.uint8)

    _,buffer = cv2.imencode(".png", overlay)
    overlay_base64 = base64.b64encode(buffer).decode("utf-8")

    return {

        "overlay_image": overlay_base64,
        "area_ratio": float(area_ratio),
        "groundwater_factor": float(groundwater_factor),
        "landslide_level": level

    }