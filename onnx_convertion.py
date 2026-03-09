import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("satellite_model.onnx")

dummy = np.random.randn(1,14,128,128).astype(np.float32)

output = session.run(None, {"input": dummy})

print("ONNX output shape:", output[0].shape)
