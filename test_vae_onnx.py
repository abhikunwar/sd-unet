import onnxruntime as ort
import numpy as np
from PIL import Image

# -----------------------------
# 1) Load ONNX model
# -----------------------------
session = ort.InferenceSession("kolors_vae_decoder.onnx")

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print("Input shape :", session.get_inputs()[0].shape)
print("Output shape:", session.get_outputs()[0].shape)

# -----------------------------
# 2) Create dummy latent
# -----------------------------
latent = np.random.randn(1, 4, 64, 64).astype(np.float32)

# Stable Diffusion / Kolors scaling factor
latent = latent * 0.18215

# -----------------------------
# 3) Run inference
# -----------------------------
image = session.run([output_name], {input_name: latent})[0]

print("Raw min :", image.min())
print("Raw max :", image.max())
print("Raw mean:", image.mean())

# -----------------------------
# 4) Standard SD postprocess
# -----------------------------
image_sd = (image / 2 + 0.5)
image_sd = np.clip(image_sd, 0, 1)

image_sd = (image_sd * 255).astype(np.uint8)
image_sd = image_sd[0]
image_sd = np.transpose(image_sd, (1, 2, 0))

Image.fromarray(image_sd).save("vae_output_sd.png")

# -----------------------------
# 5) Enhanced contrast version
# -----------------------------
img = image[0]
img = np.transpose(img, (1, 2, 0))

# Min-max normalization (for visibility)
img = (img - img.min()) / (img.max() - img.min())
img = (img * 255).astype(np.uint8)

Image.fromarray(img).save("vae_output_contrast.png")

print("✅ Saved:")
print(" - vae_output_sd.png (standard SD output)")
print(" - vae_output_contrast.png (contrast stretched)")
