import onnxruntime as ort
import numpy as np

# Load ONNX model
session = ort.InferenceSession("final_onnx/kolors_unet.onnx")

# Create dummy inputs (adjust shapes if needed)
batch = 1
latent = np.random.randn(batch, 4, 64, 64).astype(np.float32)
timestep = np.array([1], dtype=np.int64)
encoder_hidden_states = np.random.randn(batch, 77, 4096).astype(np.float32)  # 768 is common cross_attention_dim

# Run inference
outputs = session.run(
    None,
    {
        "sample": latent,
        "timestep": timestep,
        "encoder_hidden_states": encoder_hidden_states,
    },
)

# Print output info
print("Output shape:", outputs[0].shape)
print("Output mean:", outputs[0].mean())
print("ONNX model ran successfully ✅")

