import torch
from diffusers import AutoencoderKL

model_id = r"weights/Kolors"

vae = AutoencoderKL.from_pretrained(
    model_id,
    subfolder="vae",
    torch_dtype=torch.float32
)

vae.eval()

# -----------------------------
# Wrap decoder properly
# -----------------------------
class VAEDecoderWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latent):
        return self.vae.decode(latent).sample

decoder = VAEDecoderWrapper(vae)

# -----------------------------
# Dummy latent
# -----------------------------
dummy_latent = torch.randn(1, 4, 64, 64)

# -----------------------------
# Export
# -----------------------------
torch.onnx.export(
    decoder,
    dummy_latent,
    "kolors_vae_decoder.onnx",
    opset_version=17,
    input_names=["latent"],
    output_names=["image"],
    dynamic_axes={
        "latent": {0: "batch"},
        "image": {0: "batch"},
    },
)

print("Export successful.")
