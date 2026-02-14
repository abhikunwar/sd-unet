import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel


# ==========================================================
# 1️⃣ Load Kolors UNet
# ==========================================================
model_path = "weights/Kolors"  # change if needed

unet = UNet2DConditionModel.from_pretrained(
    model_path,
    subfolder="unet",
    torch_dtype=torch.float32,
)

unet.eval()
unet.to("cpu")

# Disable memory-efficient attention for ONNX stability
unet.set_default_attn_processor()

print("cross_attention_dim:", unet.config.cross_attention_dim)
print("projection_class_embeddings_input_dim:",
      unet.config.projection_class_embeddings_input_dim)
print("addition_time_embed_dim:",
      unet.config.addition_time_embed_dim)
print("addition_embed_type:",
      unet.config.addition_embed_type)


# ==========================================================
# 2️⃣ Derive Correct Dimensions (CRITICAL)
# ==========================================================
cross_dim = unet.config.cross_attention_dim
proj_dim = unet.config.projection_class_embeddings_input_dim
time_embed_dim = unet.config.addition_time_embed_dim

# THIS is the real text embedding size
text_embed_dim = proj_dim - time_embed_dim

print("Derived text_embed_dim:", text_embed_dim)


# ==========================================================
# 3️⃣ ONNX Wrapper
# ==========================================================
class UNetOnnxWrapper(nn.Module):
    def __init__(self, unet_model):
        super().__init__()
        self.unet = unet_model

    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        text_embeds,
        time_ids,
    ):
        noise_pred = self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs={
                "text_embeds": text_embeds,
                "time_ids": time_ids,
            },
            return_dict=False,
        )[0]

        return noise_pred


wrapper = UNetOnnxWrapper(unet)


# ==========================================================
# 4️⃣ Correct Dummy Inputs
# ==========================================================
batch = 1
height = 64
width = 64
seq_len = 77

sample = torch.randn(batch, 4, height, width, dtype=torch.float32)

# MUST be long
timestep = torch.tensor([1], dtype=torch.long)

encoder_hidden_states = torch.randn(
    batch, seq_len, cross_dim, dtype=torch.float32
)

# Correct derived dimension
text_embeds = torch.randn(
    batch, text_embed_dim, dtype=torch.float32
)

# MUST be long and shape (B, 6)
time_ids = torch.zeros(batch, 6, dtype=torch.long)


# ==========================================================
# 5️⃣ Export to ONNX
# ==========================================================
torch.onnx.export(
    wrapper,
    (
        sample,
        timestep,
        encoder_hidden_states,
        text_embeds,
        time_ids,
    ),
    "kolors_unet.onnx",
    input_names=[
        "sample",
        "timestep",
        "encoder_hidden_states",
        "text_embeds",
        "time_ids",
    ],
    output_names=["noise_pred"],
    dynamic_axes={
        "sample": {0: "batch", 2: "height", 3: "width"},
        "encoder_hidden_states": {0: "batch", 1: "sequence"},
        "text_embeds": {0: "batch"},
        "time_ids": {0: "batch"},
        "noise_pred": {0: "batch", 2: "height", 3: "width"},
    },
    opset_version=17,
    do_constant_folding=True,
)

print("\n✅ SUCCESS: kolors_unet.onnx exported correctly")
