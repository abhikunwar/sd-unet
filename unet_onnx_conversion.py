import torch
from diffusers import UNet2DConditionModel
import torch.nn as nn

model_path = "weights/Kolors"

unet = UNet2DConditionModel.from_pretrained(
    model_path,
    subfolder="unet",
    torch_dtype=torch.float32
)

unet.eval()

# Check cross attention dim
print("Cross attention dim:", unet.config.cross_attention_dim)

batch = 1
seq = 77
latent = torch.randn(batch, 4, 64, 64)
timestep = torch.tensor([1], dtype=torch.int64)
# Build `encoder_hidden_states` to match the model config when possible.
# If the UNet config defines `encoder_hid_dim` and `encoder_hid_dim_type`,
# the encoder projection expects inputs of shape (batch, seq, encoder_hid_dim).
# Otherwise default to `cross_attention_dim` which is commonly used.
encoder_hid_dim = getattr(unet.config, "encoder_hid_dim", None)
encoder_hid_dim_type = getattr(unet.config, "encoder_hid_dim_type", None)
if encoder_hid_dim is not None and encoder_hid_dim_type is not None:
    feat_dim = encoder_hid_dim
else:
    feat_dim = unet.config.cross_attention_dim

encoder_hidden_states = torch.randn(batch, seq, feat_dim)


# Wrap the UNet so tracing/export always provides `added_cond_kwargs` (diffusers
# expects this to be a dict; when it's None tracing raises ``TypeError``).
class UNetOnnxWrapper(nn.Module):
    def __init__(self, unet_model: UNet2DConditionModel):
        super().__init__()
        self.unet = unet_model

    def forward(self, sample, timestep, encoder_hidden_states):
        # Debug shapes during tracing to help diagnose dimension issues
        try:
            print("[DEBUG] sample.shape=", getattr(sample, "shape", None))
            print("[DEBUG] timestep.shape=", getattr(timestep, "shape", None))
            print("[DEBUG] encoder_hidden_states.shape=", getattr(encoder_hidden_states, "shape", None))
        except Exception:
            pass

        # Some UNet configs (e.g. `addition_embed_type == 'text_time'`) build
        # extra embeddings inside `get_aug_embed` which have complex, model-
        # specific shapes. Rather than guessing those shapes here (which can
        # change across diffusers versions), temporarily disable the
        # `addition_embed_type` config during tracing so the model skips that
        # logic and tracing remains stable.
        orig_add = getattr(self.unet.config, "addition_embed_type", None)
        disable_temp = orig_add is not None
        if disable_temp:
            self.unet.config.addition_embed_type = None

        try:
            return self.unet(
                sample,
                timestep,
                encoder_hidden_states,
                added_cond_kwargs={},
            )
        finally:
            if disable_temp:
                self.unet.config.addition_embed_type = orig_add


wrapper = UNetOnnxWrapper(unet)

torch.onnx.export(
    wrapper,
    (latent, timestep, encoder_hidden_states),
    "kolors_unet.onnx",
    input_names=["sample", "timestep", "encoder_hidden_states"],
    output_names=["out_sample"],
    opset_version=17,
    dynamic_axes={
        "sample": {0: "batch", 2: "height", 3: "width"},
        "encoder_hidden_states": {0: "batch"},
        "out_sample": {0: "batch"},
    },
    export_params=True,
    do_constant_folding=True,
    external_data=False  
)

print("✅ Export successful")
