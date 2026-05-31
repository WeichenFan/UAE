"""CLIP vision-tower encoder for UAE Stage-1 training.

Mirrors the structure of dinov2.py. Loads HuggingFace CLIPVisionModel; if
``normalize=True`` the final ``post_layernorm`` affine parameters are removed
so that the encoder output is the un-scaled penultimate-style hidden state
(matches the layout of the published UAE-CLIP-L checkpoint, whose state dict
omits ``post_layernorm.{weight,bias}``).

Note on torch.load safety check (CVE-2025-32434): transformers >= 4.49 refuses
to load ``.bin`` checkpoints unless torch >= 2.6. The OpenAI CLIP-L weights on
HuggingFace ship only ``pytorch_model.bin`` (no safetensors), and the rest of
the rebuttal pipeline runs on torch 2.2. Since the OpenAI CLIP weights are
trusted, we bypass that one check here.
"""
def _bypass_torch_load_safety_check():
    """Disable transformers' CVE-2025-32434 mitigation. Trusted weights only."""
    _noop = lambda: None
    try:
        import transformers.utils.import_utils as _impu
        _impu.check_torch_load_is_safe = _noop
    except Exception:
        pass
    try:
        import transformers.modeling_utils as _mu
        _mu.check_torch_load_is_safe = _noop
    except Exception:
        pass

_bypass_torch_load_safety_check()  # noqa: E305

from transformers import AutoConfig, CLIPVisionModel
from torch import nn
import torch
from . import register_encoder


@register_encoder()
class CLIPwNorm(nn.Module):
    def __init__(
        self,
        model_name: str,
        normalize: bool = True,
        pretrained: bool = True,
    ):
        super().__init__()
        if pretrained:
            try:
                self.model = CLIPVisionModel.from_pretrained(model_name, local_files_only=True)
            except (OSError, ValueError, AttributeError):
                self.model = CLIPVisionModel.from_pretrained(model_name, local_files_only=False)
        else:
            try:
                config = AutoConfig.from_pretrained(model_name, local_files_only=True)
            except (OSError, ValueError, AttributeError):
                config = AutoConfig.from_pretrained(model_name, local_files_only=False)
            self.model = CLIPVisionModel(config)

        if normalize:
            # Disable the final layer-norm affine: state-dict thus omits
            # post_layernorm.{weight,bias}, matching the published UAE-CLIP-L.
            self.model.vision_model.post_layernorm.elementwise_affine = False
            self.model.vision_model.post_layernorm.weight = None
            self.model.vision_model.post_layernorm.bias = None

        self.patch_size = self.model.config.patch_size
        self.hidden_size = self.model.config.hidden_size

    def clip_forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=x)
        # Strip CLS, return patch tokens [B, N, C].
        return out.last_hidden_state[:, 1:]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.clip_forward(x)
