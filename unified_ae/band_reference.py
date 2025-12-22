from __future__ import annotations

import math
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Dinov2WithRegistersModel, AutoImageProcessor

from .uae_residual_split_flow import (
    UAEResidualSplitFlowConfig,
    UAEResidualSplitFlowProcessor,
    FFTBandProjector,
    LearnedConvProjector,
)


class DinoBandReference(nn.Module):
    """
    Utility module that extracts the first latent band from a pretrained Dinov2 encoder.

    Matches the behaviour used in the original RAE training pipeline so we can reuse the
    same band-regularisation loss.
    """

    def __init__(
        self,
        model_name: str,
        device: torch.device,
        input_size: int,
        schedule: List[float],
        use_residual_split: bool,
        residual_projector: str,
        residual_projector_kwargs: Optional[dict] = None,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        encoder = Dinov2WithRegistersModel.from_pretrained(model_name)
        encoder.eval()
        encoder.requires_grad_(False)
        if normalize:
            layernorm = getattr(encoder, "layernorm", None)
            if layernorm is not None:
                layernorm.elementwise_affine = False
                layernorm.weight = None
                layernorm.bias = None
        self.encoder = encoder.to(device)

        processor = AutoImageProcessor.from_pretrained(model_name)
        mean = torch.tensor(processor.image_mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        std = torch.tensor(processor.image_std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self.register_buffer("img_mean", mean, persistent=False)
        self.register_buffer("img_std", std, persistent=False)

        self.input_size = int(input_size)
        self.patch_size = self.encoder.config.patch_size
        num_registers = getattr(self.encoder.config, "num_register_tokens", 4)
        self.unused_tokens = 1 + num_registers

        projector_name = residual_projector.lower()
        projector_kwargs = dict(residual_projector_kwargs or {})
        if projector_name == "fft":
            projector = FFTBandProjector()
        elif projector_name in {"learned_conv", "learned-conv", "learnedconv"}:
            projector = LearnedConvProjector(**projector_kwargs)
        else:
            raise ValueError(f"Unsupported residual projector '{residual_projector}' for band reference.")
        self.band_processor = UAEResidualSplitFlowProcessor(
            UAEResidualSplitFlowConfig(num_bands=len(schedule), patch_schedule=schedule),
            projector=projector,
        ).to(device)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = images
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bicubic", align_corners=False)
        x = (x - self.img_mean) / self.img_std
        outputs = self.encoder(x, output_hidden_states=True)
        tokens = outputs.last_hidden_state[:, self.unused_tokens :, :]
        b, n, c = tokens.shape
        side = int(math.sqrt(n))
        latent = tokens.transpose(1, 2).view(b, c, side, side)
        bands, _ = self.band_processor(latent)
        if not bands:
            raise RuntimeError("Band processor returned no bands.")
        return bands[0]
