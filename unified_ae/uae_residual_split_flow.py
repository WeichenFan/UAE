from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class UAEResidualSplitFlowConfig:
    num_bands: int = 10
    patch_schedule: Optional[Sequence[int]] = None
    dtype: torch.dtype = torch.float32
    condition_strategy: str = "mask"
    condition_eps: float = 0.0
    clamp_condition: bool = True


class BandProjector(nn.Module):
    """Abstract projector interface for residual split flow."""

    def precompute(self, x: torch.Tensor, num_bands: int, schedule: Optional[Sequence[int]]) -> None:
        raise NotImplementedError

    def project(self, x: torch.Tensor, band_idx: int) -> torch.Tensor:
        raise NotImplementedError


class FFTBandProjector(BandProjector):
    """FFT-based ring projector matching the legacy implementation."""

    def __init__(self) -> None:
        super().__init__()
        self._radius_grid: Optional[torch.Tensor] = None
        self._radii: Optional[torch.Tensor] = None

    def precompute(self, x: torch.Tensor, num_bands: int, schedule: Optional[Sequence[int]]) -> None:
        device, dtype = x.device, x.dtype
        _, _, h, w = x.shape
        centre_h, centre_w = h // 2, w // 2
        y_loc, x_loc = torch.meshgrid(
            torch.arange(-centre_h, h - centre_h, device=device, dtype=dtype),
            torch.arange(-centre_w, w - centre_w, device=device, dtype=dtype),
            indexing="ij",
        )
        radius_grid = torch.sqrt(y_loc.square() + x_loc.square())
        max_radius = float(min(centre_h, centre_w))

        if schedule is not None and len(schedule) >= num_bands:
            schedule_tensor = torch.tensor(schedule[:num_bands], device=device, dtype=dtype)
            if schedule_tensor.numel() > 1 and torch.all(schedule_tensor[1:] >= schedule_tensor[:-1]):
                max_val = schedule_tensor[-1].clamp_min(1e-6)
                radii = schedule_tensor / max_val * max_radius
                radii = torch.cat([torch.zeros(1, device=device, dtype=dtype), radii])
            else:
                weights = torch.clamp(schedule_tensor, min=0.0)
                total = weights.sum()
                if total <= 0:
                    radii = torch.linspace(0.0, max_radius, num_bands + 1, device=device, dtype=dtype)
                else:
                    weights = weights / total
                    cumulative = torch.cumsum(weights, dim=0) * max_radius
                    radii = torch.cat([torch.zeros(1, device=device, dtype=dtype), cumulative])
        else:
            radii = torch.linspace(0.0, max_radius, num_bands + 1, device=device, dtype=dtype)

        self._radius_grid = radius_grid
        self._radii = radii

    def _band_mask(self, band_idx: int, x: torch.Tensor) -> torch.Tensor:
        assert self._radius_grid is not None and self._radii is not None, "Projector not initialised."
        inner, outer = self._radii[band_idx], self._radii[band_idx + 1]
        mask = ((self._radius_grid >= inner) & (self._radius_grid <= outer)).to(dtype=x.dtype, device=x.device)
        return mask.expand_as(x)

    def project(self, x: torch.Tensor, band_idx: int) -> torch.Tensor:
        mask = self._band_mask(band_idx, x)
        x_fft = fft.fftshift(fft.fft2(torch.complex(x, torch.zeros_like(x)), dim=(-2, -1)), dim=(-2, -1))
        filtered = fft.ifftshift(x_fft * mask, dim=(-2, -1))
        return fft.ifft2(filtered, dim=(-2, -1)).real


class LearnedConvProjector(BandProjector):
    """
    Depthwise convolutional projector that learns soft spatial gates per band.
    """

    def __init__(
        self,
        kernel_size: int = 3,
        dilation_growth: float = 1.5,
        temperature: float = 1.0,
        activation: str = "sigmoid",
        init_scale: float = 0.02,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("LearnedConvProjector.kernel_size must be odd for 'same' padding.")
        if temperature <= 0:
            raise ValueError("LearnedConvProjector.temperature must be > 0.")
        self.kernel_size = int(kernel_size)
        self.dilation_growth = float(max(1.0, dilation_growth))
        self.temperature = float(temperature)
        self.activation = activation.lower()
        self.init_scale = float(init_scale)

        self.convs = nn.ModuleList()
        self._channels: Optional[int] = None
        self._num_bands: Optional[int] = None
        self._device: Optional[torch.device] = None
        self._dtype: Optional[torch.dtype] = None

    def _build_layers(
        self,
        channels: int,
        num_bands: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        needs_rebuild = (
            self._channels != channels
            or self._num_bands != num_bands
            or len(self.convs) != num_bands
            or self._device != device
            or self._dtype != dtype
        )
        if not needs_rebuild:
            for conv in self.convs:
                if conv.weight.dtype != dtype or conv.weight.device != device:
                    conv.to(device=device, dtype=dtype)
            self._device = device
            self._dtype = dtype
            return

        convs: List[nn.Conv2d] = []
        for idx in range(num_bands):
            dilation = max(1, int(round(self.dilation_growth ** idx)))
            padding = ((self.kernel_size - 1) * dilation) // 2
            conv = nn.Conv2d(
                channels,
                channels,
                kernel_size=self.kernel_size,
                padding=padding,
                dilation=dilation,
                groups=channels,
                bias=True,
            )
            nn.init.normal_(conv.weight, mean=0.0, std=self.init_scale)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)
            convs.append(conv.to(device=device, dtype=dtype))
        self.convs = nn.ModuleList(convs)
        self._channels = channels
        self._num_bands = num_bands
        self._device = device
        self._dtype = dtype

    def precompute(self, x: torch.Tensor, num_bands: int, schedule: Optional[Sequence[int]]) -> None:
        del schedule
        _, channels, _, _ = x.shape
        self._build_layers(int(channels), int(num_bands), x.device, x.dtype)

    def _activate(self, logits: torch.Tensor) -> torch.Tensor:
        temperature = logits.new_tensor(self.temperature, dtype=logits.dtype)
        scaled = logits / temperature
        if self.activation == "sigmoid":
            return torch.sigmoid(scaled)
        if self.activation == "softplus":
            return F.softplus(scaled)
        if self.activation == "tanh":
            return torch.tanh(scaled)
        raise ValueError(f"Unsupported activation '{self.activation}' for LearnedConvProjector.")

    def project(self, x: torch.Tensor, band_idx: int) -> torch.Tensor:
        if band_idx >= len(self.convs):
            raise IndexError(f"Band index {band_idx} out of range for {len(self.convs)} learned conv filters.")
        conv = self.convs[band_idx]
        device = x.device
        if torch.is_autocast_enabled():
            conv = conv.to(device=device, dtype=torch.float32)
            self.convs[band_idx] = conv
            self._device = conv.weight.device
            self._dtype = conv.weight.dtype
            x_conv = x.to(dtype=torch.float32)
            with torch.cuda.amp.autocast(enabled=False):
                logits = conv(x_conv)
        else:
            if conv.weight.dtype != x.dtype or conv.weight.device != device:
                conv = conv.to(device=device, dtype=x.dtype)
                self.convs[band_idx] = conv
                self._device = conv.weight.device
                self._dtype = conv.weight.dtype
            logits = conv(x)
        mask = self._activate(logits)
        if mask.dtype != x.dtype:
            mask = mask.to(dtype=x.dtype)
        return x * mask


class UAEResidualSplitFlowProcessor(nn.Module):
    """Invertible projection + residual decomposition with optional band conditioning."""

    def __init__(
        self,
        config: Optional[UAEResidualSplitFlowConfig] = None,
        projector: Optional[BandProjector] = None,
    ) -> None:
        super().__init__()
        self.config = config or UAEResidualSplitFlowConfig()
        self.num_bands = max(1, int(self.config.num_bands))
        self.patch_schedule = (
            list(self.config.patch_schedule) if self.config.patch_schedule is not None else None
        )
        self.projector = projector or FFTBandProjector()
        self.condition_strategy = self.config.condition_strategy.lower()
        if self.condition_strategy not in {"mask", "scale", "residual", "noise_only"}:
            raise ValueError(
                f"Unsupported condition_strategy '{self.config.condition_strategy}'. "
                "Expected one of: 'mask', 'scale', 'residual', 'noise_only'."
            )

    def forward(self, latent: torch.Tensor, band_condition: Optional[torch.Tensor] = None) -> Tuple[
        List[torch.Tensor], List[torch.Tensor]
    ]:
        return self.decompose(latent, band_condition=band_condition)

    def decompose(
        self,
        latent: torch.Tensor,
        band_condition: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        x = latent.to(dtype=self.config.dtype)
        self.projector.precompute(x, self.num_bands, self.patch_schedule)

        residual = x
        cumulative_acc = torch.zeros_like(x)
        bands: List[torch.Tensor] = []
        cumulative: List[torch.Tensor] = []

        condition = None
        if band_condition is not None:
            condition = band_condition.to(dtype=self.config.dtype)
            if self.config.clamp_condition:
                condition = condition.clamp_(0.0, 1.0)

        for band_idx in range(self.num_bands):
            band_component = self.projector.project(residual, band_idx)
            residual = residual - band_component

            if condition is not None:
                cond_val = condition[:, band_idx]
                cond_val = cond_val.view(-1, 1, 1, 1).to(dtype=band_component.dtype, device=band_component.device)
                if self.condition_strategy in {"mask", "scale"}:
                    band_component = band_component * cond_val
                elif self.condition_strategy == "residual":
                    keep = band_component * cond_val
                    residual = residual + (band_component - keep)
                    band_component = keep
                elif self.condition_strategy == "noise_only":
                    pass
                else:
                    raise ValueError(f"Unsupported condition_strategy '{self.condition_strategy}'.")

            bands.append(band_component.to(latent.dtype))
            cumulative_acc = cumulative_acc + band_component
            cumulative.append(cumulative_acc.to(latent.dtype).clone())

        return bands, cumulative
