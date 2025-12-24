from __future__ import annotations

import logging
import numpy as np
import torch
import torch.nn as nn

from typing import Dict, List, Optional, Sequence, Tuple, Union

from .uae_residual_split_flow import (
    FFTBandProjector,
    LearnedConvProjector,
    UAEResidualSplitFlowConfig,
    UAEResidualSplitFlowProcessor,
)


class BandSpectralLayer(nn.Module):
    def __init__(self, num_bands: int, identity_init: bool = False):
        super().__init__()
        weight = torch.eye(num_bands, dtype=torch.float32)
        if not identity_init:
            weight = weight + 0.01 * torch.randn_like(weight)
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, c, h, w = x.shape
        flat = x.reshape(b, s, -1)
        mixed = torch.einsum("ij,bjk->bik", self.weight, flat)
        return mixed.reshape(b, s, c, h, w)


class BandSpectralTransform(nn.Module):
    def __init__(self, num_bands: int, num_layers: int, identity_init: bool = False):
        super().__init__()
        self.layers = nn.ModuleList(
            [BandSpectralLayer(num_bands, identity_init=identity_init) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            if x.dtype != layer.weight.dtype:
                x = x.to(layer.weight.dtype)
            x = layer(x)
        return x


class FrequencyBandModulator(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        using_znorm: bool = True,
        beta: float = 0.25,
        default_qresi_counts: int = 0,
        v_patch_nums: Optional[Sequence[float]] = None,
        quant_resi: float = 0.0,
        share_quant_resi: int = 4,
        num_latent_tokens: int = 256,
        codebook_drop: float = 0.0,
        spectral_transform_layers: int = 0,
        spectral_identity_init: bool = True,
        freq_first_patch: float = 1.0,
        freq_num_bands: int = 10,
        freq_max_patch: float = 16.0,
        use_residual_split: bool = True,
        residual_projector: str = "fft",
        residual_projector_kwargs: Optional[dict] = None,
        band_noise_strategy: str = "mask",
        band_noise_tau: float = 0.0,
        dynamic_schedule: Optional[dict] = None,
    ):
        super().__init__()
        del using_znorm, beta, default_qresi_counts, share_quant_resi, num_latent_tokens

        self.embed_dim = embed_dim
        self.codebook_drop = codebook_drop
        self.freq_first_patch = float(freq_first_patch)
        self.freq_num_bands = int(freq_num_bands)
        self.freq_max_patch = float(freq_max_patch)
        self.use_residual_split = bool(use_residual_split)
        self.residual_projector = residual_projector
        self.residual_projector_kwargs = dict(residual_projector_kwargs or {})
        self.spectral_identity_init = bool(spectral_identity_init)
        self.band_transform_layers = int(spectral_transform_layers)
        self.band_noise_strategy = str(band_noise_strategy or "mask").lower()
        if self.band_noise_strategy not in {"mask", "noise"}:
            raise ValueError(f"Unsupported band_noise_strategy '{band_noise_strategy}'. Expected 'mask' or 'noise'.")
        self.band_noise_tau = float(max(0.0, band_noise_tau))
        self.dynamic_schedule_cfg = self._parse_dynamic_schedule(dynamic_schedule)
        self.dynamic_transforms = nn.ModuleDict()
        self.dynamic_denoisers = nn.ModuleDict()
        self.denoiser: Optional[nn.Module] = None
        self._denoiser_in_channels: Optional[int] = None
        schedule_init = [float(x) for x in v_patch_nums] if v_patch_nums else self.generate_schedule(
            self.freq_first_patch, self.freq_num_bands, self.freq_max_patch
        )
        self.freq_num_bands = len(schedule_init)

        if abs(quant_resi) > 1e-6:
            logging.getLogger(__name__).warning(
                "FrequencyBandModulator.quant_resi is deprecated; spectral transform handles refinement."
            )

        self._build_dynamic_banks()

        self.configure_bands(
            self.freq_first_patch, self.freq_num_bands, self.freq_max_patch, schedule=schedule_init
        )

    def forward(
        self,
        f_BChw: torch.Tensor,
        ret_usages: bool = False,
        dropout: Optional[torch.Tensor] = None,
        band_ratio: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[List[float]], torch.Tensor, torch.Tensor, torch.Tensor]:
        processed_bands, quant, energy_post = self._process_latent(
            f_BChw, dropout, self.training, band_ratio=band_ratio
        )
        zero = f_BChw.new_zeros(())
        usages = [value * 100.0 for value in energy_post] if ret_usages else None
        return quant, usages, zero, zero, zero

    def decompose_latent(
        self,
        z: torch.Tensor,
        to_fhat: bool,
        v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None,
        return_bands: bool = False,
        band_ratio: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        del v_patch_nums
        processed_bands, quant, _ = self._process_latent(z, None, False, band_ratio=band_ratio)
        if to_fhat:
            return processed_bands if return_bands else [quant]

        features: List[torch.Tensor] = []
        for lat in (quant,):
            b, c, h, w = lat.shape
            features.append(lat.view(b, c, h * w).transpose(1, 2).contiguous())
        return features

    def reconstruct_from_bands(
        self,
        band_tensor: torch.Tensor,
        *,
        band_ratio: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if band_tensor.ndim != 5:
            raise ValueError(
                f"Expected band tensor with shape (B, num_bands, C, H, W); got {tuple(band_tensor.shape)}."
            )
        b, num_bands, channels, h, w = band_tensor.shape
        if channels != self.embed_dim:
            raise ValueError(
                f"Band tensor channel dimension {channels} does not match embed_dim={self.embed_dim}."
            )
        device = band_tensor.device
        original_dtype = band_tensor.dtype
        working_dtype = original_dtype if original_dtype in (torch.float32, torch.float64) else torch.float32
        bands_fp = band_tensor if band_tensor.dtype == working_dtype else band_tensor.to(dtype=working_dtype)

        ratio = band_ratio
        mask_tensor: Optional[torch.Tensor] = None
        if ratio is not None:
            ratio = ratio.to(device=device, dtype=working_dtype).view(-1)
            ratio = torch.clamp(ratio, 0.0, 1.0)
            keep_counts = torch.ceil(ratio * num_bands).to(dtype=torch.int64)
            keep_counts = torch.clamp(keep_counts, min=1)
            band_indices = torch.arange(num_bands, device=device, dtype=torch.int64)
            mask_tensor = (keep_counts.view(-1, 1) > band_indices.view(1, -1)).to(dtype=working_dtype)
            mask_tensor = mask_tensor.view(b, num_bands, 1, 1, 1)
            bands_fp = bands_fp * mask_tensor

        processed = [bands_fp[:, idx, ...].contiguous() for idx in range(num_bands)]

        stacked = torch.cat(processed, dim=1)
        denoiser = self._select_denoiser(num_bands, stacked.device, stacked.dtype)
        denoised = denoiser(stacked)
        residual = sum(processed)
        combined = denoised + residual

        if combined.dtype != original_dtype:
            combined = combined.to(dtype=original_dtype)
        return combined

    def _build_dynamic_banks(self) -> None:
        if self.dynamic_schedule_cfg is None:
            self._ensure_band_transform()
            self._ensure_denoiser(self.freq_num_bands)
            return
        min_b = int(self.dynamic_schedule_cfg["min_bands"])
        max_b = int(self.dynamic_schedule_cfg["max_bands"])
        for num in range(min_b, max_b + 1):
            key = str(num)
            if self.band_transform_layers > 0 and key not in self.dynamic_transforms:
                self.dynamic_transforms[key] = BandSpectralTransform(
                    num,
                    self.band_transform_layers,
                    identity_init=self.spectral_identity_init,
                )
            if key not in self.dynamic_denoisers:
                self.dynamic_denoisers[key] = self._build_denoiser_module(num)

    def _parse_dynamic_schedule(self, cfg: Optional[dict]) -> Optional[Dict[str, float]]:
        if not cfg:
            return None
        cfg = dict(cfg)
        if not cfg.get("enabled"):
            return None
        min_bands = int(cfg.get("min_bands", self.freq_num_bands))
        max_bands = int(cfg.get("max_bands", self.freq_num_bands))
        if min_bands < 1:
            min_bands = 1
        if max_bands < min_bands:
            max_bands = min_bands
        first_patch = float(cfg.get("first_patch", self.freq_first_patch))
        max_patch = float(cfg.get("max_patch", self.freq_max_patch))
        min_step = float(cfg.get("min_step", 1e-6))
        max_patch = max(first_patch + min_step, max_patch)
        return {
            "min_bands": min_bands,
            "max_bands": max_bands,
            "first_patch": first_patch,
            "max_patch": max_patch,
            "train_only": bool(cfg.get("train_only", True)),
            "randomize_inner": bool(cfg.get("randomize_inner", True)),
            "min_step": min_step,
        }

    @staticmethod
    def generate_schedule(first_patch: float, num_bands: int, max_patch: float) -> List[float]:
        first_patch = float(first_patch)
        max_patch = float(max_patch)
        num_bands = max(1, int(num_bands))
        if num_bands == 1:
            return [max(first_patch, max_patch)]
        max_patch = max(first_patch, max_patch)
        tokens = np.linspace(first_patch ** 2, max_patch ** 2, num_bands)
        schedule: List[float] = []
        last = None
        for value in tokens:
            patch = float(np.sqrt(value))
            if last is not None and patch <= last:
                patch = last + 1e-6
            schedule.append(patch)
            last = patch
        schedule[0] = first_patch
        schedule[-1] = max_patch
        return schedule

    def configure_bands(
        self,
        first_patch: float,
        num_bands: int,
        max_patch: float,
        schedule: Optional[Sequence[float]] = None,
    ) -> None:
        first_patch = float(first_patch)
        max_patch = float(max_patch)
        num_bands = max(1, int(num_bands))
        if schedule is None:
            schedule = self.generate_schedule(first_patch, num_bands, max_patch)
        else:
            schedule = [float(x) for x in schedule]
            if len(schedule) == 0:
                schedule = self.generate_schedule(first_patch, num_bands, max_patch)
            else:
                num_bands = len(schedule)
        self.freq_first_patch = first_patch
        self.freq_num_bands = num_bands
        self.freq_max_patch = max_patch
        self.v_patch_nums = schedule

        projector_name = self.residual_projector.lower()
        projector_kwargs = dict(self.residual_projector_kwargs)
        condition_strategy = projector_kwargs.pop("condition_strategy", "mask")
        condition_eps = float(projector_kwargs.pop("condition_eps", 0.0))
        clamp_condition = bool(projector_kwargs.pop("clamp_condition", True))
        if projector_name == "fft":
            projector = FFTBandProjector()
        elif projector_name in {"learned_conv", "learned-conv", "learnedconv"}:
            projector = LearnedConvProjector(**projector_kwargs)
        else:
            raise ValueError(f"Unsupported residual projector '{self.residual_projector}'.")
        self.processor = UAEResidualSplitFlowProcessor(
            UAEResidualSplitFlowConfig(
                num_bands=len(schedule),
                patch_schedule=schedule,
                dtype=torch.float32,
                condition_strategy=condition_strategy,
                condition_eps=condition_eps,
                clamp_condition=clamp_condition,
            ),
            projector=projector,
        )

        self._ensure_band_transform()
        self._reset_denoiser()

    def _reset_denoiser(self) -> None:
        if self.dynamic_schedule_cfg is not None:
            self.denoiser = None
            self._denoiser_in_channels = None
            return
        self.denoiser = None
        self._denoiser_in_channels = None
        if getattr(self, "freq_num_bands", 0) > 0:
            self._ensure_denoiser(self.freq_num_bands)

    def set_band_splits(
        self,
        num_bands: Optional[int] = None,
        first_patch: Optional[float] = None,
        max_patch: Optional[float] = None,
        schedule: Optional[Sequence[float]] = None,
    ) -> None:
        if schedule is not None and len(schedule) == 0:
            schedule = None
        if schedule is not None:
            num = len(schedule)
        else:
            num = num_bands if num_bands is not None else self.freq_num_bands
        first = first_patch if first_patch is not None else self.freq_first_patch
        max_p = max_patch if max_patch is not None else self.freq_max_patch
        self.configure_bands(first, num, max_p, schedule=schedule)

    def _build_dynamic_schedule(self, num_bands: int, cfg: Dict[str, float], device: torch.device) -> List[float]:
        if num_bands <= 1:
            return [max(cfg["first_patch"], cfg["max_patch"])]
        num_inner = max(0, num_bands - 2)
        epsilon = cfg["min_step"]
        first_patch = cfg["first_patch"]
        max_patch = max(first_patch + epsilon, cfg["max_patch"])
        inner: List[float] = []
        if num_inner > 0:
            if cfg.get("randomize_inner", True):
                span = max_patch - first_patch
                samples = torch.rand(num_inner, device=device) * span + first_patch
                inner = torch.sort(samples).values.cpu().tolist()
            else:
                base = self.generate_schedule(first_patch, num_bands, max_patch)
                inner = base[1:-1]
        schedule = [first_patch] + inner + [max_patch]
        for idx in range(1, len(schedule)):
            if schedule[idx] <= schedule[idx - 1]:
                schedule[idx] = schedule[idx - 1] + epsilon
        return schedule

    def _maybe_resample_schedule(self, training: bool, device: torch.device) -> None:
        cfg = self.dynamic_schedule_cfg
        if not cfg:
            return
        if cfg["train_only"] and not training:
            return
        min_bands = cfg["min_bands"]
        max_bands = cfg["max_bands"]
        if max_bands <= min_bands:
            num_bands = min_bands
        else:
            num_bands = int(torch.randint(min_bands, max_bands + 1, (1,), device=device).item())
        schedule = self._build_dynamic_schedule(num_bands, cfg, device)
        processor = getattr(self, "processor", None)
        if processor is None:
            return
        processor.num_bands = len(schedule)
        processor.patch_schedule = list(schedule)
        if hasattr(processor, "config"):
            processor.config.num_bands = len(schedule)
            processor.config.patch_schedule = list(schedule)
        self.freq_num_bands = len(schedule)
        self.v_patch_nums = list(schedule)

    def _ensure_band_transform(self) -> None:
        if self.band_transform_layers <= 0 or self.freq_num_bands <= 0:
            self.band_transform = None
            return
        self.band_transform = BandSpectralTransform(
            self.freq_num_bands,
            self.band_transform_layers,
            identity_init=self.spectral_identity_init,
        )

    def _ensure_denoiser(self, num_bands: int) -> None:
        if self.dynamic_schedule_cfg is not None:
            return
        in_channels = self.embed_dim * num_bands
        if self.denoiser is not None and self._denoiser_in_channels == in_channels:
            return
        self.denoiser = self._build_denoiser_module(num_bands)
        self._denoiser_in_channels = in_channels

    def _build_denoiser_module(self, num_bands: int) -> nn.Sequential:
        in_channels = self.embed_dim * num_bands
        hidden = max(self.embed_dim * 2, 512)
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, self.embed_dim, kernel_size=3, padding=1),
        )

    def _select_band_transform(
        self,
        num_bands: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[nn.Module]:
        if self.band_transform_layers <= 0:
            return None
        if self.dynamic_schedule_cfg is None:
            module = self.band_transform
            if module is not None:
                module = module.to(device=device, dtype=dtype)
            return module
        key = str(num_bands)
        if key not in self.dynamic_transforms:
            self.dynamic_transforms[key] = BandSpectralTransform(
                num_bands,
                self.band_transform_layers,
                identity_init=self.spectral_identity_init,
            )
        module = self.dynamic_transforms[key].to(device=device, dtype=dtype)
        self.dynamic_transforms[key] = module
        return module

    def _select_denoiser(self, num_bands: int, device: torch.device, dtype: torch.dtype) -> nn.Module:
        if self.dynamic_schedule_cfg is None:
            self._ensure_denoiser(num_bands)
            assert self.denoiser is not None
            return self.denoiser
        key = str(num_bands)
        if key not in self.dynamic_denoisers:
            self.dynamic_denoisers[key] = self._build_denoiser_module(num_bands)
        module = self.dynamic_denoisers[key].to(device=device, dtype=dtype)
        self.dynamic_denoisers[key] = module
        return module

    def _prepare_raw_bands(
        self,
        latent: torch.Tensor,
        band_ratio: Optional[torch.Tensor],
        training: bool,
    ) -> Tuple[List[torch.Tensor], Optional[torch.Tensor], int]:
        device = latent.device
        b = latent.size(0)
        self._maybe_resample_schedule(training, device)

        processor = getattr(self, "processor", None)
        if processor is None:
            raise RuntimeError("FrequencyBandModulator processor is not initialised; call configure_bands first.")

        num_processor_bands = max(1, processor.num_bands)

        if band_ratio is not None:
            band_ratio = band_ratio.to(device=device, dtype=latent.dtype).view(-1)
            band_ratio = torch.clamp(band_ratio, 0.0, 1.0)
        else:
            band_ratio = torch.ones(b, device=device, dtype=latent.dtype)

        keep_counts = torch.ceil(band_ratio * num_processor_bands).to(dtype=torch.int64)
        keep_counts = torch.clamp(keep_counts, min=1)

        band_indices = torch.arange(num_processor_bands, device=device, dtype=torch.int64)
        band_condition = (keep_counts.view(-1, 1) > band_indices.view(1, -1)).to(dtype=latent.dtype)

        raw_bands, _ = processor.decompose(latent, band_condition=band_condition)
        if not raw_bands:
            raw_bands = [latent]
            band_condition = torch.ones(b, 1, device=device, dtype=latent.dtype)

        return raw_bands, band_condition, num_processor_bands

    def _process_latent(
        self,
        latent: torch.Tensor,
        dropout: Optional[torch.Tensor],
        training: bool,
        *,
        band_ratio: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, List[float]]:
        device = latent.device
        b = latent.size(0)
        del dropout
        raw_bands, band_condition, _ = self._prepare_raw_bands(latent, band_ratio, training)
        num_bands = len(raw_bands)

        corrupted_bands: List[torch.Tensor] = []
        use_noise = self.band_noise_strategy == "noise" and training
        for si, band in enumerate(raw_bands):
            if si < band_condition.shape[1]:
                band_keep = band_condition[:, si]
            else:
                band_keep = torch.ones(b, device=device, dtype=latent.dtype)
            if use_noise:
                keep_mask = band_keep.view(b, 1, 1, 1)
                if torch.all(keep_mask.bool()):
                    corrupted = band
                else:
                    if self.band_noise_tau > 0.0:
                        noise_sigma = self.band_noise_tau * torch.rand(
                            (b,) + (1,) * (band.dim() - 1),
                            device=device,
                            dtype=latent.dtype,
                        )
                        noise = noise_sigma * torch.randn_like(band)
                    else:
                        noise = torch.zeros_like(band)
                    corrupted = band + (1.0 - keep_mask) * noise
            else:
                corrupted = band
            corrupted_bands.append(corrupted)

        band_transform = self._select_band_transform(len(corrupted_bands), device, latent.dtype)
        if band_transform is not None:
            stacked = torch.stack(corrupted_bands, dim=1)
            transformed_stack = band_transform(stacked)
            transformed_bands = [transformed_stack[:, si, ...] for si in range(num_bands)]
        else:
            transformed_bands = corrupted_bands

        energy_post: List[float] = [band.pow(2).mean().detach().item() for band in transformed_bands]

        processed_bands: List[torch.Tensor] = [band.clone() for band in transformed_bands]

        stacked_features = torch.cat(processed_bands, dim=1)
        denoiser = self._select_denoiser(len(processed_bands), stacked_features.device, stacked_features.dtype)
        denoised = denoiser(stacked_features)
        residual = sum(processed_bands)
        f_hat = denoised + residual

        return processed_bands, f_hat, energy_post



__all__ = [
    "BandSpectralTransform",
    "FrequencyBandModulator",
]
