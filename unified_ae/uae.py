from __future__ import annotations

import torch
import torch.nn as nn

from contextlib import nullcontext
from math import sqrt
from typing import Any, Dict, Optional, Protocol

from transformers import AutoConfig, AutoImageProcessor

from .decoders import GeneralDecoder
from .encoders import ARCHS
from .uae_frequency import FrequencyBandModulator


class Stage1Protocal(Protocol):
    patch_size: int
    hidden_size: int

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        ...


class UnifiedUAE(nn.Module):
    def __init__(
        self,
        # ---- encoder configs ----
        encoder_cls: str = "Dinov2withNorm",
        encoder_config_path: str = "facebook/dinov2-base",
        encoder_input_size: int = 224,
        encoder_params: Optional[Dict[str, Any]] = None,
        encoder_trainable: bool = False,
        # ---- decoder configs ----
        decoder_config_path: str = "vit_mae-base",
        decoder_patch_size: int = 16,
        pretrained_decoder_path: Optional[str] = None,
        # ---- noising, reshaping and normalization ----
        noise_tau: float = 0.8,
        add_train_noise: bool = False,
        reshape_to_2d: bool = True,
        normalization_stat_path: Optional[str] = None,
        eps: float = 1e-5,
        # ---- frequency modulation ----
        frequency_config: Optional[Dict[str, Any]] = None,
        frequency_trainable: bool = True,
    ):
        super().__init__()
        encoder_params = dict(encoder_params or {})
        encoder_cls = ARCHS[encoder_cls]
        self.encoder: Stage1Protocal = encoder_cls(**encoder_params)
        self.encoder_trainable = bool(encoder_trainable)
        if self.encoder_trainable:
            self.encoder.requires_grad_(True)
        else:
            self.encoder.requires_grad_(False)

        print(f"encoder_config_path: {encoder_config_path}")
        proc = AutoImageProcessor.from_pretrained(encoder_config_path)
        self.encoder_mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1)
        self.encoder_std = torch.tensor(proc.image_std).view(1, 3, 1, 1)
        encoder_config = AutoConfig.from_pretrained(encoder_config_path)

        self.encoder_input_size = encoder_input_size
        self.encoder_patch_size = self.encoder.patch_size
        self.latent_dim = self.encoder.hidden_size
        assert (
            self.encoder_input_size % self.encoder_patch_size == 0
        ), f"encoder_input_size {self.encoder_input_size} must be divisible by encoder_patch_size {self.encoder_patch_size}"
        self.base_patches = (self.encoder_input_size // self.encoder_patch_size) ** 2

        decoder_config = AutoConfig.from_pretrained(decoder_config_path)
        decoder_config.hidden_size = self.latent_dim
        decoder_config.patch_size = decoder_patch_size
        decoder_config.image_size = int(decoder_patch_size * sqrt(self.base_patches))
        self.decoder = GeneralDecoder(decoder_config, num_patches=self.base_patches)
        self.noise_tau = noise_tau
        self.add_train_noise = bool(add_train_noise)
        self.reshape_to_2d = bool(reshape_to_2d)
        if frequency_config is not None and not self.reshape_to_2d:
            raise ValueError("Frequency modulation requires reshape_to_2d=True.")
        band_mask_fusion = "noise"
        if frequency_config is not None:
            freq_cfg = dict(frequency_config)
            freq_cfg.setdefault("vocab_size", 1)
            freq_cfg.setdefault("embed_dim", self.latent_dim)
            freq_cfg.setdefault("default_qresi_counts", 0)
            band_mask_fusion = freq_cfg.pop("band_mask_fusion", "noise")
            band_noise_strategy = freq_cfg.pop("band_noise_strategy", None)
            if band_noise_strategy is None:
                band_noise_strategy = "noise" if band_mask_fusion == "noise" else "mask"
            freq_cfg["band_noise_strategy"] = band_noise_strategy
            if "band_noise_tau" not in freq_cfg:
                freq_cfg["band_noise_tau"] = self.noise_tau
            residual_kwargs = dict(freq_cfg.get("residual_projector_kwargs", {}))
            band_condition_strategy = freq_cfg.pop("band_condition_strategy", residual_kwargs.pop("condition_strategy", "mask"))
            band_condition_eps = float(freq_cfg.pop("band_condition_eps", residual_kwargs.pop("condition_eps", 0.0)))
            clamp_condition = bool(residual_kwargs.pop("clamp_condition", True))
            residual_kwargs.setdefault("condition_strategy", band_condition_strategy)
            residual_kwargs.setdefault("condition_eps", band_condition_eps)
            residual_kwargs.setdefault("clamp_condition", clamp_condition)
            freq_cfg["residual_projector_kwargs"] = residual_kwargs
            self.freq_modulator = FrequencyBandModulator(**freq_cfg)
            self.freq_modulator.requires_grad_(bool(frequency_trainable))
            self.decoder.band_mask_fusion = "noise"
            self.decoder.set_band_mask_dim(0)
        else:
            self.freq_modulator = None
            self.decoder.band_mask_fusion = "none"
            self.decoder.set_band_mask_dim(0)

        self.latent_mean: Optional[torch.Tensor] = None
        self.latent_var: Optional[torch.Tensor] = None
        self.do_normalization = False
        self.eps = eps
        if normalization_stat_path is not None:
            stats = torch.load(normalization_stat_path, map_location="cpu")
            self.latent_mean = stats.get("mean", None)
            self.latent_var = stats.get("var", None)
            self.do_normalization = True
            print(f"Loaded normalization stats from {normalization_stat_path}")

        if pretrained_decoder_path is not None:
            print(f"Loading pretrained decoder from {pretrained_decoder_path}")
            state_dict = torch.load(pretrained_decoder_path, map_location="cpu")
            if isinstance(state_dict, dict):
                if "model" in state_dict and isinstance(state_dict["model"], dict):
                    decoder_state = {
                        key.removeprefix("decoder."): value
                        for key, value in state_dict["model"].items()
                        if key.startswith("decoder.")
                    }
                    if decoder_state:
                        state_dict = decoder_state
                elif "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
                    decoder_state = {
                        key.removeprefix("decoder."): value
                        for key, value in state_dict["state_dict"].items()
                        if key.startswith("decoder.")
                    }
                    if decoder_state:
                        state_dict = decoder_state
            decoder_current = self.decoder.state_dict()
            filtered_state: Dict[str, torch.Tensor] = {}
            for key, tensor in state_dict.items():
                if key in decoder_current and decoder_current[key].shape == tensor.shape:
                    filtered_state[key] = tensor
            keys = self.decoder.load_state_dict(filtered_state, strict=False)
            if len(keys.missing_keys) > 0:
                print(f"Missing keys when loading pretrained decoder: {keys.missing_keys}")

    def encode(
        self,
        x: torch.Tensor,
        *,
        apply_frequency: Optional[bool] = None,
        dropout: Optional[torch.Tensor] = None,
        freq_ratio: Optional[torch.Tensor] = None,
        freeze_encoder: bool = False,
        return_pre_frequency: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        apply_frequency = self.freq_modulator is not None if apply_frequency is None else apply_frequency
        if freq_ratio is not None:
            apply_frequency = True
        encoder_ctx = torch.no_grad() if freeze_encoder else nullcontext()
        with encoder_ctx:
            _, _, h, w = x.shape
            if h != self.encoder_input_size or w != self.encoder_input_size:
                x = nn.functional.interpolate(
                    x,
                    size=(self.encoder_input_size, self.encoder_input_size),
                    mode="bicubic",
                    align_corners=False,
                )
            x = (x - self.encoder_mean.to(x.device)) / self.encoder_std.to(x.device)
            z = self.encoder(x)

            if self.reshape_to_2d:
                b, n, c = z.shape
                h = w = int(sqrt(n))
                z = z.transpose(1, 2).view(b, c, h, w)
        if freeze_encoder:
            z = z.detach()
        if self.do_normalization:
            mean = self._format_latent_stat(self.latent_mean, z.device, z.dtype)
            var = self._format_latent_stat(self.latent_var, z.device, z.dtype)
            if mean is not None and var is not None:
                z = (z - mean) / torch.sqrt(var + z.new_tensor(self.eps))
        pre_frequency_latent = z
        if apply_frequency:
            z = self._apply_frequency(z, dropout=dropout, freq_ratio=freq_ratio)

        if self.training and self.add_train_noise and self.noise_tau > 0:
            noise_sigma = self.noise_tau * torch.rand(
                (z.size(0),) + (1,) * (z.dim() - 1),
                device=z.device,
                dtype=z.dtype,
            )
            z = z + noise_sigma * torch.randn_like(z)

        if return_pre_frequency:
            return pre_frequency_latent, z
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.do_normalization:
            mean = self._format_latent_stat(self.latent_mean, z.device, z.dtype)
            var = self._format_latent_stat(self.latent_var, z.device, z.dtype)
            if mean is not None and var is not None:
                z = z * torch.sqrt(var + z.new_tensor(self.eps)) + mean
        if self.reshape_to_2d:
            b, c, h, w = z.shape
            n = h * w
            z = z.view(b, c, n).transpose(1, 2)
        output = self.decoder(z, band_mask=None, drop_cls_token=False).logits
        x_rec = self.decoder.unpatchify(output)
        x_rec = x_rec * self.encoder_std.to(x_rec.device) + self.encoder_mean.to(x_rec.device)
        return x_rec

    def _combine_band_latents(
        self,
        bands: torch.Tensor,
        *,
        freq_ratio: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if bands.ndim != 5:
            raise ValueError(f"Expected band tensor with shape (B, num_bands, C, H, W); got {tuple(bands.shape)}.")
        if bands.size(2) != self.latent_dim:
            raise ValueError(
                f"Band tensor channel dimension {bands.size(2)} does not match latent_dim={self.latent_dim}."
            )
        if self.freq_modulator is None:
            return bands.sum(dim=1)
        combined = self.freq_modulator.reconstruct_from_bands(bands, band_ratio=freq_ratio)
        if combined.dtype != bands.dtype:
            combined = combined.to(dtype=bands.dtype)
        return combined

    def _apply_frequency(
        self,
        latent: torch.Tensor,
        *,
        dropout: Optional[torch.Tensor] = None,
        freq_ratio: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.freq_modulator is None:
            return latent
        b, c, h, w = latent.shape
        if c % self.latent_dim == 0 and c != self.latent_dim:
            num_bands = c // self.latent_dim
            band_tensor = latent.view(b, num_bands, self.latent_dim, h, w)
            combined = self._combine_band_latents(
                band_tensor,
                freq_ratio=freq_ratio,
            )
            if combined.dtype != latent.dtype:
                combined = combined.to(dtype=latent.dtype)
            return combined
        original_dtype = latent.dtype
        needs_cast = original_dtype not in (torch.float32, torch.float64)
        if needs_cast:
            latent = latent.float()
        if latent.is_cuda and torch.is_autocast_enabled():
            autocast_ctx = torch.cuda.amp.autocast(enabled=False)
        else:
            autocast_ctx = nullcontext()
        with autocast_ctx:
            latent_fp = latent if latent.dtype == torch.float32 else latent.float()
            ratio_fp = freq_ratio
            if ratio_fp is not None and ratio_fp.dtype != latent_fp.dtype:
                ratio_fp = ratio_fp.to(dtype=latent_fp.dtype)
            freq_module = self.freq_modulator
            target_dtype = latent_fp.dtype
            ref_tensor = None
            for tensor in freq_module.parameters():
                if torch.is_floating_point(tensor):
                    ref_tensor = tensor
                    break
            if ref_tensor is None:
                for tensor in freq_module.buffers():
                    if torch.is_floating_point(tensor):
                        ref_tensor = tensor
                        break
            if ref_tensor is not None and ref_tensor.dtype != target_dtype:
                freq_module.to(dtype=target_dtype)
            quant, _, _, _, _ = freq_module(latent_fp, dropout=dropout, band_ratio=ratio_fp)
        if quant.dtype != original_dtype:
            quant = quant.to(dtype=original_dtype)
        return quant

    def _format_latent_stat(self, stat: Optional[torch.Tensor], device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        if stat is None:
            return None
        tensor = stat.to(device=device, dtype=dtype)
        if not self.reshape_to_2d:
            if tensor.ndim == 2 and tensor.shape[0] == self.base_patches and tensor.shape[1] == self.latent_dim:
                return tensor.unsqueeze(0)
            if tensor.ndim == 1 and tensor.numel() == self.latent_dim:
                return tensor.view(1, 1, self.latent_dim)
            return tensor
        h = w = int(sqrt(self.base_patches))
        if tensor.ndim == 1 and tensor.numel() == self.latent_dim:
            return tensor.view(1, self.latent_dim, 1, 1)
        if tensor.ndim == 2 and tensor.shape[0] == self.base_patches and tensor.shape[1] == self.latent_dim:
            return tensor.view(h, w, self.latent_dim).permute(2, 0, 1).unsqueeze(0)
        if tensor.ndim == 3 and tensor.shape[0] == h and tensor.shape[1] == w and tensor.shape[2] == self.latent_dim:
            return tensor.permute(2, 0, 1).unsqueeze(0)
        if tensor.ndim == 4:
            if tensor.shape[0] == 1:
                return tensor
            return tensor.view(1, *tensor.shape)
        return tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x, apply_frequency=None)
        return self.decode(z)
