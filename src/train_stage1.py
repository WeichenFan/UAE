# Copyright (c) Meta Platforms.
# Licensed under the MIT license.
"""
Stage-1 UAE training script with reconstruction, LPIPS, and GAN losses.

This script adapts the training logic from the Kakao Brain VQGAN trainer while
targeting the UAE autoencoder architecture used in this repository.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.models import VGG16_Weights
from torchvision.datasets import ImageFolder
from glob import glob
from torchvision.utils import make_grid
from omegaconf import OmegaConf
from eval import evaluate_reconstruction_distributed
from disc import (
    DiffAug,
    LPIPS,
    build_discriminator,
    hinge_d_loss,
    vanilla_d_loss,
    vanilla_g_loss,
)
from stage1 import UAE
from stage1.encoders import ARCHS

##### general utils
from utils import wandb_utils
from utils.model_utils import instantiate_from_config
from utils.train_utils import *
from utils.optim_utils import *
from utils.resume_utils import *
from utils.io_utils import safe_torch_save
from utils.wandb_utils import *
from utils.dist_utils import *
from PIL import Image
import numpy as np

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage-1 UAE with GAN and LPIPS losses.")
    parser.add_argument("--config", type=str, required=True, help="YAML config containing a stage_1 section.")
    parser.add_argument("--data-path", type=Path, required=True, help="Directory with ImageFolder structure.")
    parser.add_argument("--results-dir", type=str, default="ckpts", help="Directory to store training outputs.")
    parser.add_argument("--image-size", type=int, default=256, help="Image resolution (assumes square images).")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--global-seed", type=int, default=None, help="Override training.global_seed from the config.")    
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases for logging if set.')
    parser.add_argument("--compile", action="store_true", help="Use torch compile (for uae.encode, uae.forward).")
    return parser.parse_args()


def calculate_adaptive_weight(
    recon_loss: torch.Tensor,
    gan_loss: torch.Tensor,
    layer: torch.nn.Parameter,
    max_d_weight: float = 1e4,
) -> torch.Tensor:
    recon_grads = torch.autograd.grad(recon_loss, layer, retain_graph=True)[0]
    gan_grads = torch.autograd.grad(gan_loss, layer, retain_graph=True)[0]
    d_weight = torch.norm(recon_grads) / (torch.norm(gan_grads) + 1e-6)
    d_weight = torch.clamp(d_weight, 0.0, max_d_weight)
    return d_weight.detach()


_DCT_CACHE: dict[tuple[int, int, str, int, str], tuple[torch.Tensor, torch.Tensor]] = {}
_WEIGHT_CACHE: dict[tuple[Any, ...], torch.Tensor] = {}
_PERM_CACHE: dict[tuple[int, int, str, int, str], torch.Tensor] = {}


def _build_dct_matrix(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    k = torch.arange(n, device=device, dtype=dtype).view(n, 1)
    n_idx = torch.arange(n, device=device, dtype=dtype).view(1, n)
    mat = torch.cos((torch.pi / n) * (n_idx + 0.5) * k)
    mat[0] *= 1.0 / math.sqrt(n)
    if n > 1:
        mat[1:] *= math.sqrt(2.0 / n)
    return mat


def _get_dct_matrices(
    h: int,
    w: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (h, w, device.type, device.index or -1, str(dtype))
    cached = _DCT_CACHE.get(key)
    if cached is not None:
        return cached
    mat_h = _build_dct_matrix(h, device, dtype)
    mat_w = _build_dct_matrix(w, device, dtype)
    _DCT_CACHE[key] = (mat_h, mat_w)
    return mat_h, mat_w


def _dct2(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    mat_h, mat_w = _get_dct_matrices(h, w, x.device, x.dtype)
    x_flat = x.reshape(-1, h, w)
    x_flat = torch.matmul(mat_h, x_flat)
    x_flat = torch.matmul(x_flat, mat_w.t())
    return x_flat.reshape(b, c, h, w)


def _compute_band_edges(
    h: int,
    w: int,
    num_bands: int,
    schedule: Optional[list[float]],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    max_radius = float(math.sqrt((h - 1) ** 2 + (w - 1) ** 2))
    if schedule:
        schedule_tensor = torch.tensor(schedule[:num_bands], device=device, dtype=dtype)
        if schedule_tensor.numel() > 0:
            max_val = torch.clamp(schedule_tensor.max(), min=1e-6)
            edges = schedule_tensor / max_val * max_radius
            edges = torch.cat([torch.zeros(1, device=device, dtype=dtype), edges])
        else:
            edges = torch.linspace(0.0, max_radius, num_bands + 1, device=device, dtype=dtype)
    else:
        edges = torch.linspace(0.0, max_radius, num_bands + 1, device=device, dtype=dtype)
    edges[-1] = edges[-1] + torch.finfo(dtype).eps
    return edges


def _get_radius_grid(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    y_loc, x_loc = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.sqrt(y_loc.square() + x_loc.square())


def _build_zigzag_perm_top_left(h: int, w: int, device: torch.device) -> torch.Tensor:
    coords = []
    for s in range(h + w - 1):
        if s % 2 == 0:
            x = min(s, w - 1)
            y = s - x
            while x >= 0 and y < h:
                coords.append((y, x))
                x -= 1
                y += 1
        else:
            y = min(s, h - 1)
            x = s - y
            while y >= 0 and x < w:
                coords.append((y, x))
                y -= 1
                x += 1
    linear = [y * w + x for y, x in coords]
    if len(linear) != h * w:
        raise ValueError(f"Zigzag order produced {len(linear)} indices for {(h, w)}.")
    return torch.tensor(linear, device=device, dtype=torch.long)


def _get_freq_perm(h: int, w: int, device: torch.device, order: str) -> torch.Tensor:
    key = (h, w, device.type, device.index or -1, order)
    cached = _PERM_CACHE.get(key)
    if cached is not None:
        return cached
    if order == "zigzag":
        perm = _build_zigzag_perm_top_left(h, w, device)
    elif order == "radial":
        radius = _get_radius_grid(h, w, device, torch.float32).flatten()
        perm = torch.argsort(radius, dim=0)
    else:
        raise ValueError(f"Unsupported token order '{order}'. Use one of: zigzag, radial.")
    _PERM_CACHE[key] = perm
    return perm


def compute_band_reg_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    *,
    mode: str,
    freq_transform: str,
    soft_curve: str,
    soft_power: float,
    soft_mid: float,
    soft_min_weight: float,
    freq_num_bands: int,
    freq_schedule: Optional[list[float]],
    token_limit: Optional[int],
    token_order: str,
) -> torch.Tensor:
    if recon.shape != target.shape:
        raise ValueError(f"Band reg expects same shapes, got {tuple(recon.shape)} vs {tuple(target.shape)}.")
    if recon.ndim != 4:
        raise ValueError(f"Band reg expects image tensors (B,C,H,W), got {tuple(recon.shape)}.")
    if freq_transform == "dct":
        recon_freq = _dct2(recon)
        target_freq = _dct2(target)
        loss_map = (recon_freq - target_freq).pow(2)
    elif freq_transform == "fft":
        recon_freq = torch.fft.fftshift(torch.fft.fft2(recon, norm="ortho"), dim=(-2, -1))
        target_freq = torch.fft.fftshift(torch.fft.fft2(target, norm="ortho"), dim=(-2, -1))
        diff = recon_freq - target_freq
        loss_map = diff.real.pow(2) + diff.imag.pow(2)
    else:
        raise ValueError(f"Unsupported band_reg.freq_transform '{freq_transform}'.")

    b, c, h, w = loss_map.shape
    device = loss_map.device
    dtype = loss_map.dtype
    if token_limit is not None:
        k = max(0, int(token_limit))
        if k == 0:
            return loss_map.new_zeros(())
        perm = _get_freq_perm(h, w, device, token_order)
        k = min(k, h * w)
        idx = perm[:k]
        flat_loss = loss_map.view(b, c, h * w).index_select(-1, idx)
        return flat_loss.mean()

    radius_grid = _get_radius_grid(h, w, device, dtype)
    max_radius = radius_grid.max().clamp_min(1e-6)
    ratio = (radius_grid / max_radius).clamp(0.0, 1.0)

    if mode == "soft":
        cache_sig = (h, w, device, dtype, mode, soft_curve, soft_power, soft_mid, soft_min_weight)
        weight = _WEIGHT_CACHE.get(cache_sig)
        if weight is None:
            if soft_curve == "sigmoid":
                weight = torch.sigmoid((soft_mid - ratio) * soft_power)
            else:
                weight = (1.0 - ratio).clamp_min(0.0).pow(soft_power)
            if soft_min_weight > 0.0:
                weight = weight * (1.0 - soft_min_weight) + soft_min_weight
            weight = weight / weight.sum().clamp_min(1e-6)
            _WEIGHT_CACHE[cache_sig] = weight
        weight = weight.to(device=device, dtype=dtype)
        return (loss_map * weight).sum(dim=(-2, -1)).mean()

    if mode == "band0":
        edges = _compute_band_edges(h, w, freq_num_bands, freq_schedule, device, dtype)
        band_mask = ((radius_grid >= edges[0]) & (radius_grid < edges[1])).to(dtype)
        norm = band_mask.sum().clamp_min(1e-6)
        return (loss_map * band_mask).sum(dim=(-2, -1)).mean() / norm

    raise ValueError(f"Unsupported band_reg.mode '{mode}'. Supported: soft, band0.")


@torch.no_grad()
def encode_reference_pre_frequency(
    encoder: torch.nn.Module,
    images: torch.Tensor,
    *,
    input_size: int,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    x = images
    _, _, h, w = x.shape
    if h != input_size or w != input_size:
        x = F.interpolate(x, size=(input_size, input_size), mode="bicubic", align_corners=False)
    x = (x - mean.to(device=x.device, dtype=x.dtype)) / std.to(device=x.device, dtype=x.dtype)
    tokens = encoder(x)
    if isinstance(tokens, tuple):
        tokens = tokens[0]
    if tokens.ndim != 3:
        raise RuntimeError(f"Reference encoder must return token tensor [B,N,C], got {tuple(tokens.shape)}.")
    b, n, c = tokens.shape
    hw = int(math.sqrt(n))
    if hw * hw != n:
        raise RuntimeError(f"Reference encoder token count must be square, got N={n}.")
    return tokens.transpose(1, 2).contiguous().view(b, c, hw, hw)


def resize_target_like(target: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    if target.shape[-2:] == recon.shape[-2:]:
        return target
    return F.interpolate(target, size=recon.shape[-2:], mode="bilinear", align_corners=False)


def build_stage1_optimizer(
    decoder: torch.nn.Module,
    encoder: torch.nn.Module,
    training_cfg: Dict[str, Any],
    *,
    encoder_trainable: bool,
    encoder_lr_ratio: float,
) -> tuple[torch.optim.Optimizer, str]:
    opt_cfg: Dict[str, Any] = dict(training_cfg.get("optimizer", {}))
    opt_type = str(opt_cfg.get("type", "adamw")).lower()
    if opt_type != "adamw":
        raise ValueError(f"Unsupported optimizer '{opt_type}'. Only AdamW is currently available.")

    base_lr = float(opt_cfg.get("lr", training_cfg.get("base_lr", 2e-4)))
    betas_raw = opt_cfg.get("betas", opt_cfg.get("beta", (0.9, 0.95)))
    if isinstance(betas_raw, (list, tuple)):
        if len(betas_raw) != 2:
            raise ValueError(f"Expected optimizer.betas of length 2, got {len(betas_raw)}.")
        betas = (float(betas_raw[0]), float(betas_raw[1]))
    else:
        betas = (float(betas_raw), float(betas_raw))
    weight_decay = float(opt_cfg.get("weight_decay", opt_cfg.get("wd", 0.0)))
    eps = float(opt_cfg.get("eps", 1e-8))

    dec_params = [p for p in decoder.parameters() if p.requires_grad]
    if not dec_params:
        raise RuntimeError("No trainable decoder parameters found.")
    param_groups: list[dict[str, Any]] = [
        {"params": dec_params, "lr": base_lr, "lr_scale": 1.0, "name": "decoder"},
    ]
    message = (
        f"Optimizer: AdamW with decoder lr={base_lr}, betas={betas}, "
        f"weight_decay={weight_decay}, eps={eps}"
    )
    if encoder_trainable:
        enc_params = [p for p in encoder.parameters() if p.requires_grad]
        if not enc_params:
            raise RuntimeError("encoder_trainable=true but no trainable encoder parameters found.")
        enc_lr = base_lr * float(encoder_lr_ratio)
        param_groups.append(
            {"params": enc_params, "lr": enc_lr, "lr_scale": float(encoder_lr_ratio), "name": "encoder"}
        )
        message = (
            f"Optimizer: AdamW with decoder lr={base_lr}, encoder lr={enc_lr} "
            f"(ratio={encoder_lr_ratio}), betas={betas}, weight_decay={weight_decay}, eps={eps}"
        )

    supported_fused_devices = {"cuda", "xpu", "privateuseone"}
    use_fused = all(
        p.is_floating_point() and p.device.type in supported_fused_devices
        for group in param_groups
        for p in group["params"]
    )

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=base_lr,
        betas=betas,
        weight_decay=weight_decay,
        eps=eps,
        fused=use_fused,
    )
    training_cfg.setdefault("base_lr", base_lr)
    training_cfg.setdefault("final_lr", float(training_cfg.get("final_lr", base_lr)))
    return optimizer, message


def disable_unused_encoder_params(encoder: torch.nn.Module) -> list[str]:
    """
    Disable known-unused encoder parameters for this stage-1 objective.

    Dinov2's `mask_token` is only used for masked-image modeling inputs and is
    not part of the plain reconstruction forward used here. Keeping it trainable
    triggers DDP unused-parameter errors when encoder training is enabled.

    SigLIP2's `vision_model.head.*` branch is also unused when this project's
    encoder wrapper returns `last_hidden_state`.
    """
    disabled: list[str] = []
    encoder_name = encoder.__class__.__name__.lower()
    is_siglip2_encoder = encoder_name.startswith("siglip2")
    for name, param in encoder.named_parameters():
        should_disable = False
        if "mask_token" in name:
            should_disable = True
        elif is_siglip2_encoder and name.startswith("model.head."):
            should_disable = True
        if should_disable:
            if param.requires_grad:
                param.requires_grad_(False)
            disabled.append(name)
    return disabled


def ensure_lpips_vgg16_weights_available(local_rank: int, logger: logging.Logger) -> None:
    """
    Ensure torchvision VGG16 weights are ready before LPIPS init.

    We only allow one process per node (local_rank 0) to download, then use a
    distributed barrier so the remaining ranks never race the same URL.
    """
    hub_dir = Path(torch.hub.get_dir())
    checkpoint_dir = hub_dir / "checkpoints"
    checkpoint_path = checkpoint_dir / "vgg16-397923af.pth"
    if local_rank == 0:
        if checkpoint_path.exists():
            logger.info(f"LPIPS VGG16 checkpoint found at {checkpoint_path}.")
        else:
            logger.info(
                "LPIPS VGG16 checkpoint missing. Downloading once on local_rank 0 "
                f"to {checkpoint_path}."
            )
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            try:
                load_state_dict_from_url(
                    VGG16_Weights.IMAGENET1K_V1.url,
                    model_dir=str(checkpoint_dir),
                    progress=True,
                    check_hash=True,
                )
                logger.info(f"LPIPS VGG16 checkpoint downloaded to {checkpoint_path}.")
            except Exception as exc:
                raise RuntimeError(
                    "Failed to prepare LPIPS VGG16 checkpoint. "
                    f"Expected path: {checkpoint_path}. "
                    "Pre-download it or set TORCH_HOME to a shared cache."
                ) from exc
    if dist.is_available() and dist.is_initialized():
        dist.barrier()




def select_gan_losses(disc_kind: str, gen_kind: str):
    if disc_kind == "hinge":
        disc_loss_fn = hinge_d_loss
    elif disc_kind == "vanilla":
        disc_loss_fn = vanilla_d_loss
    else:
        raise ValueError(f"Unsupported discriminator loss '{disc_kind}'")

    if gen_kind == "vanilla":
        gen_loss_fn = vanilla_g_loss
    else:
        raise ValueError(f"Unsupported generator loss '{gen_kind}'")
    return disc_loss_fn, gen_loss_fn


def save_checkpoint(
    path: str,
    step: int,
    epoch: int,
    model: DDP,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
    disc: torch.nn.Module,
    disc_optimizer: torch.optim.Optimizer,
    disc_scheduler: Optional[LambdaLR],
    logger: Optional[logging.Logger] = None,
) -> bool:
    state = {
        "step": step,
        "epoch": epoch,
        "model": model.module.state_dict(),
        "ema": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "disc": disc.state_dict(),
        "disc_optimizer": disc_optimizer.state_dict(),
        "disc_scheduler": disc_scheduler.state_dict() if disc_scheduler is not None else None,
    }
    return safe_torch_save(
        state,
        path,
        logger=logger,
        context=f"stage1 checkpoint (epoch={epoch}, step={step})",
    )


def load_checkpoint(
    path: str,
    model: DDP,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
    disc: torch.nn.Module,
    disc_optimizer: torch.optim.Optimizer,
    disc_scheduler: Optional[LambdaLR],
) -> Tuple[int, int]:
    checkpoint = torch.load(path, map_location="cpu")
    model.module.load_state_dict(checkpoint["model"])
    ema_model.load_state_dict(checkpoint["ema"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    disc.load_state_dict(checkpoint["disc"])
    disc_optimizer.load_state_dict(checkpoint["disc_optimizer"])
    if disc_scheduler is not None and checkpoint.get("disc_scheduler") is not None:
        disc_scheduler.load_state_dict(checkpoint["disc_scheduler"])
    return checkpoint.get("epoch", 0), checkpoint.get("step", 0)

def main():
    args = parse_args()
    #### Dist Init
    rank, world_size, device = setup_distributed()
    
    #### Config init
    full_cfg = OmegaConf.load(args.config)
    (uae_config, *_) = parse_configs(full_cfg)
    if uae_config is None:
        raise ValueError("Config must define a top-level 'stage_1' section.")
    uae_params_section = uae_config.get("params", None)
    uae_params_cfg = OmegaConf.to_container(uae_params_section, resolve=True) if uae_params_section is not None else {}
    uae_params_cfg = dict(uae_params_cfg) if isinstance(uae_params_cfg, dict) else {}
    freq_cfg = dict(uae_params_cfg.get("frequency_config", {})) if isinstance(uae_params_cfg.get("frequency_config", {}), dict) else {}
    training_section = full_cfg.get("training", None)
    training_cfg = OmegaConf.to_container(training_section, resolve=True) if training_section is not None else {}
    training_cfg = dict(training_cfg) if isinstance(training_cfg, dict) else {}

    gan_section = full_cfg.get("gan", None)
    gan_cfg = OmegaConf.to_container(gan_section, resolve=True) if gan_section is not None else {}
    if not gan_cfg:
        raise ValueError("Config must define a top-level 'gan' section for stage-1 training.")
    disc_cfg = gan_cfg.get("disc", {})
    if not disc_cfg:
        raise ValueError("gan.disc configuration is required for stage-1 training.")
    loss_cfg = gan_cfg.get("loss", {})
    perceptual_weight = float(loss_cfg.get("perceptual_weight", 0.0))
    disc_weight = float(loss_cfg.get("disc_weight", 0.0))
    gan_start_epoch = int(loss_cfg.get("disc_start", 0))
    disc_update_epoch = int(loss_cfg.get("disc_upd_start", gan_start_epoch))
    lpips_start_epoch = int(loss_cfg.get("lpips_start", 0))
    
    disc_updates = int(loss_cfg.get("disc_updates", 1))
    max_d_weight = float(loss_cfg.get("max_d_weight", 1e4))
    disc_loss_type = loss_cfg.get("disc_loss", "hinge")
    gen_loss_type = loss_cfg.get("gen_loss", "vanilla")
    band_reg_cfg = dict(training_cfg.get("band_reg", {}))
    band_reg_enabled = bool(band_reg_cfg.get("enabled", False))
    band_reg_weight = float(band_reg_cfg.get("weight", 0.0)) if band_reg_enabled else 0.0
    band_reg_mode = str(band_reg_cfg.get("mode", "band0")).lower()
    band_reg_freq_transform = str(band_reg_cfg.get("freq_transform", freq_cfg.get("freq_transform", "dct"))).lower()
    soft_curve = str(band_reg_cfg.get("soft_curve", "sigmoid")).lower()
    if soft_curve == "linear":
        soft_curve = "power"
    soft_power = float(band_reg_cfg.get("soft_power", 1.0))
    soft_mid = float(band_reg_cfg.get("soft_mid", 0.2))
    soft_min_weight = float(band_reg_cfg.get("soft_min_weight", 0.0))
    first_tokens_src = band_reg_cfg.get("first_tokens", band_reg_cfg.get("first_k_tokens", band_reg_cfg.get("token_count", None)))
    band_reg_first_tokens = int(first_tokens_src) if first_tokens_src is not None else None
    band_reg_token_order = str(band_reg_cfg.get("token_order", freq_cfg.get("freq_order", "zigzag"))).lower()
    band_reg_num_bands = int(band_reg_cfg.get("freq_num_bands", freq_cfg.get("freq_num_bands", 2)))
    schedule_src = band_reg_cfg.get("freq_schedule", band_reg_cfg.get("v_patch_nums", freq_cfg.get("v_patch_nums", None)))
    band_reg_schedule = [float(v) for v in schedule_src] if schedule_src is not None else None
    encoder_params_cfg = dict(uae_params_cfg.get("encoder_params", {})) if isinstance(uae_params_cfg.get("encoder_params", {}), dict) else {}
    band_reg_reference_encoder_cls = str(band_reg_cfg.get("reference_encoder_cls", uae_params_cfg.get("encoder_cls", "Dinov2withNorm")))
    band_reg_reference_model = str(
        band_reg_cfg.get(
            "reference_model",
            encoder_params_cfg.get("dinov2_path", uae_params_cfg.get("encoder_config_path", "facebook/dinov2-with-registers-base")),
        )
    )
    band_reg_reference_normalize = bool(band_reg_cfg.get("reference_normalize", encoder_params_cfg.get("normalize", True)))
    if band_reg_enabled and band_reg_mode not in {"band0", "soft"}:
        raise ValueError("Unsupported training.band_reg.mode. Use one of: band0, soft.")
    if band_reg_enabled and band_reg_freq_transform not in {"dct", "fft"}:
        raise ValueError("Unsupported training.band_reg.freq_transform. Use one of: dct, fft.")
    if band_reg_enabled and band_reg_mode == "soft" and soft_curve not in {"sigmoid", "power"}:
        raise ValueError("training.band_reg.soft_curve must be one of: sigmoid, power.")
    if band_reg_enabled and soft_power <= 0.0:
        raise ValueError("training.band_reg.soft_power must be > 0.")
    if band_reg_enabled and band_reg_mode == "soft" and soft_curve == "sigmoid" and not (0.0 <= soft_mid <= 1.0):
        raise ValueError("training.band_reg.soft_mid must be between 0 and 1 for sigmoid curve.")
    if band_reg_enabled and not (0.0 <= soft_min_weight <= 1.0):
        raise ValueError("training.band_reg.soft_min_weight must be between 0 and 1.")
    if band_reg_enabled and band_reg_first_tokens is not None and band_reg_first_tokens <= 0:
        raise ValueError("training.band_reg.first_tokens must be > 0 when provided.")
    if band_reg_enabled and band_reg_token_order not in {"zigzag", "radial"}:
        raise ValueError("training.band_reg.token_order must be one of: zigzag, radial.")
    clean_recon_cfg = dict(training_cfg.get("clean_recon", {}))
    clean_recon_enabled = bool(clean_recon_cfg.get("enabled", False))
    clean_recon_l1_weight = float(clean_recon_cfg.get("l1_weight", 1.0))
    clean_recon_lpips_weight = float(clean_recon_cfg.get("lpips_weight", 1.0))
    if clean_recon_enabled and clean_recon_l1_weight < 0.0:
        raise ValueError("training.clean_recon.l1_weight must be >= 0.")
    if clean_recon_enabled and clean_recon_lpips_weight < 0.0:
        raise ValueError("training.clean_recon.lpips_weight must be >= 0.")
    skip_gan_on_small_recon = bool(training_cfg.get("skip_gan_on_small_recon", False))
    encoder_trainable = bool(training_cfg.get("encoder_trainable", False))
    encoder_lr_ratio = float(training_cfg.get("encoder_lr_ratio", training_cfg.get("optimizer", {}).get("encoder_lr_ratio", 0.1)))
    if encoder_trainable and encoder_lr_ratio <= 0.0:
        raise ValueError("training.encoder_lr_ratio must be > 0 when training.encoder_trainable=true.")
    batch_size = int(training_cfg.get("batch_size", 16))
    global_batch_size = training_cfg.get("global_batch_size", None) # optional global batch size for override
    if global_batch_size is not None:
        global_batch_size = int(global_batch_size)
        assert global_batch_size % world_size == 0, "global_batch_size must be divisible by world_size"
        batch_size = global_batch_size // world_size
    else:
        global_batch_size = batch_size * world_size
    num_workers = int(training_cfg.get("num_workers", 4))
    clip_grad_val = training_cfg.get("clip_grad", 1.0)
    clip_grad = float(clip_grad_val) if clip_grad_val is not None else None
    if clip_grad is not None and clip_grad <= 0:
        clip_grad = None
    log_interval = int(training_cfg.get("log_interval", 100))
    sample_every = int(training_cfg.get("sample_every", 1250)) 
    checkpoint_interval = int(training_cfg.get("checkpoint_interval", 4)) # ckpt interval is epoch based
    ema_decay = float(training_cfg.get("ema_decay", 0.9999))
    num_epochs = int(training_cfg.get("epochs", 200))
    default_seed = int(training_cfg.get("global_seed", 0))
    eval_section = full_cfg.get("eval", None)
    
    if eval_section:
        do_eval = True
        eval_interval = int(eval_section.get("eval_interval", 5000))
        eval_model = eval_section.get("eval_model", False) # by default eval ema. This decides whether to **additionally** eval the non-ema model.
        eval_metrics = eval_section.get("metrics", ("rfid", "psnr", "ssim")) # by default eval all
        eval_data = eval_section.get("data_path", None)
        reference_npz_path = eval_section.get("reference_npz_path", None)
        reference_npz_key = eval_section.get("reference_npz_key", None)
        assert eval_data, "eval.data_path must be specified to enable evaluation."
        assert reference_npz_path, "eval.reference_npz_path must be specified to enable evaluation."
        assert len(eval_metrics) > 0, "eval.metrics must contain at least one metric to compute."
    else:
        do_eval = False
        reference_npz_key = None
    global_seed = args.global_seed if args.global_seed is not None else default_seed
    seed = global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    experiment_dir, checkpoint_dir, logger = configure_experiment_dirs(args, rank)
    # update args as a dict to full_cfg
    full_cfg.cmd_args = vars(args)
    full_cfg.experiment_dir = experiment_dir
    full_cfg.checkpoint_dir = checkpoint_dir
    
    
    #### Model init
    uae: UAE = instantiate_from_config(uae_config).to(device)
    if args.compile:
        uae.encode = torch.compile(uae.encode)
        uae.forward = torch.compile(uae.forward)
    uae.decoder.train()
    if encoder_trainable:
        uae.encoder.train()
    else:
        uae.encoder.eval()
    ema_model = deepcopy(uae).to(device).eval()
    ema_model.requires_grad_(False)
    # decoder is always trainable; encoder can be optionally trained with a smaller lr.
    uae.encoder.requires_grad_(encoder_trainable)
    disabled_encoder_params = disable_unused_encoder_params(uae.encoder)
    uae.decoder.requires_grad_(True)
    ddp_model = DDP(uae, device_ids=[device.index], broadcast_buffers=False, find_unused_parameters=False)  # type: ignore[arg-type]
    uae = ddp_model.module
    clean_recon_active = clean_recon_enabled and hasattr(uae, "noise_tau") and float(getattr(uae, "noise_tau", 0.0)) > 0.0
    decoder = uae.decoder
    encoder = uae.encoder
    band_reference_encoder: Optional[torch.nn.Module] = None
    if band_reg_enabled and band_reg_weight > 0.0:
        ref_encoder_ctor = ARCHS.get(band_reg_reference_encoder_cls)
        if ref_encoder_ctor is None:
            raise ValueError(f"Unknown band_reg.reference_encoder_cls '{band_reg_reference_encoder_cls}'.")
        ref_encoder_params = dict(encoder_params_cfg)
        if "dinov2_path" in ref_encoder_params:
            ref_encoder_params["dinov2_path"] = band_reg_reference_model
        if "normalize" in ref_encoder_params:
            ref_encoder_params["normalize"] = band_reg_reference_normalize
        band_reference_encoder = ref_encoder_ctor(**ref_encoder_params).to(device).eval()
        band_reference_encoder.requires_grad_(False)
    discriminator, disc_aug = build_discriminator(disc_cfg, device)
    ddp_disc = DDP(discriminator, device_ids=[device.index], broadcast_buffers=False, find_unused_parameters=False)  # type: ignore[arg-type]
    discriminator = ddp_disc.module
    disc_scheduler: LambdaLR | None = None
    disc_sched_msg: Optional[str] = None

    discriminator.train()
    disc_loss_fn, gen_loss_fn = select_gan_losses(disc_loss_type, gen_loss_type)

    local_rank = int(os.environ.get("LOCAL_RANK", device.index if device.index is not None else 0))
    ensure_lpips_vgg16_weights_available(local_rank, logger)
    
    lpips = LPIPS().to(device)
    lpips.eval()
    
    #### Opt, Schedl init
    optimizer, optim_msg = build_stage1_optimizer(
        decoder,
        encoder,
        training_cfg,
        encoder_trainable=encoder_trainable,
        encoder_lr_ratio=encoder_lr_ratio,
    )
    disc_params = [p for p in discriminator.parameters() if p.requires_grad]
    disc_optimizer, disc_optim_msg = build_optimizer(disc_params, disc_cfg)
    
    #### AMP init
    scaler, autocast_kwargs = get_autocast_scaler(args)
    
    
    #### Data init
    first_crop_size = 384 if args.image_size == 256 else int(args.image_size * 1.5)
    stage1_transform = transforms.Compose(
        [
            transforms.Resize(first_crop_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomCrop(args.image_size),
            transforms.ToTensor(),
        ]
    )    
    loader, sampler = prepare_dataloader(
        args.data_path, batch_size, num_workers, rank, world_size, transform=stage1_transform
    )
    if do_eval:
        eval_dataset = ImageFolder(
            str(eval_data),
            transform=transforms.Compose([
                transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
                transforms.ToTensor(),
            ])
        )
        logger.info(f"Evaluation dataset loaded from {eval_data}, containing {len(eval_dataset)} images.")
    
    steps_per_epoch = len(loader)
    if steps_per_epoch == 0:
        raise RuntimeError("Dataloader returned zero batches. Check dataset and batch size settings.")
    
    # Schedl init after knowing dataset length
    scheduler: LambdaLR | None = None
    sched_msg: Optional[str] = None
    if training_cfg.get("scheduler"):
        scheduler, sched_msg = build_scheduler(optimizer, steps_per_epoch, training_cfg)
    if disc_cfg.get("scheduler"):
        disc_scheduler, disc_sched_msg = build_scheduler(disc_optimizer, steps_per_epoch, disc_cfg)
    
    ### Resuming and checkpointing
    start_epoch = 0
    global_step = 0
    maybe_resume_ckpt_path = find_resume_checkpoint(experiment_dir)
    if maybe_resume_ckpt_path is not None:
        logger.info(f"Experiment resume checkpoint found at {maybe_resume_ckpt_path}, automatically resuming...")
        ckpt_path = Path(maybe_resume_ckpt_path)
        if ckpt_path.is_file():
            start_epoch, global_step = load_checkpoint(
                ckpt_path,
                ddp_model,
                ema_model,
                optimizer,
                scheduler,
                discriminator,
                disc_optimizer,
                disc_scheduler,
            )
            logger.info(f"[Rank {rank}] Resumed from {ckpt_path} (epoch={start_epoch}, step={global_step}).")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    else:
        # starting from fresh, save worktree and configs
        if rank == 0:
            saved = save_worktree(experiment_dir, full_cfg)
            if saved:
                logger.info(f"Saved training worktree and config to {experiment_dir}.")
            else:
                logger.warning(
                    "Skipped saving training worktree/config snapshot due to I/O constraints."
                )
    
    ### Logging experiment details
    if rank == 0:
        num_params = sum(p.numel() for p in ddp_model.parameters() if p.requires_grad)
        logger.info(f"Stage-1 UAE trainable parameters: {num_params/1e6:.2f}M")
        logger.info(f"Discriminator architecture:\n{discriminator}")
        num_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
        logger.info(f"Discriminator trainable parameters: {num_params/1e6:.2f}M")
        logger.info(f"Using {disc_loss_type} discriminator loss and {gen_loss_type} generator loss.")
        logger.info(f"Perceptual (LPIPS) weight: {perceptual_weight:.6f}, GAN weight: {disc_weight:.6f}")
        if encoder_trainable:
            logger.info(f"Encoder is trainable with lr ratio {encoder_lr_ratio:.4f} (encoder lr = ratio * decoder lr).")
            if disabled_encoder_params:
                logger.info(
                    "Disabled %d known-unused encoder parameter(s) to keep DDP stable: %s",
                    len(disabled_encoder_params),
                    ", ".join(disabled_encoder_params),
                )
        else:
            logger.info("Encoder is frozen (decoder-only training).")
        if band_reg_enabled and band_reg_weight > 0.0:
            logger.info(
                "Band regularization enabled: weight=%.6f, mode=%s, transform=%s, first_tokens=%s, token_order=%s, reference=%s (%s, normalize=%s)",
                band_reg_weight,
                band_reg_mode,
                band_reg_freq_transform,
                str(band_reg_first_tokens),
                band_reg_token_order,
                band_reg_reference_model,
                band_reg_reference_encoder_cls,
                str(band_reg_reference_normalize),
            )
        else:
            logger.info("Band regularization disabled.")
        if clean_recon_enabled:
            if clean_recon_active:
                logger.info(
                    "Clean reconstruction enabled: l1_weight=%.6f, lpips_weight=%.6f",
                    clean_recon_l1_weight,
                    clean_recon_lpips_weight,
                )
            else:
                logger.info(
                    "Clean reconstruction requested but inactive because model noise is disabled (noise_tau <= 0)."
                )
        logger.info(f"GAN training starts at epoch {gan_start_epoch}, discriminator updates start at epoch {disc_update_epoch}, LPIPS loss starts at epoch {lpips_start_epoch}.")
        if disc_aug is not None:
            logger.info(f"Using DiffAug with policies: {disc_aug}")
        else:
            logger.info("Not using DiffAug.")
        if clip_grad is not None:
            logger.info(f"Clipping gradients to max norm {clip_grad}.")
        else:
            logger.info("Not clipping gradients.")
        # print optim and schel
        logger.info(optim_msg)
        print(sched_msg if sched_msg else "No LR scheduler for generator.")
        logger.info(disc_optim_msg)
        print(disc_sched_msg if disc_sched_msg else "No LR scheduler for discriminator.")
        logger.info(f"Training for {num_epochs} epochs, batch size {batch_size} per GPU.")
        logger.info(f"Dataset contains {len(loader.dataset)} samples, {steps_per_epoch} steps per epoch.")
        logger.info(f"Running with world size {world_size}, starting from epoch {start_epoch} to {num_epochs}.")


    last_layer = decoder.decoder_pred.weight
    gan_start_step = gan_start_epoch * steps_per_epoch
    disc_update_step = disc_update_epoch * steps_per_epoch
    lpips_start_step = lpips_start_epoch * steps_per_epoch
    dist.barrier()
    for epoch in range(start_epoch, num_epochs):
        ddp_model.train()
        sampler.set_epoch(epoch)
        epoch_metrics: Dict[str, torch.Tensor] = defaultdict(lambda: torch.zeros(1, device=device))
        num_batches = 0
        if checkpoint_interval > 0 and epoch % checkpoint_interval == 0  and rank == 0:
            logger.info(f"Saving checkpoint at epoch {epoch}...")
            ckpt_path = f"{checkpoint_dir}/ep-{epoch:07d}.pt" 
            save_checkpoint(
                ckpt_path,
                global_step,
                epoch,
                ddp_model,
                ema_model,
                optimizer,
                scheduler,
                discriminator,
                disc_optimizer,
                disc_scheduler,
                logger=logger,
            )
        for step, (images, _) in enumerate(loader):
            use_gan_base = global_step >= gan_start_step and disc_weight > 0.0
            train_disc_base = global_step >= disc_update_step and disc_weight > 0.0
            use_lpips = global_step >= lpips_start_step and perceptual_weight > 0.0
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            discriminator.eval()
            model_core = ddp_model.module
            pre_frequency_latent: Optional[torch.Tensor] = None
            band_reg_pre_frequency_latent: Optional[torch.Tensor] = None
            need_clean_band_reg_latent = (
                band_reg_enabled
                and band_reg_weight > 0.0
                and hasattr(model_core, "noise_tau")
                and float(getattr(model_core, "noise_tau", 0.0)) > 0.0
            )
            with autocast(**autocast_kwargs):
                if band_reg_enabled and band_reg_weight > 0.0:
                    recon, pre_frequency_latent, _ = ddp_model(images, return_pre_frequency=True)  # keep gradient synced
                else:
                    recon = ddp_model(images)  # keep gradient synced
                skip_gan_for_map2d = (
                    skip_gan_on_small_recon
                    and str(getattr(model_core, "band_split_cutoff_mode", "")).lower() == "map2d_square"
                )
                small_recon = skip_gan_for_map2d and (
                    recon.shape[-2] < images.shape[-2] or recon.shape[-1] < images.shape[-1]
                )
                use_gan = use_gan_base and not small_recon
                target_for_recon = resize_target_like(images, recon)
                recon_normed = recon * 2.0 - 1.0
                target_for_recon_normed = target_for_recon * 2.0 - 1.0
                rec_loss = (recon - target_for_recon).abs().mean() # L1
                if use_lpips:
                    lpips_loss = lpips(target_for_recon_normed, recon_normed)
                else:
                    lpips_loss = rec_loss.new_zeros(())
                clean_rec_loss = rec_loss.new_zeros(())
                clean_lpips_loss = rec_loss.new_zeros(())
                band_reg_loss = rec_loss.new_zeros(())
                recon_total = rec_loss + perceptual_weight * lpips_loss
                if clean_recon_active:
                    prev_noise_tau = float(model_core.noise_tau)
                    model_core.noise_tau = 0.0
                    try:
                        if need_clean_band_reg_latent:
                            clean_recon, band_reg_pre_frequency_latent, _ = ddp_model(
                                images,
                                apply_band_mask=False,
                                return_pre_frequency=True,
                            )
                        else:
                            clean_recon = ddp_model(images, apply_band_mask=False)
                    finally:
                        model_core.noise_tau = prev_noise_tau
                    clean_target = resize_target_like(images, clean_recon)
                    if clean_recon_l1_weight > 0.0:
                        clean_rec_loss = (clean_recon - clean_target).abs().mean()
                        recon_total = recon_total + clean_recon_l1_weight * clean_rec_loss
                    if clean_recon_lpips_weight > 0.0 and use_lpips:
                        clean_target_normed = clean_target * 2.0 - 1.0
                        clean_recon_normed = clean_recon * 2.0 - 1.0
                        clean_lpips_loss = lpips(clean_target_normed, clean_recon_normed)
                        recon_total = recon_total + clean_recon_lpips_weight * clean_lpips_loss
                elif need_clean_band_reg_latent:
                    prev_noise_tau = float(model_core.noise_tau)
                    model_core.noise_tau = 0.0
                    try:
                        _, band_reg_pre_frequency_latent, _ = ddp_model(
                            images,
                            apply_band_mask=False,
                            return_pre_frequency=True,
                        )
                    finally:
                        model_core.noise_tau = prev_noise_tau
                if use_gan:
                    fake_aug = disc_aug.aug(recon_normed)
                    logits_fake, _ = ddp_disc(fake_aug, None)
                    gan_loss = gen_loss_fn(logits_fake)
                else:
                    gan_loss = torch.zeros_like(recon_total)
            if (
                band_reg_enabled
                and band_reg_weight > 0.0
                and pre_frequency_latent is not None
                and band_reference_encoder is not None
            ):
                with torch.no_grad():
                    ref_pre_frequency = encode_reference_pre_frequency(
                        band_reference_encoder,
                        images.float(),
                        input_size=int(model_core.encoder_input_size),
                        mean=model_core.encoder_mean,
                        std=model_core.encoder_std,
                    )
                if ref_pre_frequency.shape[1] != pre_frequency_latent.shape[1]:
                    raise RuntimeError("Band reference latent channel mismatch.")
                band_reg_recon = (
                    band_reg_pre_frequency_latent
                    if band_reg_pre_frequency_latent is not None
                    else pre_frequency_latent
                )
                band_reg_loss = compute_band_reg_loss(
                    band_reg_recon.float(),
                    ref_pre_frequency.float(),
                    mode=band_reg_mode,
                    freq_transform=band_reg_freq_transform,
                    soft_curve=soft_curve,
                    soft_power=soft_power,
                    soft_mid=soft_mid,
                    soft_min_weight=soft_min_weight,
                    freq_num_bands=band_reg_num_bands,
                    freq_schedule=band_reg_schedule,
                    token_limit=band_reg_first_tokens,
                    token_order=band_reg_token_order,
                ).to(rec_loss.dtype)
                recon_total = recon_total + band_reg_weight * band_reg_loss
            # Calculate adaptive weight outside autocast (autograd operation, not forward pass)
            if use_gan:
                adaptive_weight = calculate_adaptive_weight(
                    recon_total, gan_loss, last_layer, max_d_weight
                )
                total_loss = recon_total + disc_weight * adaptive_weight * gan_loss
            else:
                adaptive_weight = torch.zeros_like(recon_total)
                total_loss = recon_total
            total_loss.float()
            if scaler:
                scaler.scale(total_loss).backward()
                if clip_grad is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), clip_grad)
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            update_ema(ema_model, ddp_model.module, ema_decay)

            disc_metrics: Dict[str, torch.Tensor] = {}
            train_disc = train_disc_base and not (skip_gan_on_small_recon and small_recon)
            if train_disc:
                # Set model to eval mode and get fresh reconstruction with updated weights
                ddp_model.eval()
                ddp_disc.train()
                for _ in range(disc_updates):
                    disc_optimizer.zero_grad(set_to_none=True)
                    with autocast(**autocast_kwargs):
                        # Fresh forward pass with updated model weights (no gradient)
                        with torch.no_grad():
                            if small_recon:
                                recon_disc = ddp_model(images, apply_band_mask=False)
                            else:
                                recon_disc = ddp_model(images)
                            real_target_for_disc = resize_target_like(images, recon_disc)
                            real_normed_for_disc = real_target_for_disc * 2.0 - 1.0
                            recon_disc_normed = recon_disc * 2.0 - 1.0
                        # discretize
                        fake_detached = recon_disc_normed.clamp(-1.0, 1.0)
                        fake_detached = torch.round((fake_detached + 1.0) * 127.5) / 127.5 - 1.0
                        fake_input = disc_aug.aug(fake_detached)
                        real_input = disc_aug.aug(real_normed_for_disc)
                        logits_fake, logits_real = discriminator(fake_input, real_input)
                        d_loss = disc_loss_fn(logits_real, logits_fake)
                        accuracy = (logits_real > logits_fake).float().mean() # accuracy of the discriminator
                    d_loss.float()
                    if scaler:
                        scaler.scale(d_loss).backward()
                        scaler.step(disc_optimizer)
                        scaler.update()
                    else:
                        d_loss.backward()
                        disc_optimizer.step()
                    disc_metrics = {
                        "disc_loss": d_loss.detach(),
                        "logits_real": logits_real.detach().mean(),
                        "logits_fake": logits_fake.detach().mean(),
                        "disc_accuracy": accuracy.detach(),
                    }
                    epoch_metrics["disc_loss"] += d_loss.detach()
                    epoch_metrics["disc_accuracy"] += accuracy.detach()
                    if disc_scheduler is not None:
                        disc_scheduler.step()
                ddp_disc.eval()
                # Set model back to train mode
                ddp_model.train()

            epoch_metrics["recon"] += rec_loss.detach()
            epoch_metrics["lpips"] += lpips_loss.detach()
            epoch_metrics["clean_recon"] += clean_rec_loss.detach()
            epoch_metrics["clean_lpips"] += clean_lpips_loss.detach()
            epoch_metrics["band_reg"] += band_reg_loss.detach()
            epoch_metrics["gan"] += gan_loss.detach()
            epoch_metrics["total"] += total_loss.detach()
            num_batches += 1

            if log_interval > 0 and global_step % log_interval == 0 and rank == 0:
                stats = {
                    "loss/total": total_loss.detach().item(),
                    "loss/recon": rec_loss.detach().item(),
                    "loss/lpips": lpips_loss.detach().item(),
                    "loss/clean_recon": clean_rec_loss.detach().item(),
                    "loss/clean_lpips": clean_lpips_loss.detach().item(),
                    "loss/band_reg": band_reg_loss.detach().item(),
                    "loss/gan": gan_loss.detach().item(),
                    "lr/generator": optimizer.param_groups[0]["lr"],
                }
                if disc_metrics:
                    stats.update(
                        {
                            "loss/disc": disc_metrics["disc_loss"].item(),
                            "disc/logits_real": disc_metrics["logits_real"].item(),
                            "disc/logits_fake": disc_metrics["logits_fake"].item(),
                            "lr/discriminator": disc_optimizer.param_groups[0]["lr"],
                            "disc/accuracy": disc_metrics["disc_accuracy"].item(),
                            "disc/weight": adaptive_weight.item(),
                        }
                    )
                logger.info(
                    f"[Epoch {epoch} | Step {global_step}] "
                    + ", ".join(f"{k}: {v:.4f}" for k, v in stats.items())
                )
                if args.wandb:
                    wandb_utils.log(stats, step=global_step)
            if global_step % sample_every == 0:
                logger.info("Generating EMA samples...")
                with torch.no_grad():
                    # only keep first 4 sample
                    sample_images = images[:4]
                    samples = ema_model.decode(ema_model.encode(sample_images))
                    sample_targets = resize_target_like(sample_images, samples)
                    # also concat input and reconstruction
                    comparison = torch.cat([sample_targets, samples], dim=0).cpu().float()
                    # reshape to grid (sample at first row, recon at second row)
                    n = sample_images.size(0)
                    grid = make_grid(comparison, nrow=n)
                    if args.wandb:
                        wandb_utils.log_image(grid, step=global_step)
                logger.info("Generating EMA samples done.")
            if do_eval and (eval_interval > 0 and global_step % eval_interval == 0):
                logger.info("Starting evaluation...")
                eval_models = [(ema_model, "ema")]
                if eval_model:
                    eval_models.append((ddp_model.module, "model"))
                for eval_mod, mod_name in eval_models:
                    eval_stats = evaluate_reconstruction_distributed(
                        eval_mod,
                        eval_dataset,
                        len(eval_dataset),
                        rank = rank,
                        world_size = world_size,
                        device = device,
                        batch_size = batch_size,
                        metrics_to_compute = eval_metrics,
                        experiment_dir = experiment_dir,
                        global_step = global_step,
                        autocast_kwargs = autocast_kwargs,
                        reference_npz_path = reference_npz_path,
                        reference_npz_key = reference_npz_key,
                    )
                    # log with prefix
                    eval_stats = {f"eval_{mod_name}/{k}": v for k, v in eval_stats.items()} if eval_stats is not None else {}
                    if args.wandb:
                        wandb_utils.log(eval_stats, step=global_step)
                logger.info("Evaluation done.")
            global_step += 1
        if rank == 0 and num_batches > 0:
            avg_recon = (epoch_metrics["recon"] / num_batches).item()
            avg_lpips = (epoch_metrics["lpips"] / num_batches).item()
            avg_clean_recon = (epoch_metrics["clean_recon"] / num_batches).item()
            avg_clean_lpips = (epoch_metrics["clean_lpips"] / num_batches).item()
            avg_band_reg = (epoch_metrics["band_reg"] / num_batches).item()
            avg_gan = (epoch_metrics["gan"] / num_batches).item()
            avg_total = (epoch_metrics["total"] / num_batches).item()
            epoch_stats = {
                "epoch/loss_total": avg_total,
                "epoch/loss_recon": avg_recon,
                "epoch/loss_lpips": avg_lpips,
                "epoch/loss_clean_recon": avg_clean_recon,
                "epoch/loss_clean_lpips": avg_clean_lpips,
                "epoch/loss_band_reg": avg_band_reg,
                "epoch/loss_gan": avg_gan,
            }
            if disc_metrics:
                epoch_stats.update(
                    {
                        "epoch/loss_disc": (epoch_metrics["disc_loss"] / num_batches).item(),
                        "epoch/disc_logits_real": disc_metrics["logits_real"].item(),
                        "epoch/disc_logits_fake": disc_metrics["logits_fake"].item(),
                        "epoch/disc_accuracy": (epoch_metrics["disc_accuracy"] / num_batches).item(),
                    }
                )
            logger.info(
                f"[Epoch {epoch}] "
                + ", ".join(f"{k}: {v:.4f}" for k, v in epoch_stats.items())
            )
            if args.wandb:
                wandb_utils.log(epoch_stats, step=global_step)
    # save the final ckpt
    if rank == 0:
        logger.info(f"Saving final checkpoint at epoch {num_epochs}...")
        ckpt_path = f"{checkpoint_dir}/ep-last.pt" 
        save_checkpoint(
            ckpt_path,
            global_step,
            num_epochs,
            ddp_model,
            ema_model,
            optimizer,
            scheduler,
            discriminator,
            disc_optimizer,
            disc_scheduler,
            logger=logger,
        )
    dist.barrier()
    logger.info("Done!")
    cleanup_distributed()


if __name__ == "__main__":
    main()
