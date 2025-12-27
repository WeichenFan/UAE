#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import sys
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image
from torchvision.models import inception_v3, Inception_V3_Weights
from PIL import Image
from scipy import linalg

MODULE_DIR = Path(__file__).resolve().parent / "unified_ae"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from unified_ae import UnifiedUAE  
from unified_ae.band_reference import DinoBandReference  
from model_utils import instantiate_from_config_2  
from unified_ae.gan import (
    DiffAug,
    LPIPS,
    build_discriminator,
    hinge_d_loss,
    vanilla_d_loss,
    vanilla_g_loss,
)  
from unified_ae.optim import (  
    build_optimizer as build_cfg_optimizer,
    build_scheduler as build_cfg_scheduler,
)


def autocast_context(device: torch.device, enabled: bool, dtype: Optional[torch.dtype]):
    if device.type == "cuda" and enabled:
        return torch.cuda.amp.autocast(dtype=dtype)
    return nullcontext()


def disable_autocast(device: torch.device):
    if device.type == "cuda":
        return torch.cuda.amp.autocast(enabled=False)
    return nullcontext()


def _load_model_state_strict(model: nn.Module, state_dict: Dict[str, torch.Tensor], logger: Optional[logging.Logger] = None) -> None:
    """
    Load a checkpoint strictly while gracefully handling extra keys (e.g., dynamic band modules).
    """

    def attempt(sd: Dict[str, torch.Tensor]) -> bool:
        try:
            model.load_state_dict(sd, strict=True)
            return True
        except RuntimeError as err:
            attempt.last_error = err  
            return False

    attempt.last_error = RuntimeError("Unhandled state dict load failure.")  

    if attempt(state_dict):
        return

    if all(key.startswith("module.") for key in state_dict.keys()):
        stripped_state = {k[len("module.") :]: v for k, v in state_dict.items()}
        if attempt(stripped_state):
            return
        candidate = stripped_state
    else:
        candidate = state_dict

    model_keys = set(model.state_dict().keys())
    filtered = {k: v for k, v in candidate.items() if k in model_keys}
    dropped = [k for k in candidate.keys() if k not in model_keys]

    if filtered:
        if dropped and logger is not None:
            sample = ", ".join(dropped[:5])
            logger.warning(
                "Dropping %d unmatched checkpoint keys when loading model (e.g., %s).",
                len(dropped),
                sample,
            )
        model.load_state_dict(filtered, strict=True)
        return

    raise attempt.last_error


def center_crop_arr(pil_image, image_size: int):
    """
    Center cropping implementation matching the legacy Stage-1 pipeline.
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def build_transform(image_size: int, augment: bool) -> transforms.Compose:
    ops = [
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
    ]
    if augment:
        ops.insert(0, transforms.RandomHorizontalFlip())
    ops.append(transforms.ToTensor())
    return transforms.Compose(ops)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the unified autoencoder.")
    parser.add_argument("--config", type=Path, required=True, help="YAML config with a stage_1 section.")
    parser.add_argument(
        "--stage-key",
        type=str,
        default=None,
        help="If the config defines a 'stages' mapping, pick which stage to train.",
    )
    parser.add_argument("--data-path", type=Path, required=True, help="ImageFolder root for training data.")
    parser.add_argument("--results-dir", type=Path, default=Path("unified_results"), help="Where to store runs.")
    parser.add_argument("--image-size", type=int, default=256, help="Square image size.")
    parser.add_argument("--global-seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs from config.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size from config.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate from config.")
    parser.add_argument("--log-every", type=int, default=100, help="Log interval in steps.")
    parser.add_argument("--ckpt-every", type=int, default=5000, help="Checkpoint interval in steps.")
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="no")
    parser.add_argument("--resume-from", type=Path, default=None, help="Checkpoint to resume from.")
    parser.add_argument("--val-path", type=Path, default=None, help="Optional validation ImageFolder root.")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging (overrides config).")
    parser.add_argument("--wandb-project", type=str, default=None, help="W&B project name.")
    parser.add_argument("--wandb-name", type=str, default=None, help="W&B run name.")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity/organization.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging regardless of config.")
    return parser.parse_args()


def create_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "train.log"),
        ],
    )
    return logging.getLogger("unified-ae")


def compute_freq_ratio(mode: str, batch: int, cfg: dict, device: torch.device) -> torch.Tensor:
    mode = (mode or "full").lower()
    if mode == "full":
        return torch.ones(batch, device=device)
    if mode == "fixed":
        value = float(cfg.get("freq_ratio_value", 1.0))
        return torch.full((batch,), value, device=device)
    if mode == "random_uniform":
        low = float(min(cfg.get("freq_ratio_min", 0.0), cfg.get("freq_ratio_max", 1.0)))
        high = float(max(cfg.get("freq_ratio_min", 0.0), cfg.get("freq_ratio_max", 1.0)))
        return torch.rand(batch, device=device) * (high - low) + low
    if mode == "mixed":
        prob = float(cfg.get("freq_ratio_mix_prob", 0.5))
        low = float(min(cfg.get("freq_ratio_min", 0.0), cfg.get("freq_ratio_max", 1.0)))
        high = float(max(cfg.get("freq_ratio_min", 0.0), cfg.get("freq_ratio_max", 1.0)))
        keep_full = torch.rand(batch, device=device) >= prob
        sampled = torch.rand(batch, device=device) * (high - low) + low
        return torch.where(keep_full, torch.ones_like(sampled), sampled)
    raise ValueError(f"Unsupported freq_ratio_mode '{mode}'.")


def build_eval_loader(
    val_path: Optional[Path],
    image_size: int,
    batch_size: int,
    num_workers: int,
    center_crop: bool,
) -> Optional[DataLoader]:
    if val_path is None:
        return None
    if not val_path.exists():
        raise FileNotFoundError(f"Validation path does not exist: {val_path}")
    if center_crop:
        transform = transforms.Compose(
            [
                transforms.Lambda(lambda pil: center_crop_arr(pil, image_size)),
                transforms.ToTensor(),
            ]
        )
    else:
        transform = build_transform(image_size, augment=False)
    dataset = ImageFolder(str(val_path), transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return loader


class FeatureStats:
    def __init__(self, dim: int):
        self.dim = dim
        self.n = 0
        self.sum = np.zeros(dim, dtype=np.float64)
        self.sum_sq = np.zeros((dim, dim), dtype=np.float64)

    def update(self, feats: torch.Tensor) -> None:
        feats_np = feats.detach().cpu().numpy().astype(np.float64)
        if feats_np.size == 0:
            return
        self.n += feats_np.shape[0]
        self.sum += feats_np.sum(axis=0)
        self.sum_sq += feats_np.T @ feats_np

    def mean(self) -> np.ndarray:
        if self.n == 0:
            raise ValueError("No features accumulated for mean computation.")
        return self.sum / self.n

    def covariance(self) -> np.ndarray:
        if self.n == 0:
            raise ValueError("No features accumulated for covariance computation.")
        mu = self.mean()
        cov = self.sum_sq / self.n - np.outer(mu, mu)
        cov = (cov + cov.T) * 0.5
        return cov


class InceptionFeatureExtractor:
    def __init__(self, device: torch.device):
        weights = Inception_V3_Weights.IMAGENET1K_V1
        model = inception_v3(weights=weights, aux_logits=True, transform_input=False)
        if hasattr(model, "AuxLogits"):
            model.AuxLogits = None
        if hasattr(model, "aux_logits"):
            model.aux_logits = False
        model.fc = torch.nn.Identity()
        model.eval()
        self.model = model.to(device)
        self.device = device
        preprocess = weights.transforms()
        norm_mean = None
        norm_std = None
        if hasattr(preprocess, "transforms"):
            for t in preprocess.transforms:
                if isinstance(t, transforms.Normalize):
                    norm_mean = t.mean
                    norm_std = t.std
        if norm_mean is None or norm_std is None:
            norm_mean = (0.485, 0.456, 0.406)
            norm_std = (0.229, 0.224, 0.225)
        mean = torch.tensor(norm_mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        std = torch.tensor(norm_std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self.mean = mean
        self.std = std

    @torch.no_grad()
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        imgs = images.to(self.device, non_blocking=True).to(dtype=torch.float32)
        imgs = F.interpolate(imgs, size=(299, 299), mode="bilinear", align_corners=False)
        imgs = (imgs - self.mean) / self.std
        return self.model(imgs)


def compute_fid(
    mu_real: np.ndarray,
    sigma_real: np.ndarray,
    mu_fake: np.ndarray,
    sigma_fake: np.ndarray,
    eps: float = 1e-6,
) -> float:
    diff = mu_real - mu_fake
    cov_prod = sigma_real @ sigma_fake
    covmean, info = linalg.sqrtm(cov_prod, disp=False)
    if not np.isfinite(covmean).all() or info != 0:
        offset = np.eye(sigma_real.shape[0], dtype=np.float64) * eps
        covmean, _ = linalg.sqrtm((sigma_real + offset) @ (sigma_fake + offset), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma_real) + np.trace(sigma_fake) - 2.0 * np.trace(covmean)
    return float(max(fid, 0.0))


def select_gan_losses(disc_kind: str, gen_kind: str):
    if hinge_d_loss is None or vanilla_d_loss is None or vanilla_g_loss is None:
        raise RuntimeError("GAN loss functions are unavailable; check unified_ae.gan installation.")
    if disc_kind == "hinge":
        disc_loss_fn = hinge_d_loss
    elif disc_kind == "vanilla":
        disc_loss_fn = vanilla_d_loss
    else:
        raise ValueError(f"Unsupported discriminator loss '{disc_kind}'.")

    if gen_kind == "vanilla":
        gen_loss_fn = vanilla_g_loss
    else:
        raise ValueError(f"Unsupported generator loss '{gen_kind}'.")
    return disc_loss_fn, gen_loss_fn


def calculate_adaptive_weight(
    recon_loss: torch.Tensor,
    gan_loss: torch.Tensor,
    layer: torch.nn.Parameter,
    max_d_weight: float = 1e4,
) -> torch.Tensor:
    recon_grads = torch.autograd.grad(recon_loss, layer, retain_graph=True, allow_unused=True)[0]
    gan_grads = torch.autograd.grad(gan_loss, layer, retain_graph=True, allow_unused=True)[0]
    if recon_grads is None or gan_grads is None:
        return torch.tensor(1.0, device=layer.device)
    d_weight = torch.norm(recon_grads) / (torch.norm(gan_grads) + 1e-6)
    d_weight = torch.clamp(d_weight, 0.0, max_d_weight)
    return d_weight.detach()


@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, current_model: torch.nn.Module, decay: float) -> None:
    ema_params = dict(ema_model.named_parameters())
    model_params = dict(current_model.named_parameters())
    for name, param in model_params.items():
        if name in ema_params:
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


@torch.no_grad()
def evaluate_model(
    model: UnifiedUAE,
    accelerator: Accelerator,
    loader: Optional[DataLoader],
    training_cfg: Dict[str, object],
    logger: logging.Logger,
    global_step: int,
    experiment_dir: Optional[Path],
    wandb_run,
    autocast_enabled: bool,
    autocast_dtype: Optional[torch.dtype],
) -> None:
    if loader is None:
        return
    model.eval()
    device = accelerator.device
    base_model = getattr(model, "module", model)
    freq_ratio_mode = training_cfg.get("eval_freq_ratio_mode", training_cfg.get("freq_ratio_mode", "full"))
    max_batches = training_cfg.get("eval_max_batches")
    vis_samples = int(training_cfg.get("frequency_vis_samples", 4))
    compute_fid_flag = bool(training_cfg.get("eval_compute_fid", True))

    psnr_sum = torch.tensor(0.0, device=device)
    count_sum = torch.tensor(0.0, device=device)
    grid_tensor = None

    if compute_fid_flag:
        if not hasattr(evaluate_model, "_inception") or evaluate_model._inception is None:
            evaluate_model._inception = InceptionFeatureExtractor(device)
        inception = evaluate_model._inception
        stats_real = FeatureStats(2048)
        stats_fake = FeatureStats(2048)
        fid_max_images = training_cfg.get("eval_fid_max_images")
    else:
        inception = None
        stats_real = stats_fake = None
        fid_max_images = None

    for batch_idx, (images, _) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        apply_frequency = base_model.freq_modulator is not None
        freq_ratio_tensor: Optional[torch.Tensor] = None
        if apply_frequency:
            freq_ratio_tensor = compute_freq_ratio(
                freq_ratio_mode,
                images.size(0),
                training_cfg,
                device,
            ).clamp_(0.0, 1.0)
        with autocast_context(device, autocast_enabled, autocast_dtype):
            latent_output = base_model.encode(
                images,
                apply_frequency=apply_frequency,
                freq_ratio=freq_ratio_tensor,
            )
        latent = latent_output[1] if isinstance(latent_output, tuple) else latent_output
        with autocast_context(device, autocast_enabled, autocast_dtype):
            recon = base_model.decode(latent)
        recon = recon.clamp_(0.0, 1.0)
        images_f32 = images.float()
        recon_f32 = recon.float()
        mse = torch.mean((recon_f32 - images_f32) ** 2, dim=(1, 2, 3))
        psnr = 10.0 * torch.log10(1.0 / (mse + 1e-8))
        psnr_sum += psnr.sum()
        count_sum += torch.tensor(float(psnr.numel()), device=device)

        if inception is not None:
            feats_real = inception(images_f32)
            feats_fake = inception(recon_f32)
            stats_real.update(feats_real)
            stats_fake.update(feats_fake)

        if grid_tensor is None and accelerator.is_main_process:
            num_vis = min(vis_samples, images.size(0))
            vis = torch.cat([images_f32[:num_vis], recon_f32[:num_vis]], dim=0)
            grid_tensor = make_grid(vis, nrow=num_vis, padding=2)
        if max_batches is not None and batch_idx + 1 >= int(max_batches):
            break
        if fid_max_images is not None and stats_real is not None and stats_real.n >= int(fid_max_images):
            break

    total_psnr = accelerator.reduce(psnr_sum, reduction="sum").item()
    total_count = accelerator.reduce(count_sum, reduction="sum").item()
    mean_psnr = total_psnr / max(total_count, 1.0)

    fid_value = None
    if inception is not None and stats_real is not None and stats_real.n > 0 and stats_fake.n > 0:
        sum_real = torch.tensor(stats_real.sum, device=device)
        sum_fake = torch.tensor(stats_fake.sum, device=device)
        sum_sq_real = torch.tensor(stats_real.sum_sq, device=device)
        sum_sq_fake = torch.tensor(stats_fake.sum_sq, device=device)
        n_real = torch.tensor(float(stats_real.n), device=device)
        n_fake = torch.tensor(float(stats_fake.n), device=device)

        sum_real = accelerator.reduce(sum_real, reduction="sum").cpu().numpy()
        sum_fake = accelerator.reduce(sum_fake, reduction="sum").cpu().numpy()
        sum_sq_real = accelerator.reduce(sum_sq_real, reduction="sum").cpu().numpy()
        sum_sq_fake = accelerator.reduce(sum_sq_fake, reduction="sum").cpu().numpy()
        n_real = accelerator.reduce(n_real, reduction="sum").item()
        n_fake = accelerator.reduce(n_fake, reduction="sum").item()

        if accelerator.is_main_process and n_real > 0 and n_fake > 0:
            mean_real = sum_real / n_real
            mean_fake = sum_fake / n_fake
            cov_real = sum_sq_real / n_real - np.outer(mean_real, mean_real)
            cov_fake = sum_sq_fake / n_fake - np.outer(mean_fake, mean_fake)
            cov_real = (cov_real + cov_real.T) * 0.5
            cov_fake = (cov_fake + cov_fake.T) * 0.5
            fid_value = compute_fid(mean_real, cov_real, mean_fake, cov_fake)

    if accelerator.is_main_process:
        logger.info(f"Eval step {global_step}: PSNR={mean_psnr:.3f} dB")
        log_payload = {"eval/psnr": mean_psnr}
        if fid_value is not None:
            logger.info(f"Eval step {global_step}: FID={fid_value:.3f}")
            log_payload["eval/fid"] = fid_value
        if wandb_run is not None:
            wandb_run.log(log_payload, step=global_step)
        if grid_tensor is not None:
            grid_cpu = grid_tensor.detach().cpu()
            if experiment_dir is not None:
                save_dir = experiment_dir / "eval_samples"
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / f"recon_step_{global_step:07d}.png"
                save_image(grid_cpu, save_path)
                logger.info(f"Saved reconstruction grid to {save_path}")
            if wandb_run is not None:
                import wandb

                grid_np = grid_cpu.permute(1, 2, 0).numpy()
                wandb_run.log({"eval/recon": wandb.Image(grid_np)}, step=global_step)

    model.train()


evaluate_model._inception = None  


def main() -> None:
    args = parse_args()
    accelerator = Accelerator()
    set_seed(args.global_seed, device_specific=True)
    device = accelerator.device

    precision_mode = args.mixed_precision
    precision_warning = None
    if device.type != "cuda" and precision_mode != "no":
        precision_warning = "Mixed precision requested but CUDA unavailable; falling back to fp32."
        precision_mode = "no"
    if precision_mode == "fp16":
        scaler: Optional[GradScaler] = GradScaler()
        autocast_enabled = True
        autocast_dtype: Optional[torch.dtype] = torch.float16
    elif precision_mode == "bf16":
        scaler = None
        autocast_enabled = True
        autocast_dtype = torch.bfloat16
    else:
        scaler = None
        autocast_enabled = False
        autocast_dtype = None

    config = OmegaConf.load(args.config)
    selected_stage_key: Optional[str] = None
    stages_cfg = config.get("stages")
    if stages_cfg is not None:
        stage_keys = list(stages_cfg.keys())
        if not stage_keys:
            raise ValueError("Config includes a 'stages' section but it contains no entries.")
        selected_stage_key = args.stage_key or stage_keys[0]
        if selected_stage_key not in stages_cfg:
            available = ", ".join(stage_keys)
            raise ValueError(f"Stage '{selected_stage_key}' not found in config. Available stages: {available}")
        shared_cfg = config.get("shared")
        chosen_cfg = stages_cfg[selected_stage_key]
        if shared_cfg is not None:
            chosen_cfg = OmegaConf.merge(shared_cfg, chosen_cfg)
        config = OmegaConf.create(chosen_cfg)
        accelerator.print(f"Selected stage '{selected_stage_key}' from multi-stage config '{args.config}'.")
    elif args.stage_key:
        raise ValueError("--stage-key was provided but the config does not define a 'stages' mapping.")
    stage_cfg = config.get("stage_1")
    if stage_cfg is None:
        raise ValueError("Config must contain a 'stage_1' section describing the model.")
    model: UnifiedUAE = instantiate_from_config_2(stage_cfg)

    training_cfg_obj = config.get("training", {})
    if isinstance(training_cfg_obj, dict):
        training_cfg_dict = dict(training_cfg_obj)
    else:
        training_cfg_dict = dict(OmegaConf.to_container(training_cfg_obj, resolve=True))

    epochs = args.epochs or int(training_cfg_dict.get("epochs", 10))
    batch_size = args.batch_size or int(training_cfg_dict.get("batch_size", 32))
    augment = bool(training_cfg_dict.get("augment", False))
    freq_ratio_mode = training_cfg_dict.get("freq_ratio_mode", "full")

    opt_cfg = training_cfg_dict.setdefault("optimizer", {})
    if args.lr is not None:
        opt_cfg["lr"] = args.lr

    clip_grad_val = training_cfg_dict.get("clip_grad", 1.0)
    clip_grad = float(clip_grad_val) if clip_grad_val is not None else None
    if clip_grad is not None and clip_grad <= 0:
        clip_grad = None

    use_gan = bool(training_cfg_dict.get("use_gan", False))
    use_lpips = bool(training_cfg_dict.get("use_lpips", False))
    gan_on_corrupt_only = bool(training_cfg_dict.get("gan_on_corrupt_only", False))
    use_ema = bool(training_cfg_dict.get("use_ema", False))
    ema_decay = float(training_cfg_dict.get("ema_decay", 0.9999))

    dynamic_band_cfg = dict(training_cfg_dict.get("dynamic_band_weights", {}))
    dynamic_band_enabled = bool(dynamic_band_cfg.get("enabled", False))
    recon_min = float(dynamic_band_cfg.get("recon_min", 1.0))
    recon_full = float(dynamic_band_cfg.get("recon_full", 1.0))
    perceptual_min = float(dynamic_band_cfg.get("perceptual_min", 1.0))
    perceptual_full = float(dynamic_band_cfg.get("perceptual_full", 1.0))
    disc_min_scale = float(dynamic_band_cfg.get("disc_min", 1.0))
    disc_full_scale = float(dynamic_band_cfg.get("disc_full", 1.0))

    model = model.to(device)
    ema_model = None
    if use_ema:
        ema_model = deepcopy(model).to(device)
        ema_model.requires_grad_(False)

    transform = build_transform(args.image_size, augment=augment)
    dataset = ImageFolder(str(args.data_path), transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(training_cfg_dict.get("num_workers", 8)),
        pin_memory=True,
        drop_last=True,
    )
    steps_per_epoch = len(loader)
    if steps_per_epoch == 0:
        raise RuntimeError("Dataloader returned zero batches. Check dataset and batch size settings.")

    val_path_cfg = training_cfg_dict.get("val_path")
    val_path = args.val_path if args.val_path is not None else val_path_cfg
    if isinstance(val_path, str):
        val_path = Path(val_path)
    eval_loader = None
    if val_path is not None:
        eval_batch_size = int(training_cfg_dict.get("eval_batch_size", batch_size))
        eval_num_workers = int(training_cfg_dict.get("eval_num_workers", training_cfg_dict.get("num_workers", 8)))
        eval_center_crop = bool(training_cfg_dict.get("eval_center_crop", True))
        eval_loader = build_eval_loader(val_path, args.image_size, eval_batch_size, eval_num_workers, eval_center_crop)

    eval_interval = int(training_cfg_dict.get("eval_interval", 0))

    wandb_cfg = config.get("wandb")
    if not wandb_cfg:
        wandb_cfg = training_cfg_dict.get("wandb")
    if wandb_cfg:
        if not isinstance(wandb_cfg, dict):
            wandb_cfg = OmegaConf.to_container(wandb_cfg, resolve=True)
        wandb_cfg = dict(wandb_cfg)
    else:
        wandb_cfg = {}
    cli_wandb_settings: Dict[str, Any] = {}
    if args.no_wandb:
        cli_wandb_settings["enable"] = False
    else:
        if args.wandb or args.wandb_project or args.wandb_name or args.wandb_entity:
            cli_wandb_settings["enable"] = True
        if args.wandb_project:
            cli_wandb_settings["project"] = args.wandb_project
        if args.wandb_name:
            cli_wandb_settings["name"] = args.wandb_name
        if args.wandb_entity:
            cli_wandb_settings["entity"] = args.wandb_entity
    if cli_wandb_settings:
        wandb_cfg.update(cli_wandb_settings)
    if not wandb_cfg:
        wandb_cfg = None
    wandb_run = None

    band_reg_cfg = dict(training_cfg_dict.get("band_reg", {}))
    band_reg_enabled = bool(band_reg_cfg.get("enabled", False))
    band_reg_weight = float(band_reg_cfg.get("weight", 0.0)) if band_reg_enabled else 0.0
    band_reference = None
    if band_reg_enabled and band_reg_weight > 0.0:
        freq_mod = getattr(model, "freq_modulator", None)
        if freq_mod is None:
            raise ValueError("Band regularization requires a frequency modulator.")
        schedule = list(getattr(freq_mod, "v_patch_nums", []))
        if not schedule:
            raise ValueError("Frequency schedule is empty; cannot apply band regularization.")
        use_residual = bool(getattr(freq_mod, "use_residual_split", False))
        residual_projector = getattr(freq_mod, "residual_projector", "fft")
        residual_kwargs = dict(getattr(freq_mod, "residual_projector_kwargs", {}) or {})
        stage_params = stage_cfg.get("params", {})
        default_reference = stage_params.get("encoder_config_path", "facebook/dinov2-with-registers-base")
        reference_model_name = band_reg_cfg.get("reference_model", default_reference)
        encoder_params_cfg = stage_params.get("encoder_params", {})
        normalize_encoder = bool(encoder_params_cfg.get("normalize", True))
        band_reference = DinoBandReference(
            reference_model_name,
            device,
            model.encoder_input_size,
            schedule,
            use_residual,
            residual_projector,
            residual_projector_kwargs=residual_kwargs,
            normalize=normalize_encoder,
        )
        band_reference.eval()
    else:
        band_reg_enabled = False
        band_reg_weight = 0.0

    optimizer, optim_msg = build_cfg_optimizer(model.parameters(), training_cfg_dict)

    scheduler = None
    sched_msg = None
    if training_cfg_dict.get("scheduler"):
        scheduler, sched_msg = build_cfg_scheduler(optimizer, steps_per_epoch, training_cfg_dict)

    gan_cfg_obj = config.get("gan")
    if not gan_cfg_obj:
        gan_cfg_obj = training_cfg_dict.get("gan")
    if gan_cfg_obj:
        if isinstance(gan_cfg_obj, dict):
            gan_cfg = dict(gan_cfg_obj)
        else:
            gan_cfg = dict(OmegaConf.to_container(gan_cfg_obj, resolve=True))
    else:
        gan_cfg = {}
    disc_cfg = dict(gan_cfg.get("disc", {})) if gan_cfg else {}
    loss_cfg = dict(gan_cfg.get("loss", {})) if gan_cfg else {}

    perceptual_weight = float(loss_cfg.get("perceptual_weight", training_cfg_dict.get("perceptual_weight", 0.0)))
    disc_weight = float(loss_cfg.get("disc_weight", training_cfg_dict.get("disc_weight", 0.0)))
    disc_updates = int(loss_cfg.get("disc_updates", training_cfg_dict.get("disc_updates", 1)))
    max_d_weight = float(loss_cfg.get("max_d_weight", training_cfg_dict.get("max_d_weight", 1e4)))
    gan_start_epoch = int(loss_cfg.get("disc_start", training_cfg_dict.get("gan_start_epoch", 0)))
    disc_update_epoch = int(loss_cfg.get("disc_upd_start", training_cfg_dict.get("disc_upd_start", gan_start_epoch)))
    lpips_start_epoch = int(loss_cfg.get("lpips_start", training_cfg_dict.get("lpips_start", 0)))
    disc_loss_type = str(loss_cfg.get("disc_loss", training_cfg_dict.get("disc_loss", "hinge"))).lower()
    gen_loss_type = str(loss_cfg.get("gen_loss", training_cfg_dict.get("gen_loss", "vanilla"))).lower()
    gan_start_step = gan_start_epoch * steps_per_epoch
    disc_update_step = disc_update_epoch * steps_per_epoch
    lpips_start_step = lpips_start_epoch * steps_per_epoch

    discriminator = None
    disc_optimizer = None
    disc_scheduler = None
    disc_aug = None
    disc_optim_msg = None
    disc_sched_msg = None
    disc_loss_fn = None
    gen_loss_fn = None
    if use_gan:
        if not disc_cfg:
            raise ValueError("GAN is enabled but 'gan.disc' configuration is missing.")
        discriminator, disc_aug = build_discriminator(disc_cfg, device)
        discriminator.eval()
        disc_training_cfg = {
            "optimizer": disc_cfg.get("optimizer", {}),
            "scheduler": disc_cfg.get("scheduler", {}),
        }
        disc_optimizer, disc_optim_msg = build_cfg_optimizer(discriminator.parameters(), disc_training_cfg)
        if disc_training_cfg["scheduler"]:
            disc_scheduler, disc_sched_msg = build_cfg_scheduler(disc_optimizer, steps_per_epoch, disc_training_cfg)
        disc_loss_fn, gen_loss_fn = select_gan_losses(disc_loss_type, gen_loss_type)

    gan_active = use_gan and disc_weight > 0.0 and discriminator is not None

    lpips_model = None
    lpips_enabled = use_lpips or perceptual_weight > 0.0
    if lpips_enabled:
        if LPIPS is None:
            raise RuntimeError("LPIPS is requested but the dependency is unavailable.")
        lpips_model = LPIPS().to(device)
        lpips_model.eval()
    else:
        perceptual_weight = 0.0

    prepare_args = [model]
    if gan_active and discriminator is not None:
        prepare_args.append(discriminator)
    prepare_args.append(optimizer)
    if gan_active and disc_optimizer is not None:
        prepare_args.append(disc_optimizer)
    prepare_args.append(loader)
    if eval_loader is not None:
        prepare_args.append(eval_loader)
    prepared_items = accelerator.prepare(*prepare_args)
    if not isinstance(prepared_items, (list, tuple)):
        prepared_items = [prepared_items]
    prepared_iter = iter(prepared_items)
    model = next(prepared_iter)
    if gan_active and discriminator is not None:
        discriminator = next(prepared_iter)
    optimizer = next(prepared_iter)
    if gan_active and disc_optimizer is not None:
        disc_optimizer = next(prepared_iter)
    loader = next(prepared_iter)
    if eval_loader is not None:
        eval_loader = next(prepared_iter)

    if accelerator.is_main_process:
        run_dir = args.results_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        run_index = len(list(run_dir.glob("*")))
        model_name = stage_cfg.get("target", "uae.UnifiedUAE").split(".")[-1]
        if selected_stage_key:
            model_name = f"{selected_stage_key}-{model_name}"
        experiment_dir = run_dir / f"{run_index:03d}-{model_name}"
        checkpoint_dir = experiment_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger = create_logger(experiment_dir)
        if selected_stage_key:
            logger.info("Training stage '%s' from multi-stage config.", selected_stage_key)
        logger.info("Experiment directory: %s", experiment_dir)
        if precision_warning:
            logger.warning(precision_warning)
        logger.info("Mixed precision mode: %s", precision_mode)
        logger.info(optim_msg)
        if sched_msg:
            logger.info(sched_msg)
        if gan_active and disc_optim_msg:
            logger.info(disc_optim_msg)
            if disc_sched_msg:
                logger.info(disc_sched_msg)
        if wandb_cfg and wandb_cfg.get("enable", False):
            try:
                import wandb

                wandb_kwargs = {k: v for k, v in wandb_cfg.items() if k != "enable"}
                wandb_kwargs.setdefault("project", "unified-ae")
                wandb_kwargs.setdefault("config", OmegaConf.to_container(config, resolve=True))
                wandb_run = wandb.init(**wandb_kwargs)
                logger.info("Initialized Weights&Biases run: %s", wandb_run.name)
            except ImportError:
                logger.warning("wandb is not installed; skipping wandb logging.")
                wandb_run = None
    else:
        experiment_dir = None
        checkpoint_dir = None
        logger = logging.getLogger("dummy")
        logger.addHandler(logging.NullHandler())

    start_epoch = 0
    global_step = 0

    if args.resume_from is not None:
        ckpt = torch.load(args.resume_from, map_location="cpu")
        base_model = accelerator.unwrap_model(model)
        state_dict = ckpt["model"]
        _load_model_state_strict(base_model, state_dict, logger)
        optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        if gan_active and discriminator is not None:
            disc_state = ckpt.get("discriminator")
            if disc_state is not None:
                accelerator.unwrap_model(discriminator).load_state_dict(disc_state)
            if disc_optimizer is not None and ckpt.get("disc_optimizer") is not None:
                disc_optimizer.load_state_dict(ckpt["disc_optimizer"])
            if disc_scheduler is not None and ckpt.get("disc_scheduler") is not None:
                disc_scheduler.load_state_dict(ckpt["disc_scheduler"])
        if use_ema and ema_model is not None and ckpt.get("ema") is not None:
            ema_model.load_state_dict(ckpt["ema"])
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("global_step", 0)
        accelerator.print(f"Resumed from checkpoint {args.resume_from} at epoch {start_epoch}, step {global_step}.")

    last_layer_param = None
    if gan_active:
        base_model = accelerator.unwrap_model(model)
        try:
            last_layer_param = base_model.decoder.decoder_pred.weight
        except AttributeError as exc:
            raise AttributeError(
                "UnifiedUAE decoder is missing decoder_pred.weight required for GAN training."
            ) from exc

    for epoch in range(start_epoch, epochs):
        model.train()
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            model_core = accelerator.unwrap_model(model)
            optimizer.zero_grad(set_to_none=True)
            if gan_active and discriminator is not None:
                discriminator.eval()

            freq_ratio_tensor = None
            encoder_trainable = getattr(model_core, "encoder_trainable", True)

            with autocast_context(device, autocast_enabled, autocast_dtype):
                apply_frequency = model_core.freq_modulator is not None
                if apply_frequency:
                    freq_ratio_tensor = compute_freq_ratio(
                        freq_ratio_mode,
                        images.size(0),
                        training_cfg_dict,
                        device,
                    ).clamp_(0.0, 1.0)
                encode_kwargs = {
                    "apply_frequency": apply_frequency,
                    "freq_ratio": freq_ratio_tensor,
                    "return_pre_frequency": apply_frequency,
                    "freeze_encoder": not encoder_trainable,
                }
                encoded_output = model_core.encode(images, **encode_kwargs)
                if isinstance(encoded_output, tuple):
                    encoded_latent, latent_for_decode = encoded_output
                else:
                    encoded_latent = encoded_output
                    latent_for_decode = encoded_output
                recon = model_core.decode(latent_for_decode)

            recon_autocast = recon
            recon_fp32 = recon_autocast.float()
            images_fp32 = images.float()
            rec_loss = F.l1_loss(recon_fp32, images_fp32)
            real_normed = images_fp32 * 2.0 - 1.0
            real_normed_disc = images.to(dtype=recon_autocast.dtype) * 2.0 - 1.0
            if freq_ratio_tensor is not None:
                freq_ratio_tensor_fp32 = freq_ratio_tensor.float()
            else:
                freq_ratio_tensor_fp32 = None

            lpips_loss = torch.zeros_like(rec_loss)
            if lpips_model is not None and global_step >= lpips_start_step and perceptual_weight > 0.0:
                with disable_autocast(device):
                    images_lpips = (images_fp32 * 2.0 - 1.0).clamp(-1.0, 1.0)
                    recon_lpips = (recon_fp32 * 2.0 - 1.0).clamp(-1.0, 1.0)
                    lpips_value = lpips_model(images_lpips, recon_lpips)
                lpips_loss = lpips_value.to(rec_loss.dtype)

            keep_fraction = 1.0
            if freq_ratio_tensor_fp32 is not None:
                keep_fraction = float(freq_ratio_tensor_fp32.mean().detach().item())
            keep_fraction = max(0.0, min(1.0, keep_fraction))

            recon_scale = recon_min + keep_fraction * (recon_full - recon_min) if dynamic_band_enabled else 1.0
            perceptual_scale = (
                perceptual_min + keep_fraction * (perceptual_full - perceptual_min) if dynamic_band_enabled else 1.0
            )
            disc_scale = (
                disc_min_scale + keep_fraction * (disc_full_scale - disc_min_scale) if dynamic_band_enabled else 1.0
            )

            recon_total = rec_loss * recon_scale
            perceptual_weight_scaled = perceptual_weight * perceptual_scale
            if perceptual_weight_scaled > 0.0:
                recon_total = recon_total + perceptual_weight_scaled * lpips_loss

            band_reg_loss = None
            if band_reference is not None and model_core.freq_modulator is not None:
                encoded_latent_fp32 = encoded_latent.float()
                ae_bands = model_core.freq_modulator.decompose_latent(
                    encoded_latent_fp32,
                    to_fhat=True,
                    return_bands=True,
                    band_ratio=freq_ratio_tensor_fp32,
                )
                if ae_bands:
                    ae_band0 = ae_bands[0]
                    with torch.no_grad():
                        with disable_autocast(device):
                            ref_band0 = band_reference(images_fp32)
                    ref_band0 = ref_band0.to(device=ae_band0.device, dtype=ae_band0.dtype)
                    if ref_band0.shape[-2:] != ae_band0.shape[-2:]:
                        ref_band0 = F.interpolate(
                            ref_band0,
                            size=ae_band0.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                    band_reg_loss = F.mse_loss(ae_band0, ref_band0)
                    recon_total = recon_total + band_reg_weight * band_reg_loss

            disc_weight_effective = disc_weight * disc_scale
            gan_loss = torch.zeros_like(recon_total)
            adaptive_weight = torch.zeros_like(recon_total)
            recon_normed = recon_autocast * 2.0 - 1.0
            use_gan_step = gan_active and global_step >= gan_start_step
            train_disc = gan_active and global_step >= disc_update_step
            if gan_on_corrupt_only and keep_fraction >= 0.999:
                use_gan_step = False
                train_disc = False
            total_loss = recon_total
            if use_gan_step and discriminator is not None and gen_loss_fn is not None and last_layer_param is not None:
                fake_input = disc_aug.aug(recon_normed) if disc_aug is not None else recon_normed
                with autocast_context(device, autocast_enabled, autocast_dtype):
                    logits_fake, _ = discriminator(fake_input, None)
                gan_loss = gen_loss_fn(logits_fake).to(recon_total.dtype)
                adaptive_weight = calculate_adaptive_weight(recon_total, gan_loss, last_layer_param, max_d_weight)
                total_loss = total_loss + disc_weight_effective * adaptive_weight * gan_loss

            if scaler is not None:
                scaler.scale(total_loss).backward()
                if clip_grad is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            if use_ema and ema_model is not None:
                update_ema(ema_model, accelerator.unwrap_model(model), ema_decay)

            disc_metrics: Dict[str, torch.Tensor] = {}
            if (
                train_disc
                and discriminator is not None
                and disc_optimizer is not None
                and disc_loss_fn is not None
            ):
                discriminator.train()
                for _ in range(max(1, disc_updates)):
                    disc_optimizer.zero_grad(set_to_none=True)
                    with autocast_context(device, autocast_enabled, autocast_dtype):
                        fake_detached = recon_normed.detach()
                        fake_detached = fake_detached.clamp(-1.0, 1.0)
                        fake_detached = torch.round((fake_detached + 1.0) * 127.5) / 127.5 - 1.0
                        fake_input = disc_aug.aug(fake_detached) if disc_aug is not None else fake_detached
                        real_input = disc_aug.aug(real_normed_disc) if disc_aug is not None else real_normed_disc
                        logits_fake, logits_real = discriminator(fake_input, real_input)
                        d_loss = disc_loss_fn(logits_real, logits_fake)
                    if scaler is not None:
                        scaler.scale(d_loss).backward()
                        scaler.step(disc_optimizer)
                        scaler.update()
                    else:
                        d_loss.backward()
                        disc_optimizer.step()
                    if disc_scheduler is not None:
                        disc_scheduler.step()
                        disc_metrics = {
                            "disc_loss": d_loss.detach(),
                            "logits_real": logits_real.detach().mean() if logits_real is not None else torch.zeros(
                                (), device=d_loss.device
                            ),
                            "logits_fake": logits_fake.detach().mean(),
                        }
                discriminator.eval()

            global_step += 1
            if accelerator.is_main_process and global_step % args.log_every == 0:
                stats: Dict[str, Any] = {
                    "loss/total": float(total_loss.item()),
                    "loss/recon": float(rec_loss.item()),
                    "train/freq_keep": float(keep_fraction),
                    "lr/generator": optimizer.param_groups[0]["lr"],
                }
                if lpips_model is not None and perceptual_weight_scaled > 0.0:
                    stats["loss/lpips"] = float(lpips_loss.item())
                if band_reg_loss is not None:
                    stats["loss/band_reg"] = float(band_reg_loss.item())
                if gan_active:
                    stats["loss/gan"] = float(gan_loss.item())
                    stats["gan/weight"] = float(adaptive_weight.item()) if use_gan_step else 0.0
                    stats["gan/coef"] = float(disc_weight_effective)
                if freq_ratio_tensor_fp32 is not None:
                    stats["freq_ratio/mean"] = float(freq_ratio_tensor_fp32.mean().detach().item())
                if dynamic_band_enabled:
                    stats["band_weights/recon_scale"] = float(recon_scale)
                    stats["band_weights/perceptual_scale"] = float(perceptual_scale)
                    stats["band_weights/disc_scale"] = float(disc_scale)
                if scheduler is not None:
                    stats["lr/scheduler"] = scheduler.get_last_lr()[0]
                if disc_metrics:
                    stats["loss/disc"] = float(disc_metrics["disc_loss"].item())
                    stats["disc/logits_real"] = float(disc_metrics["logits_real"].item())
                    stats["disc/logits_fake"] = float(disc_metrics["logits_fake"].item())
                    stats["lr/discriminator"] = disc_optimizer.param_groups[0]["lr"]
                logger.info(
                    "epoch=%d step=%d total=%.5f freq_keep=%.3f",
                    epoch,
                    global_step,
                    total_loss.item(),
                    keep_fraction,
                )
                if wandb_run is not None:
                    wandb_run.log(stats, step=global_step)

            if (
                eval_loader is not None
                and eval_interval > 0
                and global_step % eval_interval == 0
                and global_step > 0
            ):
                eval_wandb = wandb_run if accelerator.is_main_process else None
                eval_dir = experiment_dir if accelerator.is_main_process else None
                eval_target = ema_model if (use_ema and ema_model is not None) else model
                evaluate_model(
                    eval_target,
                    accelerator,
                    eval_loader,
                    training_cfg_dict,
                    logger,
                    global_step,
                    eval_dir,
                    eval_wandb,
                    autocast_enabled,
                    autocast_dtype,
                )

            if (
                accelerator.is_main_process
                and checkpoint_dir is not None
                and args.ckpt_every > 0
                and global_step % args.ckpt_every == 0
            ):
                ckpt_path = checkpoint_dir / "final.pt" #checkpoint_dir / f"{global_step:07d}.pt"
                to_save = {
                    "model": accelerator.unwrap_model(model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "config": OmegaConf.to_container(config, resolve=True),
                }
                if scheduler is not None:
                    to_save["scheduler"] = scheduler.state_dict()
                if gan_active and discriminator is not None:
                    to_save["discriminator"] = accelerator.unwrap_model(discriminator).state_dict()
                    to_save["disc_optimizer"] = disc_optimizer.state_dict() if disc_optimizer is not None else None
                    if disc_scheduler is not None:
                        to_save["disc_scheduler"] = disc_scheduler.state_dict()
                if use_ema and ema_model is not None:
                    to_save["ema"] = ema_model.state_dict()
                torch.save(to_save, ckpt_path)
                logger.info("Saved checkpoint to %s", ckpt_path)

        accelerator.wait_for_everyone()

    eval_target = ema_model if (use_ema and ema_model is not None) else model
    if eval_loader is not None:
        eval_wandb = wandb_run if accelerator.is_main_process else None
        eval_dir = experiment_dir if accelerator.is_main_process else None
        evaluate_model(
            eval_target,
            accelerator,
            eval_loader,
            training_cfg_dict,
            logger,
            global_step,
            eval_dir,
            eval_wandb,
            autocast_enabled,
            autocast_dtype,
        )

    if accelerator.is_main_process and checkpoint_dir is not None:
        final_ckpt = checkpoint_dir / "final.pt"
        to_save = {
            "model": accelerator.unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epochs,
            "global_step": global_step,
            "config": OmegaConf.to_container(config, resolve=True),
        }
        if scheduler is not None:
            to_save["scheduler"] = scheduler.state_dict()
        if gan_active and discriminator is not None:
            to_save["discriminator"] = accelerator.unwrap_model(discriminator).state_dict()
            to_save["disc_optimizer"] = disc_optimizer.state_dict() if disc_optimizer is not None else None
            if disc_scheduler is not None:
                to_save["disc_scheduler"] = disc_scheduler.state_dict()
        if use_ema and ema_model is not None:
            to_save["ema"] = ema_model.state_dict()
        torch.save(to_save, final_ckpt)
        logger.info("Training complete. Final checkpoint: %s", final_ckpt)

    if accelerator.is_main_process and wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
