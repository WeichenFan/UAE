#!/usr/bin/env python3
"""
Evaluate Unified UAE reconstructions on datasets with PSNR / SSIM / rFID.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.models import inception_v3, Inception_V3_Weights
from scipy import linalg

from model_utils import instantiate_from_config_2

try:
    from torchmetrics.functional import structural_similarity_index_measure
except ImportError as exc:  # pragma: no cover
    raise ImportError("torchmetrics is required for SSIM computation") from exc


def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


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


class ImageFolderFlat(Dataset):
    def __init__(self, root: Path, transform: transforms.Compose, extensions: Optional[List[str]] = None):
        self.root = Path(root)
        self.transform = transform
        if extensions is None:
            extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
        self.paths = sorted([p for p in self.root.rglob("*") if p.suffix.lower() in extensions])
        if not self.paths:
            raise FileNotFoundError(f"No images with extensions {extensions} found under {root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
        return self.transform(img)


def build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Lambda(lambda img: center_crop_arr(img, image_size)),
            transforms.ToTensor(),
        ]
    )


def load_uae(config_path: Path, checkpoint: Optional[Path], device: torch.device):
    cfg = OmegaConf.load(str(config_path))
    stage_cfg = cfg.get("stage_1", None)
    if stage_cfg is None:
        raise ValueError("Config missing stage_1 definition")
    if checkpoint is not None:
        stage_cfg.ckpt = str(checkpoint)
    model = instantiate_from_config_2(stage_cfg).to(device)
    model.eval()
    return model


def reconstruct(model, images: torch.Tensor, freq_ratio: float) -> torch.Tensor:
    with torch.no_grad():
        freq_mod = getattr(model, "freq_modulator", None)
        freq_ratio_tensor = None
        apply_frequency = freq_mod is not None
        if apply_frequency:
            value = float(max(0.0, min(1.0, freq_ratio)))
            freq_ratio_tensor = torch.full(
                (images.size(0),),
                value,
                device=images.device,
                dtype=images.dtype,
            )
        latent = model.encode(
            images,
            apply_frequency=apply_frequency,
            freq_ratio=freq_ratio_tensor,
        )
        recon = model.decode(latent)
    return recon.clamp(0.0, 1.0)


def evaluate_dataset(
    name: str,
    model,
    dataset_root: Path,
    args,
    device: torch.device,
) -> dict:
    transform = build_transform(args.image_size)
    dataset = ImageFolderFlat(dataset_root, transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    total_psnr = 0.0
    total_ssim = 0.0
    count = 0
    inception = InceptionFeatureExtractor(device)
    stats_real = FeatureStats(2048)
    stats_fake = FeatureStats(2048)

    for images in loader:
        if args.max_images is not None and count >= args.max_images:
            break
        if args.max_images is not None and count + images.size(0) > args.max_images:
            images = images[: args.max_images - count]
        images = images.to(device)
        recon = reconstruct(model, images, args.freq_ratio)
        images_f32 = images.float()
        recon_f32 = recon.float()

        mse = F.mse_loss(recon_f32, images_f32, reduction="none").view(images.size(0), -1).mean(dim=1)
        psnr = 10 * torch.log10(1.0 / mse.clamp_min(1e-8))
        ssim = structural_similarity_index_measure(recon_f32, images_f32)

        total_psnr += psnr.sum().item()
        total_ssim += ssim.item() * images.size(0)
        count += images.size(0)

        feats_real = inception(images_f32)
        feats_fake = inception(recon_f32)
        stats_real.update(feats_real)
        stats_fake.update(feats_fake)

    metrics = {
        "dataset": name,
        "psnr": total_psnr / count,
        "ssim": total_ssim / count,
        "rfid": compute_fid(stats_real.mean(), stats_real.covariance(), stats_fake.mean(), stats_fake.covariance()),
        "images": count,
    }
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Unified UAE reconstructions.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--imagenet-path", type=Path, required=True)
    parser.add_argument("--coco-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap per dataset")
    parser.add_argument("--log-file", type=Path, default=Path("uae_eval_metrics.txt"))
    parser.add_argument("--freq-ratio", type=float, default=1.0, help="Frequency ratio applied during decoding.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_uae(args.config, args.checkpoint, device)

    results = []
    for name, path in [("ImageNet", args.imagenet_path), ("MS-COCO", args.coco_path)]:
        print(f"Evaluating on {name}...")
        metrics = evaluate_dataset(name, model, path, args, device)
        print(
            f"{name}: PSNR={metrics['psnr']:.3f} dB | SSIM={metrics['ssim']:.4f} | rFID={metrics['rfid']:.3f} | images={metrics['images']}"
        )
        results.append(metrics)

    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.log_file, "w", encoding="utf-8") as f:
        f.write("dataset,images,psnr,ssim,rfid\n")
        for m in results:
            f.write(f"{m['dataset']},{m['images']},{m['psnr']:.6f},{m['ssim']:.6f},{m['rfid']:.6f}\n")
    print(f"Saved metrics to {args.log_file}")


if __name__ == "__main__":
    main()
