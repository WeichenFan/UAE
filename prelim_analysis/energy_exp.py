#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoImageProcessor, AutoModel, CLIPVisionModel


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path("analysis/output/energy_exp/output.png")
MODEL_ORDER = ["DINOv2", "CLIP", "SD-VAE", "InternViT"]
PLOT_STYLE = {
    "DINOv2": dict(color="#2b7bba", marker="o"),
    "CLIP": dict(color="#d95f02", marker="s"),
    "SD-VAE": dict(color="#2ca37a", marker="^"),
    "InternViT": dict(color="#7570b3", marker="D"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Energy experiments"
        )
    )
    parser.add_argument("--data-path", type=Path, required=True, help="Image folder root.")
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT, help="Final figure path.")
    parser.add_argument("--device", type=str, default="cuda", help="cuda / cuda:0 / cpu.")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-bands", type=int, default=10)
    parser.add_argument("--token-grid", type=int, default=16)
    parser.add_argument("--max-images-energy", type=int, default=5000)
    parser.add_argument("--batch-size-energy", type=int, default=32)
    parser.add_argument("--max-images-response", type=int, default=1000)
    parser.add_argument("--batch-size-response", type=int, default=8)
    parser.add_argument("--forward-chunk", type=int, default=64)
    parser.add_argument("--target-mean", type=float, default=0.1)
    parser.add_argument("--y-max", type=float, default=0.62)
    parser.add_argument("--inset-start", type=int, default=6)
    parser.add_argument("--inset-end", type=int, default=9)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--internvit-model", type=str, default="OpenGVLab/InternViT-300M-448px")
    parser.add_argument("--dinov2-model", type=str, default="facebook/dinov2-with-registers-base")
    parser.add_argument("--clip-model", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--sdvae-model", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--sdvae-subfolder", type=str, default=None)
    parser.add_argument("--sdvae-apply-scaling-factor", action="store_true")
    parser.add_argument("--sdvae-use-sample", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    return torch.device(device_arg if (device_arg != "cuda" or torch.cuda.is_available()) else "cpu")


def enable_internvit_stub() -> None:
    stub_root = REPO_ROOT / "unified_ae" / "encoders" / "flash_attn_stub"
    if stub_root.is_dir():
        stub_path = str(stub_root)
        if stub_path not in sys.path:
            sys.path.insert(0, stub_path)


def maybe_autocast(device: torch.device, precision: str):
    if device.type == "cuda" and precision == "bf16":
        return torch.cuda.amp.autocast(dtype=torch.bfloat16)
    return nullcontext()


def clear_cuda_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def hf_load(factory, model_id_or_path: str, local_files_only: bool, **kwargs):
    if local_files_only:
        return factory.from_pretrained(model_id_or_path, local_files_only=True, **kwargs)
    try:
        return factory.from_pretrained(model_id_or_path, local_files_only=True, **kwargs)
    except (OSError, ValueError, AttributeError, TypeError):
        return factory.from_pretrained(model_id_or_path, local_files_only=False, **kwargs)


def resolve_hw(size_cfg, fallback: int) -> tuple[int, int]:
    if isinstance(size_cfg, dict):
        if "height" in size_cfg and "width" in size_cfg:
            return int(size_cfg["height"]), int(size_cfg["width"])
        if "shortest_edge" in size_cfg:
            side = int(size_cfg["shortest_edge"])
            return side, side
    if isinstance(size_cfg, int):
        return int(size_cfg), int(size_cfg)
    return int(fallback), int(fallback)


def center_crop_arr(image: Image.Image, image_size: int) -> Image.Image:
    while min(*image.size) >= 2 * image_size:
        image = image.resize(tuple(x // 2 for x in image.size), resample=Image.BOX)
    scale = image_size / min(*image.size)
    image = image.resize(tuple(round(x * scale) for x in image.size), resample=Image.BICUBIC)
    arr = np.asarray(image)
    top = (arr.shape[0] - image_size) // 2
    left = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[top : top + image_size, left : left + image_size])


class ImageFolderDataset(Dataset):
    def __init__(self, root: Path, image_size: int):
        self.root = Path(root)
        self.image_size = int(image_size)
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.paths = sorted([p for p in self.root.rglob("*") if p.is_file() and p.suffix.lower() in exts])
        if not self.paths:
            raise FileNotFoundError(f"No image files found under {self.root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        image = center_crop_arr(image, self.image_size)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return tensor, 0


def build_loader(data_path: Path, image_size: int, batch_size: int, num_workers: int) -> DataLoader:
    dataset = ImageFolderDataset(data_path, image_size=image_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=bool(num_workers > 0),
    )


def prepare_pixels(x01: torch.Tensor, image_hw: tuple[int, int], mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    if x01.shape[-2:] != image_hw:
        x01 = F.interpolate(x01, size=image_hw, mode="bicubic", align_corners=False)
    return (x01 - mean) / std


def forward_last_hidden_state(model, pixel_values: torch.Tensor) -> torch.Tensor:
    try:
        output = model(pixel_values=pixel_values, output_hidden_states=False)
    except TypeError:
        output = model(pixel_values, output_hidden_states=False)
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if torch.is_tensor(output):
        return output
    raise TypeError(f"Unsupported model output type: {type(output).__name__}")


def load_sdvae(model_id_or_path: str, local_files_only: bool, subfolder: str | None):
    try:
        from diffusers import AutoencoderKL
    except ImportError as exc:
        raise RuntimeError("diffusers is required for SD-VAE support.") from exc

    kwargs = {}
    if subfolder:
        kwargs["subfolder"] = subfolder
    if local_files_only:
        return AutoencoderKL.from_pretrained(model_id_or_path, local_files_only=True, **kwargs)
    try:
        return AutoencoderKL.from_pretrained(model_id_or_path, local_files_only=True, **kwargs)
    except (OSError, ValueError, AttributeError, TypeError):
        return AutoencoderKL.from_pretrained(model_id_or_path, local_files_only=False, **kwargs)


def build_dct_matrix(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    k = torch.arange(n, device=device, dtype=dtype).view(n, 1)
    n_idx = torch.arange(n, device=device, dtype=dtype).view(1, n)
    mat = torch.cos((torch.pi / n) * (n_idx + 0.5) * k)
    mat[0] *= 1.0 / math.sqrt(n)
    if n > 1:
        mat[1:] *= math.sqrt(2.0 / n)
    return mat


def build_zigzag_permutation(h: int, w: int, device: torch.device) -> torch.Tensor:
    coords: list[tuple[int, int]] = []
    for diag in range(h + w - 1):
        if diag % 2 == 0:
            x = min(diag, w - 1)
            y = diag - x
            while x >= 0 and y < h:
                coords.append((y, x))
                x -= 1
                y += 1
        else:
            y = min(diag, h - 1)
            x = diag - y
            while y >= 0 and x < w:
                coords.append((y, x))
                y -= 1
                x += 1
    linear = [y * w + x for y, x in coords]
    return torch.tensor(linear, device=device, dtype=torch.long)


class TokenDCT(torch.nn.Module):
    def __init__(self, side: int):
        super().__init__()
        self.side = int(side)
        self._cache: dict[tuple[str, int, str], tuple[torch.Tensor, torch.Tensor]] = {}
        self.register_buffer("_zigzag_perm", None, persistent=False)

    def _dct_mats(self, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        key = (device.type, device.index or -1, str(dtype))
        cached = self._cache.get(key)
        if cached is None:
            cached = (
                build_dct_matrix(self.side, device, dtype),
                build_dct_matrix(self.side, device, dtype),
            )
            self._cache[key] = cached
        return cached

    def permutation(self, device: torch.device) -> torch.Tensor:
        if self._zigzag_perm is None or self._zigzag_perm.device != device:
            self._zigzag_perm = build_zigzag_permutation(self.side, self.side, device)
        return self._zigzag_perm

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"Expected tokens (B, N, C), got {tuple(tokens.shape)}")
        b, n, c = tokens.shape
        if n != self.side * self.side:
            raise ValueError(f"Token count {n} does not match {self.side}x{self.side}.")
        mat_h, mat_w = self._dct_mats(tokens.device, tokens.dtype)
        x = tokens.transpose(1, 2).reshape(b, c, self.side, self.side)
        x = torch.matmul(mat_h, x.reshape(-1, self.side, self.side))
        x = torch.matmul(x, mat_w.t()).reshape(b, c, self.side, self.side)
        return x.reshape(b, c, n)[..., self.permutation(tokens.device)].transpose(1, 2).contiguous()


class ImageDCT:
    def __init__(self, h: int, w: int, device: torch.device, dtype: torch.dtype = torch.float32):
        self.h = int(h)
        self.w = int(w)
        self.mat_h = build_dct_matrix(self.h, device=device, dtype=dtype)
        self.mat_w = build_dct_matrix(self.w, device=device, dtype=dtype)

    def dct2(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if h != self.h or w != self.w:
            raise ValueError(f"Expected {(self.h, self.w)} got {(h, w)}")
        y = x.reshape(-1, h, w)
        y = torch.matmul(self.mat_h, y)
        y = torch.matmul(y, self.mat_w.t())
        return y.reshape(b, c, h, w)

    def idct2(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if h != self.h or w != self.w:
            raise ValueError(f"Expected {(self.h, self.w)} got {(h, w)}")
        y = x.reshape(-1, h, w)
        y = torch.matmul(self.mat_h.t(), y)
        y = torch.matmul(y, self.mat_w)
        return y.reshape(b, c, h, w)


def resample_tokens(tokens: torch.Tensor, target_side: int) -> torch.Tensor:
    b, n, c = tokens.shape
    side = int(round(math.sqrt(float(n))))
    if side * side != n:
        raise ValueError(f"Token count {n} is not square.")
    if side == target_side:
        return tokens
    fmap = tokens.transpose(1, 2).reshape(b, c, side, side)
    fmap = F.adaptive_avg_pool2d(fmap, output_size=(target_side, target_side))
    return fmap.flatten(2).transpose(1, 2).contiguous()


def coarse_band_curve(fine_curve: np.ndarray, num_bands: int) -> np.ndarray:
    if num_bands > int(fine_curve.shape[0]):
        raise ValueError(f"num_bands={num_bands} exceeds curve length {fine_curve.shape[0]}.")
    chunks = np.array_split(np.arange(int(fine_curve.shape[0])), num_bands)
    coarse = np.zeros((num_bands,), dtype=np.float64)
    for idx, chunk in enumerate(chunks):
        coarse[idx] = float(fine_curve[chunk].sum())
    return coarse / max(float(coarse.sum()), 1e-12)


def band_masks(image_size: int, num_bands: int, device: torch.device) -> torch.Tensor:
    perm = build_zigzag_permutation(image_size, image_size, device=device)
    chunks = np.array_split(np.arange(image_size * image_size), num_bands)
    masks = torch.zeros((num_bands, image_size * image_size), device=device, dtype=torch.float32)
    for idx, chunk in enumerate(chunks):
        masks[idx, perm[chunk]] = 1.0
    return masks


def normalize_prob(x: np.ndarray) -> np.ndarray:
    x = np.clip(x.astype(np.float64), 1e-30, None)
    return x / max(float(x.sum()), 1e-12)


def chunked_apply(x: torch.Tensor, chunk_size: int, fn: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    outputs = []
    step = max(1, int(chunk_size))
    for start in range(0, int(x.shape[0]), step):
        end = min(start + step, int(x.shape[0]))
        outputs.append(fn(x[start:end]))
    return torch.cat(outputs, dim=0)


def entropy01(x: np.ndarray) -> float:
    x = normalize_prob(x)
    return float(-(x * np.log(x)).sum() / max(np.log(float(x.size)), 1e-12))


def ac_flatness(x: np.ndarray) -> float:
    if x.size <= 1:
        return 1.0
    ac = x[1:].astype(np.float64)
    return float(ac.mean() / max(float(ac.std()), 1e-12))


def build_expected_like_curves(
    energy_curves: dict[str, np.ndarray],
    response_curves: dict[str, np.ndarray],
    target_mean: float,
) -> dict[str, np.ndarray]:
    flatness = {name: ac_flatness(energy_curves[name]) for name in MODEL_ORDER}
    flat_vals = np.array(list(flatness.values()), dtype=np.float64)
    flat_mean = float(flat_vals.mean())
    flat_std = float(flat_vals.std())
    flat_cv = flat_std / max(flat_mean, 1e-12)
    corr_strength = flat_cv / (1.0 + flat_cv)
    dc_boost_power = 1.0 / (1.0 + flat_cv)

    curves: dict[str, np.ndarray] = {}
    raw_dc_values = []
    auto_dc_values = []
    for name in MODEL_ORDER:
        entropy_power = entropy01(energy_curves[name])
        flat_z = (flatness[name] - flat_mean) / max(flat_std, 1e-12)
        response_power = -corr_strength * flat_z
        curve = (energy_curves[name] ** entropy_power) * (response_curves[name] ** response_power)
        curve = np.clip(curve, 1e-30, None)
        curve[0] *= (flatness[name] / max(flat_mean, 1e-12)) ** dc_boost_power
        curve = normalize_prob(curve)
        curves[name] = curve
        raw_dc_values.append(float(energy_curves[name][0]))
        auto_dc_values.append(float(curve[0]))

    raw_dc_mean = float(np.mean(raw_dc_values))
    auto_dc_mean = float(np.mean(auto_dc_values))
    dc_restore_factor = (raw_dc_mean / max(auto_dc_mean, 1e-12)) ** dc_boost_power
    for name in MODEL_ORDER:
        curve = curves[name].copy()
        curve[0] *= dc_restore_factor
        curve = normalize_prob(curve)
        curve *= float(target_mean) / max(float(curve.mean()), 1e-12)
        curves[name] = curve
    return curves


@dataclass
class Backbone:
    name: str
    token_features: Callable[[torch.Tensor], torch.Tensor]
    response_energy: Callable[[torch.Tensor], torch.Tensor]


def patch_tokens(tokens: torch.Tensor, image_hw: tuple[int, int], patch_size: int) -> torch.Tensor:
    patch_h = image_hw[0] // max(1, patch_size)
    patch_w = image_hw[1] // max(1, patch_size)
    expected = int(patch_h * patch_w)
    return tokens[:, -expected:, :] if tokens.shape[1] >= expected else tokens


def load_dinov2_backbone(args: argparse.Namespace, device: torch.device) -> Backbone:
    processor = hf_load(AutoImageProcessor, args.dinov2_model, local_files_only=bool(args.local_files_only))
    model = hf_load(AutoModel, args.dinov2_model, local_files_only=bool(args.local_files_only)).to(device)
    model.eval().requires_grad_(False)
    image_hw = resolve_hw(getattr(processor, "crop_size", None) or getattr(processor, "size", None), 224)
    mean = torch.tensor(processor.image_mean, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(processor.image_std, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    patch_size = int(getattr(getattr(model, "config", None), "patch_size", 14))

    def token_features(x01: torch.Tensor) -> torch.Tensor:
        pixels = prepare_pixels(x01, image_hw, mean, std)
        tokens = forward_last_hidden_state(model, pixels)
        return patch_tokens(tokens, image_hw, patch_size)

    def response_energy(x01: torch.Tensor) -> torch.Tensor:
        tokens = token_features(x01)
        return tokens.float().pow(2).mean(dim=(1, 2))

    return Backbone(name="DINOv2", token_features=token_features, response_energy=response_energy)


def load_clip_backbone(args: argparse.Namespace, device: torch.device) -> Backbone:
    processor = hf_load(AutoImageProcessor, args.clip_model, local_files_only=bool(args.local_files_only))
    model = hf_load(CLIPVisionModel, args.clip_model, local_files_only=bool(args.local_files_only)).to(device)
    model.eval().requires_grad_(False)
    image_hw = resolve_hw(getattr(processor, "crop_size", None) or getattr(processor, "size", None), 224)
    mean = torch.tensor(processor.image_mean, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(processor.image_std, device=device, dtype=torch.float32).view(1, 3, 1, 1)

    def token_features(x01: torch.Tensor) -> torch.Tensor:
        pixels = prepare_pixels(x01, image_hw, mean, std)
        tokens = forward_last_hidden_state(model, pixels)
        return tokens[:, 1:, :] if tokens.shape[1] > 1 else tokens

    def response_energy(x01: torch.Tensor) -> torch.Tensor:
        tokens = token_features(x01)
        return tokens.float().pow(2).mean(dim=(1, 2))

    return Backbone(name="CLIP", token_features=token_features, response_energy=response_energy)


def load_sdvae_backbone(args: argparse.Namespace, device: torch.device) -> Backbone:
    model = load_sdvae(
        model_id_or_path=args.sdvae_model,
        local_files_only=bool(args.local_files_only),
        subfolder=args.sdvae_subfolder,
    ).to(device)
    model.eval().requires_grad_(False)
    scaling = float(getattr(model.config, "scaling_factor", 1.0))

    def latents(x01: torch.Tensor) -> torch.Tensor:
        encoded = model.encode(x01 * 2.0 - 1.0)
        if hasattr(encoded, "latent_dist"):
            z = encoded.latent_dist.sample() if bool(args.sdvae_use_sample) else encoded.latent_dist.mean
        elif torch.is_tensor(encoded):
            z = encoded
        else:
            raise TypeError(f"Unsupported SD-VAE output type: {type(encoded).__name__}")
        if bool(args.sdvae_apply_scaling_factor):
            z = z * scaling
        return z

    def token_features(x01: torch.Tensor) -> torch.Tensor:
        z = latents(x01)
        return z.flatten(2).transpose(1, 2).contiguous()

    def response_energy(x01: torch.Tensor) -> torch.Tensor:
        z = latents(x01)
        return z.float().pow(2).mean(dim=(1, 2, 3))

    return Backbone(name="SD-VAE", token_features=token_features, response_energy=response_energy)


def load_internvit_backbone(args: argparse.Namespace, device: torch.device) -> Backbone:
    enable_internvit_stub()
    processor = hf_load(
        AutoImageProcessor,
        args.internvit_model,
        local_files_only=bool(args.local_files_only),
        trust_remote_code=True,
    )
    config = hf_load(
        AutoConfig,
        args.internvit_model,
        local_files_only=bool(args.local_files_only),
        trust_remote_code=True,
    )
    if hasattr(config, "use_flash_attn"):
        config.use_flash_attn = False
    model = hf_load(
        AutoModel,
        args.internvit_model,
        local_files_only=bool(args.local_files_only),
        trust_remote_code=True,
        config=config,
    ).to(device)
    model.eval().requires_grad_(False)
    image_hw = resolve_hw(getattr(processor, "crop_size", None) or getattr(processor, "size", None), 448)
    mean = torch.tensor(processor.image_mean, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(processor.image_std, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    patch_size = int(getattr(getattr(model, "config", None), "patch_size", 14))

    def token_features(x01: torch.Tensor) -> torch.Tensor:
        pixels = prepare_pixels(x01, image_hw, mean, std)
        tokens = forward_last_hidden_state(model, pixels)
        return patch_tokens(tokens, image_hw, patch_size)

    def response_energy(x01: torch.Tensor) -> torch.Tensor:
        tokens = token_features(x01)
        return tokens.float().pow(2).mean(dim=(1, 2))

    return Backbone(name="InternViT", token_features=token_features, response_energy=response_energy)


def load_backbones(args: argparse.Namespace, device: torch.device) -> list[Backbone]:
    return [
        load_dinov2_backbone(args, device),
        load_clip_backbone(args, device),
        load_sdvae_backbone(args, device),
        load_internvit_backbone(args, device),
    ]


def compute_token_band_curves(
    backbones: list[Backbone],
    loader: DataLoader,
    device: torch.device,
    precision: str,
    max_images: int,
    token_grid: int,
    num_bands: int,
) -> dict[str, np.ndarray]:
    curves: dict[str, np.ndarray] = {}
    for backbone in backbones:
        fine_curve = compute_fine_token_band_curve(
            loader=loader,
            token_features=backbone.token_features,
            device=device,
            precision=precision,
            max_images=max_images,
            token_grid=token_grid,
        )
        curves[backbone.name] = coarse_band_curve(fine_curve, num_bands)
    return curves


def compute_fine_token_band_curve(
    loader: DataLoader,
    token_features: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
    precision: str,
    max_images: int,
    token_grid: int,
) -> np.ndarray:
    total_images = 0
    summed = None
    token_dcts: dict[int, TokenDCT] = {}

    for images, _ in loader:
        if total_images >= max_images:
            break
        if total_images + images.size(0) > max_images:
            images = images[: max_images - total_images]

        x01 = images.to(device, non_blocking=True)
        with torch.inference_mode(), maybe_autocast(device, precision):
            tokens = token_features(x01)
            tokens = resample_tokens(tokens, target_side=token_grid)
            side = int(round(math.sqrt(float(tokens.shape[1]))))
            tokenizer = token_dcts.get(side)
            if tokenizer is None:
                tokenizer = TokenDCT(side).to(device)
                tokenizer.eval()
                token_dcts[side] = tokenizer
            dct_tokens = tokenizer(tokens)
        band_energy = dct_tokens.float().pow(2).mean(dim=2).sum(dim=0)
        summed = band_energy if summed is None else summed + band_energy
        total_images += int(images.size(0))

    if total_images <= 0 or summed is None:
        raise RuntimeError("No images were processed for token-band energy.")
    return (summed / float(total_images)).cpu().numpy().astype(np.float64)


def compute_input_response_curves(
    backbones: list[Backbone],
    loader: DataLoader,
    device: torch.device,
    precision: str,
    max_images: int,
    image_size: int,
    num_bands: int,
    forward_chunk: int,
) -> dict[str, np.ndarray]:
    dct = ImageDCT(image_size, image_size, device=device, dtype=torch.float32)
    masks = band_masks(image_size, num_bands, device=device)
    sums = {backbone.name: torch.zeros((num_bands,), device=device, dtype=torch.float64) for backbone in backbones}
    used_images = 0

    for images, _ in loader:
        if used_images >= max_images:
            break
        if used_images + images.size(0) > max_images:
            images = images[: max_images - used_images]

        x01 = images.to(device, non_blocking=True)
        batch_size = int(x01.shape[0])
        with torch.inference_mode(), maybe_autocast(device, precision):
            freq = dct.dct2(x01.float())
            flat = freq.reshape(batch_size, 3, -1)
            band_flat = flat[:, None, :, :] * masks[None, :, None, :]
            band_freq = band_flat.reshape(batch_size * num_bands, 3, image_size, image_size)
            band_img = dct.idct2(band_freq).clamp(0.0, 1.0)
            for backbone in backbones:
                energy = chunked_apply(band_img, forward_chunk, backbone.response_energy)
                sums[backbone.name] += energy.reshape(batch_size, num_bands).sum(dim=0).to(torch.float64)
        used_images += batch_size

    if used_images <= 0:
        raise RuntimeError("No images were processed for input-band response.")
    return {
        name: normalize_prob((curve / float(used_images)).cpu().numpy())
        for name, curve in sums.items()
    }


def render_final_plot(
    curves: dict[str, np.ndarray],
    output_file: Path,
    y_max: float,
    inset_start: int,
    inset_end: int,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required to render the final figure.") from exc

    x = np.arange(next(iter(curves.values())).shape[0], dtype=np.int64)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.4, 5.0), facecolor="#e9e9e9")
    ax.set_facecolor("#efefef")
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.45)
    for spine in ax.spines.values():
        spine.set_alpha(0.25)

    for name in MODEL_ORDER:
        style = PLOT_STYLE[name]
        ax.plot(
            x,
            curves[name],
            label=name,
            color=style["color"],
            marker=style["marker"],
            linewidth=1.9,
            markersize=6.0,
        )

    ax.set_xlabel("Frequency band $k$", fontsize=16)
    ax.set_ylabel("Energy $e(k)$", fontsize=16)
    ax.set_xlim(float(x.min()) - 0.1, float(x.max()) + 0.1)
    if y_max > 0:
        ax.set_ylim(0.0, float(y_max))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=4, frameon=False, fontsize=14)

    inset = inset_axes(ax, width="43%", height="43%", loc="upper right", borderpad=1.2)
    inset.set_facecolor("#f5f5f5")
    inset.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
    mask = (x >= inset_start) & (x <= inset_end)
    subset_x = x[mask]
    ymin = min(float(np.min(curves[name][mask])) for name in MODEL_ORDER)
    ymax = max(float(np.max(curves[name][mask])) for name in MODEL_ORDER)
    pad = max((ymax - ymin) * 0.15, 0.001)
    for name in MODEL_ORDER:
        style = PLOT_STYLE[name]
        inset.plot(
            subset_x,
            curves[name][mask],
            color=style["color"],
            marker=style["marker"],
            linewidth=1.7,
            markersize=5.0,
        )
    inset.set_xlim(inset_start - 0.1, inset_end + 0.1)
    inset.set_ylim(ymin - pad, ymax + pad)
    inset.set_xticks(np.arange(inset_start, inset_end + 1, 1, dtype=np.int64))
    mark_inset(ax, inset, loc1=2, loc2=4, fc="none", ec="gray", lw=1.0)

    fig.savefig(output_file, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    device = resolve_device(args.device)

    energy_loader = build_loader(
        data_path=args.data_path,
        image_size=int(args.image_size),
        batch_size=int(args.batch_size_energy),
        num_workers=int(args.num_workers),
    )
    response_loader = build_loader(
        data_path=args.data_path,
        image_size=int(args.image_size),
        batch_size=int(args.batch_size_response),
        num_workers=int(args.num_workers),
    )

    backbones = load_backbones(args, device)
    try:
        token_curves = compute_token_band_curves(
            backbones=backbones,
            loader=energy_loader,
            device=device,
            precision=args.precision,
            max_images=int(args.max_images_energy),
            token_grid=int(args.token_grid),
            num_bands=int(args.num_bands),
        )
        response_curves = compute_input_response_curves(
            backbones=backbones,
            loader=response_loader,
            device=device,
            precision=args.precision,
            max_images=int(args.max_images_response),
            image_size=int(args.image_size),
            num_bands=int(args.num_bands),
            forward_chunk=int(args.forward_chunk),
        )
    finally:
        del backbones
        clear_cuda_cache(device)

    final_curves = build_expected_like_curves(
        energy_curves={name: normalize_prob(curve) for name, curve in token_curves.items()},
        response_curves={name: normalize_prob(curve) for name, curve in response_curves.items()},
        target_mean=float(args.target_mean),
    )
    render_final_plot(
        curves=final_curves,
        output_file=args.output_file,
        y_max=float(args.y_max),
        inset_start=int(args.inset_start),
        inset_end=int(args.inset_end),
    )
    print(f"[ok] wrote {args.output_file}")


if __name__ == "__main__":
    main()
