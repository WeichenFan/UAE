#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import random
import re
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights
from transformers import AutoProcessor, CLIPModel, SiglipModel

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DEFAULT_OUTPUT = Path("analysis/output/retrival_exp/output.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "T2I retrival experiments."
        )
    )
    parser.add_argument("--data-path", type=Path, default=Path("data/imagenet/val"))
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-images", type=int, default=0, help="0 means use all images.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--text-template", type=str, default="the image of {}")
    parser.add_argument("--class-names-file", type=Path, default=None)
    parser.add_argument(
        "--cutoffs",
        type=str,
        default="0.05,0.10,0.15,0.20,0.30,0.40,0.60,0.80,1.00",
    )
    parser.add_argument(
        "--retrieval-filter-shape",
        type=str,
        default="radial",
        choices=["square", "radial", "zigzag"],
    )
    parser.add_argument("--retrieval-cutoff-power", type=float, default=1.0)
    parser.add_argument(
        "--retrieval-radial-norm",
        type=str,
        default="axis",
        choices=["axis", "corner"],
    )
    parser.add_argument("--clip-model", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--siglip2-model", type=str, default="google/siglip2-base-patch16-224")
    parser.add_argument("--topk", type=int, default=1)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_model_path(model_id_or_path: str) -> str:
    path = Path(model_id_or_path)
    if path.exists():
        return str(path)
    mirror = Path("hf_models") / model_id_or_path
    if mirror.exists():
        return str(mirror)
    return model_id_or_path


def hf_load(factory, model_id_or_path: str, local_files_only: bool, **kwargs):
    resolved = resolve_model_path(model_id_or_path)
    if local_files_only:
        return factory.from_pretrained(resolved, local_files_only=True, **kwargs)
    try:
        return factory.from_pretrained(resolved, local_files_only=True, **kwargs)
    except (OSError, ValueError, AttributeError, TypeError):
        return factory.from_pretrained(resolved, local_files_only=False, **kwargs)


def center_crop_arr(image: Image.Image, image_size: int) -> Image.Image:
    while min(*image.size) >= 2 * image_size:
        image = image.resize(tuple(side // 2 for side in image.size), resample=Image.BOX)
    scale = image_size / min(*image.size)
    image = image.resize(tuple(round(side * scale) for side in image.size), resample=Image.BICUBIC)
    array = np.asarray(image)
    top = (array.shape[0] - image_size) // 2
    left = (array.shape[1] - image_size) // 2
    return Image.fromarray(array[top : top + image_size, left : left + image_size])


class ImageFolderDataset(Dataset):
    def __init__(self, root: Path, image_size: int):
        self.root = Path(root)
        self.image_size = int(image_size)
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.class_names = sorted(path.name for path in self.root.iterdir() if path.is_dir())

        if self.class_names:
            class_to_index = {name: idx for idx, name in enumerate(self.class_names)}
            samples: list[tuple[Path, int]] = []
            for class_name in self.class_names:
                class_dir = self.root / class_name
                for path in class_dir.rglob("*"):
                    if path.is_file() and path.suffix.lower() in exts:
                        samples.append((path, class_to_index[class_name]))
            self.samples = sorted(samples, key=lambda item: str(item[0]))
        else:
            paths = sorted(path for path in self.root.rglob("*") if path.is_file() and path.suffix.lower() in exts)
            self.class_names = ["unknown"]
            self.samples = [(path, 0) for path in paths]

        if not self.samples:
            raise FileNotFoundError(f"No image files found under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        image = center_crop_arr(image, self.image_size)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        return tensor, int(label)


def build_loader(
    data_path: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, list[str]]:
    dataset = ImageFolderDataset(data_path, image_size=image_size)
    generator = torch.Generator()
    generator.manual_seed(0)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=bool(num_workers > 0),
        generator=generator,
    )
    return loader, dataset.class_names


def resolve_hw(size_cfg, fallback: int) -> tuple[int, int]:
    if isinstance(size_cfg, dict):
        if "height" in size_cfg and "width" in size_cfg:
            return int(size_cfg["height"]), int(size_cfg["width"])
        if "shortest_edge" in size_cfg:
            edge = int(size_cfg["shortest_edge"])
            return edge, edge
    if isinstance(size_cfg, int):
        return int(size_cfg), int(size_cfg)
    return int(fallback), int(fallback)


def prepare_pixels(
    images_01: torch.Tensor,
    target_hw: tuple[int, int],
    image_mean: torch.Tensor,
    image_std: torch.Tensor,
) -> torch.Tensor:
    if images_01.shape[-2:] != target_hw:
        images_01 = F.interpolate(images_01, size=target_hw, mode="bicubic", align_corners=False)
    return (images_01 - image_mean) / image_std


def build_zigzag_perm_top_left(h: int, w: int, device: torch.device) -> torch.Tensor:
    coords = []
    for diagonal in range(h + w - 1):
        if diagonal % 2 == 0:
            x = min(diagonal, w - 1)
            y = diagonal - x
            while x >= 0 and y < h:
                coords.append((y, x))
                x -= 1
                y += 1
        else:
            y = min(diagonal, h - 1)
            x = diagonal - y
            while y >= 0 and x < w:
                coords.append((y, x))
                y -= 1
                x += 1
    linear = [y * w + x for y, x in coords]
    return torch.tensor(linear, device=device, dtype=torch.long)


class ImageDCT:
    def __init__(self, h: int, w: int, device: torch.device):
        self.h = int(h)
        self.w = int(w)
        self.device = device
        self.mat_h = self._build_dct_matrix(self.h, device, torch.float32)
        self.mat_w = self._build_dct_matrix(self.w, device, torch.float32)
        yy, xx = torch.meshgrid(
            torch.arange(self.h, device=device, dtype=torch.float32),
            torch.arange(self.w, device=device, dtype=torch.float32),
            indexing="ij",
        )
        self.radius = torch.sqrt(yy.square() + xx.square())
        self.max_radius = float(self.radius.max().item())
        self.max_axis_radius = float(min(self.h - 1, self.w - 1))
        self.zigzag_perm = build_zigzag_perm_top_left(self.h, self.w, device)
        self.mask_cache: dict[tuple, torch.Tensor] = {}

    def _build_dct_matrix(self, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        k = torch.arange(n, device=device, dtype=dtype).view(n, 1)
        n_idx = torch.arange(n, device=device, dtype=dtype).view(1, n)
        mat = torch.cos((torch.pi / n) * (n_idx + 0.5) * k)
        mat[0] *= 1.0 / math.sqrt(n)
        if n > 1:
            mat[1:] *= math.sqrt(2.0 / n)
        return mat

    def dct2(self, images: torch.Tensor) -> torch.Tensor:
        _, _, h, w = images.shape
        if (h, w) != (self.h, self.w):
            raise ValueError(f"Expected image size {(self.h, self.w)}, got {(h, w)}")
        flat = images.reshape(-1, h, w)
        flat = torch.matmul(self.mat_h, flat)
        flat = torch.matmul(flat, self.mat_w.t())
        return flat.reshape_as(images)

    def idct2(self, coeffs: torch.Tensor) -> torch.Tensor:
        _, _, h, w = coeffs.shape
        if (h, w) != (self.h, self.w):
            raise ValueError(f"Expected image size {(self.h, self.w)}, got {(h, w)}")
        flat = coeffs.reshape(-1, h, w)
        flat = torch.matmul(self.mat_h.t(), flat)
        flat = torch.matmul(flat, self.mat_w)
        return flat.reshape_as(coeffs)

    def _square_mask(self, cutoff: float) -> torch.Tensor:
        if cutoff <= 0.0:
            return torch.zeros((self.h, self.w), device=self.device, dtype=torch.float32)
        if cutoff >= 1.0:
            return torch.ones((self.h, self.w), device=self.device, dtype=torch.float32)
        side = int(round(cutoff * float(min(self.h, self.w))))
        side = max(1, min(side, min(self.h, self.w)))
        mask = torch.zeros((self.h, self.w), device=self.device, dtype=torch.float32)
        mask[:side, :side] = 1.0
        return mask

    def _zigzag_mask(self, cutoff: float) -> torch.Tensor:
        if cutoff <= 0.0:
            return torch.zeros((self.h, self.w), device=self.device, dtype=torch.float32)
        if cutoff >= 1.0:
            return torch.ones((self.h, self.w), device=self.device, dtype=torch.float32)
        total = self.h * self.w
        keep = int(round(cutoff * float(total)))
        keep = max(1, min(keep, total))
        flat = torch.zeros((total,), device=self.device, dtype=torch.float32)
        flat[self.zigzag_perm[:keep]] = 1.0
        return flat.view(self.h, self.w)

    def _radial_mask(self, cutoff: float, radial_norm: str) -> torch.Tensor:
        threshold = cutoff * (self.max_axis_radius if radial_norm == "axis" else self.max_radius)
        return (self.radius <= threshold).to(torch.float32)

    def mask(
        self,
        cutoff: float,
        shape: str,
        cutoff_power: float,
        radial_norm: str,
    ) -> torch.Tensor:
        cutoff = float(max(0.0, min(1.0, cutoff)))
        cutoff_eff = float(cutoff ** float(max(cutoff_power, 1e-8)))
        key = (shape, cutoff, cutoff_eff, radial_norm)
        cached = self.mask_cache.get(key)
        if cached is not None:
            return cached

        if shape == "square":
            mask = self._square_mask(cutoff_eff)
        elif shape == "zigzag":
            mask = self._zigzag_mask(cutoff_eff)
        elif shape == "radial":
            mask = self._radial_mask(cutoff_eff, radial_norm)
        else:
            raise ValueError(f"Unknown filter shape: {shape}")

        mask = mask.view(1, 1, self.h, self.w)
        self.mask_cache[key] = mask
        return mask

    def lowpass(
        self,
        images_01: torch.Tensor,
        cutoff: float,
        shape: str,
        cutoff_power: float,
        radial_norm: str,
    ) -> torch.Tensor:
        coeffs = self.dct2(images_01.float())
        mask = self.mask(cutoff, shape, cutoff_power, radial_norm)
        return self.idct2(coeffs * mask).clamp(0.0, 1.0)


def load_class_names(class_ids: list[str], class_names_file: Path | None) -> list[str]:
    if class_names_file is not None:
        names = [line.strip() for line in class_names_file.read_text().splitlines() if line.strip()]
        if len(names) != len(class_ids):
            raise ValueError(
                f"class-names-file has {len(names)} lines, but dataset has {len(class_ids)} classes"
            )
        return names

    if len(class_ids) == 1000 and all(re.fullmatch(r"n\d{8}", class_id) for class_id in class_ids):
        try:
            return list(ResNet50_Weights.IMAGENET1K_V1.meta["categories"])
        except Exception:
            pass
    return [class_id.replace("_", " ") for class_id in class_ids]


def collect_query_class_indices(loader: DataLoader, max_images: int) -> np.ndarray:
    labels = []
    seen = 0
    for _, batch_labels in loader:
        if max_images > 0 and seen >= max_images:
            break
        if max_images > 0 and seen + batch_labels.shape[0] > max_images:
            batch_labels = batch_labels[: max_images - seen]
        labels.append(batch_labels.numpy())
        seen += int(batch_labels.shape[0])
    if not labels:
        raise RuntimeError("No labels found in dataset")
    return np.unique(np.concatenate(labels, axis=0)).astype(np.int64)


def build_text_features(
    processor: AutoProcessor,
    model,
    class_names: list[str],
    text_template: str,
    device: torch.device,
    padding,
) -> np.ndarray:
    prompts = [text_template.format(name) for name in class_names]
    encoded = processor(text=prompts, padding=padding, truncation=True, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items() if torch.is_tensor(value)}
    with torch.no_grad():
        features = model.get_text_features(**encoded)
    return F.normalize(features.float(), dim=-1).cpu().numpy()


def recall_at_k(
    query_features: np.ndarray,
    query_class_indices: np.ndarray,
    image_features: np.ndarray,
    image_labels: np.ndarray,
    topk: int,
) -> float:
    similarity = query_features @ image_features.T
    if topk >= similarity.shape[1]:
        topk_idx = np.argsort(similarity, axis=1)[:, ::-1]
    else:
        part = np.argpartition(similarity, kth=similarity.shape[1] - topk, axis=1)[:, -topk:]
        part_scores = np.take_along_axis(similarity, part, axis=1)
        order = np.argsort(part_scores, axis=1)[:, ::-1]
        topk_idx = np.take_along_axis(part, order, axis=1)
    hits = []
    for row_index, class_index in enumerate(query_class_indices):
        labels = image_labels[topk_idx[row_index]]
        hits.append(float(np.any(labels == int(class_index))))
    return float(np.mean(hits))


def maybe_autocast(device: torch.device, precision: str):
    if device.type == "cuda" and precision == "bf16":
        return torch.cuda.amp.autocast(dtype=torch.bfloat16)
    return nullcontext()


@torch.no_grad()
def compute_image_features(
    loader: DataLoader,
    model,
    image_hw: tuple[int, int],
    image_mean: torch.Tensor,
    image_std: torch.Tensor,
    dct: ImageDCT,
    cutoff: float,
    filter_shape: str,
    cutoff_power: float,
    radial_norm: str,
    device: torch.device,
    precision: str,
    max_images: int,
) -> tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []
    seen = 0
    amp_context = maybe_autocast(device, precision)

    for images, batch_labels in loader:
        if max_images > 0 and seen >= max_images:
            break
        if max_images > 0 and seen + images.shape[0] > max_images:
            keep = max_images - seen
            images = images[:keep]
            batch_labels = batch_labels[:keep]

        images_01 = images.to(device, non_blocking=True)
        images_01 = dct.lowpass(
            images_01,
            cutoff=cutoff,
            shape=filter_shape,
            cutoff_power=cutoff_power,
            radial_norm=radial_norm,
        )
        pixel_values = prepare_pixels(images_01, image_hw, image_mean, image_std)
        with amp_context:
            batch_features = model.get_image_features(pixel_values=pixel_values)
        batch_features = F.normalize(batch_features.float(), dim=-1)
        features.append(batch_features.cpu().numpy())
        labels.append(batch_labels.numpy())
        seen += int(images.shape[0])

    if not features:
        raise RuntimeError("No image features were computed")
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def plot_curves(
    output_file: Path,
    cutoffs: list[float],
    curves: dict[str, list[float]],
    backbones: list["Backbone"],
    topk: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.2, 5.6), facecolor="#e9e9e9")
    ax.set_facecolor("#efefef")
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.45)
    for spine in ax.spines.values():
        spine.set_alpha(0.2)

    x = np.asarray(cutoffs, dtype=np.float64)
    for spec in backbones:
        y = np.asarray(curves[spec.key], dtype=np.float64)
        ax.plot(
            x,
            y,
            marker=spec.marker,
            markersize=5.0,
            linewidth=1.8,
            color=spec.color,
            label=spec.label,
        )

    ymax = max(float(np.max(curves[spec.key])) for spec in backbones)
    ax.set_xlabel("Low-pass cutoff (fraction of Nyquist)", fontsize=14)
    ax.set_ylabel(f"T2I Recall@{int(topk)}", fontsize=14)
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, max(0.42, ymax + 0.04))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=len(backbones), frameon=True, fontsize=12)
    fig.savefig(output_file, dpi=220, bbox_inches="tight")
    plt.close(fig)


@dataclass(frozen=True)
class Backbone:
    key: str
    label: str
    marker: str
    color: str
    model_id: str
    padding: str | bool
    model_cls: type


@dataclass
class LoadedBackbone:
    spec: Backbone
    model: torch.nn.Module
    image_hw: tuple[int, int]
    image_mean: torch.Tensor
    image_std: torch.Tensor
    query_features: np.ndarray


def load_backbone(
    spec: Backbone,
    class_names: list[str],
    text_template: str,
    device: torch.device,
    local_files_only: bool,
) -> LoadedBackbone:
    processor = hf_load(AutoProcessor, spec.model_id, local_files_only=local_files_only)
    model = hf_load(spec.model_cls, spec.model_id, local_files_only=local_files_only).to(device)
    model.eval().requires_grad_(False)

    image_processor = processor.image_processor
    image_hw = resolve_hw(
        getattr(image_processor, "crop_size", None) or getattr(image_processor, "size", None),
        224,
    )
    image_mean = torch.tensor(image_processor.image_mean, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    image_std = torch.tensor(image_processor.image_std, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    query_features = build_text_features(
        processor=processor,
        model=model,
        class_names=class_names,
        text_template=text_template,
        device=device,
        padding=spec.padding,
    )
    return LoadedBackbone(
        spec=spec,
        model=model,
        image_hw=image_hw,
        image_mean=image_mean,
        image_std=image_std,
        query_features=query_features,
    )


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    output_file = args.output_file
    output_file.parent.mkdir(parents=True, exist_ok=True)

    cutoffs = [float(value.strip()) for value in str(args.cutoffs).split(",") if value.strip()]
    if not cutoffs:
        raise ValueError("--cutoffs is empty")
    if int(args.topk) <= 0:
        raise ValueError("--topk must be positive")

    device_name = args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    loader, class_ids = build_loader(
        data_path=args.data_path,
        image_size=int(args.image_size),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
    )
    effective_max_images = int(args.max_images) if int(args.max_images) > 0 else int(len(loader.dataset))
    query_class_indices = collect_query_class_indices(loader, effective_max_images)
    class_names = load_class_names(class_ids, args.class_names_file)
    dct = ImageDCT(h=int(args.image_size), w=int(args.image_size), device=device)

    backbones = [
        Backbone(
            key="clip",
            label="CLIP",
            marker="s",
            color="#8a9a00",
            model_id=str(args.clip_model),
            padding=True,
            model_cls=CLIPModel,
        ),
        Backbone(
            key="siglip2",
            label="SigLIP2",
            marker="v",
            color="#1f78b4",
            model_id=str(args.siglip2_model),
            padding="max_length",
            model_cls=SiglipModel,
        ),
    ]

    loaded_backbones = [
        load_backbone(
            spec=spec,
            class_names=class_names,
            text_template=str(args.text_template),
            device=device,
            local_files_only=bool(args.local_files_only),
        )
        for spec in backbones
    ]

    curves = {spec.key: [] for spec in backbones}
    for cutoff in cutoffs:
        for loaded in loaded_backbones:
            image_features, image_labels = compute_image_features(
                loader=loader,
                model=loaded.model,
                image_hw=loaded.image_hw,
                image_mean=loaded.image_mean,
                image_std=loaded.image_std,
                dct=dct,
                cutoff=float(cutoff),
                filter_shape=str(args.retrieval_filter_shape),
                cutoff_power=float(args.retrieval_cutoff_power),
                radial_norm=str(args.retrieval_radial_norm),
                device=device,
                precision=str(args.precision),
                max_images=effective_max_images,
            )
            score = recall_at_k(
                query_features=loaded.query_features[query_class_indices],
                query_class_indices=query_class_indices,
                image_features=image_features,
                image_labels=image_labels,
                topk=int(args.topk),
            )
            curves[loaded.spec.key].append(score)

    for loaded in loaded_backbones:
        del loaded.model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    plot_curves(
        output_file=output_file,
        cutoffs=cutoffs,
        curves=curves,
        backbones=backbones,
        topk=int(args.topk),
    )
    print(f"[ok] wrote {output_file}")


if __name__ == "__main__":
    main()
