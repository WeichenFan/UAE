from .ref_iqa import calculate_psnr, calculate_lpips, calculate_ssim
from .fid import calculate_rfid, calculate_gfid
import inspect
import numpy as np
import torch
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from typing import Callable, Dict, Optional
import os
import sys
from utils.io_utils import safe_np_savez, is_disk_full_error


def _forward_reconstruction_eval(model, images: torch.Tensor) -> torch.Tensor:
    """
    Reconstruction eval should not apply stochastic/structured band masking.
    If the model supports `apply_band_mask`, force it off.
    """
    try:
        signature = inspect.signature(model.forward)
        if "apply_band_mask" in signature.parameters:
            return model(images, apply_band_mask=False)
    except (TypeError, ValueError, AttributeError):
        pass
    return model(images)


def _resize_reference_array_like(
    reference: np.ndarray,
    reconstruction: np.ndarray,
    *,
    chunk_size: int = 512,
) -> np.ndarray:
    if reference.shape[1:3] == reconstruction.shape[1:3]:
        return reference
    target_hw = reconstruction.shape[1:3]
    resized_chunks = []
    for start in range(0, reference.shape[0], chunk_size):
        end = min(start + chunk_size, reference.shape[0])
        chunk = torch.from_numpy(reference[start:end]).permute(0, 3, 1, 2).float() / 255.0
        chunk = F.interpolate(chunk, size=target_hw, mode="bilinear", align_corners=False)
        chunk = (chunk.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
        resized_chunks.append(chunk.permute(0, 2, 3, 1).cpu().numpy())
    return np.concatenate(resized_chunks, axis=0)


def compute_reconstruction_metrics(
    ref_arr: np.ndarray,
    rec_arr: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
    metrics_to_compute = ("psnr", "ssim", "rfid"),
    disable_bar: bool = True
) -> Dict[str, float]:
    """
    Compute reconstruction metrics between reference and reconstructed images.

    Args:
        ref_arr: Reference images [N, H, W, C] uint8
        rec_arr: Reconstructed images [N, H, W, C] uint8
        device: Device for computation
        batch_size: Batch size for metric computation

    Returns:
        Dictionary with metrics: eval/psnr, eval/ssim, eval/rfid
        Note: LPIPS is not computed here since it's already tracked during training
    """
    ref_arr = _resize_reference_array_like(ref_arr, rec_arr)
    device_str = "cuda" if device.type == "cuda" else "cpu"
    results_dict = {}
    if 'psnr' in metrics_to_compute:
        psnr = calculate_psnr(ref_arr, rec_arr, batch_size, device_str, disable_bar=disable_bar)
        results_dict["psnr"] = psnr
    if 'ssim' in metrics_to_compute:
        ssim = calculate_ssim(ref_arr, rec_arr, batch_size, device_str, disable_bar=disable_bar)
        results_dict["ssim"] = ssim
    if 'rfid' in metrics_to_compute:
        rfid = calculate_rfid(ref_arr, rec_arr, batch_size, device_str)
        results_dict["rfid"] = rfid
    assert len(results_dict) > 0, "No metrics were computed."
    return results_dict
def compute_generation_metrics(
    ref_arr: np.ndarray,
    rec_arr: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
):
    device_str = "cuda" if device.type == "cuda" else "cpu"
    # only eval FID
    fid = calculate_gfid(rec_arr, ref_arr, batch_size, device_str) 
    return {
        'fid': fid
    }


def _load_npz_array(
    npz_path: str,
    preferred_key: Optional[str] = None,
) -> tuple[np.ndarray, str]:
    """
    Load an image array from an NPZ file.

    Selection order:
    1. preferred_key (if provided)
    2. common defaults: arr_0, images, data, x
    3. the only key if there is exactly one key
    """
    npz = np.load(npz_path)
    try:
        keys = list(npz.files)
        if len(keys) == 0:
            raise KeyError(f"{npz_path} has no arrays.")
        if preferred_key is not None:
            if preferred_key not in keys:
                raise KeyError(
                    f"{preferred_key} is not a file in the archive. Available keys: {keys}"
                )
            return npz[preferred_key], preferred_key

        for key in ("arr_0", "images", "data", "x"):
            if key in keys:
                return npz[key], key
        if len(keys) == 1:
            only_key = keys[0]
            return npz[only_key], only_key

        raise KeyError(
            f"Could not infer image key from {npz_path}. "
            f"Available keys: {keys}. Set eval.reference_npz_key in config."
        )
    finally:
        npz.close()


def _sync_success(local_ok: bool, device: torch.device) -> bool:
    ok_tensor = torch.tensor(1 if local_ok else 0, device=device, dtype=torch.int32)
    dist.all_reduce(ok_tensor, op=dist.ReduceOp.MIN)
    return bool(ok_tensor.item())


@torch.no_grad()
def evaluate_generation_distributed(
    model_fn,
    sample_fn,
    latent_size, # for noise 
    additional_model_kwargs,
    use_guidance: bool,
    uae, 
    val_dataset,
    num_samples: int,
    batch_size: int,
    rank: int,
    world_size: int,
    device: torch.device,
    experiment_dir: str,
    global_step: int,
    autocast_kwargs: dict,
    metric_batch_size: int = 128,
    reference_npz_path: Optional[str] = None,
    latent_postprocess_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> Optional[Dict[str, float]]:
    """
    Evaluate reconstruction metrics using all GPUs in a distributed manner.

    Args:
        val_dataset: Validation dataset
        batch_size: Batch size per GPU for reconstruction
        rank: Current GPU rank
        world_size: Total number of GPUs
        device: Device to use
        experiment_dir: Experiment directory
        global_step: Current training step
        autocast_kwargs: Autocast configuration
        metric_batch_size: Batch size for metric computation (on rank 0)
        reference_npz_path: Optional path to existing reference NPZ file

    Returns:
        Dictionary of metrics (only on rank 0, None on other ranks)
    """
    # model.eval()
    # Save shard NPZ
    temp_dir = os.path.join(experiment_dir, "eval_npzs")
    temp_dir_ready = True
    if rank == 0:
        print(f"\n[Eval] Starting distributed sampling evaluation at step {global_step}")
        try:
            os.makedirs(temp_dir, exist_ok=True)
        except Exception as exc:
            if is_disk_full_error(exc):
                temp_dir_ready = False
            else:
                raise
    if not _sync_success(temp_dir_ready, device):
        if rank == 0:
            print(
                f"[Eval] Skipping generation evaluation at step {global_step}: "
                "disk full while creating eval directory."
            )
        dist.barrier()
        return None

    # Wait for rank 0 to create the directory before other ranks try to save
    dist.barrier()
    # print(f"[Rank {rank}] Starting sampling...")
    # Each rank processes its shard
    N = min(len(val_dataset), num_samples)
    chunk = N // world_size

    if rank < world_size - 1:
        start = rank * chunk
        end   = (rank + 1) * chunk
    else:
        # Last rank takes the remainder (and handles N < world_size gracefully)
        start = rank * chunk
        end   = N

    rank_indices = list(range(start, end))
    subset = Subset(val_dataset, rank_indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    # Reconstruct images on this rank
    generations = []
    iterator = tqdm(loader, desc=f"[Rank {rank}] Sampling", file=sys.stdout) if rank == 0 else loader

    with torch.inference_mode():
        for _, label in iterator: # don't actually need images at sampling time
            n = label.size(0)
            z = torch.randn(n, *latent_size, device = device)
            y = label.to(device)
            if use_guidance:
                z = torch.cat([z, z], dim=0)
                y_null = torch.full((n,), null_label, device=device)
                y = torch.cat([y, y_null], dim=0)
            model_kwargs = dict(y=y, **additional_model_kwargs)
            with autocast(**autocast_kwargs):
                samples = sample_fn(z, model_fn, **model_kwargs)[-1]
                if use_guidance:
                    samples, _ = samples.chunk(2, dim = 0)
                if latent_postprocess_fn is not None:
                    samples = latent_postprocess_fn(samples)
                samples = uae.decode(samples).clamp(0,1)
            gen_np = samples.mul(255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            for img in gen_np:
                generations.append(img)

    generations = np.stack(generations)
    shard_path = os.path.join(temp_dir, f"gen_{global_step:07d}_{rank:02d}.npz")
    shard_saved = safe_np_savez(
        shard_path,
        arr_0=generations,
        context=f"generation eval shard (step={global_step}, rank={rank})",
    )
    if not _sync_success(shard_saved, device):
        if rank == 0:
            print(
                f"[Eval] Skipping generation evaluation at step {global_step}: "
                "disk full while writing shard NPZ files."
            )
        dist.barrier()
        return None

    if rank == 0:
        print(f"[Rank {rank}] Saved {len(generations)} generation to {shard_path}")

    # Wait for all ranks to finish generation
    dist.barrier()

    # Rank 0 computes metrics
    metrics = None
    if rank == 0:
        # Combine all generation shards
        all_gens = []
        for r in range(world_size):
            shard_file = os.path.join(temp_dir, f"gen_{global_step:07d}_{r:02d}.npz")
            shard_data = np.load(shard_file)["arr_0"]
            all_gens.append(shard_data)

        combined_recons = np.concatenate(all_gens, axis=0)[:num_samples]
        print(f"[Eval] Combined generation NPZ shape: {combined_recons.shape}")

        # Load reference NPZ
        ref_npz_path = reference_npz_path

        if not os.path.exists(ref_npz_path):
            raise FileNotFoundError(f"Reference NPZ not found at {ref_npz_path}")

        ref_stats = np.load(ref_npz_path)
        print(f"[Eval] Loaded reference NPZ from {ref_npz_path}")

        # Compute metrics
        print("[Eval] Computing metrics...")
        metrics = compute_generation_metrics(
            ref_stats,
            combined_recons,
            device,
            metric_batch_size,
        )

        # Print results
        print(f"[Eval] Step {global_step} Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.6f}")

        # Cleanup reconstruction shards
        for r in range(world_size):
            shard_file = os.path.join(temp_dir, f"gen_{global_step:07d}_{r:02d}.npz")
            if os.path.exists(shard_file):
                os.remove(shard_file)

    dist.barrier()
    return metrics
@torch.no_grad()
def evaluate_reconstruction_distributed(
    model,
    val_dataset,
    num_samples: int,
    batch_size: int,
    rank: int,
    world_size: int,
    device: torch.device,
    experiment_dir: str,
    global_step: int,
    autocast_kwargs: dict,
    metric_batch_size: int = 128,
    reference_npz_path: Optional[str] = None,
    reference_npz_key: Optional[str] = None,
    metrics_to_compute: Optional[list] = ("psnr", "ssim", "rfid")
) -> Optional[Dict[str, float]]:
    """
    Evaluate reconstruction metrics using all GPUs in a distributed manner.

    Args:
        model: Model to evaluate (should be in eval mode)
        val_dataset: Validation dataset
        batch_size: Batch size per GPU for reconstruction
        rank: Current GPU rank
        world_size: Total number of GPUs
        device: Device to use
        experiment_dir: Experiment directory
        global_step: Current training step
        autocast_kwargs: Autocast configuration
        metric_batch_size: Batch size for metric computation (on rank 0)
        reference_npz_path: Optional path to existing reference NPZ file
        reference_npz_key: Optional key to read from reference NPZ

    Returns:
        Dictionary of metrics (only on rank 0, None on other ranks)
    """
    # model.eval()
    # Save shard NPZ
    temp_dir = os.path.join(experiment_dir, "eval_npzs")
    if rank == 0:
        print(f"\n[Eval] Starting distributed reconstruction evaluation at step {global_step}")
    # Make sure all ranks can write their shard files.
    temp_dir_ready = True
    try:
        os.makedirs(temp_dir, exist_ok=True)
    except Exception as exc:
        if is_disk_full_error(exc):
            temp_dir_ready = False
        else:
            raise
    if not _sync_success(temp_dir_ready, device):
        if rank == 0:
            print(
                f"[Eval] Skipping reconstruction evaluation at step {global_step}: "
                "disk full while creating eval directory."
            )
        dist.barrier()
        return None

    # Sync before reconstruction/saving.
    dist.barrier()
    # print(f"[Rank {rank}] Starting reconstruction...")
    # Each rank processes its shard
    N = min(len(val_dataset), num_samples)
    chunk = N // world_size

    if rank < world_size - 1:
        start = rank * chunk
        end   = (rank + 1) * chunk
    else:
        # Last rank takes the remainder (and handles N < world_size gracefully)
        start = rank * chunk
        end   = N

    rank_indices = list(range(start, end))
    subset = Subset(val_dataset, rank_indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    # Reconstruct images on this rank
    reconstructions = []
    iterator = tqdm(loader, desc=f"[Rank {rank}] Reconstructing", file=sys.stdout) if rank == 0 else loader

    with torch.inference_mode():
        for images, _ in iterator:
            images = images.to(device, non_blocking=True)
            with autocast(**autocast_kwargs):
                recon = _forward_reconstruction_eval(model, images)

            # Convert to numpy uint8 [H, W, C]
            recon = recon.clamp(0, 1)
            recon_np = recon.mul(255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

            for img in recon_np:
                reconstructions.append(img)

    reconstructions = np.stack(reconstructions)
    shard_path = os.path.join(temp_dir, f"recon_{global_step:07d}_{rank:02d}.npz")
    shard_saved = safe_np_savez(
        shard_path,
        arr_0=reconstructions,
        context=f"reconstruction eval shard (step={global_step}, rank={rank})",
    )
    if not _sync_success(shard_saved, device):
        if rank == 0:
            print(
                f"[Eval] Skipping reconstruction evaluation at step {global_step}: "
                "disk full while writing shard NPZ files."
            )
        dist.barrier()
        return None

    if rank == 0:
        print(f"[Rank {rank}] Saved {len(reconstructions)} reconstructions to {shard_path}")

    # Wait for all ranks to finish reconstruction
    dist.barrier()

    # Rank 0 computes metrics
    metrics = None
    if rank == 0:
        # Combine all reconstruction shards
        all_recons = []
        for r in range(world_size):
            shard_file = os.path.join(temp_dir, f"recon_{global_step:07d}_{r:02d}.npz")
            shard_data = np.load(shard_file)["arr_0"]
            all_recons.append(shard_data)

        combined_recons = np.concatenate(all_recons, axis=0)[:num_samples]
        print(f"[Eval] Combined reconstruction NPZ shape: {combined_recons.shape}")

        # Load reference NPZ
        ref_npz_path = reference_npz_path

        if not os.path.exists(ref_npz_path):
            raise FileNotFoundError(f"Reference NPZ not found at {ref_npz_path}")

        ref_images, used_key = _load_npz_array(ref_npz_path, preferred_key=reference_npz_key)
        print(
            f"[Eval] Loaded reference NPZ from {ref_npz_path} "
            f"(key={used_key}), shape: {ref_images.shape}"
        )

        # Compute metrics
        print("[Eval] Computing metrics...")
        metrics = compute_reconstruction_metrics(
            ref_images,
            combined_recons,
            device,
            metric_batch_size,
            metrics_to_compute=metrics_to_compute,
            disable_bar= True, # by default no bar
        )

        # Print results
        print(f"[Eval] Step {global_step} Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.6f}")

        # Cleanup reconstruction shards
        for r in range(world_size):
            shard_file = os.path.join(temp_dir, f"recon_{global_step:07d}_{r:02d}.npz")
            if os.path.exists(shard_file):
                os.remove(shard_file)

    dist.barrier()
    # model.train()

    return metrics
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-img", type=str, default="samples/imagenet-256-val.npz")
    parser.add_argument("--rec-img", type=str, default="samples/sdvae-ft-mse-f8d4.npz")
    parser.add_argument("--bs", type=int, default=128)
    args = parser.parse_args()

    # Load images
    device = "cuda"
    ref_img = np.load(args.ref_img)["arr_0"]
    rec_img = np.load(args.rec_img)["arr_0"]
    print(f"Loaded images: ref: {ref_img.shape}, rec: {rec_img.shape}")
    
    psnr = calculate_psnr(ref_img, rec_img, args.bs, device)
    print(f"PSNR: {psnr:.6f}")
    lpips = calculate_lpips(ref_img, rec_img, args.bs, device)
    print(f"LPIPS: {lpips:.6f}")
    ssim_val = calculate_ssim(ref_img, rec_img, args.bs, device)
    print(f"SSIM: {ssim_val:.6f}")
    rfid = calculate_rfid(ref_img, rec_img, args.bs, device)
    print(f"rFID: {rfid:.6f}")
