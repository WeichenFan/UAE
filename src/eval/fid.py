import argparse
from pathlib import Path
import numpy as np

from torch_fidelity import calculate_metrics
from .utils import ImgArrDataset

from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3  # original torch-fidelity :contentReference[oaicite:2]{index=2}
import scipy.linalg
import torch

def _fid_from_moments(mu1, sigma1, mu2, sigma2) -> float:
    # FID formula :contentReference[oaicite:3]{index=3}

    mu1 = np.asarray(mu1, dtype=np.float64)
    mu2 = np.asarray(mu2, dtype=np.float64)
    sigma1 = np.asarray(sigma1, dtype=np.float64)
    sigma2 = np.asarray(sigma2, dtype=np.float64)

    diff = mu1 - mu2
    covmean = scipy.linalg.sqrtm(sigma1 @ sigma2)

    if np.iscomplexobj(covmean):  # numerical noise
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(max(fid, 0.0))


@torch.no_grad()
def _compute_inception_moments_from_arr(arr: np.ndarray, batch_size: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Uses torch-fidelity's InceptionV3 feature extractor to get 2048-d pool features.
    Assumes arr is [N,H,W,C] or [N,C,H,W], uint8 (0..255) or float (0..1 or 0..255).
    """
    # Convert to torch in NCHW uint8 as the safest default.
    x = arr
    if x.ndim != 4:
        raise ValueError(f"Expected 4D array, got shape {x.shape}")

    if x.shape[-1] == 3:  # NHWC -> NCHW
        x = np.transpose(x, (0, 3, 1, 2))

    if x.dtype != np.uint8:
        # If float in [0,1], scale up; otherwise assume already 0..255-ish
        x_f = x.astype(np.float32)
        if x_f.max() <= 1.5:
            x_f = x_f * 255.0
        x = np.clip(x_f, 0, 255).astype(np.uint8)

    xt = torch.from_numpy(x).to(device=device, dtype=torch.uint8)

    fe = FeatureExtractorInceptionV3(name="inception-v3-compat", features_list=['2048']).to(device).eval()  # preregistered extractor name :contentReference[oaicite:4]{index=4}

    feats = []
    for i in range(0, xt.shape[0], batch_size):
        batch = xt[i : i + batch_size]
        f = fe(batch)[0]              # (B, 2048)
        feats.append(f.detach().cpu())

    feats = torch.cat(feats, dim=0).double().numpy()  # (N, 2048) float64
    mu = feats.mean(axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


def calculate_gfid(
    arr1: np.ndarray,
    ref_arr: dict,
    batch_size: int = 64,
    device: str = "cuda",
) -> float:
    mu_ref, sigma_ref = ref_arr['mu'], ref_arr['sigma']
    mu_gen, sigma_gen = _compute_inception_moments_from_arr(arr1, batch_size=batch_size, device=device)
    return _fid_from_moments(mu_gen, sigma_gen, mu_ref, sigma_ref)

def calculate_rfid(
    arr1,
    arr2=None,
    bs=64,
    device="cuda",
    fid_statistics_file=None,
):
    arr1_ds = ImgArrDataset(arr1)

    if fid_statistics_file is not None:
        metrics_kwargs = dict(
            input1=arr1_ds,
            input2=None,
            fid_statistics_file=fid_statistics_file,
            batch_size=bs,
            fid=True,
            cuda=(device == "cuda"),
        )
    else:
        if arr2 is None:
            raise ValueError("Either arr2 or fid_statistics_file must be provided.")
        arr2_ds = ImgArrDataset(arr2)
        metrics_kwargs = dict(
            input1=arr1_ds,
            input2=arr2_ds,
            batch_size=bs,
            fid=True,
            cuda=(device == "cuda"),
        )

    metrics = calculate_metrics(**metrics_kwargs)
    return metrics["frechet_inception_distance"]


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    import numpy as np
    import torch

    parser = argparse.ArgumentParser(description="Compute FID using original torch-fidelity")
    parser.add_argument("--arr1", type=str, required=True,
                        help="Path to generated images array (.npy or .npz). Array should be N x H x W x 3 or N x 3 x H x W.")
    parser.add_argument("--arr2", type=str, default=None,
                        help="Optional path to reference images array (.npy or .npz). If set, uses torch_fidelity.calculate_metrics.")

    # reference stats path (npz with mu/sigma keys)
    parser.add_argument("--ref", "--fid-statistics-file", dest="ref", type=str, default=None,
                        help="Path to reference stats (.npz with mu/sigma, mu_s/sigma_s, mu_clip/sigma_clip, etc.). "
                             "If set, computes moments for arr1 using torch-fidelity InceptionV3 and FID vs these stats.")

    # which keys in the ref npz to use
    parser.add_argument("--ref-mu-key", type=str, default="mu",
                        help="Key in --ref .npz for reference mean (default: mu).")
    parser.add_argument("--ref-sigma-key", type=str, default="sigma",
                        help="Key in --ref .npz for reference covariance (default: sigma).")

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--no-scipy", action="store_true",
                        help="Do not use scipy.linalg.sqrtm; use torch eig fallback (slower/less identical).")

    args = parser.parse_args()

    # exactly one of arr2 or ref must be provided
    if (args.arr2 is None) == (args.ref is None):
        parser.error("Specify exactly one of --arr2 or --ref/--fid-statistics-file")

    def load_array(path: str) -> np.ndarray:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        if p.suffix == ".npy":
            return np.load(p)
        elif p.suffix == ".npz":
            z = np.load(p)
            # common conventions: 'arr_0' or 'images'
            if "arr_0" in z.files:
                return z["arr_0"]
            if "images" in z.files:
                return z["images"]
            raise KeyError(f"{p} is .npz but has no 'arr_0' or 'images'. Keys={z.files}")
        else:
            raise ValueError(f"Unsupported array file type: {p.suffix} (expected .npy or .npz)")

    arr1 = load_array(args.arr1)
    print("[INFO] arr1:", arr1.shape, arr1.dtype)

    if args.arr2 is not None:
        arr2 = load_array(args.arr2)
        print("[INFO] arr2:", arr2.shape, arr2.dtype)

        fid = calculate_rfid(
            arr1=arr1,
            arr2=arr2,
            bs=args.batch_size,
            device=args.device,
            fid_statistics_file=None,   # unused in upstream path
        )
        print(f"[RESULT] FID: {fid:.6f}")
        raise SystemExit(0)

    # stats mode: arr1 vs ref npz moments
    ref_path = Path(args.ref)
    if not ref_path.exists():
        raise FileNotFoundError(f"Ref stats not found: {ref_path}")

    if ref_path.suffix != ".npz":
        raise ValueError(
            f"--ref must be a .npz containing mu/sigma. Got: {ref_path.suffix}. "
            "Original torch-fidelity cannot use .pt stats without input2."
        )

    stats = np.load(ref_path)
    if args.ref_mu_key not in stats.files or args.ref_sigma_key not in stats.files:
        raise KeyError(
            f"Missing '{args.ref_mu_key}'/'{args.ref_sigma_key}' in {ref_path}. "
            f"Available keys: {list(stats.files)}"
        )
    print(f"[INFO] ref stats: {ref_path}")
    # If you want to respect --no-scipy, thread it through:
    # (edit your fid_arr_vs_npzstats / _fid_from_moments accordingly)
    fid = calculate_gfid(
        arr1=arr1,
        ref_arr=stats,
        batch_size=args.batch_size,
        device=args.device,
        # optionally add: use_scipy=(not args.no_scipy)
    )
    print(f"[RESULT] FID: {fid:.6f}")