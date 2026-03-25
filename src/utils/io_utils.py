import errno
import os
from typing import Any

import numpy as np
import torch


def is_disk_full_error(exc: BaseException) -> bool:
    if isinstance(exc, OSError) and getattr(exc, "errno", None) == errno.ENOSPC:
        return True
    msg = str(exc).lower()
    return (
        "no space left on device" in msg
        or "errno 28" in msg
        or "enospc" in msg
    )


def _warn(logger, message: str) -> None:
    if logger is not None:
        logger.warning(message)
    else:
        print(message)


def safe_torch_save(
    obj: Any,
    path: str,
    *,
    logger=None,
    context: str = "artifact",
) -> bool:
    directory = os.path.dirname(path)
    if directory:
        try:
            os.makedirs(directory, exist_ok=True)
        except Exception as exc:
            if is_disk_full_error(exc):
                _warn(
                    logger,
                    f"[I/O] Disk full while creating directory for {context}: {directory}. Skipping save.",
                )
                return False
            raise

    tmp_path = f"{path}.tmp"
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
        return True
    except Exception as exc:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        if is_disk_full_error(exc):
            _warn(
                logger,
                f"[I/O] Disk full while saving {context} to {path}. Skipping this save and continuing.",
            )
            return False
        raise


def safe_np_savez(
    path: str,
    *,
    arr_0,
    logger=None,
    context: str = "npz shard",
) -> bool:
    directory = os.path.dirname(path)
    if directory:
        try:
            os.makedirs(directory, exist_ok=True)
        except Exception as exc:
            if is_disk_full_error(exc):
                _warn(
                    logger,
                    f"[I/O] Disk full while creating directory for {context}: {directory}. Skipping save.",
                )
                return False
            raise

    tmp_path = f"{path}.tmp.npz"
    try:
        np.savez(tmp_path, arr_0=arr_0)
        os.replace(tmp_path, path)
        return True
    except Exception as exc:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        if is_disk_full_error(exc):
            _warn(
                logger,
                f"[I/O] Disk full while saving {context} to {path}. Skipping this save and continuing.",
            )
            return False
        raise
