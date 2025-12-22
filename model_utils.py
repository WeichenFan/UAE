import importlib
from dataclasses import dataclass
from typing import Union, Tuple, Optional, Dict
import torch.nn as nn
from omegaconf import OmegaConf
import torch

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def _safe_load_state(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    def attempt(sd: Dict[str, torch.Tensor]) -> bool:
        try:
            model.load_state_dict(sd, strict=True)
            return True
        except RuntimeError:
            return False

    candidate = state_dict
    if attempt(candidate):
        return
    if all(k.startswith("module.") for k in candidate.keys()):
        stripped = {k[len("module.") :]: v for k, v in candidate.items()}
        if attempt(stripped):
            return
        candidate = stripped
    model_keys = set(model.state_dict().keys())
    filtered = {k: v for k, v in candidate.items() if k in model_keys}
    if filtered:
        missing = [k for k in candidate.keys() if k not in model_keys]
        if missing:
            sample = ", ".join(missing[:5])
            print(f"[instantiate_from_config] Dropping {len(missing)} unexpected keys (e.g., {sample}).")
        model.load_state_dict(filtered, strict=True)
        return
    model.load_state_dict(candidate, strict=True)


def instantiate_from_config_2(config) -> object:
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    model = get_obj_from_str(config["target"])(**config.get("params", dict()))
    ckpt_path = config.get("ckpt", None)

    if ckpt_path is not None:
        state_dict = torch.load(ckpt_path, map_location="cpu")
        # see if it's a ckpt from training by checking for "model"
        # if "ema" in state_dict:
        #     state_dict = state_dict["ema"]
        # elif "model" in state_dict:
        state_dict = state_dict["model"]
        _safe_load_state(model, state_dict)
        print(f'target {config["target"]} loaded from {ckpt_path}')
    return model
