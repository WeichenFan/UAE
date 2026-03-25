import torch
import torch.nn as nn
from .decoders import GeneralDecoder
from .encoders import ARCHS
from transformers import AutoConfig, AutoImageProcessor
from typing import Optional
from math import sqrt
from typing import Protocol

class Stage1Protocal(Protocol):
    # must have patch size attribute
    patch_size: int
    hidden_size: int 
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        ...

class UAE(nn.Module):
    def __init__(self, 
        # ---- encoder configs ----
        encoder_cls: str = 'Dinov2withNorm',
        encoder_config_path: str = 'facebook/dinov2-base',
        encoder_input_size: int = 224,
        encoder_params: dict = {},
        # ---- decoder configs ----
        decoder_config_path: str = 'vit_mae-base',
        decoder_patch_size: int = 16,
        pretrained_decoder_path: Optional[str] = None,
        # ---- noising, reshaping and normalization-----
        noise_tau: float = 0.8,
        reshape_to_2d: bool = True,
        frequency_config: Optional[dict] = None,
        normalization_stat_path: Optional[str] = None,
        eps: float = 1e-5,
    ):
        super().__init__()
        encoder_cls = ARCHS[encoder_cls]
        self.encoder: Stage1Protocal = encoder_cls(**encoder_params)
        print(f"encoder_config_path: {encoder_config_path}")
        proc = AutoImageProcessor.from_pretrained(encoder_config_path)
        self.encoder_mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1)
        self.encoder_std = torch.tensor(proc.image_std).view(1, 3, 1, 1)
        encoder_config = AutoConfig.from_pretrained(encoder_config_path)
        # see if the encoder has patch size attribute            
        self.encoder_input_size = encoder_input_size
        self.encoder_patch_size = self.encoder.patch_size
        self.latent_dim = self.encoder.hidden_size
        assert self.encoder_input_size % self.encoder_patch_size == 0, f"encoder_input_size {self.encoder_input_size} must be divisible by encoder_patch_size {self.encoder_patch_size}"
        self.base_patches = (self.encoder_input_size // self.encoder_patch_size) ** 2 # number of patches of the latent
        
        # decoder
        decoder_config = AutoConfig.from_pretrained(decoder_config_path)
        decoder_config.hidden_size = self.latent_dim # set the hidden size of the decoder to be the same as the encoder's output
        decoder_config.patch_size = decoder_patch_size
        decoder_config.image_size = int(decoder_patch_size * sqrt(self.base_patches)) 
        self.decoder = GeneralDecoder(decoder_config, num_patches=self.base_patches)
        freq_cfg = dict(frequency_config or {})
        self.freq_transform = str(freq_cfg.get("freq_transform", "none")).lower()
        self.freq_order = str(freq_cfg.get("freq_order", "zigzag")).lower()
        self.freq_num_bands = max(1, int(freq_cfg.get("freq_num_bands", 2)))
        schedule = freq_cfg.get("v_patch_nums")
        self.freq_schedule = [float(x) for x in schedule] if schedule is not None else None
        self.use_frequency_module = self.freq_transform == "dct"
        if self.freq_transform not in {"none", "dct"}:
            raise ValueError(f"Unsupported freq_transform '{self.freq_transform}'. Supported: none, dct.")
        if self.freq_order not in {"zigzag", "radial"}:
            raise ValueError(f"Unsupported freq_order '{self.freq_order}'. Supported: zigzag, radial.")
        split_cfg = dict(freq_cfg.get("band_split", {}))
        self.band_split_enabled = bool(split_cfg.get("enabled", False))
        self.band_split_mode = str(split_cfg.get("mode", "low")).lower()
        self.band_split_count = split_cfg.get("count", split_cfg.get("keep_count", None))
        self.band_split_ratio = split_cfg.get("ratio", split_cfg.get("keep_ratio", None))
        self.band_split_band_count = split_cfg.get("band_count", split_cfg.get("keep_bands", None))
        self.band_split_apply_prob = float(split_cfg.get("apply_prob", split_cfg.get("prob", 1.0)))
        if not (0.0 <= self.band_split_apply_prob <= 1.0):
            raise ValueError("frequency_config.band_split.apply_prob must be between 0 and 1.")
        self.band_split_uniform = bool(split_cfg.get("uniform", split_cfg.get("sample_uniform", False)))
        self.band_split_per_sample = bool(split_cfg.get("per_sample", False))
        self.band_split_count_min = split_cfg.get("count_min", split_cfg.get("keep_count_min", None))
        self.band_split_count_max = split_cfg.get("count_max", split_cfg.get("keep_count_max", None))
        self.band_split_decode_from_split_band = bool(
            split_cfg.get("decode_from_split_band", split_cfg.get("decode_from_split", False))
        )
        self.band_split_decode_from_split_band_prob = float(
            split_cfg.get(
                "decode_from_split_band_prob",
                split_cfg.get(
                    "decode_from_split_prob",
                    1.0 if self.band_split_decode_from_split_band else 0.0,
                ),
            )
        )
        if not (0.0 <= self.band_split_decode_from_split_band_prob <= 1.0):
            raise ValueError("band_split.decode_from_split_band_prob must be between 0 and 1.")
        self.band_split_decode_keep_count = split_cfg.get(
            "decode_keep_count",
            split_cfg.get("decode_count", None),
        )
        self.band_split_decode_keep_ratio = split_cfg.get(
            "decode_keep_ratio",
            split_cfg.get("decode_ratio", None),
        )
        cutoff_mode = str(
            split_cfg.get(
                "cutoff_mode",
                split_cfg.get("cutoff", split_cfg.get("cutoff_type", "reorder_1d")),
            )
        ).lower()
        if cutoff_mode in {"reorder_1d", "1d", "token_1d", "flat_1d"}:
            self.band_split_cutoff_mode = "reorder_1d"
        elif cutoff_mode in {"map2d_square", "2d", "2d_square", "map2d", "dct2d"}:
            self.band_split_cutoff_mode = "map2d_square"
        else:
            raise ValueError(
                f"Unsupported band_split cutoff mode '{cutoff_mode}'. "
                "Supported: reorder_1d, map2d_square."
            )
        if self.band_split_decode_from_split_band:
            if not self.band_split_enabled:
                raise ValueError("band_split.decode_from_split_band requires band_split.enabled=true.")
            if not self.use_frequency_module:
                raise ValueError("band_split.decode_from_split_band requires frequency_config.freq_transform='dct'.")
            if self.band_split_cutoff_mode != "map2d_square":
                raise ValueError("band_split.decode_from_split_band requires cutoff_mode=map2d_square.")
            if self.band_split_decode_keep_count is not None and int(self.band_split_decode_keep_count) <= 0:
                raise ValueError("band_split.decode_keep_count must be > 0 when provided.")
            if self.band_split_decode_keep_ratio is not None:
                decode_ratio = float(self.band_split_decode_keep_ratio)
                if not (0.0 < decode_ratio <= 1.0):
                    raise ValueError("band_split.decode_keep_ratio must be in (0, 1].")
            if self.band_split_per_sample:
                raise ValueError(
                    "band_split.decode_from_split_band is incompatible with band_split.per_sample=true."
                )
        self.register_buffer("_freq_radius_grid", None, persistent=False)
        self.register_buffer("_freq_edges", None, persistent=False)
        self.register_buffer("_perm", None, persistent=False)
        self._dct_cache: dict[tuple[int, int, str, int, str], tuple[torch.Tensor, torch.Tensor]] = {}
        self._last_band_keep_count: Optional[int] = None
        self._decode_from_split_band_active = bool(
            self.band_split_decode_from_split_band and self.band_split_decode_from_split_band_prob > 0.0
        )
        # load pretrained decoder weights
        if pretrained_decoder_path is not None:
            print(f"Loading pretrained decoder from {pretrained_decoder_path}")
            state_dict = torch.load(pretrained_decoder_path, map_location='cpu')
            keys = self.decoder.load_state_dict(state_dict, strict=False)
            if len(keys.missing_keys) > 0:
                print(f"Missing keys when loading pretrained decoder: {keys.missing_keys}")
        self.noise_tau = noise_tau
        self.reshape_to_2d = reshape_to_2d
        if normalization_stat_path is not None:
            stats = torch.load(normalization_stat_path, map_location='cpu')
            self.latent_mean = stats.get('mean', None)
            self.latent_var = stats.get('var', None)
            self.do_normalization = True
            self.eps = eps
            print(f"Loaded normalization stats from {normalization_stat_path}")
        else:
            self.do_normalization = False

    def _preprocess_encoder_input(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        if h != self.encoder_input_size or w != self.encoder_input_size:
            x = nn.functional.interpolate(
                x,
                size=(self.encoder_input_size, self.encoder_input_size),
                mode='bicubic',
                align_corners=False,
            )
        x = (x - self.encoder_mean.to(x.device)) / self.encoder_std.to(x.device)
        return x

    def _tokens_to_map(self, z: torch.Tensor) -> torch.Tensor:
        b, n, c = z.shape
        latent_hw = int(sqrt(n))
        if latent_hw * latent_hw != n:
            raise ValueError(f"Expected square token count for frequency module, got {n}.")
        return z.transpose(1, 2).view(b, c, latent_hw, latent_hw)

    def _map_to_tokens(self, z_map: torch.Tensor) -> torch.Tensor:
        b, c, h, w = z_map.shape
        return z_map.view(b, c, h * w).transpose(1, 2)

    def _forward_frequency_module(
        self,
        z_map: torch.Tensor,
        *,
        return_frequency_tokens: bool = False,
        apply_band_mask: Optional[bool] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not self.use_frequency_module:
            return z_map, None
        dtype = z_map.dtype
        dct_input = z_map.float() if dtype in (torch.float16, torch.bfloat16) else z_map
        freq = self._dct2(dct_input)
        mask = self._build_band_mask(
            z_map.shape[-2],
            z_map.shape[-1],
            z_map.shape[0],
            z_map.device,
            freq.dtype,
            apply_band_mask=apply_band_mask,
        )
        if mask is not None:
            freq = freq * mask
        spatial = self._idct2(freq)
        if spatial.dtype != dtype:
            spatial = spatial.to(dtype)
        if not return_frequency_tokens:
            return spatial, None
        freq_tokens = freq.to(dtype) if freq.dtype != dtype else freq
        return spatial, freq_tokens

    def _build_dct_matrix(self, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        k = torch.arange(n, device=device, dtype=dtype).view(n, 1)
        n_idx = torch.arange(n, device=device, dtype=dtype).view(1, n)
        mat = torch.cos((torch.pi / n) * (n_idx + 0.5) * k)
        mat[0] *= 1.0 / sqrt(n)
        if n > 1:
            mat[1:] *= sqrt(2.0 / n)
        return mat

    def _get_dct_matrices(
        self,
        h: int,
        w: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (h, w, device.type, device.index or -1, str(dtype))
        cached = self._dct_cache.get(key)
        if cached is not None:
            return cached
        mat_h = self._build_dct_matrix(h, device, dtype)
        mat_w = self._build_dct_matrix(w, device, dtype)
        self._dct_cache[key] = (mat_h, mat_w)
        return mat_h, mat_w

    def _dct2(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        mat_h, mat_w = self._get_dct_matrices(h, w, x.device, x.dtype)
        x_flat = x.reshape(-1, h, w)
        x_flat = torch.matmul(mat_h, x_flat)
        x_flat = torch.matmul(x_flat, mat_w.t())
        return x_flat.reshape(b, c, h, w)

    def _idct2(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        mat_h, mat_w = self._get_dct_matrices(h, w, x.device, x.dtype)
        x_flat = x.reshape(-1, h, w)
        x_flat = torch.matmul(mat_h.t(), x_flat)
        x_flat = torch.matmul(x_flat, mat_w)
        return x_flat.reshape(b, c, h, w)

    def _build_zigzag_perm_top_left(self, h: int, w: int, device: torch.device) -> torch.Tensor:
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

    def _compute_edges(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        max_radius = float(sqrt((h - 1) ** 2 + (w - 1) ** 2))
        if self.freq_schedule:
            schedule_tensor = torch.tensor(self.freq_schedule[: self.freq_num_bands], device=device, dtype=dtype)
            if schedule_tensor.numel() > 0:
                max_val = torch.clamp(schedule_tensor.max(), min=1e-6)
                edges = schedule_tensor / max_val * max_radius
                edges = torch.cat([torch.zeros(1, device=device, dtype=dtype), edges])
            else:
                edges = torch.linspace(0.0, max_radius, self.freq_num_bands + 1, device=device, dtype=dtype)
        else:
            edges = torch.linspace(0.0, max_radius, self.freq_num_bands + 1, device=device, dtype=dtype)
        if edges.numel() > 0:
            edges[-1] = edges[-1] + torch.finfo(edges.dtype).eps
        return edges

    def _refresh_frequency_cache(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> None:
        need_rebuild = (
            self._freq_radius_grid is None
            or self._freq_radius_grid.shape != (h, w)
            or self._freq_radius_grid.device != device
            or self._freq_radius_grid.dtype != dtype
            or self._perm is None
            or self._perm.numel() != h * w
            or self._freq_edges is None
        )
        if not need_rebuild:
            return
        y_loc, x_loc = torch.meshgrid(
            torch.arange(h, device=device, dtype=dtype),
            torch.arange(w, device=device, dtype=dtype),
            indexing="ij",
        )
        radius_grid = torch.sqrt(y_loc.square() + x_loc.square())
        if self.freq_order == "zigzag":
            perm = self._build_zigzag_perm_top_left(h, w, device)
        else:
            perm = torch.argsort(radius_grid.flatten(), dim=0)
        edges = self._compute_edges(h, w, device, dtype)
        self._freq_radius_grid = radius_grid
        self._perm = perm
        self._freq_edges = edges

    def _resolve_keep_count(self, token_count: int) -> int:
        if self.band_split_count is not None:
            keep = int(self.band_split_count)
        elif self.band_split_ratio is not None:
            keep = int(round(float(self.band_split_ratio) * token_count))
        else:
            keep = token_count
        keep = max(0, min(keep, token_count))
        if (
            self.band_split_cutoff_mode == "map2d_square"
            and self.band_split_mode in {"low", "keep_low", "high", "keep_high", "drop_high", "drop_low"}
            and keep not in {0, token_count}
        ):
            side = int(sqrt(keep))
            if side * side != keep:
                raise ValueError(
                    f"band_split.keep_count={keep} must be a perfect square when cutoff_mode=map2d_square."
                )
        return keep

    def _resolve_square_hw(self, token_count: int) -> int:
        hw = int(sqrt(token_count))
        if hw * hw != token_count:
            raise ValueError(
                f"cutoff_mode=map2d_square requires square token count, got {token_count}."
            )
        return hw

    def _ceil_sqrt_int(self, value: int) -> int:
        value = max(0, int(value))
        root = int(sqrt(value))
        return root if root * root == value else root + 1

    def _keep_count_to_square_side(self, keep: int, token_count: int) -> int:
        keep = max(0, min(int(keep), token_count))
        if keep == 0:
            return 0
        hw = self._resolve_square_hw(token_count)
        side = int(sqrt(keep))
        if side * side != keep:
            raise ValueError(
                f"Keep count {keep} is not a perfect square for cutoff_mode=map2d_square."
            )
        if side > hw:
            raise ValueError(f"Keep side {side} exceeds latent side {hw}.")
        return side

    def _build_square_region_mask(
        self,
        side: int,
        h: int,
        w: int,
        *,
        low_region: bool,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        side = max(0, min(int(side), min(h, w)))
        mask2d = torch.zeros((h, w), device=device, dtype=dtype)
        if side <= 0:
            return mask2d.view(1, h, w)
        if low_region:
            mask2d[:side, :side] = 1.0
        else:
            mask2d[h - side :, w - side :] = 1.0
        return mask2d.view(1, h, w)

    def _resolve_uniform_keep_bounds(self, token_count: int) -> tuple[int, int]:
        low = self.band_split_count_min if self.band_split_count_min is not None else 0
        high = self.band_split_count_max if self.band_split_count_max is not None else token_count
        low = max(0, min(int(low), token_count))
        high = max(0, min(int(high), token_count))
        if high < low:
            low, high = high, low
        return low, high

    def _sample_keep_counts(self, token_count: int, batch_size: int, device: torch.device) -> torch.Tensor:
        sample_uniform = self.band_split_uniform and (
            self.training or self.band_split_decode_from_split_band
        )
        if sample_uniform:
            low, high = self._resolve_uniform_keep_bounds(token_count)
            sample_count = batch_size if self.band_split_per_sample else 1
            if (
                self.band_split_cutoff_mode == "map2d_square"
                and self.band_split_mode in {"low", "keep_low", "high", "keep_high", "drop_high", "drop_low"}
            ):
                hw = self._resolve_square_hw(token_count)
                side_low = min(self._ceil_sqrt_int(low), hw)
                side_high = min(int(sqrt(high)), hw)
                if side_high < side_low:
                    raise ValueError(
                        "Invalid band_split uniform bounds for cutoff_mode=map2d_square: "
                        f"count_min={low}, count_max={high} produce no valid NxN range."
                    )
                if side_high == side_low:
                    sides = torch.full((sample_count,), side_low, device=device, dtype=torch.long)
                else:
                    sides = torch.randint(side_low, side_high + 1, (sample_count,), device=device, dtype=torch.long)
                return sides * sides
            if high == low:
                return torch.full((sample_count,), low, device=device, dtype=torch.long)
            return torch.randint(low, high + 1, (sample_count,), device=device, dtype=torch.long)
        return torch.tensor([self._resolve_keep_count(token_count)], device=device, dtype=torch.long)

    def _build_mask_from_keep_count(
        self,
        keep: int,
        token_count: int,
        h: int,
        w: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        mask = torch.zeros(token_count, device=device, dtype=dtype)
        keep = max(0, min(int(keep), token_count))
        mode = self.band_split_mode
        if (
            self.band_split_cutoff_mode == "map2d_square"
            and mode in {"low", "keep_low", "high", "keep_high", "drop_high", "drop_low"}
        ):
            if mode in {"low", "keep_low"}:
                active_count = keep
                low_region = True
            elif mode == "drop_high":
                active_count = token_count - keep
                low_region = True
            elif mode in {"high", "keep_high"}:
                active_count = keep
                low_region = False
            else:  # drop_low
                active_count = token_count - keep
                low_region = False
            active_count = max(0, min(active_count, token_count))
            side = self._keep_count_to_square_side(active_count, token_count)
            return self._build_square_region_mask(
                side,
                h,
                w,
                low_region=low_region,
                device=device,
                dtype=dtype,
            )
        if keep <= 0:
            if mode in {"drop_low", "drop_high"}:
                mask.fill_(1.0)
            return mask.view(1, h, w)
        if mode in {"low", "keep_low"}:
            mask[self._perm[:keep]] = 1.0
        elif mode in {"high", "keep_high"}:
            mask[self._perm[token_count - keep :]] = 1.0
        elif mode in {"drop_high"}:
            mask[self._perm[: token_count - keep]] = 1.0
        elif mode in {"drop_low"}:
            mask[self._perm[keep:]] = 1.0
        else:
            raise ValueError(
                f"Unsupported band_split.mode '{self.band_split_mode}'. "
                "Supported: low, high, drop_high, drop_low, all."
            )
        return mask.view(1, h, w)

    def _resolve_decode_keep_count(self, token_count: int) -> int:
        if self.band_split_decode_keep_count is not None:
            keep = int(self.band_split_decode_keep_count)
        elif self.band_split_decode_keep_ratio is not None:
            keep = int(round(float(self.band_split_decode_keep_ratio) * token_count))
        elif self._last_band_keep_count is not None:
            keep = int(self._last_band_keep_count)
        else:
            keep = self._resolve_keep_count(token_count)
            if self.band_split_mode in {"drop_high", "drop_low"}:
                keep = token_count - keep
        keep = max(0, min(keep, token_count))
        if keep not in {0, token_count}:
            side = int(sqrt(keep))
            if side * side != keep:
                raise ValueError(
                    "band_split.decode_from_split_band requires square keep count, "
                    f"got {keep}."
                )
        return keep

    def _build_band_mask(
        self,
        h: int,
        w: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        apply_band_mask: Optional[bool] = None,
    ) -> Optional[torch.Tensor]:
        token_count = h * w
        self._decode_from_split_band_active = bool(
            self.band_split_decode_from_split_band and self.band_split_decode_from_split_band_prob > 0.0
        )
        if not self.band_split_enabled:
            self._last_band_keep_count = token_count
            return None
        if apply_band_mask is False:
            self._last_band_keep_count = token_count
            return None
        if apply_band_mask is None and self.training:
            prob = float(self.band_split_apply_prob)
            if prob <= 0.0:
                self._last_band_keep_count = token_count
                return None
            if prob < 1.0 and torch.rand((), device=device).item() >= prob:
                self._last_band_keep_count = token_count
                return None
        self._refresh_frequency_cache(h, w, device, torch.float32)
        if self._perm is None or self._freq_radius_grid is None or self._freq_edges is None:
            self._last_band_keep_count = token_count
            return None
        mode = self.band_split_mode
        if mode in {"none", "all"}:
            self._last_band_keep_count = token_count
            return None
        if self.band_split_band_count is not None:
            band_count = int(self.band_split_band_count)
            if band_count <= 0:
                keep = 0
            else:
                max_band = min(band_count, int(self._freq_edges.numel()) - 1)
                threshold = self._freq_edges[max_band]
                radius_flat = self._freq_radius_grid.flatten()
                keep = int((radius_flat < threshold).sum().item())
            keep_counts = torch.tensor([keep], device=device, dtype=torch.long)
        else:
            keep_counts = self._sample_keep_counts(token_count, batch_size, device)
        decode_keep = int(keep_counts[0].item()) if keep_counts.numel() > 0 else token_count
        if mode in {"drop_high", "drop_low"}:
            decode_keep = token_count - decode_keep
        self._last_band_keep_count = max(0, min(decode_keep, token_count))
        if self.band_split_decode_from_split_band:
            prob = float(self.band_split_decode_from_split_band_prob)
            if prob <= 0.0:
                self._decode_from_split_band_active = False
            elif prob >= 1.0:
                self._decode_from_split_band_active = True
            else:
                self._decode_from_split_band_active = bool(
                    torch.rand((), device=device).item() < prob
                )
        else:
            self._decode_from_split_band_active = False
        masks = [
            self._build_mask_from_keep_count(
                keep=int(keep.item()),
                token_count=token_count,
                h=h,
                w=w,
                device=device,
                dtype=dtype,
            )
            for keep in keep_counts
        ]
        if len(masks) == 1:
            return masks[0].unsqueeze(0)
        return torch.stack(masks, dim=0)

    def _apply_frequency_band_module(self, z_map: torch.Tensor) -> torch.Tensor:
        spatial, _ = self._forward_frequency_module(z_map, return_frequency_tokens=False)
        return spatial

    def noising(self, x: torch.Tensor) -> torch.Tensor:
        noise_sigma = self.noise_tau * torch.rand((x.size(0),) + (1,) * (len(x.shape) - 1), device=x.device)
        noise = noise_sigma * torch.randn_like(x)
        return x + noise

    def encode(
        self,
        x: torch.Tensor,
        *,
        apply_frequency: Optional[bool] = None,
        apply_band_mask: Optional[bool] = None,
        return_pre_frequency: bool = False,
        return_frequency_tokens: bool = False,
    ):
        x = self._preprocess_encoder_input(x)
        z_tokens = self.encoder(x)
        if self.training and self.noise_tau > 0:
            z_tokens = self.noising(z_tokens)
        pre_frequency = self._tokens_to_map(z_tokens)
        use_frequency = self.use_frequency_module if apply_frequency is None else bool(apply_frequency)
        frequency_tokens: Optional[torch.Tensor] = None
        if use_frequency:
            z_map, frequency_tokens = self._forward_frequency_module(
                pre_frequency,
                return_frequency_tokens=return_frequency_tokens,
                apply_band_mask=apply_band_mask,
            )
            if self.reshape_to_2d:
                z = z_map
            else:
                z = self._map_to_tokens(z_map)
        elif self.reshape_to_2d:
            z = pre_frequency
        else:
            z = z_tokens
        if self.reshape_to_2d and z.ndim == 3:
            z = self._tokens_to_map(z)
        if self.do_normalization:
            latent_mean = self.latent_mean.to(z.device) if self.latent_mean is not None else 0
            latent_var = self.latent_var.to(z.device) if self.latent_var is not None else 1
            z = (z - latent_mean) / torch.sqrt(latent_var + self.eps)
        if return_pre_frequency and return_frequency_tokens:
            return z, pre_frequency, frequency_tokens
        if return_pre_frequency:
            return z, pre_frequency
        if return_frequency_tokens:
            return z, frequency_tokens
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.do_normalization:
            latent_mean = self.latent_mean.to(z.device) if self.latent_mean is not None else 0
            latent_var = self.latent_var.to(z.device) if self.latent_var is not None else 1
            z = z * torch.sqrt(latent_var + self.eps) + latent_mean
        compact_decode = False
        decode_image_size: Optional[tuple[int, int]] = None
        if self.reshape_to_2d:
            b, c, h, w = z.shape
            token_count = h * w
            if self.band_split_decode_from_split_band and self._decode_from_split_band_active:
                mode = self.band_split_mode
                if mode not in {"low", "keep_low", "high", "keep_high", "drop_high", "drop_low"}:
                    raise ValueError(
                        "band_split.decode_from_split_band only supports modes: "
                        "low, keep_low, high, keep_high, drop_high, drop_low."
                    )
                keep = self._resolve_decode_keep_count(token_count)
                if keep <= 0:
                    raise ValueError("Resolved decode keep count is 0; cannot decode empty latent.")
                if keep == token_count:
                    z = z.reshape(b, c, token_count).transpose(1, 2)
                else:
                    side = self._keep_count_to_square_side(keep, token_count)
                    if mode in {"low", "keep_low", "drop_high"}:
                        z = z[:, :, :side, :side]
                    else:
                        z = z[:, :, h - side :, w - side :]
                    z = z.reshape(b, c, side * side).transpose(1, 2)
                    compact_decode = True
                    out_hw = side * int(self.decoder.config.patch_size)
                    decode_image_size = (out_hw, out_hw)
            else:
                z = z.reshape(b, c, token_count).transpose(1, 2)
        output = self.decoder(
            z,
            drop_cls_token=False,
            interpolate_pos_encoding=compact_decode,
            interpolate_latent_tokens=not compact_decode,
        ).logits
        x_rec = self.decoder.unpatchify(output, original_image_size=decode_image_size)
        x_rec = x_rec * self.encoder_std.to(x_rec.device) + self.encoder_mean.to(x_rec.device)
        return x_rec
    
    def forward(
        self,
        x: torch.Tensor,
        *,
        return_pre_frequency: bool = False,
        return_frequency_tokens: bool = False,
        apply_frequency: Optional[bool] = None,
        apply_band_mask: Optional[bool] = None,
    ):
        if return_pre_frequency or return_frequency_tokens:
            enc_out = self.encode(
                x,
                apply_frequency=apply_frequency,
                apply_band_mask=apply_band_mask,
                return_pre_frequency=return_pre_frequency,
                return_frequency_tokens=return_frequency_tokens,
            )
            if return_pre_frequency and return_frequency_tokens:
                z, pre_frequency, frequency_tokens = enc_out
            elif return_pre_frequency:
                z, pre_frequency = enc_out
                frequency_tokens = None
            else:
                z, frequency_tokens = enc_out
                pre_frequency = None
            x_rec = self.decode(z)
            return x_rec, pre_frequency, frequency_tokens
        z = self.encode(x, apply_frequency=apply_frequency, apply_band_mask=apply_band_mask)
        x_rec = self.decode(z)
        return x_rec
