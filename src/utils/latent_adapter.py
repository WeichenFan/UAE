from __future__ import annotations

from dataclasses import dataclass
from math import isqrt
from typing import Iterable

import torch


def _as_tuple_int(values: Iterable[int | float]) -> tuple[int, ...]:
    return tuple(int(v) for v in values)


@dataclass
class Stage2LatentAdapter:
    """Adapter that keeps stage2 model IO in 4D while supporting freq-token latents."""

    mode: str
    channels: int
    token_count: int | None
    height: int
    width: int

    @property
    def model_latent_size(self) -> tuple[int, int, int]:
        return (self.channels, self.height, self.width)

    @classmethod
    def from_misc(cls, misc: dict) -> "Stage2LatentAdapter":
        mode = str(misc.get("latent_format", "spatial")).lower()
        latent_size_raw = misc.get("latent_size", (768, 16, 16))
        if latent_size_raw is None:
            raise ValueError("misc.latent_size is required.")
        latent_size = _as_tuple_int(latent_size_raw)

        if mode == "spatial":
            if len(latent_size) != 3:
                raise ValueError(
                    f"spatial latent_format expects misc.latent_size=[C,H,W], got {latent_size}."
                )
            c, h, w = latent_size
            return cls(mode=mode, channels=c, token_count=None, height=h, width=w)

        if mode == "freq_tokens":
            if len(latent_size) == 2:
                c, n = latent_size
            elif len(latent_size) == 3:
                c, h, w = latent_size
                n = h * w
            else:
                raise ValueError(
                    f"freq_tokens latent_format expects misc.latent_size=[C,N] or [C,H,W], got {latent_size}."
                )

            hw = misc.get("freq_token_hw", None)
            if hw is not None:
                hw_tuple = _as_tuple_int(hw)
                if len(hw_tuple) != 2:
                    raise ValueError("misc.freq_token_hw must be [H,W].")
                h, w = hw_tuple
                if h * w != n:
                    raise ValueError(
                        f"misc.freq_token_hw product ({h*w}) must equal token count ({n})."
                    )
            else:
                side = isqrt(n)
                if side * side != n:
                    raise ValueError(
                        f"freq token count {n} is not square. Please set misc.freq_token_hw=[H,W]."
                    )
                h = side
                w = side

            return cls(mode=mode, channels=c, token_count=n, height=h, width=w)

        if mode == "freq_tokens_1d":
            if len(latent_size) == 2:
                c, n = latent_size
            elif len(latent_size) == 3:
                c, h, w = latent_size
                n = h * w
            else:
                raise ValueError(
                    f"freq_tokens_1d latent_format expects misc.latent_size=[C,N] or [C,H,W], got {latent_size}."
                )

            hw = misc.get("freq_token_hw", None)
            if hw is not None:
                hw_tuple = _as_tuple_int(hw)
                if len(hw_tuple) != 2:
                    raise ValueError("misc.freq_token_hw must be [H,W].")
                h, w = hw_tuple
                if h * w != n:
                    raise ValueError(
                        f"misc.freq_token_hw product ({h*w}) must equal token count ({n})."
                    )
                if h != 1 and w != 1:
                    raise ValueError(
                        f"freq_tokens_1d expects one axis to be 1, got freq_token_hw={hw_tuple}."
                    )
            else:
                h, w = 1, n

            return cls(mode=mode, channels=c, token_count=n, height=h, width=w)

        raise ValueError(
            f"Unsupported misc.latent_format '{mode}'. Use 'spatial', 'freq_tokens', or 'freq_tokens_1d'."
        )

    def to_model_space(self, z: torch.Tensor) -> torch.Tensor:
        """Convert stage1 latent output to stage2 model input."""
        if self.mode == "spatial":
            if z.ndim != 4:
                raise ValueError(f"spatial mode expects 4D latent [B,C,H,W], got {tuple(z.shape)}.")
            return z

        if self.mode == "freq_tokens_1d":
            if z.ndim == 3:
                b, c, n = z.shape
                if c != self.channels:
                    raise ValueError(f"Expected channels={self.channels}, got {c}.")
                if self.token_count is not None and n != self.token_count:
                    raise ValueError(f"Expected token_count={self.token_count}, got {n}.")
                return z.reshape(b, c, self.height, self.width)
            if z.ndim == 4:
                b, c, h, w = z.shape
                if c != self.channels:
                    raise ValueError(f"Expected channels={self.channels}, got {c}.")
                n = h * w
                if self.token_count is not None and n != self.token_count:
                    raise ValueError(f"Expected token_count={self.token_count}, got {n}.")
                return z.reshape(b, c, self.height, self.width)
            raise ValueError(f"freq_tokens_1d mode expects latent ndim 3 or 4, got {z.ndim}.")

        if self.mode != "freq_tokens":
            raise ValueError(f"Unsupported latent mode '{self.mode}'.")

        if z.ndim == 3:
            b, c, n = z.shape
            if c != self.channels:
                raise ValueError(f"Expected channels={self.channels}, got {c}.")
            if self.token_count is not None and n != self.token_count:
                raise ValueError(f"Expected token_count={self.token_count}, got {n}.")
            return z.reshape(b, c, self.height, self.width)

        if z.ndim == 4:
            b, c, h, w = z.shape
            if c != self.channels:
                raise ValueError(f"Expected channels={self.channels}, got {c}.")
            if (h, w) == (self.height, self.width):
                return z
            if self.token_count is not None and h * w == self.token_count:
                return z.reshape(b, c, self.height, self.width)
            raise ValueError(
                f"Cannot map latent shape {tuple(z.shape)} to freq_token_hw=({self.height},{self.width})."
            )

        raise ValueError(f"freq_tokens mode expects latent ndim 3 or 4, got {z.ndim}.")

    def from_model_space(self, z: torch.Tensor) -> torch.Tensor:
        """Convert stage2 model output to stage1 decode input."""
        if self.mode == "spatial":
            return z
        if self.mode == "freq_tokens_1d":
            if z.ndim == 4:
                b, c, h, w = z.shape
                if c != self.channels:
                    raise ValueError(f"Expected channels={self.channels}, got {c}.")
                if h * w != self.height * self.width:
                    raise ValueError(
                        f"Expected model latent spatial size product {self.height*self.width}, got {h*w}."
                    )
                return z.reshape(b, c, h * w)
            if z.ndim == 3:
                return z
            raise ValueError(f"freq_tokens_1d mode expects model output ndim 3 or 4, got {z.ndim}.")
        if self.mode != "freq_tokens":
            raise ValueError(f"Unsupported latent mode '{self.mode}'.")

        if z.ndim == 4:
            b, c, h, w = z.shape
            if c != self.channels:
                raise ValueError(f"Expected channels={self.channels}, got {c}.")
            if h * w != self.height * self.width:
                raise ValueError(
                    f"Expected model latent spatial size product {self.height*self.width}, got {h*w}."
                )
            return z.reshape(b, c, h * w)
        if z.ndim == 3:
            return z
        raise ValueError(f"freq_tokens mode expects model output ndim 3 or 4, got {z.ndim}.")
