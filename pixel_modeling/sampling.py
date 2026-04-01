import torch
from typing import Tuple


class DiffusionSampler:
    """Sampling utilities split out of the Denoiser to keep training code clean."""

    def __init__(
        self,
        model,
        img_size: int,
        num_classes: int,
        noise_scale: float,
        t_eps: float,
        method: str,
        steps: int,
        cfg_scale: float,
        cfg_interval: Tuple[float, float],
    ):
        self.model = model
        self.img_size = img_size
        self.num_classes = num_classes
        self.noise_scale = noise_scale
        self.t_eps = t_eps
        self.method = method
        self.steps = steps
        self.cfg_scale = cfg_scale
        self.cfg_interval = cfg_interval

    @torch.no_grad()
    def generate(self, labels: torch.Tensor) -> torch.Tensor:
        device = labels.device
        bsz = labels.size(0)
        z = self.noise_scale * torch.randn(bsz, 3, self.img_size, self.img_size, device=device)
        timesteps = torch.linspace(0.0, 1.0, self.steps + 1, device=device).view(
            -1, *([1] * z.ndim)
        ).expand(-1, bsz, -1, -1, -1)

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError(f"Unknown sampling method {self.method}")

        for i in range(self.steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = stepper(z, t, t_next, labels)
        z = self._euler_step(z, timesteps[-2], timesteps[-1], labels)
        return z

    @torch.no_grad()
    def _forward_sample(self, z: torch.Tensor, t: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        v_cond = self.model.predict_v(z, t, labels)
        v_uncond = self.model.predict_v(z, t, torch.full_like(labels, self.num_classes))

        low, high = self.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)

        return v_uncond + cfg_scale_interval * (v_cond - v_uncond)

    @torch.no_grad()
    def _euler_step(self, z: torch.Tensor, t: torch.Tensor, t_next: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        v_pred = self._forward_sample(z, t, labels)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z: torch.Tensor, t: torch.Tensor, t_next: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        v_pred_t = self._forward_sample(z, t, labels)
        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, labels)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next
