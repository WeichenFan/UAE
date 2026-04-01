import torch
import torch.nn as nn
from model_jit import JiT_models


class Denoiser(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        self.net = JiT_models[args.model](
            input_size=args.img_size,
            window_size=getattr(args, "window_size", None),
            in_channels=3,
            num_classes=args.class_num,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
        )
        self.img_size = args.img_size
        self.num_classes = args.class_num

        self.label_drop_prob = args.label_drop_prob
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale
        self.prediction = getattr(args, "prediction", "x")
        if self.prediction == "epsilon":
            self.prediction = "eps"

        # ema
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None  # dict[name -> tensor]
        self.ema_params2 = None  # dict[name -> tensor]

        # generation hyper params
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)

    def _format_t(self, t, x):
        if t.ndim == 1:
            t_vec = t
            t_broadcast = t.view(-1, *([1] * (x.ndim - 1)))
        else:
            t_broadcast = t
            t_vec = t.reshape(t.shape[0], -1)[:, 0].contiguous()
        return t_vec, t_broadcast

    def _pred_to_v(self, pred, z, t):
        if self.prediction == "x":
            return (pred - z) / (1 - t).clamp_min(self.t_eps)
        if self.prediction == "v":
            return pred
        if self.prediction == "eps":
            return (z - pred) / t.clamp_min(self.t_eps)
        raise ValueError(f"Unknown prediction type: {self.prediction}")

    def predict_v(self, z, t, labels):
        t_vec, t_broadcast = self._format_t(t, z)
        pred = self.net(z, t_vec, labels)
        return self._pred_to_v(pred, z, t_broadcast)

    def drop_labels(self, labels):
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        out = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
        return out

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, x, labels):
        labels_dropped = self.drop_labels(labels) if self.training else labels

        t_vec = self.sample_t(x.size(0), device=x.device)  # shape (B,)
        t = t_vec.view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale

        z = t * x + (1 - t) * e
        v = (x - z) / (1 - t).clamp_min(self.t_eps)

        pred = self.net(z, t_vec, labels_dropped)
        v_pred = self._pred_to_v(pred, z, t)

        # l2 loss
        loss = (v - v_pred) ** 2
        loss = loss.mean(dim=(1, 2, 3)).mean()

        return loss

    @torch.no_grad()
    def update_ema(self):
        if self.ema_params1 is None or self.ema_params2 is None:
            return
        for name, param in self.named_parameters():
            if name in self.ema_params1:
                self.ema_params1[name].detach().mul_(self.ema_decay1).add_(param, alpha=1 - self.ema_decay1)
            if name in self.ema_params2:
                self.ema_params2[name].detach().mul_(self.ema_decay2).add_(param, alpha=1 - self.ema_decay2)
