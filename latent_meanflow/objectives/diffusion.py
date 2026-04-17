import math

import torch
import torch.nn as nn

from .common import weighted_regression_loss


def build_beta_schedule(
    *,
    num_train_steps,
    beta_schedule="cosine",
    beta_start=1.0e-4,
    beta_end=2.0e-2,
    cosine_s=0.008,
):
    num_train_steps = int(num_train_steps)
    if num_train_steps <= 0:
        raise ValueError(f"num_train_steps must be positive, got {num_train_steps}")

    beta_schedule = str(beta_schedule).lower()
    if beta_schedule == "linear":
        betas = torch.linspace(float(beta_start), float(beta_end), num_train_steps, dtype=torch.float32)
    elif beta_schedule == "cosine":
        steps = num_train_steps + 1
        x = torch.linspace(0, num_train_steps, steps, dtype=torch.float64)
        alphas_cumprod = torch.cos(((x / num_train_steps) + float(cosine_s)) / (1.0 + float(cosine_s)) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = betas.clamp(min=1.0e-6, max=0.999).to(dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported beta_schedule: {beta_schedule}")
    return betas


def normalize_diffusion_timesteps(timesteps, num_train_steps, *, dtype=torch.float32):
    num_train_steps = int(num_train_steps)
    if num_train_steps <= 1:
        return torch.zeros_like(timesteps, dtype=dtype)
    return timesteps.to(dtype=dtype) / float(num_train_steps - 1)


def extract_schedule_value(buffer, timesteps, reference):
    values = buffer.gather(0, timesteps.to(device=buffer.device, dtype=torch.long))
    return values.view(-1, *([1] * (reference.ndim - 1))).to(device=reference.device, dtype=reference.dtype)


class GaussianDiffusionObjective(nn.Module):
    name = "diffusion"
    prediction_type = "epsilon"

    def __init__(
        self,
        num_train_steps=1000,
        beta_schedule="cosine",
        beta_start=1.0e-4,
        beta_end=2.0e-2,
        cosine_s=0.008,
        loss_type="mse",
        min_snr_gamma=None,
    ):
        super().__init__()
        self.num_train_steps = int(num_train_steps)
        self.beta_schedule = str(beta_schedule)
        self.loss_type = str(loss_type)
        self.min_snr_gamma = None if min_snr_gamma is None else float(min_snr_gamma)

        betas = build_beta_schedule(
            num_train_steps=self.num_train_steps,
            beta_schedule=self.beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            cosine_s=cosine_s,
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        snr = alphas_cumprod / (1.0 - alphas_cumprod).clamp_min(1.0e-12)

        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("alphas", alphas, persistent=False)
        self.register_buffer("alphas_cumprod", alphas_cumprod, persistent=False)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod), persistent=False)
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt((1.0 - alphas_cumprod).clamp_min(1.0e-12)),
            persistent=False,
        )
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod), persistent=False)
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod",
            torch.sqrt((1.0 / alphas_cumprod - 1.0).clamp_min(0.0)),
            persistent=False,
        )
        self.register_buffer("snr", snr, persistent=False)

    def q_sample(self, x_start, timesteps, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract_schedule_value(self.sqrt_alphas_cumprod, timesteps, x_start) * x_start
            + extract_schedule_value(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start) * noise
        )

    def predict_x0(self, z_t, timesteps, pred_noise):
        return (
            extract_schedule_value(self.sqrt_recip_alphas_cumprod, timesteps, z_t) * z_t
            - extract_schedule_value(self.sqrt_recipm1_alphas_cumprod, timesteps, z_t) * pred_noise
        )

    def _resolve_base_weight(self, timesteps, reference):
        if self.min_snr_gamma is None:
            return torch.ones((timesteps.shape[0],), device=reference.device, dtype=reference.dtype)
        snr = self.snr.gather(0, timesteps.to(device=self.snr.device, dtype=torch.long))
        gamma = torch.full_like(snr, float(self.min_snr_gamma))
        return (torch.minimum(snr, gamma) / snr.clamp_min(1.0e-12)).to(device=reference.device, dtype=reference.dtype)

    def forward(self, model_fn, x_lat, condition=None, global_step=None, **kwargs):
        _ = global_step, kwargs
        batch_size = x_lat.shape[0]
        timesteps = torch.randint(0, self.num_train_steps, (batch_size,), device=x_lat.device, dtype=torch.long)
        t = normalize_diffusion_timesteps(timesteps, self.num_train_steps, dtype=x_lat.dtype)
        noise = torch.randn_like(x_lat)
        z_t = self.q_sample(x_lat, timesteps, noise=noise)
        pred_noise = model_fn(z_t, t=t, condition=condition)
        base_weight = self._resolve_base_weight(timesteps, x_lat)
        loss, loss_stats = weighted_regression_loss(
            pred_noise,
            noise.detach(),
            loss_type=self.loss_type,
            base_weight=base_weight,
            weighting_mode="none",
        )
        x0_pred = self.predict_x0(z_t, timesteps, pred_noise)
        loss_dict = {
            "diffusion_loss": loss,
            "total_loss": loss,
            "t_mean": t.mean(),
            "t_index_mean": timesteps.float().mean(),
            "snr_mean": self.snr.gather(0, timesteps.to(device=self.snr.device)).to(device=x_lat.device, dtype=x_lat.dtype).mean(),
            **loss_stats,
        }
        if self.min_snr_gamma is not None:
            loss_dict["min_snr_gamma"] = torch.tensor(float(self.min_snr_gamma), device=x_lat.device, dtype=x_lat.dtype)

        return {
            "loss": loss,
            "diffusion_loss": loss,
            "loss_dict": loss_dict,
            "t": t,
            "t_index": timesteps,
            "noise": noise,
            "z_t": z_t,
            "pred_field": pred_noise,
            "target_field": noise.detach(),
            "x0_pred": x0_pred.detach(),
            "base_weight": base_weight,
        }
