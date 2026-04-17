import torch
import torch.nn as nn

from latent_meanflow.objectives.diffusion import (
    build_beta_schedule,
    extract_schedule_value,
    normalize_diffusion_timesteps,
)


class DDIMDiffusionSampler(nn.Module):
    def __init__(
        self,
        num_train_steps=1000,
        beta_schedule="cosine",
        beta_start=1.0e-4,
        beta_end=2.0e-2,
        cosine_s=0.008,
        default_nfe=8,
        eta=0.0,
        clip_denoised=True,
        clip_min=0.0,
        clip_max=1.0,
    ):
        super().__init__()
        self.num_train_steps = int(num_train_steps)
        self.default_nfe = int(default_nfe)
        self.eta = float(eta)
        self.clip_denoised = bool(clip_denoised)
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)

        betas = build_beta_schedule(
            num_train_steps=self.num_train_steps,
            beta_schedule=beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            cosine_s=cosine_s,
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

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

    def _predict_x0(self, z_t, timesteps, pred_noise):
        return (
            extract_schedule_value(self.sqrt_recip_alphas_cumprod, timesteps, z_t) * z_t
            - extract_schedule_value(self.sqrt_recipm1_alphas_cumprod, timesteps, z_t) * pred_noise
        )

    def _build_time_indices(self, nfe, device):
        nfe = self.default_nfe if nfe is None else int(nfe)
        if nfe <= 0:
            raise ValueError(f"nfe must be positive, got {nfe}")
        time_indices = torch.linspace(
            self.num_train_steps - 1,
            0,
            steps=nfe,
            device=device,
        ).round().to(dtype=torch.long)
        return torch.unique_consecutive(time_indices)

    def sample(self, model_fn, batch_size, latent_shape, device, condition=None, noise=None, nfe=None):
        if noise is None:
            z_t = torch.randn(batch_size, *latent_shape, device=device)
        else:
            z_t = noise.to(device=device)
        if z_t.shape != (batch_size, *latent_shape):
            raise ValueError(
                f"Noise shape mismatch: expected {(batch_size, *latent_shape)}, got {tuple(z_t.shape)}"
            )

        time_indices = self._build_time_indices(nfe=nfe, device=device)
        for step_idx, current_idx in enumerate(time_indices):
            timestep = torch.full((batch_size,), int(current_idx.item()), device=device, dtype=torch.long)
            t = normalize_diffusion_timesteps(timestep, self.num_train_steps, dtype=z_t.dtype)
            pred_noise = model_fn(z_t, t=t, condition=condition)
            x0_pred = self._predict_x0(z_t, timestep, pred_noise)
            if self.clip_denoised:
                x0_pred = x0_pred.clamp(min=self.clip_min, max=self.clip_max)

            is_last_step = step_idx == (len(time_indices) - 1)
            if is_last_step:
                z_t = x0_pred
                continue

            prev_idx = torch.full(
                (batch_size,),
                int(time_indices[step_idx + 1].item()),
                device=device,
                dtype=torch.long,
            )
            alpha_bar = extract_schedule_value(self.alphas_cumprod, timestep, z_t)
            alpha_bar_prev = extract_schedule_value(self.alphas_cumprod, prev_idx, z_t)
            sigma = self.eta * torch.sqrt(
                ((1.0 - alpha_bar_prev) / (1.0 - alpha_bar).clamp_min(1.0e-12))
                * (1.0 - alpha_bar / alpha_bar_prev.clamp_min(1.0e-12))
            ).clamp_min(0.0)
            direction = torch.sqrt((1.0 - alpha_bar_prev - sigma ** 2).clamp_min(0.0)) * pred_noise
            if self.eta > 0.0:
                z_t = torch.sqrt(alpha_bar_prev) * x0_pred + direction + sigma * torch.randn_like(z_t)
            else:
                z_t = torch.sqrt(alpha_bar_prev) * x0_pred + direction
        return z_t
