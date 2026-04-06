import math

import torch
import torch.nn as nn
from ldm.util import instantiate_from_config

from .common import expand_time_like, rectified_path, regression_loss, sample_interval
from .meanflow import meanflow_jvp


class SigmoidAlphaScheduler(nn.Module):
    def __init__(self, start_step, end_step, gamma=25.0, clamp_eta=0.05):
        super().__init__()
        self.start_step = int(start_step)
        self.end_step = int(end_step)
        self.gamma = float(gamma)
        self.clamp_eta = float(clamp_eta)
        if self.end_step <= self.start_step:
            raise ValueError(
                f"end_step must be greater than start_step, got {self.start_step} and {self.end_step}"
            )

    def forward(self, step):
        step = float(step)
        scale = 1.0 / float(self.end_step - self.start_step)
        offset = -float(self.start_step + self.end_step) / 2.0 / float(self.end_step - self.start_step)
        alpha = 1.0 - 1.0 / (1.0 + math.exp(-((scale * step + offset) * self.gamma)))
        if alpha > (1.0 - self.clamp_eta):
            return 1.0
        if alpha < self.clamp_eta:
            return 0.0
        return alpha


class AlphaFlowObjective(nn.Module):
    name = "alphaflow"
    prediction_type = "average_velocity"

    def __init__(
        self,
        time_eps=1.0e-4,
        min_delta=1.0e-3,
        loss_type="mse",
        alpha_schedule_config=None,
        meanflow_alpha_threshold=1.0e-8,
    ):
        super().__init__()
        self.time_eps = float(time_eps)
        self.min_delta = float(min_delta)
        self.loss_type = str(loss_type)
        self.meanflow_alpha_threshold = float(meanflow_alpha_threshold)

        if alpha_schedule_config is None:
            alpha_schedule_config = {
                "target": "latent_meanflow.objectives.alphaflow.SigmoidAlphaScheduler",
                "params": {
                    "start_step": 5_000,
                    "end_step": 60_000,
                    "gamma": 25.0,
                    "clamp_eta": 0.05,
                },
            }
        self.alpha_schedule = instantiate_from_config(alpha_schedule_config)

    def get_alpha(self, global_step):
        if global_step is None:
            global_step = 0
        if isinstance(global_step, torch.Tensor):
            global_step = int(global_step.item())
        return float(self.alpha_schedule(global_step))

    def forward(self, model_fn, x_lat, condition=None, global_step=None, **kwargs):
        batch_size = x_lat.shape[0]
        r, t, delta_t = sample_interval(
            batch_size=batch_size,
            device=x_lat.device,
            time_eps=self.time_eps,
            min_delta=self.min_delta,
        )
        z_t, velocity, noise = rectified_path(x_lat, t=t)

        alpha_value = self.get_alpha(global_step)
        alpha = torch.full((batch_size,), alpha_value, device=x_lat.device, dtype=x_lat.dtype)

        if alpha_value <= self.meanflow_alpha_threshold:
            pred_field, total_derivative = meanflow_jvp(
                model_fn=model_fn,
                z_t=z_t,
                r=r,
                t=t,
                velocity=velocity,
                condition=condition,
            )
            target_field = velocity - expand_time_like(delta_t, x_lat) * total_derivative
            s = t
            branch = "meanflow"
        else:
            pred_field = model_fn(z_t, r=r, t=t, delta_t=delta_t, condition=condition)
            s = alpha * r + (1.0 - alpha) * t
            z_s = z_t - expand_time_like(t - s, x_lat) * velocity
            shifted_field = model_fn(z_s, r=r, t=s, delta_t=s - r, condition=condition)
            target_field = expand_time_like(alpha, x_lat) * velocity + expand_time_like(1.0 - alpha, x_lat) * shifted_field
            total_derivative = None
            branch = "alphaflow"

        loss = regression_loss(pred_field, target_field.detach(), loss_type=self.loss_type)
        loss_dict = {
            "alphaflow_loss": loss,
            "total_loss": loss,
            "alpha": alpha.mean(),
            "delta_t_mean": delta_t.mean(),
        }
        return {
            "loss": loss,
            "alphaflow_loss": loss,
            "loss_dict": loss_dict,
            "alpha": alpha,
            "alpha_value": alpha.mean(),
            "r": r,
            "t": t,
            "s": s,
            "delta_t": delta_t,
            "noise": noise,
            "z_t": z_t,
            "velocity": velocity,
            "pred_field": pred_field,
            "target_field": target_field.detach(),
            "total_derivative": total_derivative,
            "objective_branch": branch,
        }
