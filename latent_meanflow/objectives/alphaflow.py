import math

import torch
import torch.nn as nn
from ldm.util import instantiate_from_config

from .common import (
    build_time_sampler,
    expand_time_like,
    rectified_path,
    sample_interval,
    weighted_regression_loss,
)
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


class ConstantAlphaScheduler(nn.Module):
    def __init__(self, value=1.0):
        super().__init__()
        self.value = float(value)

    def forward(self, step):
        _ = step
        return self.value


class AlphaFlowObjective(nn.Module):
    name = "alphaflow"
    prediction_type = "average_velocity"

    def __init__(
        self,
        time_eps=1.0e-4,
        min_delta=1.0e-3,
        loss_type="mse",
        time_sampler_config=None,
        r_equals_t_ratio=0.0,
        flow_matching_ratio=0.0,
        weighting_mode="alpha_adaptive",
        adaptive_weight_power=0.75,
        adaptive_weight_bias=1.0e-4,
        alpha_schedule_config=None,
        meanflow_alpha_threshold=1.0e-8,
        alpha_inverse_eps=1.0e-6,
    ):
        super().__init__()
        self.time_eps = float(time_eps)
        self.min_delta = float(min_delta)
        self.loss_type = str(loss_type)
        self.r_equals_t_ratio = float(r_equals_t_ratio)
        self.flow_matching_ratio = float(flow_matching_ratio)
        self.weighting_mode = str(weighting_mode)
        self.adaptive_weight_power = float(adaptive_weight_power)
        self.adaptive_weight_bias = float(adaptive_weight_bias)
        self.meanflow_alpha_threshold = float(meanflow_alpha_threshold)
        self.alpha_inverse_eps = float(alpha_inverse_eps)
        self.time_sampler = build_time_sampler(time_sampler_config=time_sampler_config, time_eps=self.time_eps)

        if alpha_schedule_config is None:
            alpha_schedule_config = {
                "target": "latent_meanflow.objectives.alphaflow.SigmoidAlphaScheduler",
                "params": {
                    "start_step": 0,
                    "end_step": 1_200_000,
                    "gamma": 25.0,
                    "clamp_eta": 0.25,
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
            time_sampler=self.time_sampler,
            min_delta=self.min_delta,
            r_equals_t_ratio=self.r_equals_t_ratio,
            dtype=x_lat.dtype,
        )
        z_t, velocity, noise = rectified_path(x_lat, t=t)
        target_model_fn = kwargs.get("target_model_fn", model_fn)

        alpha_value = self.get_alpha(global_step)
        alpha = torch.full((batch_size,), alpha_value, device=x_lat.device, dtype=x_lat.dtype)
        if self.flow_matching_ratio > 0.0:
            force_flow_matching = torch.rand(batch_size, device=x_lat.device) < self.flow_matching_ratio
            alpha = torch.where(force_flow_matching, torch.ones_like(alpha), alpha)
        else:
            force_flow_matching = torch.zeros(batch_size, device=x_lat.device, dtype=torch.bool)

        pred_field = torch.zeros_like(x_lat)
        target_field = torch.zeros_like(x_lat)
        base_weight = torch.ones(batch_size, device=x_lat.device, dtype=x_lat.dtype)
        total_derivative = None
        s = t.clone()
        objective_branch = torch.full((batch_size,), 1, device=x_lat.device, dtype=torch.long)

        meanflow_mask = alpha <= self.meanflow_alpha_threshold
        if torch.any(meanflow_mask):
            pred_meanflow, total_derivative_meanflow = meanflow_jvp(
                model_fn=model_fn,
                z_t=z_t[meanflow_mask],
                r=r[meanflow_mask],
                t=t[meanflow_mask],
                velocity=velocity[meanflow_mask],
                condition=None if condition is None else condition[meanflow_mask],
            )
            pred_field[meanflow_mask] = pred_meanflow
            target_field[meanflow_mask] = (
                velocity[meanflow_mask]
                - expand_time_like(delta_t[meanflow_mask], x_lat[meanflow_mask]) * total_derivative_meanflow
            )
            total_derivative = torch.zeros_like(x_lat)
            total_derivative[meanflow_mask] = total_derivative_meanflow
            objective_branch[meanflow_mask] = 0

        alphaflow_mask = ~meanflow_mask
        if torch.any(alphaflow_mask):
            alpha_subset = alpha[alphaflow_mask]
            pred_subset = model_fn(
                z_t[alphaflow_mask],
                r=r[alphaflow_mask],
                t=t[alphaflow_mask],
                delta_t=delta_t[alphaflow_mask],
                condition=None if condition is None else condition[alphaflow_mask],
            )
            s_subset = alpha_subset * r[alphaflow_mask] + (1.0 - alpha_subset) * t[alphaflow_mask]
            z_s_subset = z_t[alphaflow_mask] - expand_time_like(t[alphaflow_mask] - s_subset, z_t[alphaflow_mask]) * velocity[
                alphaflow_mask
            ]
            with torch.no_grad():
                shifted_subset = target_model_fn(
                    z_s_subset,
                    r=r[alphaflow_mask],
                    t=s_subset,
                    delta_t=s_subset - r[alphaflow_mask],
                    condition=None if condition is None else condition[alphaflow_mask],
                )
            target_subset = (
                expand_time_like(alpha_subset, z_t[alphaflow_mask]) * velocity[alphaflow_mask]
                + expand_time_like(1.0 - alpha_subset, z_t[alphaflow_mask]) * shifted_subset
            )

            pred_field[alphaflow_mask] = pred_subset
            target_field[alphaflow_mask] = target_subset
            s[alphaflow_mask] = s_subset
            base_weight[alphaflow_mask] = 1.0 / alpha_subset.clamp_min(self.alpha_inverse_eps)
            objective_branch[alphaflow_mask] = torch.where(
                force_flow_matching[alphaflow_mask],
                torch.full_like(alpha_subset, 2, dtype=torch.long),
                torch.full_like(alpha_subset, 1, dtype=torch.long),
            )

        loss, weighting_stats = weighted_regression_loss(
            pred_field,
            target_field.detach(),
            loss_type=self.loss_type,
            base_weight=base_weight,
            weighting_mode=self.weighting_mode,
            adaptive_power=self.adaptive_weight_power,
            adaptive_bias=self.adaptive_weight_bias,
        )
        loss_dict = {
            "alphaflow_loss": loss,
            "total_loss": loss,
            "alpha": alpha.mean(),
            "delta_t_mean": delta_t.mean(),
            "flow_matching_ratio": force_flow_matching.float().mean(),
            "r_equals_t_ratio": (delta_t == 0).float().mean(),
            **weighting_stats,
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
            "objective_branch": objective_branch,
            "weighting_stats": weighting_stats,
            "base_weight": base_weight,
            "force_flow_matching": force_flow_matching,
        }
