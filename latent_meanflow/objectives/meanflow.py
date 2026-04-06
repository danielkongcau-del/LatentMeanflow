import torch
import torch.nn as nn

from .common import (
    build_time_sampler,
    expand_time_like,
    rectified_path,
    sample_interval,
    weighted_regression_loss,
)


def meanflow_jvp(model_fn, z_t, r, t, velocity, condition=None):
    tangents = (velocity, torch.zeros_like(r), torch.ones_like(t))

    def wrapped(z_arg, r_arg, t_arg):
        return model_fn(z_arg, r=r_arg, t=t_arg, delta_t=t_arg - r_arg, condition=condition)

    average_velocity, total_derivative = torch.func.jvp(
        wrapped,
        primals=(z_t, r, t),
        tangents=tangents,
    )
    return average_velocity, total_derivative


class MeanFlowObjective(nn.Module):
    name = "meanflow"
    prediction_type = "average_velocity"

    def __init__(
        self,
        time_eps=1.0e-4,
        min_delta=1.0e-3,
        loss_type="mse",
        time_sampler_config=None,
        r_equals_t_ratio=0.0,
        border_fm_ratio=None,
        weighting_mode="paper_like",
        adaptive_weight_power=1.0,
        adaptive_weight_bias=1.0e-4,
    ):
        super().__init__()
        self.time_eps = float(time_eps)
        self.min_delta = float(min_delta)
        self.loss_type = str(loss_type)
        if border_fm_ratio is not None:
            if float(r_equals_t_ratio) != 0.0 and float(r_equals_t_ratio) != float(border_fm_ratio):
                raise ValueError("r_equals_t_ratio and border_fm_ratio disagree")
            r_equals_t_ratio = border_fm_ratio
        self.r_equals_t_ratio = float(r_equals_t_ratio)
        self.weighting_mode = str(weighting_mode)
        self.adaptive_weight_power = float(adaptive_weight_power)
        self.adaptive_weight_bias = float(adaptive_weight_bias)
        self.time_sampler = build_time_sampler(time_sampler_config=time_sampler_config, time_eps=self.time_eps)

    def forward(self, model_fn, x_lat, condition=None, **kwargs):
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
        average_velocity, total_derivative = meanflow_jvp(
            model_fn=model_fn,
            z_t=z_t,
            r=r,
            t=t,
            velocity=velocity,
            condition=condition,
        )
        target_field = velocity - expand_time_like(delta_t, x_lat) * total_derivative
        loss, weighting_stats = weighted_regression_loss(
            average_velocity,
            target_field.detach(),
            loss_type=self.loss_type,
            weighting_mode=self.weighting_mode,
            adaptive_power=self.adaptive_weight_power,
            adaptive_bias=self.adaptive_weight_bias,
        )
        border_fm_ratio = torch.mean((delta_t == 0).float())
        return {
            "loss": loss,
            "meanflow_loss": loss,
            "loss_dict": {
                "meanflow_loss": loss,
                "total_loss": loss,
                "delta_t_mean": delta_t.mean(),
                "border_fm_ratio": border_fm_ratio,
                "r_equals_t_ratio": border_fm_ratio,
                **weighting_stats,
            },
            "r": r,
            "t": t,
            "delta_t": delta_t,
            "noise": noise,
            "z_t": z_t,
            "velocity": velocity,
            "pred_field": average_velocity,
            "target_field": target_field.detach(),
            "total_derivative": total_derivative,
            "weighting_stats": weighting_stats,
            "border_fm_mask": delta_t == 0,
        }
