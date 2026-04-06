import torch
import torch.nn as nn

from .common import expand_time_like, rectified_path, regression_loss, sample_interval


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

    def __init__(self, time_eps=1.0e-4, min_delta=1.0e-3, loss_type="mse"):
        super().__init__()
        self.time_eps = float(time_eps)
        self.min_delta = float(min_delta)
        self.loss_type = str(loss_type)

    def forward(self, model_fn, x_lat, condition=None, **kwargs):
        batch_size = x_lat.shape[0]
        r, t, delta_t = sample_interval(
            batch_size=batch_size,
            device=x_lat.device,
            time_eps=self.time_eps,
            min_delta=self.min_delta,
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
        loss = regression_loss(average_velocity, target_field.detach(), loss_type=self.loss_type)
        return {
            "loss": loss,
            "meanflow_loss": loss,
            "loss_dict": {
                "meanflow_loss": loss,
                "total_loss": loss,
                "delta_t_mean": delta_t.mean(),
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
        }
