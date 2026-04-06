import torch
import torch.nn.functional as F


def regression_loss(prediction, target, loss_type="mse"):
    loss_type = str(loss_type)
    if loss_type == "mse":
        return F.mse_loss(prediction, target)
    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(prediction, target)
    raise ValueError(f"Unsupported loss_type: {loss_type}")


def expand_time_like(time_values, reference):
    return time_values.view(-1, *([1] * (reference.ndim - 1)))


def rectified_path(x_lat, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_lat)
    t_view = expand_time_like(t, x_lat)
    z_t = (1.0 - t_view) * x_lat + t_view * noise
    velocity = noise - x_lat
    return z_t, velocity, noise


def sample_time(batch_size, device, time_eps):
    return torch.rand(batch_size, device=device) * (1.0 - 2.0 * time_eps) + time_eps


def sample_interval(batch_size, device, time_eps=1.0e-4, min_delta=1.0e-3):
    time_eps = float(time_eps)
    min_delta = float(min_delta)
    if min_delta <= 0.0:
        raise ValueError(f"min_delta must be positive, got {min_delta}")
    if 2.0 * time_eps + min_delta >= 1.0:
        raise ValueError(
            f"Invalid time range: time_eps={time_eps} and min_delta={min_delta} leave no valid interval"
        )

    t = torch.rand(batch_size, device=device) * (1.0 - 2.0 * time_eps - min_delta) + (time_eps + min_delta)
    max_r = t - min_delta
    r = torch.rand(batch_size, device=device) * (max_r - time_eps) + time_eps
    delta_t = t - r
    return r, t, delta_t
