from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from ldm.util import instantiate_from_config


def expand_time_like(time_values, reference):
    return time_values.view(-1, *([1] * (reference.ndim - 1)))


def rectified_path(x_lat, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_lat)
    t_view = expand_time_like(t, x_lat)
    z_t = (1.0 - t_view) * x_lat + t_view * noise
    velocity = noise - x_lat
    return z_t, velocity, noise


class UniformTimeSampler(nn.Module):
    def __init__(self, time_eps=1.0e-4):
        super().__init__()
        self.time_eps = float(time_eps)

    def sample(self, batch_size, device, dtype=torch.float32):
        return (
            torch.rand(batch_size, device=device, dtype=dtype) * (1.0 - 2.0 * self.time_eps)
            + self.time_eps
        )

    def clamp(self, values):
        return values.clamp(min=self.time_eps, max=1.0 - self.time_eps)


class LogitNormalTimeSampler(nn.Module):
    def __init__(self, loc=0.0, scale=1.0, time_eps=1.0e-4):
        super().__init__()
        self.loc = float(loc)
        self.scale = float(scale)
        self.time_eps = float(time_eps)
        if self.scale <= 0.0:
            raise ValueError(f"scale must be positive, got {self.scale}")

    def sample(self, batch_size, device, dtype=torch.float32):
        logits = torch.randn(batch_size, device=device, dtype=dtype) * self.scale + self.loc
        values = torch.sigmoid(logits)
        return self.clamp(values)

    def clamp(self, values):
        return values.clamp(min=self.time_eps, max=1.0 - self.time_eps)


class ConstantTimeSampler(nn.Module):
    def __init__(self, value=0.5, time_eps=1.0e-4):
        super().__init__()
        self.value = float(value)
        self.time_eps = float(time_eps)
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"value must be in [0, 1], got {self.value}")

    def sample(self, batch_size, device, dtype=torch.float32):
        values = torch.full((batch_size,), self.value, device=device, dtype=dtype)
        return self.clamp(values)

    def clamp(self, values):
        return values.clamp(min=self.time_eps, max=1.0 - self.time_eps)


def build_time_sampler(time_sampler_config=None, time_eps=1.0e-4):
    if time_sampler_config is None:
        time_sampler_config = {
            "target": "latent_meanflow.objectives.common.UniformTimeSampler",
            "params": {"time_eps": float(time_eps)},
        }
    config = deepcopy(time_sampler_config)
    config.setdefault("params", {})
    config["params"].setdefault("time_eps", float(time_eps))
    return instantiate_from_config(config)


def sample_time(batch_size, device, time_sampler, dtype=torch.float32):
    return time_sampler.sample(batch_size=batch_size, device=device, dtype=dtype)


def sample_interval(
    batch_size,
    device,
    time_sampler,
    *,
    min_delta=1.0e-3,
    r_equals_t_ratio=0.0,
    dtype=torch.float32,
):
    min_delta = float(min_delta)
    r_equals_t_ratio = float(r_equals_t_ratio)
    if not 0.0 <= r_equals_t_ratio <= 1.0:
        raise ValueError(f"r_equals_t_ratio must be in [0, 1], got {r_equals_t_ratio}")
    if min_delta < 0.0:
        raise ValueError(f"min_delta must be non-negative, got {min_delta}")

    first = time_sampler.sample(batch_size=batch_size, device=device, dtype=dtype)
    second = time_sampler.sample(batch_size=batch_size, device=device, dtype=dtype)
    r = torch.minimum(first, second)
    t = torch.maximum(first, second)

    if min_delta > 0.0:
        too_close = (t - r) < min_delta
        if torch.any(too_close):
            t = torch.where(too_close, time_sampler.clamp(r + min_delta), t)
            still_too_close = (t - r) < min_delta
            if torch.any(still_too_close):
                r = torch.where(still_too_close, time_sampler.clamp(t - min_delta), r)

    if r_equals_t_ratio > 0.0:
        equal_mask = torch.rand(batch_size, device=device) < r_equals_t_ratio
        r = torch.where(equal_mask, t, r)

    delta_t = t - r
    return r, t, delta_t


def _per_sample_reduction(error_map):
    return error_map.reshape(error_map.shape[0], -1).mean(dim=1)


def per_sample_regression_error(prediction, target, loss_type="mse"):
    loss_type = str(loss_type)
    if loss_type == "mse":
        return _per_sample_reduction((prediction - target) ** 2)
    if loss_type == "smooth_l1":
        return _per_sample_reduction(F.smooth_l1_loss(prediction, target, reduction="none"))
    raise ValueError(f"Unsupported loss_type: {loss_type}")


def regression_loss(prediction, target, loss_type="mse"):
    return per_sample_regression_error(prediction, target, loss_type=loss_type).mean()


def compute_adaptive_weight(
    base_error,
    *,
    weighting_mode="none",
    adaptive_power=1.0,
    adaptive_bias=1.0e-4,
    alpha=None,
):
    weighting_mode = str(weighting_mode)
    adaptive_power = float(adaptive_power)
    adaptive_bias = float(adaptive_bias)

    if weighting_mode == "none":
        return torch.ones_like(base_error)

    if weighting_mode == "paper_like":
        # MeanFlow Eq. (22)-style adaptive weighting:
        #   base_error = ||Δ||^2
        #   w = 1 / (base_error + c)^p
        return torch.pow(base_error + adaptive_bias, -adaptive_power)

    if weighting_mode in {"alpha_adaptive_exact", "alpha_adaptive"}:
        if alpha is None:
            raise ValueError(f"weighting_mode='{weighting_mode}' requires alpha")
        alpha = alpha.to(device=base_error.device, dtype=base_error.dtype)
        # AlphaFlow exact reformulation from the base squared error:
        #   base_error = ||Δ||^2
        #   L_alpha = alpha^{-1} * ||Δ||^2
        # Choosing the adaptive loss on L_alpha and rewriting it in terms of
        # base_error yields:
        #   w_alpha = alpha^p / (base_error + alpha * c)^p
        # The final weighted objective is:
        #   stopgrad(w_alpha) * alpha^{-1} * base_error
        numerator = torch.pow(alpha, adaptive_power)
        denominator = torch.pow(base_error + alpha * adaptive_bias, adaptive_power)
        return numerator / denominator

    raise ValueError(f"Unsupported weighting_mode: {weighting_mode}")


def weighted_regression_loss(
    prediction,
    target,
    *,
    loss_type="mse",
    base_weight=None,
    weighting_mode="none",
    adaptive_power=1.0,
    adaptive_bias=1.0e-4,
    alpha=None,
):
    base_error = per_sample_regression_error(prediction, target, loss_type=loss_type)
    if base_weight is None:
        base_weight = torch.ones_like(base_error)
    else:
        base_weight = base_weight.to(device=base_error.device, dtype=base_error.dtype)

    adaptive_weight = compute_adaptive_weight(
        base_error,
        weighting_mode=weighting_mode,
        adaptive_power=adaptive_power,
        adaptive_bias=adaptive_bias,
        alpha=alpha,
    ).detach()

    scaled_error = base_weight * base_error
    weighted_error = adaptive_weight * scaled_error
    loss = weighted_error.mean()
    return loss, {
        "base_error_mean": base_error.mean(),
        "base_weight_mean": base_weight.mean(),
        "adaptive_weight_mean": adaptive_weight.mean(),
        "weighted_error_mean": weighted_error.mean(),
    }
