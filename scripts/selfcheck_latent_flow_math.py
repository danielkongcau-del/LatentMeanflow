import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
LDM_ROOT = REPO_ROOT / "third_party" / "latent-diffusion"
TAMING_ROOT = LDM_ROOT / "taming-transformers"

for path in (REPO_ROOT, LDM_ROOT, TAMING_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from latent_meanflow.objectives.alphaflow import AlphaFlowObjective
from latent_meanflow.objectives.common import (
    ConstantTimeSampler,
    expand_time_like,
    rectified_path,
    weighted_regression_loss,
)
from latent_meanflow.objectives.meanflow import meanflow_jvp
from latent_meanflow.samplers.interval import IntervalFlowSampler


def analytic_field(z_t, r=None, t=None, delta_t=None, condition=None):
    _ = condition
    t_view = expand_time_like(t, z_t)
    r_view = expand_time_like(r, z_t)
    return z_t * (t_view - 2.0 * r_view) + 0.5 * (t_view**2) + 3.0 * r_view


def check_rectified_path():
    x_lat = torch.tensor([[[[1.0, 2.0]]]])
    noise = torch.tensor([[[[5.0, 9.0]]]])
    t = torch.tensor([0.25])
    z_t, velocity, returned_noise = rectified_path(x_lat, t=t, noise=noise)
    expected_z_t = (1.0 - 0.25) * x_lat + 0.25 * noise
    expected_velocity = noise - x_lat
    torch.testing.assert_close(z_t, expected_z_t)
    torch.testing.assert_close(velocity, expected_velocity)
    torch.testing.assert_close(returned_noise, noise)


def check_meanflow_jvp_tangent():
    torch.manual_seed(0)
    z_t = torch.randn(2, 3, 4, 4)
    velocity = torch.randn_like(z_t)
    r = torch.tensor([0.15, 0.35], dtype=z_t.dtype)
    t = torch.tensor([0.65, 0.85], dtype=z_t.dtype)

    average_velocity, total_derivative = meanflow_jvp(
        model_fn=analytic_field,
        z_t=z_t,
        r=r,
        t=t,
        velocity=velocity,
        condition=None,
    )

    t_view = expand_time_like(t, z_t)
    r_view = expand_time_like(r, z_t)
    expected_average_velocity = analytic_field(z_t, r=r, t=t)
    expected_total_derivative = velocity * (t_view - 2.0 * r_view) + z_t + t_view

    torch.testing.assert_close(average_velocity, expected_average_velocity, atol=1.0e-6, rtol=1.0e-6)
    torch.testing.assert_close(total_derivative, expected_total_derivative, atol=1.0e-6, rtol=1.0e-6)


def check_alphaflow_alpha_one_degeneracy():
    torch.manual_seed(0)
    objective = AlphaFlowObjective(
        time_eps=1.0e-4,
        min_delta=0.0,
        loss_type="mse",
        time_sampler_config={
            "target": "latent_meanflow.objectives.common.UniformTimeSampler",
            "params": {"time_eps": 1.0e-4},
        },
        trajectory_fm_ratio=0.0,
        border_fm_ratio=0.0,
        weighting_mode="none",
        alpha_schedule_config={
            "target": "latent_meanflow.objectives.alphaflow.ConstantAlphaScheduler",
            "params": {"value": 1.0},
        },
    )

    x_lat = torch.randn(2, 4, 8, 8)

    def zero_model(z_t, r=None, t=None, delta_t=None, condition=None):
        _ = r, t, delta_t, condition
        return torch.zeros_like(z_t)

    outputs = objective(zero_model, x_lat, condition=None, global_step=0)
    torch.testing.assert_close(outputs["alpha"], torch.ones_like(outputs["alpha"]))
    torch.testing.assert_close(outputs["target_field"], outputs["velocity"])
    torch.testing.assert_close(outputs["base_weight"], torch.ones_like(outputs["base_weight"]))
    assert torch.all(~outputs["border_fm_mask"])


def check_border_case_flow_matching_degeneracy():
    torch.manual_seed(0)
    objective = AlphaFlowObjective(
        time_eps=1.0e-4,
        min_delta=0.0,
        loss_type="mse",
        time_sampler_config={
            "target": "latent_meanflow.objectives.common.ConstantTimeSampler",
            "params": {"value": 0.7, "time_eps": 1.0e-4},
        },
        trajectory_fm_ratio=0.0,
        border_fm_ratio=1.0,
        weighting_mode="none",
        alpha_schedule_config={
            "target": "latent_meanflow.objectives.alphaflow.ConstantAlphaScheduler",
            "params": {"value": 0.0},
        },
    )

    x_lat = torch.randn(2, 4, 8, 8)

    def zero_model(z_t, r=None, t=None, delta_t=None, condition=None):
        _ = r, t, delta_t, condition
        return torch.zeros_like(z_t)

    outputs = objective(zero_model, x_lat, condition=None, global_step=0)
    torch.testing.assert_close(outputs["target_field"], outputs["velocity"])
    assert torch.all(outputs["border_fm_mask"])
    assert torch.all(~outputs["trajectory_fm_mask"])


def check_semantics_not_confused():
    torch.manual_seed(0)
    x_lat = torch.randn(2, 4, 8, 8)

    def zero_model(z_t, r=None, t=None, delta_t=None, condition=None):
        _ = r, t, delta_t, condition
        return torch.zeros_like(z_t)

    trajectory_objective = AlphaFlowObjective(
        time_eps=1.0e-4,
        min_delta=0.0,
        loss_type="mse",
        time_sampler_config={
            "target": "latent_meanflow.objectives.common.UniformTimeSampler",
            "params": {"time_eps": 1.0e-4},
        },
        trajectory_fm_ratio=0.0,
        border_fm_ratio=0.0,
        weighting_mode="none",
        alpha_schedule_config={
            "target": "latent_meanflow.objectives.alphaflow.ConstantAlphaScheduler",
            "params": {"value": 1.0},
        },
    )
    border_objective = AlphaFlowObjective(
        time_eps=1.0e-4,
        min_delta=0.0,
        loss_type="mse",
        time_sampler_config={
            "target": "latent_meanflow.objectives.common.ConstantTimeSampler",
            "params": {"value": 0.7, "time_eps": 1.0e-4},
        },
        trajectory_fm_ratio=0.0,
        border_fm_ratio=1.0,
        weighting_mode="none",
        alpha_schedule_config={
            "target": "latent_meanflow.objectives.alphaflow.ConstantAlphaScheduler",
            "params": {"value": 0.0},
        },
    )

    trajectory_outputs = trajectory_objective(zero_model, x_lat, condition=None, global_step=0)
    border_outputs = border_objective(zero_model, x_lat, condition=None, global_step=0)
    assert torch.all(~trajectory_outputs["border_fm_mask"])
    assert torch.all(border_outputs["border_fm_mask"])
    assert not torch.allclose(trajectory_outputs["r"], border_outputs["r"])
    torch.testing.assert_close(trajectory_outputs["target_field"], trajectory_outputs["velocity"])
    torch.testing.assert_close(border_outputs["target_field"], border_outputs["velocity"])


def check_alpha_adaptive_exact_weighting():
    prediction = torch.tensor([[[[2.0]]], [[[3.0]]]])
    target = torch.tensor([[[[1.0]]], [[[1.0]]]])
    alpha = torch.tensor([1.0, 0.25])
    loss, stats = weighted_regression_loss(
        prediction,
        target,
        loss_type="mse",
        base_weight=1.0 / alpha,
        weighting_mode="alpha_adaptive_exact",
        adaptive_power=1.0,
        adaptive_bias=1.0e-4,
        alpha=alpha,
    )
    base_error = torch.tensor([1.0, 4.0])
    expected_weight = alpha / (base_error + alpha * 1.0e-4)
    expected_loss = torch.mean(expected_weight * (base_error / alpha))
    torch.testing.assert_close(loss, expected_loss)
    torch.testing.assert_close(stats["adaptive_weight_mean"], expected_weight.mean())


def check_interval_sampler_grids():
    sampler = IntervalFlowSampler(default_nfe=1, two_step_time=0.4)
    grid_1 = sampler.build_time_grid(nfe=1, device=torch.device("cpu"))
    grid_2 = sampler.build_time_grid(nfe=2, device=torch.device("cpu"))
    torch.testing.assert_close(grid_1, torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(grid_2, torch.tensor([1.0, 0.4, 0.0]))


def main():
    _ = ConstantTimeSampler
    check_rectified_path()
    check_meanflow_jvp_tangent()
    check_alphaflow_alpha_one_degeneracy()
    check_border_case_flow_matching_degeneracy()
    check_semantics_not_confused()
    check_alpha_adaptive_exact_weighting()
    check_interval_sampler_grids()
    print("Latent flow math self-check passed")


if __name__ == "__main__":
    main()
