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

from latent_meanflow.objectives.common import expand_time_like
from latent_meanflow.objectives.meanflow import meanflow_jvp


def analytic_field(z_t, r=None, t=None, delta_t=None, condition=None):
    t_view = expand_time_like(t, z_t)
    r_view = expand_time_like(r, z_t)
    return z_t * (t_view - 2.0 * r_view) + 0.5 * (t_view ** 2) + 3.0 * r_view


def main():
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

    print("MeanFlow JVP self-check passed")
    print(f"average_velocity shape: {tuple(average_velocity.shape)}")
    print(f"total_derivative shape: {tuple(total_derivative.shape)}")


if __name__ == "__main__":
    main()
