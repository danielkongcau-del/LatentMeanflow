import torch
import torch.nn as nn
import torch.nn.functional as F


class RectifiedFlowMatchingObjective(nn.Module):
    def __init__(self, time_eps=1.0e-4, loss_type="mse"):
        super().__init__()
        self.time_eps = float(time_eps)
        self.loss_type = str(loss_type)

    def sample_time(self, batch_size, device):
        return torch.rand(batch_size, device=device) * (1.0 - 2.0 * self.time_eps) + self.time_eps

    def make_path(self, x_lat, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_lat)
        t_view = t.view(-1, *([1] * (x_lat.ndim - 1)))
        z_t = (1.0 - t_view) * x_lat + t_view * noise
        target_velocity = noise - x_lat
        return z_t, target_velocity, noise

    def compute_loss(self, pred_velocity, target_velocity):
        if self.loss_type == "mse":
            return F.mse_loss(pred_velocity, target_velocity)
        if self.loss_type == "smooth_l1":
            return F.smooth_l1_loss(pred_velocity, target_velocity)
        raise ValueError(f"Unsupported flow matching loss_type: {self.loss_type}")

    def forward(self, backbone, x_lat, condition=None):
        batch_size = x_lat.shape[0]
        t = self.sample_time(batch_size, device=x_lat.device)
        z_t, target_velocity, noise = self.make_path(x_lat, t=t)
        pred_velocity = backbone(z_t, t, condition=condition)
        fm_loss = self.compute_loss(pred_velocity, target_velocity)
        return {
            "loss": fm_loss,
            "fm_loss": fm_loss,
            "t": t,
            "noise": noise,
            "z_t": z_t,
            "target_velocity": target_velocity,
            "pred_velocity": pred_velocity,
        }
