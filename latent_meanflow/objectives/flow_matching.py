import torch.nn as nn

from .common import rectified_path, regression_loss, sample_time


class RectifiedFlowMatchingObjective(nn.Module):
    name = "fm"
    prediction_type = "instantaneous_velocity"

    def __init__(self, time_eps=1.0e-4, loss_type="mse"):
        super().__init__()
        self.time_eps = float(time_eps)
        self.loss_type = str(loss_type)

    def sample_time(self, batch_size, device):
        return sample_time(batch_size, device=device, time_eps=self.time_eps)

    def make_path(self, x_lat, t, noise=None):
        z_t, target_velocity, noise = rectified_path(x_lat, t=t, noise=noise)
        return z_t, target_velocity, noise

    def compute_loss(self, pred_velocity, target_velocity):
        return regression_loss(pred_velocity, target_velocity, loss_type=self.loss_type)

    def forward(self, backbone, x_lat, condition=None, **kwargs):
        batch_size = x_lat.shape[0]
        t = self.sample_time(batch_size, device=x_lat.device)
        z_t, target_velocity, noise = self.make_path(x_lat, t=t)
        pred_velocity = backbone(z_t, t=t, condition=condition)
        fm_loss = self.compute_loss(pred_velocity, target_velocity)
        return {
            "loss": fm_loss,
            "fm_loss": fm_loss,
            "loss_dict": {
                "fm_loss": fm_loss,
                "total_loss": fm_loss,
            },
            "t": t,
            "noise": noise,
            "z_t": z_t,
            "target_velocity": target_velocity,
            "pred_velocity": pred_velocity,
            "pred_field": pred_velocity,
            "target_field": target_velocity,
        }
