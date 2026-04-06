import torch
import torch.nn as nn

from .latent_velocity_convnet import SinusoidalTimeEmbedding, ResidualTimeBlock, _group_norm_groups


def _make_time_embed_mlp(time_embed_dim):
    return nn.Sequential(
        SinusoidalTimeEmbedding(time_embed_dim),
        nn.Linear(time_embed_dim, time_embed_dim),
        nn.SiLU(),
        nn.Linear(time_embed_dim, time_embed_dim),
    )


class LatentIntervalVelocityConvNet(nn.Module):
    def __init__(
        self,
        in_channels,
        model_channels=128,
        time_embed_dim=256,
        num_res_blocks=4,
        dropout=0.0,
        condition_num_classes=None,
        time_conditioning="t_delta",
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.model_channels = int(model_channels)
        self.time_embed_dim = int(time_embed_dim)
        self.condition_num_classes = None if condition_num_classes is None else int(condition_num_classes)
        self.time_conditioning = str(time_conditioning)
        if self.time_conditioning not in {"t", "t_delta", "r_t"}:
            raise ValueError(
                f"Unsupported time_conditioning: {self.time_conditioning}. Use 't', 't_delta', or 'r_t'."
            )

        self.t_embed = _make_time_embed_mlp(self.time_embed_dim)
        self.delta_embed = _make_time_embed_mlp(self.time_embed_dim) if self.time_conditioning == "t_delta" else None
        self.r_embed = _make_time_embed_mlp(self.time_embed_dim) if self.time_conditioning == "r_t" else None

        self.cond_embed = None
        if self.condition_num_classes is not None:
            self.cond_embed = nn.Embedding(self.condition_num_classes, self.time_embed_dim)

        self.conv_in = nn.Conv2d(self.in_channels, self.model_channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [
                ResidualTimeBlock(
                    channels=self.model_channels,
                    time_embed_dim=self.time_embed_dim,
                    dropout=dropout,
                )
                for _ in range(int(num_res_blocks))
            ]
        )
        groups = _group_norm_groups(self.model_channels)
        self.norm_out = nn.GroupNorm(groups, self.model_channels)
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv2d(self.model_channels, self.in_channels, kernel_size=3, padding=1)

    def _resolve_time_embedding(self, t, r=None, delta_t=None):
        if t is None:
            raise ValueError("t must be provided")
        if t.ndim == 0:
            t = t[None]
        if t.ndim != 1:
            raise ValueError(f"Expected t with shape [B], got {tuple(t.shape)}")

        time_embed = self.t_embed(t.float())

        if self.time_conditioning == "t_delta":
            if delta_t is None:
                if r is None:
                    raise ValueError("delta_t or r must be provided when time_conditioning='t_delta'")
                delta_t = t - r
            time_embed = time_embed + self.delta_embed(delta_t.float())
        elif self.time_conditioning == "r_t":
            if r is None:
                raise ValueError("r must be provided when time_conditioning='r_t'")
            time_embed = time_embed + self.r_embed(r.float())

        return time_embed

    def forward(self, z_t, t=None, condition=None, r=None, delta_t=None):
        time_embed = self._resolve_time_embedding(t=t, r=r, delta_t=delta_t)
        if condition is not None:
            if self.cond_embed is None:
                raise ValueError("Conditioning provided but condition_num_classes is not configured")
            condition = condition.long()
            time_embed = time_embed + self.cond_embed(condition)

        h = self.conv_in(z_t)
        for block in self.blocks:
            h = block(h, time_embed)
        h = self.conv_out(self.act_out(self.norm_out(h)))
        return h
