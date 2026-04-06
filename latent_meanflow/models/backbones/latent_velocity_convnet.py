import math

import torch
import torch.nn as nn


def _group_norm_groups(num_channels, max_groups=32):
    for num_groups in range(min(max_groups, num_channels), 0, -1):
        if num_channels % num_groups == 0:
            return num_groups
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = int(dim)

    def forward(self, t):
        half_dim = self.dim // 2
        if half_dim == 0:
            return t[:, None]
        exponent = -math.log(10000.0) / max(half_dim - 1, 1)
        frequencies = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * exponent)
        angles = t[:, None] * frequencies[None, :]
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if self.dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=1)
        return embedding


class ResidualTimeBlock(nn.Module):
    def __init__(self, channels, time_embed_dim, dropout=0.0):
        super().__init__()
        groups = _group_norm_groups(channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_embed_dim, channels * 2)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x, time_embed):
        residual = x
        h = self.conv1(self.act1(self.norm1(x)))
        scale, shift = self.time_proj(time_embed).chunk(2, dim=1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        h = self.norm2(h)
        h = h * (1.0 + scale) + shift
        h = self.conv2(self.dropout(self.act2(h)))
        return residual + h


class LatentVelocityConvNet(nn.Module):
    def __init__(
        self,
        in_channels,
        model_channels=128,
        time_embed_dim=256,
        num_res_blocks=4,
        dropout=0.0,
        condition_num_classes=None,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.model_channels = int(model_channels)
        self.time_embed_dim = int(time_embed_dim)
        self.condition_num_classes = None if condition_num_classes is None else int(condition_num_classes)

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(self.time_embed_dim),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )
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

    def forward(self, z_t, t, condition=None):
        if t.ndim == 0:
            t = t[None]
        if t.ndim != 1:
            raise ValueError(f"Expected t with shape [B], got {tuple(t.shape)}")
        time_embed = self.time_embed(t.float())

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
