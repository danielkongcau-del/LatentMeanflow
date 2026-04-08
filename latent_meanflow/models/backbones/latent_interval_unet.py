import math
from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from .latent_velocity_convnet import _group_norm_groups


def _conv_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    if dims == 2:
        return nn.Conv2d(*args, **kwargs)
    if dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def _linear(*args, **kwargs):
    return nn.Linear(*args, **kwargs)


def _avg_pool_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    if dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    if dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def _zero_module(module):
    for parameter in module.parameters():
        parameter.detach().zero_()
    return module


def _normalization(channels):
    return nn.GroupNorm(_group_norm_groups(channels), channels)


def _timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    if half == 0:
        return timesteps[:, None]
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
        / max(half, 1)
    )
    args = timesteps[:, None].float() * frequencies[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def _maybe_checkpoint(function, inputs, enabled):
    if enabled and torch.is_grad_enabled():
        return torch.utils.checkpoint.checkpoint(function, *inputs, use_reentrant=False)
    return function(*inputs)


def _make_time_embed_mlp(input_dim, output_dim):
    return nn.Sequential(
        _linear(input_dim, output_dim),
        nn.SiLU(),
        _linear(output_dim, output_dim),
    )


class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        raise NotImplementedError


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = int(channels)
        self.out_channels = int(out_channels if out_channels is not None else channels)
        self.use_conv = bool(use_conv)
        self.dims = int(dims)
        self.conv = None
        if self.use_conv:
            self.conv = _conv_nd(self.dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        if x.shape[1] != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {x.shape[1]}")
        if self.dims == 3:
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.conv is not None:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = int(channels)
        self.out_channels = int(out_channels if out_channels is not None else channels)
        self.use_conv = bool(use_conv)
        self.dims = int(dims)
        stride = 2 if self.dims != 3 else (1, 2, 2)
        if self.use_conv:
            self.op = _conv_nd(
                self.dims,
                self.channels,
                self.out_channels,
                3,
                stride=stride,
                padding=1,
            )
        else:
            if self.channels != self.out_channels:
                raise ValueError("avg-pool downsample requires matching in/out channels")
            self.op = _avg_pool_nd(self.dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        if x.shape[1] != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {x.shape[1]}")
        return self.op(x)


class ResBlock(TimestepBlock):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = int(channels)
        self.emb_channels = int(emb_channels)
        self.dropout = float(dropout)
        self.out_channels = int(out_channels if out_channels is not None else channels)
        self.use_conv = bool(use_conv)
        self.use_scale_shift_norm = bool(use_scale_shift_norm)
        self.use_checkpoint = bool(use_checkpoint)

        self.in_layers = nn.Sequential(
            _normalization(self.channels),
            nn.SiLU(),
            _conv_nd(dims, self.channels, self.out_channels, 3, padding=1),
        )

        self.updown = bool(up or down)
        if up:
            self.h_upd = Upsample(self.channels, False, dims)
            self.x_upd = Upsample(self.channels, False, dims)
        elif down:
            self.h_upd = Downsample(self.channels, False, dims)
            self.x_upd = Downsample(self.channels, False, dims)
        else:
            self.h_upd = nn.Identity()
            self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            _linear(
                self.emb_channels,
                2 * self.out_channels if self.use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            _normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=self.dropout),
            _zero_module(_conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == self.channels:
            self.skip_connection = nn.Identity()
        elif self.use_conv:
            self.skip_connection = _conv_nd(dims, self.channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = _conv_nd(dims, self.channels, self.out_channels, 1)

    def forward(self, x, emb):
        return _maybe_checkpoint(self._forward, (x, emb), self.use_checkpoint and self.training)

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        emb_out = self.emb_layers(emb).type(h.dtype)
        while emb_out.ndim < h.ndim:
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class QKVAttention(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = int(num_heads)

    def forward(self, qkv):
        batch_size, width, length = qkv.shape
        if width % (3 * self.num_heads) != 0:
            raise ValueError("qkv width must be divisible by 3 * num_heads")
        channels = width // (3 * self.num_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1.0 / math.sqrt(math.sqrt(channels))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).reshape(batch_size * self.num_heads, channels, length),
            (k * scale).reshape(batch_size * self.num_heads, channels, length),
        )
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        attended = torch.einsum(
            "bts,bcs->bct",
            weight,
            v.reshape(batch_size * self.num_heads, channels, length),
        )
        return attended.reshape(batch_size, -1, length)


class AttentionBlock(nn.Module):
    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
    ):
        super().__init__()
        self.channels = int(channels)
        self.use_checkpoint = bool(use_checkpoint)
        if num_head_channels == -1:
            self.num_heads = int(num_heads)
        else:
            if self.channels % int(num_head_channels) != 0:
                raise ValueError("channels must be divisible by num_head_channels")
            self.num_heads = self.channels // int(num_head_channels)
        self.norm = _normalization(self.channels)
        self.qkv = _conv_nd(1, self.channels, self.channels * 3, 1)
        self.attention = QKVAttention(self.num_heads)
        self.proj_out = _zero_module(_conv_nd(1, self.channels, self.channels, 1))

    def forward(self, x):
        return _maybe_checkpoint(self._forward, (x,), self.use_checkpoint and self.training)

    def _forward(self, x):
        batch_size, channels, *spatial = x.shape
        h = x.reshape(batch_size, channels, -1)
        h = self.qkv(self.norm(h))
        h = self.attention(h)
        h = self.proj_out(h)
        return (x.reshape(batch_size, channels, -1) + h).reshape(batch_size, channels, *spatial)


class LatentIntervalUNet(nn.Module):
    def __init__(
        self,
        in_channels,
        model_channels=128,
        time_embed_dim=None,
        num_res_blocks=2,
        channel_mult=(1, 2, 4),
        attention_resolutions=(2, 4),
        dropout=0.0,
        condition_num_classes=None,
        time_conditioning="t_delta",
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        num_heads=4,
        num_head_channels=64,
        use_scale_shift_norm=True,
        resblock_updown=True,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.model_channels = int(model_channels)
        self.time_embed_dim = int(time_embed_dim if time_embed_dim is not None else self.model_channels * 4)
        self.num_res_blocks = int(num_res_blocks)
        self.channel_mult = tuple(int(mult) for mult in channel_mult)
        self.attention_resolutions = {int(resolution) for resolution in attention_resolutions}
        self.dropout = float(dropout)
        self.condition_num_classes = None if condition_num_classes is None else int(condition_num_classes)
        self.time_conditioning = str(time_conditioning)
        self.conv_resample = bool(conv_resample)
        self.dims = int(dims)
        self.use_checkpoint = bool(use_checkpoint)
        self.num_heads = int(num_heads)
        self.num_head_channels = int(num_head_channels)
        self.use_scale_shift_norm = bool(use_scale_shift_norm)
        self.resblock_updown = bool(resblock_updown)

        if self.time_conditioning not in {"t", "t_delta", "r_t"}:
            raise ValueError(
                f"Unsupported time_conditioning: {self.time_conditioning}. Use 't', 't_delta', or 'r_t'."
            )
        if not self.channel_mult:
            raise ValueError("channel_mult must not be empty")

        self.t_embed = _make_time_embed_mlp(self.model_channels, self.time_embed_dim)
        self.delta_embed = (
            _make_time_embed_mlp(self.model_channels, self.time_embed_dim)
            if self.time_conditioning == "t_delta"
            else None
        )
        self.r_embed = (
            _make_time_embed_mlp(self.model_channels, self.time_embed_dim)
            if self.time_conditioning == "r_t"
            else None
        )
        self.cond_embed = None
        if self.condition_num_classes is not None:
            self.cond_embed = nn.Embedding(self.condition_num_classes, self.time_embed_dim)

        ch = input_ch = int(self.channel_mult[0] * self.model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(_conv_nd(self.dims, self.in_channels, ch, 3, padding=1))]
        )
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(self.channel_mult):
            for _ in range(self.num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        self.time_embed_dim,
                        self.dropout,
                        out_channels=int(mult * self.model_channels),
                        dims=self.dims,
                        use_checkpoint=self.use_checkpoint,
                        use_scale_shift_norm=self.use_scale_shift_norm,
                    )
                ]
                ch = int(mult * self.model_channels)
                if ds in self.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=self.num_heads,
                            num_head_channels=self.num_head_channels,
                            use_checkpoint=self.use_checkpoint,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(self.channel_mult) - 1:
                out_channels = ch
                if self.resblock_updown:
                    down_block = ResBlock(
                        ch,
                        self.time_embed_dim,
                        self.dropout,
                        out_channels=out_channels,
                        dims=self.dims,
                        use_checkpoint=self.use_checkpoint,
                        use_scale_shift_norm=self.use_scale_shift_norm,
                        down=True,
                    )
                else:
                    down_block = Downsample(ch, self.conv_resample, dims=self.dims, out_channels=out_channels)
                self.input_blocks.append(TimestepEmbedSequential(down_block))
                input_block_chans.append(out_channels)
                ch = out_channels
                ds *= 2

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                self.time_embed_dim,
                self.dropout,
                dims=self.dims,
                use_checkpoint=self.use_checkpoint,
                use_scale_shift_norm=self.use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                num_heads=self.num_heads,
                num_head_channels=self.num_head_channels,
                use_checkpoint=self.use_checkpoint,
            ),
            ResBlock(
                ch,
                self.time_embed_dim,
                self.dropout,
                dims=self.dims,
                use_checkpoint=self.use_checkpoint,
                use_scale_shift_norm=self.use_scale_shift_norm,
            ),
        )

        self.output_blocks = nn.ModuleList()
        for level, mult in list(enumerate(self.channel_mult))[::-1]:
            for block_idx in range(self.num_res_blocks + 1):
                skip_channels = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + skip_channels,
                        self.time_embed_dim,
                        self.dropout,
                        out_channels=int(self.model_channels * mult),
                        dims=self.dims,
                        use_checkpoint=self.use_checkpoint,
                        use_scale_shift_norm=self.use_scale_shift_norm,
                    )
                ]
                ch = int(self.model_channels * mult)
                if ds in self.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=self.num_heads,
                            num_head_channels=self.num_head_channels,
                            use_checkpoint=self.use_checkpoint,
                        )
                    )
                if level > 0 and block_idx == self.num_res_blocks:
                    out_channels = ch
                    if self.resblock_updown:
                        up_block = ResBlock(
                            ch,
                            self.time_embed_dim,
                            self.dropout,
                            out_channels=out_channels,
                            dims=self.dims,
                            use_checkpoint=self.use_checkpoint,
                            use_scale_shift_norm=self.use_scale_shift_norm,
                            up=True,
                        )
                    else:
                        up_block = Upsample(ch, self.conv_resample, dims=self.dims, out_channels=out_channels)
                    layers.append(up_block)
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            _normalization(ch),
            nn.SiLU(),
            _zero_module(_conv_nd(self.dims, ch, self.in_channels, 3, padding=1)),
        )

    def _embed_scalar(self, values, projector):
        return projector(_timestep_embedding(values.float(), self.model_channels))

    def _resolve_time_embedding(self, t, r=None, delta_t=None):
        if t is None:
            raise ValueError("t must be provided")
        if t.ndim == 0:
            t = t[None]
        if t.ndim != 1:
            raise ValueError(f"Expected t with shape [B], got {tuple(t.shape)}")

        embedding = self._embed_scalar(t, self.t_embed)
        if self.time_conditioning == "t_delta":
            if delta_t is None:
                if r is None:
                    raise ValueError("delta_t or r must be provided when time_conditioning='t_delta'")
                delta_t = t - r
            embedding = embedding + self._embed_scalar(delta_t, self.delta_embed)
        elif self.time_conditioning == "r_t":
            if r is None:
                raise ValueError("r must be provided when time_conditioning='r_t'")
            embedding = embedding + self._embed_scalar(r, self.r_embed)
        return embedding

    def forward(self, z_t, t=None, condition=None, r=None, delta_t=None):
        emb = self._resolve_time_embedding(t=t, r=r, delta_t=delta_t)
        if condition is not None:
            if self.cond_embed is None:
                raise ValueError("Conditioning provided but condition_num_classes is not configured")
            emb = emb + self.cond_embed(condition.long())

        hs = []
        h = z_t
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(z_t.dtype)
        return self.out(h)
