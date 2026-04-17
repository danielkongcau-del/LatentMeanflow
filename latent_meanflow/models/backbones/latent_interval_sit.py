import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalize_hw(value):
    if isinstance(value, int):
        size = int(value)
        if size <= 0:
            raise ValueError(f"input_size must be positive, got {value}")
        return (size, size)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"input_size must be an int or 2-item sequence, got {value!r}")
    values = tuple(int(item) for item in value)
    if len(values) != 2:
        raise ValueError(f"input_size must be an int or 2-item sequence, got {value!r}")
    if values[0] <= 0 or values[1] <= 0:
        raise ValueError(f"input_size must be positive, got {values}")
    return values


def _modulate(x, shift, scale):
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _build_1d_sincos_pos_embed(embed_dim, positions):
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even, got {embed_dim}")
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / max(1, embed_dim // 2)))
    out = positions.reshape(-1, 1) * omega.reshape(1, -1)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)


def _build_2d_sincos_pos_embed(embed_dim, grid_h, grid_w):
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even, got {embed_dim}")
    grid_y = torch.arange(grid_h, dtype=torch.float32)
    grid_x = torch.arange(grid_w, dtype=torch.float32)
    mesh_y, mesh_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
    emb_y = _build_1d_sincos_pos_embed(embed_dim // 2, mesh_y.reshape(-1))
    emb_x = _build_1d_sincos_pos_embed(embed_dim // 2, mesh_x.reshape(-1))
    return torch.cat([emb_y, emb_x], dim=1)


class _PatchEmbed2D(nn.Module):
    def __init__(self, input_size, patch_size, in_channels, hidden_size):
        super().__init__()
        self.input_size = _normalize_hw(input_size)
        self.patch_size = int(patch_size)
        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        if self.input_size[0] % self.patch_size != 0 or self.input_size[1] % self.patch_size != 0:
            raise ValueError(
                "input_size must be divisible by patch_size, "
                f"got input_size={self.input_size}, patch_size={self.patch_size}"
            )
        self.grid_size = (self.input_size[0] // self.patch_size, self.input_size[1] // self.patch_size)
        self.num_patches = int(self.grid_size[0] * self.grid_size[1])
        self.proj = nn.Conv2d(
            int(in_channels),
            int(hidden_size),
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )

    def forward(self, x):
        if tuple(x.shape[-2:]) != self.input_size:
            raise ValueError(
                f"LatentIntervalSiT expected spatial shape {self.input_size}, got {tuple(x.shape[-2:])}"
            )
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class _TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.frequency_embedding_size = int(frequency_embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.frequency_embedding_size, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        if t is None:
            raise ValueError("LatentIntervalSiT requires timestep conditioning t")
        if t.ndim == 0:
            t = t[None]
        if t.ndim != 1:
            raise ValueError(f"Expected t with shape [B], got {tuple(t.shape)}")
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class _SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, qkv_bias=True):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_heads, got {self.hidden_size} and {self.num_heads}"
            )
        self.head_dim = self.hidden_size // self.num_heads
        self.qkv = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=bool(qkv_bias))
        self.proj = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, x):
        batch_size, token_count, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, token_count, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if hasattr(F, "scaled_dot_product_attention"):
            attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        else:
            scale = self.head_dim ** -0.5
            attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
            attn = torch.matmul(attn, v)
        attn = attn.transpose(1, 2).reshape(batch_size, token_count, self.hidden_size)
        return self.proj(attn)


class _FeedForward(nn.Module):
    def __init__(self, hidden_size, mlp_ratio=4.0):
        super().__init__()
        inner_size = int(hidden_size * float(mlp_ratio))
        self.net = nn.Sequential(
            nn.Linear(hidden_size, inner_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(inner_size, hidden_size),
        )

    def forward(self, x):
        return self.net(x)


class _SiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, qkv_bias=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1.0e-6)
        self.attn = _SelfAttention(hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1.0e-6)
        self.mlp = _FeedForward(hidden_size, mlp_ratio=mlp_ratio)
        self.ada_ln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.ada_ln(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(_modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(_modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class _FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.patch_size = int(patch_size)
        self.out_channels = int(out_channels)
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1.0e-6)
        self.ada_ln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )
        self.linear = nn.Linear(hidden_size, self.patch_size * self.patch_size * self.out_channels)

    def forward(self, x, c):
        shift, scale = self.ada_ln(c).chunk(2, dim=1)
        x = _modulate(self.norm(x), shift, scale)
        return self.linear(x)


class LatentIntervalSiT(nn.Module):
    def __init__(
        self,
        input_size=256,
        patch_size=8,
        in_channels=4,
        out_channels=None,
        hidden_size=512,
        depth=12,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        time_conditioning="t",
        time_embed_dim=256,
        t_time_scale=1.0,
        delta_time_scale=1.0,
        r_time_scale=1.0,
        condition_num_classes=None,
        spatial_condition_channels=0,
    ):
        super().__init__()
        if condition_num_classes is not None:
            raise ValueError("LatentIntervalSiT unconditional baseline does not use image-level class conditioning.")
        if int(spatial_condition_channels) > 0:
            raise ValueError("LatentIntervalSiT unconditional baseline does not use spatial conditioning.")

        self.input_size = _normalize_hw(input_size)
        self.patch_size = int(patch_size)
        self.in_channels = int(in_channels)
        self.out_channels = int(self.in_channels if out_channels is None else out_channels)
        self.hidden_size = int(hidden_size)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.time_conditioning = str(time_conditioning)
        if self.time_conditioning not in {"t", "t_delta", "r_t"}:
            raise ValueError(
                "time_conditioning must be one of {'t', 't_delta', 'r_t'}, "
                f"got {self.time_conditioning!r}"
            )
        self.t_time_scale = float(t_time_scale)
        self.delta_time_scale = float(delta_time_scale)
        self.r_time_scale = float(r_time_scale)

        self.x_embedder = _PatchEmbed2D(
            input_size=self.input_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            hidden_size=self.hidden_size,
        )
        pos_embed = _build_2d_sincos_pos_embed(
            self.hidden_size,
            grid_h=self.x_embedder.grid_size[0],
            grid_w=self.x_embedder.grid_size[1],
        )
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0), persistent=False)

        self.t_embedder = _TimestepEmbedder(self.hidden_size, frequency_embedding_size=int(time_embed_dim))
        self.delta_embedder = (
            _TimestepEmbedder(self.hidden_size, frequency_embedding_size=int(time_embed_dim))
            if self.time_conditioning == "t_delta"
            else None
        )
        self.r_embedder = (
            _TimestepEmbedder(self.hidden_size, frequency_embedding_size=int(time_embed_dim))
            if self.time_conditioning == "r_t"
            else None
        )
        self.blocks = nn.ModuleList(
            [
                _SiTBlock(
                    hidden_size=self.hidden_size,
                    num_heads=self.num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                )
                for _ in range(self.depth)
            ]
        )
        self.final_layer = _FinalLayer(
            hidden_size=self.hidden_size,
            patch_size=self.patch_size,
            out_channels=self.out_channels,
        )
        self._initialize_weights()

    def _initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight.view(module.weight.shape[0], -1))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        if self.delta_embedder is not None:
            nn.init.normal_(self.delta_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.delta_embedder.mlp[2].weight, std=0.02)
        if self.r_embedder is not None:
            nn.init.normal_(self.r_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.r_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.ada_ln[-1].weight, 0)
            nn.init.constant_(block.ada_ln[-1].bias, 0)
        nn.init.constant_(self.final_layer.ada_ln[-1].weight, 0)
        nn.init.constant_(self.final_layer.ada_ln[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def _resolve_conditioning(self, t, r=None, delta_t=None):
        if t is None:
            raise ValueError("LatentIntervalSiT requires t")
        conditioning = self.t_embedder(t * self.t_time_scale)
        if self.time_conditioning == "t_delta":
            if delta_t is None:
                if r is None:
                    raise ValueError("LatentIntervalSiT with time_conditioning='t_delta' requires delta_t or r")
                delta_t = t - r
            conditioning = conditioning + self.delta_embedder(delta_t * self.delta_time_scale)
        elif self.time_conditioning == "r_t":
            if r is None:
                raise ValueError("LatentIntervalSiT with time_conditioning='r_t' requires r")
            conditioning = conditioning + self.r_embedder(r * self.r_time_scale)
        return conditioning

    def unpatchify(self, x):
        batch_size, token_count, _ = x.shape
        grid_h, grid_w = self.x_embedder.grid_size
        if token_count != grid_h * grid_w:
            raise ValueError(
                f"LatentIntervalSiT expected {grid_h * grid_w} tokens, got {token_count}"
            )
        patch = self.patch_size
        x = x.reshape(batch_size, grid_h, grid_w, patch, patch, self.out_channels)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(batch_size, self.out_channels, grid_h * patch, grid_w * patch)

    def forward(self, z_t, t=None, condition=None, r=None, delta_t=None):
        if condition is not None:
            raise ValueError("LatentIntervalSiT unconditional baseline does not accept condition.")
        if z_t.ndim != 4:
            raise ValueError(f"Expected z_t with shape [B, C, H, W], got {tuple(z_t.shape)}")
        tokens = self.x_embedder(z_t) + self.pos_embed.to(device=z_t.device, dtype=z_t.dtype)
        conditioning = self._resolve_conditioning(t=t, r=r, delta_t=delta_t).to(device=z_t.device, dtype=z_t.dtype)
        for block in self.blocks:
            tokens = block(tokens, conditioning)
        tokens = self.final_layer(tokens, conditioning)
        return self.unpatchify(tokens)
