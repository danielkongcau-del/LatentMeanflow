import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_mask_scheduling_fn(mask_schedule_type):
    schedule = str(mask_schedule_type).strip().lower()
    if schedule == "linear":
        fn = lambda r: 1.0 - r
    elif schedule == "cosine":
        fn = lambda r: math.cos(float(r) * math.pi / 2.0)
    elif schedule == "arccos":
        fn = lambda r: math.acos(float(r)) / (math.pi / 2.0)
    elif schedule.startswith("pow"):
        exponent = float(schedule[3:])
        fn = lambda r: 1.0 - float(r) ** exponent
    else:
        raise ValueError(f"Unknown mask schedule type: {mask_schedule_type!r}")

    def wrapped(r):
        value = float(fn(r))
        if float(r) == 1.0:
            return 0.0
        return min(max(value, 0.0), 1.0)

    return wrapped


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.n_heads = int(n_heads)
        self.dropout = float(dropout)
        if self.embed_dim % self.n_heads != 0:
            raise ValueError(
                f"embed_dim must be divisible by n_heads, got embed_dim={self.embed_dim}, n_heads={self.n_heads}"
            )

        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=False)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj_dropout = nn.Dropout(self.dropout)

    def forward(self, x):
        batch_size, sequence_length, embed_dim = x.shape
        head_dim = embed_dim // self.n_heads

        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch_size, sequence_length, self.n_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, sequence_length, self.n_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, sequence_length, self.n_heads, head_dim).transpose(1, 2)

        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=False,
            dropout_p=(self.dropout if self.training else 0.0),
        )
        x = x.transpose(1, 2).contiguous().view(batch_size, sequence_length, embed_dim)
        x = self.proj_dropout(self.proj(x))
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(int(embed_dim))
        self.norm2 = nn.LayerNorm(int(embed_dim))
        self.attn = SelfAttention(embed_dim=embed_dim, n_heads=n_heads, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(int(embed_dim), 4 * int(embed_dim)),
            nn.GELU(),
            nn.Linear(4 * int(embed_dim), int(embed_dim)),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TokenCodeMaskGitTransformerBackbone(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_tokens,
        n_layer=8,
        n_head=8,
        n_embd=256,
        embd_pdrop=0.0,
        resid_pdrop=0.0,
        attn_pdrop=0.0,
        mask_schedule_type="cosine",
    ):
        super().__init__()
        self.codebook_size = int(vocab_size)
        self.mask_token_id = int(vocab_size)
        self.token_vocabulary_size = int(vocab_size) + 1
        self.n_tokens = int(n_tokens)
        self.n_layer = int(n_layer)
        self.n_head = int(n_head)
        self.n_embd = int(n_embd)
        self.embd_pdrop = float(embd_pdrop)
        self.resid_pdrop = float(resid_pdrop)
        self.attn_pdrop = float(attn_pdrop)
        self.mask_schedule_type = str(mask_schedule_type)
        self.gamma = get_mask_scheduling_fn(self.mask_schedule_type)

        self.token_emb = nn.Embedding(self.token_vocabulary_size, self.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.n_tokens, self.n_embd))
        self.drop_emb = nn.Dropout(self.embd_pdrop)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=self.n_embd,
                    n_heads=self.n_head,
                    dropout=max(self.resid_pdrop, self.attn_pdrop),
                )
                for _ in range(self.n_layer)
            ]
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.n_embd),
            nn.Linear(self.n_embd, self.codebook_size),
        )

        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, token_ids):
        if token_ids.ndim != 2:
            raise ValueError(
                "TokenCodeMaskGitTransformerBackbone expects token ids with shape [B, T], "
                f"got {tuple(token_ids.shape)}"
            )
        if int(token_ids.shape[1]) > self.n_tokens:
            raise ValueError(
                f"TokenCodeMaskGitTransformerBackbone received sequence length {int(token_ids.shape[1])}, "
                f"but n_tokens={self.n_tokens}"
            )

        x = self.token_emb(token_ids.long())
        x = x + self.pos_emb[:, : int(token_ids.shape[1]), :]
        x = self.drop_emb(x)
        for block in self.blocks:
            x = block(x)
        return self.classifier(x)

    def get_random_mask(self, batch_size, sequence_length):
        device = self.pos_emb.device
        ratio = torch.rand((), device=device, dtype=torch.float32).item()
        masked_count = max(1, int(math.ceil(float(self.gamma(ratio)) * float(sequence_length))))
        masked_count = min(masked_count, int(sequence_length))
        indices = torch.rand((int(batch_size), int(sequence_length)), device=device).topk(masked_count, dim=1).indices
        mask = torch.zeros((int(batch_size), int(sequence_length)), dtype=torch.bool, device=device)
        mask.scatter_(1, indices, torch.ones_like(mask, dtype=torch.bool))
        return mask

    def optimizer_groups(self, *, weight_decay):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear,)
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)

        for module_name, module in self.named_modules():
            for param_name, _ in module.named_parameters(recurse=False):
                full_name = f"{module_name}.{param_name}" if module_name else param_name
                if param_name.endswith("bias"):
                    no_decay.add(full_name)
                elif param_name.endswith("weight") and isinstance(module, whitelist_weight_modules):
                    decay.add(full_name)
                elif param_name.endswith("weight") and isinstance(module, blacklist_weight_modules):
                    no_decay.add(full_name)

        no_decay.add("pos_emb")

        param_dict = {name: param for name, param in self.named_parameters()}
        intersection = decay & no_decay
        if intersection:
            raise ValueError(
                f"TokenCodeMaskGitTransformerBackbone parameters landed in both decay/no_decay sets: {sorted(intersection)}"
            )
        missing = set(param_dict.keys()) - (decay | no_decay)
        if missing:
            raise ValueError(
                f"TokenCodeMaskGitTransformerBackbone parameters were not assigned an optimizer group: {sorted(missing)}"
            )

        return [
            {
                "params": [param_dict[name] for name in sorted(decay)],
                "weight_decay": float(weight_decay),
            },
            {
                "params": [param_dict[name] for name in sorted(no_decay)],
                "weight_decay": 0.0,
            },
        ]
