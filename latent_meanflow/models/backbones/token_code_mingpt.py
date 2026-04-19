from pathlib import Path
import sys

import torch
import torch.nn as nn


def ensure_taming_transformers_on_path():
    repo_root = Path(__file__).resolve().parents[3]
    ldm_root = repo_root / "third_party" / "latent-diffusion"
    taming_root = ldm_root / "taming-transformers"
    for path in (repo_root, ldm_root, taming_root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


ensure_taming_transformers_on_path()

from taming.modules.transformer.mingpt import GPT  # noqa: E402


class TokenCodeMingptBackbone(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size,
        n_layer=8,
        n_head=8,
        n_embd=256,
        embd_pdrop=0.0,
        resid_pdrop=0.0,
        attn_pdrop=0.0,
    ):
        super().__init__()
        self.gpt = GPT(
            vocab_size=int(vocab_size),
            block_size=int(block_size),
            n_layer=int(n_layer),
            n_head=int(n_head),
            n_embd=int(n_embd),
            embd_pdrop=float(embd_pdrop),
            resid_pdrop=float(resid_pdrop),
            attn_pdrop=float(attn_pdrop),
        )

    @property
    def config(self):
        return self.gpt.config

    def get_block_size(self):
        return int(self.gpt.get_block_size())

    def forward(self, token_ids, targets=None):
        return self.gpt(token_ids, targets=targets)

    def forward_with_past(self, token_ids, *, targets=None, past=None, past_length=None):
        return self.gpt.forward_with_past(token_ids, targets=targets, past=past, past_length=past_length)

    def optimizer_groups(self, *, weight_decay):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

        for module_name, module in self.gpt.named_modules():
            for param_name, _ in module.named_parameters(recurse=False):
                full_name = f"{module_name}.{param_name}" if module_name else param_name
                if param_name.endswith("bias"):
                    no_decay.add(full_name)
                elif param_name.endswith("weight") and isinstance(module, whitelist_weight_modules):
                    decay.add(full_name)
                elif param_name.endswith("weight") and isinstance(module, blacklist_weight_modules):
                    no_decay.add(full_name)

        no_decay.add("pos_emb")

        param_dict = {name: param for name, param in self.gpt.named_parameters()}
        intersection = decay & no_decay
        if intersection:
            raise ValueError(f"mingpt parameters landed in both decay/no_decay sets: {sorted(intersection)}")

        missing = set(param_dict.keys()) - (decay | no_decay)
        if missing:
            raise ValueError(f"mingpt parameters were not assigned an optimizer group: {sorted(missing)}")

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
