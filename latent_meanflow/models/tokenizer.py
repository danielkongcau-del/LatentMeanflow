from contextlib import nullcontext
from pathlib import Path

import torch
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config


class SemanticTokenizerAdapter(torch.nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    @classmethod
    def from_config(
        cls,
        config_path,
        *,
        device=None,
        eval_mode=True,
        freeze=True,
    ):
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Tokenizer config not found: {config_path}")

        config = OmegaConf.load(config_path)
        tokenizer = instantiate_from_config(config.model)

        if freeze:
            tokenizer.requires_grad_(False)
        if eval_mode:
            tokenizer.eval()
        if device is not None:
            tokenizer = tokenizer.to(device)
        return cls(tokenizer)

    @classmethod
    def from_pretrained(
        cls,
        config_path,
        ckpt_path,
        *,
        device=None,
        eval_mode=True,
        freeze=True,
    ):
        config_path = Path(config_path)
        ckpt_path = Path(ckpt_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Tokenizer config not found: {config_path}")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Tokenizer checkpoint not found: {ckpt_path}")

        config = OmegaConf.load(config_path)
        tokenizer = instantiate_from_config(config.model)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = state["state_dict"] if "state_dict" in state else state
        tokenizer.load_state_dict(state_dict, strict=False)

        if freeze:
            tokenizer.requires_grad_(False)
        if eval_mode:
            tokenizer.eval()
        if device is not None:
            tokenizer = tokenizer.to(device)
        return cls(tokenizer)

    @property
    def trainable(self):
        return any(param.requires_grad for param in self.tokenizer.parameters())

    def _maybe_no_grad(self):
        if self.trainable:
            return nullcontext()
        return torch.no_grad()

    @property
    def num_classes(self):
        return int(self.tokenizer.num_classes)

    @property
    def codebook_size(self):
        if not hasattr(self.tokenizer, "codebook_size"):
            raise AttributeError(
                f"Tokenizer '{self.tokenizer.__class__.__name__}' does not expose codebook_size."
            )
        return int(self.tokenizer.codebook_size)

    @property
    def latent_channels(self):
        return int(self.tokenizer.latent_channels)

    @property
    def latent_spatial_shape(self):
        return tuple(int(v) for v in self.tokenizer.latent_spatial_shape)

    def latent_shape(self, batch_size):
        return (int(batch_size), self.latent_channels, *self.latent_spatial_shape)

    def _resolve_latent_key(self, outputs):
        if "z" in outputs:
            return "z"
        if "z_q" in outputs:
            return "z_q"
        raise KeyError(
            "Tokenizer encode_batch() must return either 'z' or 'z_q' so downstream routes can resolve the latent view."
        )

    def encode_batch(self, batch, sample_posterior=False):
        with self._maybe_no_grad():
            outputs = self.tokenizer.encode_batch(batch, sample_posterior=sample_posterior)
        latent_key = self._resolve_latent_key(outputs)
        if not self.trainable:
            outputs[latent_key] = outputs[latent_key].detach()
        outputs["z"] = outputs[latent_key]
        return outputs

    def decode_latents(self, z):
        with self._maybe_no_grad():
            outputs = self.tokenizer.decode_latents(z)
        return outputs

    def decode_codes(self, codes):
        if not hasattr(self.tokenizer, "decode_codes"):
            raise AttributeError(
                f"Tokenizer '{self.tokenizer.__class__.__name__}' does not support decode_codes()."
            )
        with self._maybe_no_grad():
            outputs = self.tokenizer.decode_codes(codes)
        return outputs
