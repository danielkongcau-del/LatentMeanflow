from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
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
    def ignore_index(self):
        loss_module = getattr(self.tokenizer, "loss", None)
        if loss_module is None:
            return None
        ignore_index = getattr(loss_module, "ignore_index", None)
        return None if ignore_index is None else int(ignore_index)

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

    def _resolve_quantizer(self):
        quantizer = getattr(self.tokenizer, "quantizer", None)
        if quantizer is None:
            raise AttributeError(
                f"Tokenizer '{self.tokenizer.__class__.__name__}' does not expose a quantizer needed for "
                "soft code-distribution decoding."
            )
        embedding = getattr(quantizer, "embedding", None)
        if embedding is None or not hasattr(embedding, "weight"):
            raise AttributeError(
                f"Tokenizer '{self.tokenizer.__class__.__name__}' quantizer does not expose embedding weights."
            )
        return quantizer

    def _normalize_code_distribution(self, *, code_logits=None, code_probs=None):
        if (code_logits is None) == (code_probs is None):
            raise ValueError("Pass exactly one of code_logits or code_probs.")

        if code_logits is not None:
            distribution = (
                code_logits.to(dtype=torch.float32)
                if isinstance(code_logits, torch.Tensor)
                else torch.as_tensor(code_logits, dtype=torch.float32)
            )
        else:
            distribution = (
                code_probs.to(dtype=torch.float32)
                if isinstance(code_probs, torch.Tensor)
                else torch.as_tensor(code_probs, dtype=torch.float32)
            )

        if distribution.ndim != 4:
            raise ValueError(
                "Code distribution must have shape [B, K, Ht, Wt] or [B, Ht, Wt, K], "
                f"got {tuple(distribution.shape)}"
            )

        if int(distribution.shape[1]) == self.codebook_size:
            channel_first = distribution
        elif int(distribution.shape[-1]) == self.codebook_size:
            channel_first = distribution.permute(0, 3, 1, 2)
        else:
            raise ValueError(
                f"Code distribution channel mismatch: expected codebook_size={self.codebook_size}, "
                f"got {tuple(distribution.shape)}"
            )

        spatial_shape = tuple(int(v) for v in channel_first.shape[-2:])
        if spatial_shape != self.latent_spatial_shape:
            raise ValueError(
                "Code distribution spatial shape mismatch: "
                f"expected {self.latent_spatial_shape}, got {spatial_shape}"
            )

        if code_logits is not None:
            return torch.softmax(channel_first, dim=1)

        if torch.any(channel_first < 0):
            raise ValueError("code_probs must be non-negative.")
        prob_mass = channel_first.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
        return channel_first / prob_mass

    def code_distribution_to_latents(self, *, code_logits=None, code_probs=None):
        probs = self._normalize_code_distribution(code_logits=code_logits, code_probs=code_probs)
        quantizer = self._resolve_quantizer()
        codebook = quantizer.embedding.weight.to(device=probs.device, dtype=probs.dtype)
        project_embeddings = getattr(quantizer, "_project_embeddings", None)
        if callable(project_embeddings):
            codebook = project_embeddings(codebook)
        expected_latents = torch.einsum("bkhw,ke->behw", probs, codebook)
        return {
            "code_probs": probs,
            "z_q": expected_latents,
        }

    def decode_code_distribution(self, *, code_logits=None, code_probs=None):
        if not hasattr(self.tokenizer, "decode_latents"):
            raise AttributeError(
                f"Tokenizer '{self.tokenizer.__class__.__name__}' does not support decode_latents()."
            )
        distribution = self.code_distribution_to_latents(code_logits=code_logits, code_probs=code_probs)
        decoded = self.tokenizer.decode_latents(distribution["z_q"])
        decoded["code_probs"] = distribution["code_probs"]
        decoded["z_q"] = distribution["z_q"]
        return decoded

    def semantic_auxiliary_losses(self, *, mask_logits, mask_index):
        loss_module = getattr(self.tokenizer, "loss", None)
        if loss_module is None:
            raise AttributeError(
                f"Tokenizer '{self.tokenizer.__class__.__name__}' does not expose a loss module for "
                "semantic auxiliary supervision."
            )
        cross_entropy_fn = getattr(loss_module, "_cross_entropy_loss", None)
        dice_fn = getattr(loss_module, "_dice_loss", None)
        if not callable(cross_entropy_fn) or not callable(dice_fn):
            raise AttributeError(
                f"Tokenizer '{self.tokenizer.__class__.__name__}' loss module does not expose the semantic "
                "CE/Dice helpers required for auxiliary supervision."
            )
        return {
            "semantic_ce": cross_entropy_fn(mask_logits, mask_index, use_class_weights=False),
            "semantic_dice": dice_fn(mask_logits, mask_index),
        }

    def compute_mask_metrics(self, *, mask_index, mask_logits):
        metrics_fn = getattr(self.tokenizer, "_compute_mask_metrics", None)
        if not callable(metrics_fn):
            raise AttributeError(
                f"Tokenizer '{self.tokenizer.__class__.__name__}' does not expose semantic mask metrics."
            )
        return metrics_fn(mask_index, mask_logits)
