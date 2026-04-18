from copy import deepcopy

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from ldm.modules.diffusionmodules.model import Decoder, Encoder
from ldm.util import instantiate_from_config


def _resolve_group_norm_groups(num_channels, max_groups=32):
    for num_groups in range(min(max_groups, num_channels), 0, -1):
        if num_channels % num_groups == 0:
            return num_groups
    return 1


class SemanticMaskVectorQuantizer(nn.Module):
    def __init__(self, codebook_size, embed_dim):
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.embed_dim = int(embed_dim)
        if self.codebook_size <= 0:
            raise ValueError(f"codebook_size must be positive, got {self.codebook_size}")
        if self.embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {self.embed_dim}")

        self.embedding = nn.Embedding(self.codebook_size, self.embed_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / float(self.codebook_size), 1.0 / float(self.codebook_size))

    def codes_to_latents(self, codes):
        codes = torch.as_tensor(codes)
        if codes.ndim != 3:
            raise ValueError(f"Expected codes with shape [B, H, W], got {tuple(codes.shape)}")
        if not torch.is_floating_point(codes):
            codes = codes.long()
        else:
            if not torch.all(codes == torch.round(codes)):
                raise ValueError("Floating-point codes must represent integer values.")
            codes = torch.round(codes).long()
        if int(codes.numel()) > 0:
            min_code = int(codes.min().item())
            max_code = int(codes.max().item())
            if min_code < 0 or max_code >= self.codebook_size:
                raise ValueError(
                    f"Code indices out of range: min={min_code}, max={max_code}, codebook_size={self.codebook_size}"
                )
        latents = F.embedding(codes, self.embedding.weight)
        return latents.permute(0, 3, 1, 2).contiguous()

    def forward(self, z_e):
        if z_e.ndim != 4:
            raise ValueError(f"Expected z_e with shape [B, C, H, W], got {tuple(z_e.shape)}")
        if int(z_e.shape[1]) != self.embed_dim:
            raise ValueError(
                f"Quantizer input channel mismatch: expected embed_dim={self.embed_dim}, got {tuple(z_e.shape)}"
            )

        batch_size, _, height, width = z_e.shape
        z_flat = z_e.permute(0, 2, 3, 1).reshape(-1, self.embed_dim)
        codebook = self.embedding.weight
        distances = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            + codebook.pow(2).sum(dim=1).unsqueeze(0)
            - 2.0 * torch.matmul(z_flat, codebook.t())
        )
        codes_flat = torch.argmin(distances, dim=1)
        z_q = F.embedding(codes_flat, codebook).view(batch_size, height, width, self.embed_dim)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        z_q_st = z_e + (z_q - z_e).detach()

        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commitment_loss = F.mse_loss(z_e, z_q.detach())

        code_counts = torch.bincount(codes_flat, minlength=self.codebook_size).to(device=z_e.device, dtype=z_e.dtype)
        avg_probs = code_counts / code_counts.sum().clamp_min(1.0)
        perplexity = torch.exp(-(avg_probs * torch.log(avg_probs.clamp_min(1.0e-10))).sum())
        used_code_count = (code_counts > 0).sum().to(device=z_e.device, dtype=z_e.dtype)
        used_code_fraction = used_code_count / float(self.codebook_size)
        dead_code_fraction = 1.0 - used_code_fraction

        return {
            "z_q": z_q_st,
            "z_q_lookup": z_q.detach(),
            "codes": codes_flat.view(batch_size, height, width),
            "vq_codebook_loss": codebook_loss,
            "vq_commitment_loss": commitment_loss,
            "perplexity": perplexity,
            "used_code_count": used_code_count,
            "used_code_fraction": torch.as_tensor(used_code_fraction, device=z_e.device, dtype=z_e.dtype),
            "dead_code_fraction": torch.as_tensor(dead_code_fraction, device=z_e.device, dtype=z_e.dtype),
        }


class SemanticMaskVQLoss(nn.Module):
    def __init__(
        self,
        mask_ce_weight=1.0,
        mask_dice_weight=0.0,
        mask_focal_weight=0.0,
        vq_codebook_weight=1.0,
        vq_commit_weight=0.25,
        ignore_index=None,
        focal_gamma=2.0,
        ce_label_smoothing=0.0,
        dice_eps=1.0e-6,
    ):
        super().__init__()
        self.mask_ce_weight = float(mask_ce_weight)
        self.mask_dice_weight = float(mask_dice_weight)
        self.mask_focal_weight = float(mask_focal_weight)
        self.vq_codebook_weight = float(vq_codebook_weight)
        self.vq_commit_weight = float(vq_commit_weight)
        self.ignore_index = None if ignore_index is None else int(ignore_index)
        self.focal_gamma = float(focal_gamma)
        self.ce_label_smoothing = float(ce_label_smoothing)
        self.dice_eps = float(dice_eps)

    def _mask_valid(self, mask_index):
        if self.ignore_index is None:
            return torch.ones_like(mask_index, dtype=torch.bool)
        return mask_index != self.ignore_index

    def _cross_entropy_loss(self, mask_logits, mask_index):
        valid_mask = self._mask_valid(mask_index)
        if not torch.any(valid_mask):
            return mask_logits.new_tensor(0.0)

        loss = F.cross_entropy(
            mask_logits,
            mask_index,
            ignore_index=-100 if self.ignore_index is None else self.ignore_index,
            reduction="none",
            label_smoothing=self.ce_label_smoothing,
        )
        return loss[valid_mask].mean()

    def _dice_loss(self, mask_logits, mask_index):
        valid_mask = self._mask_valid(mask_index)
        if not torch.any(valid_mask):
            return mask_logits.new_tensor(0.0)

        safe_targets = mask_index.clone()
        safe_targets[~valid_mask] = 0
        probs = torch.softmax(mask_logits, dim=1)
        target = F.one_hot(safe_targets, num_classes=mask_logits.shape[1]).permute(0, 3, 1, 2).float()
        valid_mask = valid_mask.unsqueeze(1)
        probs = probs * valid_mask
        target = target * valid_mask

        intersection = (probs * target).sum(dim=(0, 2, 3))
        denominator = probs.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3))
        dice = (2.0 * intersection + self.dice_eps) / (denominator + self.dice_eps)
        return 1.0 - dice.mean()

    def _focal_loss(self, mask_logits, mask_index):
        valid_mask = self._mask_valid(mask_index)
        if not torch.any(valid_mask):
            return mask_logits.new_tensor(0.0)

        ce = F.cross_entropy(
            mask_logits,
            mask_index,
            ignore_index=-100 if self.ignore_index is None else self.ignore_index,
            reduction="none",
            label_smoothing=self.ce_label_smoothing,
        )
        ce = ce[valid_mask]
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.focal_gamma) * ce).mean()

    def forward(self, mask_index, mask_logits, *, vq_codebook_loss, vq_commitment_loss, quantizer_stats=None):
        mask_ce = self._cross_entropy_loss(mask_logits, mask_index)
        mask_dice = self._dice_loss(mask_logits, mask_index)
        mask_focal = self._focal_loss(mask_logits, mask_index)
        vq_total = (
            self.vq_codebook_weight * vq_codebook_loss
            + self.vq_commit_weight * vq_commitment_loss
        )
        total_loss = (
            self.mask_ce_weight * mask_ce
            + self.mask_dice_weight * mask_dice
            + self.mask_focal_weight * mask_focal
            + vq_total
        )

        loss_dict = {
            "mask_ce": mask_ce,
            "mask_dice": mask_dice,
            "mask_focal": mask_focal,
            "vq_codebook": vq_codebook_loss,
            "vq_commit": vq_commitment_loss,
            "vq_total": vq_total,
            "total_loss": total_loss,
        }
        if quantizer_stats is not None:
            for name in ("perplexity", "used_code_count", "used_code_fraction", "dead_code_fraction"):
                value = quantizer_stats.get(name)
                if isinstance(value, torch.Tensor) and value.ndim == 0:
                    loss_dict[f"codebook_{name}"] = value
        return total_loss, loss_dict


class SemanticMaskVQAutoencoder(pl.LightningModule):
    def __init__(
        self,
        ddconfig,
        lossconfig,
        embed_dim,
        codebook_size,
        num_classes,
        ckpt_path=None,
        ignore_keys=None,
        monitor=None,
        mask_index_key="mask_index",
        mask_onehot_key="mask_onehot",
    ):
        super().__init__()
        ignore_keys = [] if ignore_keys is None else list(ignore_keys)

        self.num_classes = int(num_classes)
        self.embed_dim = int(embed_dim)
        self.codebook_size = int(codebook_size)
        self.mask_index_key = str(mask_index_key)
        self.mask_onehot_key = str(mask_onehot_key)

        encoder_config = deepcopy(ddconfig)
        encoder_config["in_channels"] = self.num_classes
        encoder_config["out_ch"] = self.num_classes
        decoder_config = deepcopy(ddconfig)
        decoder_config["in_channels"] = self.num_classes
        decoder_config["out_ch"] = self.num_classes
        decoder_config["give_pre_end"] = True

        self.encoder = Encoder(**encoder_config)
        self.decoder = Decoder(**decoder_config)
        encoder_channels = int(encoder_config["z_channels"])
        self.pre_quant_conv = nn.Conv2d(encoder_channels, self.embed_dim, kernel_size=1)
        self.post_quant_conv = nn.Conv2d(self.embed_dim, encoder_channels, kernel_size=1)
        self.quantizer = SemanticMaskVectorQuantizer(codebook_size=self.codebook_size, embed_dim=self.embed_dim)

        if getattr(self.decoder, "give_pre_end", False):
            for module in (self.decoder.norm_out, self.decoder.conv_out):
                for param in module.parameters():
                    param.requires_grad = False

        self.decoder_feature_channels = decoder_config["ch"] * decoder_config["ch_mult"][0]
        norm_groups = _resolve_group_norm_groups(self.decoder_feature_channels)
        self.mask_head = nn.Sequential(
            nn.GroupNorm(num_groups=norm_groups, num_channels=self.decoder_feature_channels),
            nn.SiLU(),
            nn.Conv2d(self.decoder_feature_channels, self.decoder_feature_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(self.decoder_feature_channels, self.num_classes, kernel_size=3, padding=1),
        )

        self.loss = instantiate_from_config(lossconfig)
        self.learning_rate = 1.0e-4
        self.latent_channels = self.embed_dim
        self.latent_spatial_shape = tuple(int(v) for v in self.decoder.z_shape[2:])
        self.token_grid_shape = self.latent_spatial_shape

        if monitor is not None:
            self.monitor = str(monitor)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=None):
        ignore_keys = [] if ignore_keys is None else list(ignore_keys)
        state = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = state["state_dict"] if "state_dict" in state else state
        keys = list(state_dict.keys())
        for key in keys:
            for prefix in ignore_keys:
                if key.startswith(prefix):
                    del state_dict[key]
                    break
        self.load_state_dict(state_dict, strict=False)
        print(f"Restored from {path}")

    def _validate_no_posterior_sampling(self, sample_posterior):
        if sample_posterior is None:
            return
        if bool(sample_posterior):
            raise ValueError("SemanticMaskVQAutoencoder is deterministic and does not support posterior sampling.")

    def _extract_num_classes(self, batch):
        batch_num_classes = batch.get("num_classes", None)
        if batch_num_classes is None:
            return self.num_classes
        if isinstance(batch_num_classes, torch.Tensor):
            unique_values = torch.unique(torch.as_tensor(batch_num_classes).detach().cpu())
            if unique_values.numel() != 1:
                raise ValueError(f"Batch contains inconsistent num_classes values: {unique_values.tolist()}")
            batch_num_classes = int(unique_values.item())
        else:
            batch_num_classes = int(batch_num_classes)
        if batch_num_classes != self.num_classes:
            raise ValueError(
                f"Dataset num_classes ({batch_num_classes}) does not match model num_classes ({self.num_classes})"
            )
        return batch_num_classes

    def _normalize_mask_index(self, mask_index):
        mask_index = torch.as_tensor(mask_index)
        if mask_index.ndim != 3:
            raise ValueError(f"Expected mask_index tensor with shape [B, H, W], got {tuple(mask_index.shape)}")
        return mask_index.long()

    def _normalize_mask_onehot(self, mask_onehot):
        mask_onehot = torch.as_tensor(mask_onehot)
        if mask_onehot.ndim != 4:
            raise ValueError(
                f"Expected mask_onehot tensor with shape [B, H, W, K] or [B, K, H, W], got {tuple(mask_onehot.shape)}"
            )
        if int(mask_onehot.shape[-1]) == self.num_classes:
            mask_onehot = mask_onehot.permute(0, 3, 1, 2)
        elif int(mask_onehot.shape[1]) != self.num_classes:
            raise ValueError(
                f"mask_onehot channel dimension must equal num_classes={self.num_classes}, got {tuple(mask_onehot.shape)}"
            )
        return mask_onehot.to(memory_format=torch.contiguous_format).float()

    def _mask_valid(self, mask_index):
        if self.loss.ignore_index is None:
            return torch.ones_like(mask_index, dtype=torch.bool)
        return mask_index != self.loss.ignore_index

    def _make_onehot(self, mask_index):
        valid_mask = self._mask_valid(mask_index)
        safe_targets = mask_index.clone()
        safe_targets[~valid_mask] = 0
        mask_onehot = F.one_hot(safe_targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        mask_onehot = mask_onehot * valid_mask.unsqueeze(1).float()
        return mask_onehot

    def _infer_mask_index_from_onehot(self, mask_onehot):
        channel_sums = mask_onehot.sum(dim=1)
        has_signal = channel_sums > 0.0
        mask_index = torch.argmax(mask_onehot, dim=1).long()

        if self.loss.ignore_index is None:
            if not torch.all(has_signal):
                raise ValueError("mask_onehot contains zero-sum positions, but ignore_index is not configured.")
            return mask_index

        mask_index = mask_index.clone()
        mask_index[~has_signal] = int(self.loss.ignore_index)
        return mask_index

    def _validate_mask_alignment(self, mask_index, mask_onehot):
        inferred_mask_index = self._infer_mask_index_from_onehot(mask_onehot)
        valid_mask = self._mask_valid(mask_index)
        if torch.any(mask_index[valid_mask] != inferred_mask_index[valid_mask]):
            raise ValueError("mask_index and mask_onehot describe different semantic_mask targets.")
        if self.loss.ignore_index is not None:
            ignore_mask = ~valid_mask
            if torch.any(mask_onehot[ignore_mask.unsqueeze(1).expand_as(mask_onehot)] != 0):
                raise ValueError(
                    "mask_onehot must stay zero on ignore_index positions for the semantic_mask tokenizer."
                )

    def get_input(self, batch):
        self._extract_num_classes(batch)
        has_mask_index = self.mask_index_key in batch
        has_mask_onehot = self.mask_onehot_key in batch
        if not has_mask_index and not has_mask_onehot:
            raise KeyError(
                f"SemanticMaskVQAutoencoder requires '{self.mask_index_key}' and/or '{self.mask_onehot_key}' in batch."
            )

        mask_index = None
        mask_onehot = None
        if has_mask_index:
            mask_index = self._normalize_mask_index(batch[self.mask_index_key])
        if has_mask_onehot:
            mask_onehot = self._normalize_mask_onehot(batch[self.mask_onehot_key])

        if mask_index is None:
            mask_index = self._infer_mask_index_from_onehot(mask_onehot)
        if mask_onehot is None:
            mask_onehot = self._make_onehot(mask_index)
        else:
            self._validate_mask_alignment(mask_index, mask_onehot)

        return mask_index, mask_onehot

    def _get_log_batch_size(self, batch):
        if self.mask_index_key in batch:
            return int(torch.as_tensor(batch[self.mask_index_key]).shape[0])
        return int(torch.as_tensor(batch[self.mask_onehot_key]).shape[0])

    def _should_sync_dist(self):
        trainer = getattr(self, "trainer", None)
        return trainer is not None and int(getattr(trainer, "world_size", 1)) > 1

    @staticmethod
    def _detach_scalars(loss_dict):
        detached = {}
        for name, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                detached[name] = value.detach()
            else:
                detached[name] = value
        return detached

    def _compute_mask_metrics(self, mask_index, mask_logits):
        valid_mask = self._mask_valid(mask_index)
        if not torch.any(valid_mask):
            zero = mask_logits.new_tensor(0.0)
            return {
                "pixel_accuracy": zero,
                "miou": zero,
            }

        pred_mask = torch.argmax(mask_logits, dim=1)
        pixel_accuracy = (pred_mask[valid_mask] == mask_index[valid_mask]).float().mean()

        ious = []
        for class_id in range(self.num_classes):
            pred = (pred_mask == class_id) & valid_mask
            target = (mask_index == class_id) & valid_mask
            union = torch.logical_or(pred, target).sum()
            if int(union.item()) <= 0:
                continue
            intersection = torch.logical_and(pred, target).sum()
            ious.append(intersection.float() / union.float())
        miou = torch.stack(ious).mean() if ious else mask_logits.new_tensor(0.0)

        return {
            "pixel_accuracy": pixel_accuracy,
            "miou": miou,
        }

    def encode(self, mask_onehot):
        hidden = self.encoder(mask_onehot)
        z_e = self.pre_quant_conv(hidden)
        quantized = self.quantizer(z_e)
        quantized["z_e"] = z_e
        return quantized

    def decode(self, z_q):
        z_q = self.post_quant_conv(z_q)
        features = self.decoder(z_q)
        mask_logits = self.mask_head(features)
        return mask_logits

    def decode_latents(self, z_q):
        mask_logits = self.decode(z_q)
        mask_probs = torch.softmax(mask_logits, dim=1)
        mask_index = torch.argmax(mask_logits, dim=1)
        mask_onehot = F.one_hot(mask_index, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        return {
            "mask_logits": mask_logits,
            "mask_probs": mask_probs,
            "mask_index": mask_index,
            "mask_onehot": mask_onehot,
        }

    def decode_codes(self, codes):
        z_q = self.quantizer.codes_to_latents(codes)
        decoded = self.decode_latents(z_q)
        decoded["z_q"] = z_q
        decoded["codes"] = torch.as_tensor(codes, device=z_q.device, dtype=torch.long)
        return decoded

    def forward(self, batch, sample_posterior=None):
        self._validate_no_posterior_sampling(sample_posterior)
        mask_index, mask_onehot = self.get_input(batch)
        encoded = self.encode(mask_onehot)
        decoded = self.decode_latents(encoded["z_q"])
        recon_mask_index = decoded["mask_index"]
        recon_mask_onehot = decoded["mask_onehot"]
        total_loss, loss_dict = self.loss(
            mask_index,
            decoded["mask_logits"],
            vq_codebook_loss=encoded["vq_codebook_loss"],
            vq_commitment_loss=encoded["vq_commitment_loss"],
            quantizer_stats=encoded,
        )
        return {
            "z_e": encoded["z_e"],
            "z_q": encoded["z_q"],
            "z_q_lookup": encoded["z_q_lookup"],
            "codes": encoded["codes"],
            "mask_index": mask_index,
            "mask_onehot": mask_onehot,
            "mask_logits": decoded["mask_logits"],
            "mask_probs": decoded["mask_probs"],
            "recon_mask_index": recon_mask_index,
            "recon_mask_onehot": recon_mask_onehot,
            "quantizer_stats": {
                name: encoded[name]
                for name in ("perplexity", "used_code_count", "used_code_fraction", "dead_code_fraction")
            },
            "total_loss": total_loss,
            "loss_dict": loss_dict,
        }

    def encode_batch(self, batch, sample_posterior=None):
        self._validate_no_posterior_sampling(sample_posterior)
        mask_index, mask_onehot = self.get_input(batch)
        encoded = self.encode(mask_onehot)
        return {
            "z_e": encoded["z_e"],
            "z_q": encoded["z_q"],
            "codes": encoded["codes"],
            "mask_index": mask_index,
            "mask_onehot": mask_onehot,
            "quantizer_stats": {
                name: encoded[name]
                for name in ("perplexity", "used_code_count", "used_code_fraction", "dead_code_fraction")
            },
        }

    def shared_step(self, batch, split):
        outputs = self(batch)
        total_loss = outputs["total_loss"]
        batch_size = self._get_log_batch_size(batch)
        detached_loss_dict = self._detach_scalars(outputs["loss_dict"])
        metrics = self._compute_mask_metrics(outputs["mask_index"], outputs["mask_logits"])
        metrics.update(self._detach_scalars(outputs["quantizer_stats"]))
        sync_dist = self._should_sync_dist()

        prefixed_losses = {
            f"{split}/{name}": value
            for name, value in detached_loss_dict.items()
            if name != "total_loss"
        }
        prefixed_metrics = {f"{split}/{name}": value.detach() for name, value in metrics.items()}
        self.log(
            f"{split}/total_loss",
            total_loss.detach(),
            prog_bar=(split != "train"),
            logger=True,
            on_step=(split == "train"),
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=sync_dist,
        )
        self.log_dict(
            prefixed_losses,
            prog_bar=False,
            logger=True,
            on_step=(split == "train"),
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=sync_dist,
        )
        self.log_dict(
            prefixed_metrics,
            prog_bar=(split != "train"),
            logger=True,
            on_step=(split == "train"),
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=sync_dist,
        )
        return total_loss, detached_loss_dict

    def training_step(self, batch, batch_idx):
        total_loss, _ = self.shared_step(batch, split="train")
        return total_loss

    def validation_step(self, batch, batch_idx):
        _, detached_loss_dict = self.shared_step(batch, split="val")
        return detached_loss_dict["total_loss"]

    def configure_optimizers(self):
        lr = float(getattr(self, "learning_rate", 1.0e-4))
        optimizer = torch.optim.Adam(
            [
                param
                for param in (
                    list(self.encoder.parameters())
                    + list(self.decoder.parameters())
                    + list(self.pre_quant_conv.parameters())
                    + list(self.post_quant_conv.parameters())
                    + list(self.quantizer.parameters())
                    + list(self.mask_head.parameters())
                )
                if param.requires_grad
            ],
            lr=lr,
            betas=(0.5, 0.9),
        )
        return optimizer

    @torch.no_grad()
    def log_images(self, batch, sample_posterior=None, **kwargs):
        del kwargs
        outputs = self(batch, sample_posterior=sample_posterior)
        target_mask = outputs["mask_index"].unsqueeze(1).float()
        pred_mask = outputs["recon_mask_index"].unsqueeze(1).float()
        return {
            "inputs_mask_index": target_mask.to(self.device),
            "reconstructions_mask_index": pred_mask,
        }
