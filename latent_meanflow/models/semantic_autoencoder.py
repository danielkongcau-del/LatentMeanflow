from copy import deepcopy

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from ldm.modules.diffusionmodules.model import Decoder, Encoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.util import instantiate_from_config


def _resolve_group_norm_groups(num_channels, max_groups=32):
    for num_groups in range(min(max_groups, num_channels), 0, -1):
        if num_channels % num_groups == 0:
            return num_groups
    return 1


class SemanticPairLoss(nn.Module):
    def __init__(
        self,
        rgb_l1_weight=1.0,
        rgb_lpips_weight=0.0,
        mask_ce_weight=1.0,
        mask_dice_weight=0.0,
        mask_focal_weight=0.0,
        kl_weight=1.0e-6,
        ignore_index=None,
        focal_gamma=2.0,
        ce_label_smoothing=0.0,
        dice_eps=1.0e-6,
    ):
        super().__init__()
        self.rgb_l1_weight = float(rgb_l1_weight)
        self.rgb_lpips_weight = float(rgb_lpips_weight)
        self.mask_ce_weight = float(mask_ce_weight)
        self.mask_dice_weight = float(mask_dice_weight)
        self.mask_focal_weight = float(mask_focal_weight)
        self.kl_weight = float(kl_weight)
        self.ignore_index = None if ignore_index is None else int(ignore_index)
        self.focal_gamma = float(focal_gamma)
        self.ce_label_smoothing = float(ce_label_smoothing)
        self.dice_eps = float(dice_eps)

        self.lpips = None
        if self.rgb_lpips_weight > 0.0:
            from taming.modules.losses.lpips import LPIPS

            self.lpips = LPIPS().eval()
            for param in self.lpips.parameters():
                param.requires_grad = False

    def _mask_valid(self, mask_index):
        if self.ignore_index is None:
            return torch.ones_like(mask_index, dtype=torch.bool)
        return mask_index != self.ignore_index

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

    def forward(self, rgb_target, rgb_recon, mask_index, mask_logits, posterior):
        rgb_l1 = F.l1_loss(rgb_recon, rgb_target)

        if self.lpips is None:
            rgb_lpips = rgb_recon.new_tensor(0.0)
        else:
            rgb_lpips = self.lpips(rgb_recon.contiguous(), rgb_target.contiguous()).mean()

        mask_ce = F.cross_entropy(
            mask_logits,
            mask_index,
            ignore_index=-100 if self.ignore_index is None else self.ignore_index,
            reduction="mean",
            label_smoothing=self.ce_label_smoothing,
        )
        mask_dice = self._dice_loss(mask_logits, mask_index)
        mask_focal = self._focal_loss(mask_logits, mask_index)
        kl_loss = posterior.kl().mean()

        total_loss = (
            self.rgb_l1_weight * rgb_l1
            + self.rgb_lpips_weight * rgb_lpips
            + self.mask_ce_weight * mask_ce
            + self.mask_dice_weight * mask_dice
            + self.mask_focal_weight * mask_focal
            + self.kl_weight * kl_loss
        )

        loss_dict = {
            "rgb_l1": rgb_l1,
            "rgb_lpips": rgb_lpips,
            "mask_ce": mask_ce,
            "mask_dice": mask_dice,
            "mask_focal": mask_focal,
            "kl": kl_loss,
            "total_loss": total_loss,
        }
        return total_loss, loss_dict


class SemanticPairAutoencoder(pl.LightningModule):
    def __init__(
        self,
        ddconfig,
        lossconfig,
        embed_dim,
        num_classes,
        rgb_channels=3,
        sample_posterior=True,
        ckpt_path=None,
        ignore_keys=None,
        monitor=None,
        image_key="image",
        mask_index_key="mask_index",
        mask_onehot_key="mask_onehot",
    ):
        super().__init__()
        ignore_keys = [] if ignore_keys is None else list(ignore_keys)

        self.rgb_channels = int(rgb_channels)
        self.num_classes = int(num_classes)
        self.sample_posterior = bool(sample_posterior)
        self.image_key = image_key
        self.mask_index_key = mask_index_key
        self.mask_onehot_key = mask_onehot_key
        self.embed_dim = int(embed_dim)

        expected_in_channels = self.rgb_channels + self.num_classes
        encoder_config = deepcopy(ddconfig)
        encoder_config["in_channels"] = expected_in_channels
        encoder_config["out_ch"] = expected_in_channels
        decoder_config = deepcopy(ddconfig)
        decoder_config["in_channels"] = expected_in_channels
        decoder_config["out_ch"] = expected_in_channels
        decoder_config["give_pre_end"] = True

        self.encoder = Encoder(**encoder_config)
        self.decoder = Decoder(**decoder_config)
        self.quant_conv = nn.Conv2d(2 * encoder_config["z_channels"], 2 * self.embed_dim, kernel_size=1)
        self.post_quant_conv = nn.Conv2d(self.embed_dim, encoder_config["z_channels"], kernel_size=1)

        self.decoder_feature_channels = decoder_config["ch"] * decoder_config["ch_mult"][0]
        norm_groups = _resolve_group_norm_groups(self.decoder_feature_channels)
        self.rgb_head = nn.Sequential(
            nn.GroupNorm(num_groups=norm_groups, num_channels=self.decoder_feature_channels),
            nn.SiLU(),
            nn.Conv2d(self.decoder_feature_channels, self.decoder_feature_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(self.decoder_feature_channels, self.rgb_channels, kernel_size=3, padding=1),
        )
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

        if monitor is not None:
            self.monitor = monitor
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

    def _extract_num_classes(self, batch):
        batch_num_classes = batch.get("num_classes", None)
        if batch_num_classes is None:
            return self.num_classes
        if isinstance(batch_num_classes, torch.Tensor):
            unique_values = torch.unique(batch_num_classes.detach().cpu())
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

    def get_input(self, batch):
        self._extract_num_classes(batch)

        image = batch[self.image_key]
        if image.ndim != 4:
            raise ValueError(f"Expected image tensor with shape [B, H, W, C], got {tuple(image.shape)}")
        image = image.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()

        mask_onehot = batch[self.mask_onehot_key]
        if mask_onehot.ndim != 4:
            raise ValueError(
                f"Expected mask_onehot tensor with shape [B, H, W, K], got {tuple(mask_onehot.shape)}"
            )
        mask_onehot = mask_onehot.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()

        mask_index = batch[self.mask_index_key]
        if mask_index.ndim != 3:
            raise ValueError(f"Expected mask_index tensor with shape [B, H, W], got {tuple(mask_index.shape)}")
        mask_index = mask_index.long()

        joint_input = torch.cat([image, mask_onehot], dim=1)
        return image, mask_index, mask_onehot, joint_input

    def _get_log_batch_size(self, batch):
        image = batch[self.image_key]
        return int(image.shape[0])

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

    def encode(self, joint_input):
        hidden = self.encoder(joint_input)
        moments = self.quant_conv(hidden)
        return DiagonalGaussianDistribution(moments)

    def decode(self, z):
        z = self.post_quant_conv(z)
        features = self.decoder(z)
        rgb_recon = torch.tanh(self.rgb_head(features))
        mask_logits = self.mask_head(features)
        return rgb_recon, mask_logits

    def forward(self, batch, sample_posterior=None):
        image, mask_index, mask_onehot, joint_input = self.get_input(batch)
        posterior = self.encode(joint_input)
        if sample_posterior is None:
            sample_posterior = self.sample_posterior
        z = posterior.sample() if sample_posterior else posterior.mode()
        rgb_recon, mask_logits = self.decode(z)
        total_loss, loss_dict = self.loss(image, rgb_recon, mask_index, mask_logits, posterior)
        return {
            "z": z,
            "posterior": posterior,
            "joint_input": joint_input,
            "mask_onehot": mask_onehot,
            "rgb_recon": rgb_recon,
            "mask_logits": mask_logits,
            "total_loss": total_loss,
            "loss_dict": loss_dict,
        }

    def encode_batch(self, batch, sample_posterior=None):
        image, mask_index, mask_onehot, joint_input = self.get_input(batch)
        posterior = self.encode(joint_input)
        if sample_posterior is None:
            sample_posterior = self.sample_posterior
        z = posterior.sample() if sample_posterior else posterior.mode()
        return {
            "z": z,
            "posterior": posterior,
            "image": image,
            "mask_index": mask_index,
            "mask_onehot": mask_onehot,
            "joint_input": joint_input,
        }

    def decode_latents(self, z):
        rgb_recon, mask_logits = self.decode(z)
        mask_index = torch.argmax(mask_logits, dim=1)
        return {
            "rgb_recon": rgb_recon,
            "mask_logits": mask_logits,
            "mask_index": mask_index,
        }

    def shared_step(self, batch, split):
        outputs = self(batch)
        total_loss = outputs["total_loss"]
        batch_size = self._get_log_batch_size(batch)
        detached_loss_dict = self._detach_scalars(outputs["loss_dict"])
        sync_dist = self._should_sync_dist()

        prefixed = {
            f"{split}/{name}": value
            for name, value in detached_loss_dict.items()
            if name != "total_loss"
        }
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
            prefixed,
            prog_bar=False,
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
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.quant_conv.parameters())
            + list(self.post_quant_conv.parameters())
            + list(self.rgb_head.parameters())
            + list(self.mask_head.parameters()),
            lr=lr,
            betas=(0.5, 0.9),
        )
        return optimizer

    @torch.no_grad()
    def log_images(self, batch, sample_posterior=False, **kwargs):
        outputs = self(batch, sample_posterior=sample_posterior)
        image, mask_index, _, _ = self.get_input(batch)
        pred_mask = torch.argmax(outputs["mask_logits"], dim=1, keepdim=True).float()
        target_mask = mask_index.unsqueeze(1).float()

        return {
            "inputs_image": image.to(self.device),
            "reconstructions_image": outputs["rgb_recon"],
            "inputs_mask_index": target_mask.to(self.device),
            "reconstructions_mask_index": pred_mask,
        }
