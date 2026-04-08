from copy import deepcopy

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from ldm.modules.diffusionmodules.model import Decoder, Encoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.util import instantiate_from_config

from latent_meanflow.models.semantic_autoencoder import _resolve_group_norm_groups


class ImageAutoencoderLoss(nn.Module):
    def __init__(
        self,
        rgb_l1_weight=1.0,
        rgb_lpips_weight=0.0,
        kl_weight=1.0e-6,
    ):
        super().__init__()
        self.rgb_l1_weight = float(rgb_l1_weight)
        self.rgb_lpips_weight = float(rgb_lpips_weight)
        self.kl_weight = float(kl_weight)

        self.lpips = None
        if self.rgb_lpips_weight > 0.0:
            from taming.modules.losses.lpips import LPIPS

            self.lpips = LPIPS().eval()
            for param in self.lpips.parameters():
                param.requires_grad = False

    def forward(self, rgb_target, rgb_recon, posterior):
        rgb_l1 = F.l1_loss(rgb_recon, rgb_target)

        if self.lpips is None:
            rgb_lpips = rgb_recon.new_tensor(0.0)
        else:
            rgb_lpips = self.lpips(rgb_recon.contiguous(), rgb_target.contiguous()).mean()

        kl_loss = posterior.kl().mean()

        total_loss = (
            self.rgb_l1_weight * rgb_l1
            + self.rgb_lpips_weight * rgb_lpips
            + self.kl_weight * kl_loss
        )

        loss_dict = {
            "rgb_l1": rgb_l1,
            "rgb_lpips": rgb_lpips,
            "kl": kl_loss,
            "total_loss": total_loss,
        }
        return total_loss, loss_dict


class ImageAutoencoder(pl.LightningModule):
    def __init__(
        self,
        ddconfig,
        lossconfig,
        embed_dim,
        rgb_channels=3,
        sample_posterior=True,
        ckpt_path=None,
        ignore_keys=None,
        monitor=None,
        image_key="image",
    ):
        super().__init__()
        ignore_keys = [] if ignore_keys is None else list(ignore_keys)

        self.rgb_channels = int(rgb_channels)
        self.num_classes = 0
        self.sample_posterior = bool(sample_posterior)
        self.image_key = image_key
        self.embed_dim = int(embed_dim)

        encoder_config = deepcopy(ddconfig)
        encoder_config["in_channels"] = self.rgb_channels
        encoder_config["out_ch"] = self.rgb_channels
        decoder_config = deepcopy(ddconfig)
        decoder_config["in_channels"] = self.rgb_channels
        decoder_config["out_ch"] = self.rgb_channels
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
        state = torch.load(path, map_location="cpu")
        state_dict = state["state_dict"] if "state_dict" in state else state
        keys = list(state_dict.keys())
        for key in keys:
            for prefix in ignore_keys:
                if key.startswith(prefix):
                    del state_dict[key]
                    break
        self.load_state_dict(state_dict, strict=False)
        print(f"Restored from {path}")

    def get_input(self, batch):
        image = batch[self.image_key]
        if image.ndim != 4:
            raise ValueError(f"Expected image tensor with shape [B, H, W, C], got {tuple(image.shape)}")
        image = image.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return image

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

    def encode(self, image):
        hidden = self.encoder(image)
        moments = self.quant_conv(hidden)
        return DiagonalGaussianDistribution(moments)

    def decode(self, z):
        z = self.post_quant_conv(z)
        features = self.decoder(z)
        rgb_recon = torch.tanh(self.rgb_head(features))
        return rgb_recon

    def forward(self, batch, sample_posterior=None):
        image = self.get_input(batch)
        posterior = self.encode(image)
        if sample_posterior is None:
            sample_posterior = self.sample_posterior
        z = posterior.sample() if sample_posterior else posterior.mode()
        rgb_recon = self.decode(z)
        total_loss, loss_dict = self.loss(image, rgb_recon, posterior)
        return {
            "z": z,
            "posterior": posterior,
            "rgb_recon": rgb_recon,
            "total_loss": total_loss,
            "loss_dict": loss_dict,
        }

    def encode_batch(self, batch, sample_posterior=None):
        image = self.get_input(batch)
        posterior = self.encode(image)
        if sample_posterior is None:
            sample_posterior = self.sample_posterior
        z = posterior.sample() if sample_posterior else posterior.mode()
        return {
            "z": z,
            "posterior": posterior,
            "image": image,
        }

    def decode_latents(self, z):
        rgb_recon = self.decode(z)
        return {
            "rgb_recon": rgb_recon,
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
            + list(self.rgb_head.parameters()),
            lr=lr,
            betas=(0.5, 0.9),
        )
        return optimizer

    @torch.no_grad()
    def log_images(self, batch, sample_posterior=False, **kwargs):
        outputs = self(batch, sample_posterior=sample_posterior)
        image = self.get_input(batch)
        return {
            "inputs_image": image.to(self.device),
            "reconstructions_image": outputs["rgb_recon"],
        }
