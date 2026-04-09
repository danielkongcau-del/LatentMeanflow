from copy import deepcopy

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from ldm.modules.diffusionmodules.model import Decoder, Encoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.util import instantiate_from_config

from latent_meanflow.models.semantic_autoencoder import _resolve_group_norm_groups


def _apply_optional_spectral_norm(module, enabled):
    return nn.utils.spectral_norm(module) if enabled else module


def _hinge_discriminator_loss(logits_real, logits_fake):
    loss_real = F.relu(1.0 - logits_real).mean()
    loss_fake = F.relu(1.0 + logits_fake).mean()
    return 0.5 * (loss_real + loss_fake)


def _hinge_generator_loss(logits_fake):
    return -logits_fake.mean()


class PatchDiscriminator(nn.Module):
    def __init__(
        self,
        in_channels=3,
        base_channels=64,
        max_channels=256,
        num_layers=4,
        use_spectral_norm=True,
        norm_type="instance",
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.base_channels = int(base_channels)
        self.max_channels = int(max_channels)
        self.num_layers = int(num_layers)
        self.use_spectral_norm = bool(use_spectral_norm)
        self.norm_type = str(norm_type)

        layers = []
        in_ch = self.in_channels
        out_ch = self.base_channels
        layers.extend(
            [
                _apply_optional_spectral_norm(
                    nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                    enabled=self.use_spectral_norm,
                ),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        )

        current_channels = out_ch
        for layer_index in range(1, self.num_layers):
            next_channels = min(self.base_channels * (2 ** layer_index), self.max_channels)
            stride = 1 if layer_index == self.num_layers - 1 else 2
            layers.append(
                _apply_optional_spectral_norm(
                    nn.Conv2d(current_channels, next_channels, kernel_size=4, stride=stride, padding=1),
                    enabled=self.use_spectral_norm,
                )
            )
            if self.norm_type == "instance":
                layers.append(nn.InstanceNorm2d(next_channels, affine=True))
            elif self.norm_type == "group":
                layers.append(
                    nn.GroupNorm(
                        num_groups=_resolve_group_norm_groups(next_channels),
                        num_channels=next_channels,
                    )
                )
            else:
                raise ValueError(f"Unsupported discriminator norm_type: {self.norm_type}")
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            current_channels = next_channels

        layers.append(
            _apply_optional_spectral_norm(
                nn.Conv2d(current_channels, 1, kernel_size=4, stride=1, padding=1),
                enabled=self.use_spectral_norm,
            )
        )
        self.model = nn.Sequential(*layers)

    def forward(self, image):
        return self.model(image)


class ImageAutoencoderLoss(nn.Module):
    def __init__(
        self,
        rgb_l1_weight=1.0,
        rgb_lpips_weight=0.0,
        kl_weight=1.0e-6,
        latent_channel_std_floor_weight=0.0,
        latent_channel_std_floor=0.0,
        latent_utilization_threshold=0.05,
        latent_health_eps=1.0e-6,
    ):
        super().__init__()
        self.rgb_l1_weight = float(rgb_l1_weight)
        self.rgb_lpips_weight = float(rgb_lpips_weight)
        self.kl_weight = float(kl_weight)
        self.latent_channel_std_floor_weight = float(latent_channel_std_floor_weight)
        self.latent_channel_std_floor = float(latent_channel_std_floor)
        self.latent_utilization_threshold = float(latent_utilization_threshold)
        self.latent_health_eps = float(latent_health_eps)

        self.lpips = None
        if self.rgb_lpips_weight > 0.0:
            from taming.modules.losses.lpips import LPIPS

            self.lpips = LPIPS().eval()
            for param in self.lpips.parameters():
                param.requires_grad = False

    def _latent_health_metrics(self, latent_tensor):
        if latent_tensor.ndim != 4:
            raise ValueError(f"Expected latent tensor with shape [B, C, H, W], got {tuple(latent_tensor.shape)}")

        channel_values = latent_tensor.permute(1, 0, 2, 3).reshape(latent_tensor.shape[1], -1)
        channel_std = channel_values.std(dim=1, unbiased=False)
        std_mean = channel_std.mean()
        std_min = channel_std.min()
        std_max = channel_std.max()
        std_cv = channel_std.std(unbiased=False) / (std_mean.abs() + self.latent_health_eps)
        utilized = channel_std >= self.latent_utilization_threshold
        utilized_count = utilized.float().sum()
        utilized_fraction = utilized.float().mean()

        if self.latent_channel_std_floor_weight > 0.0 and self.latent_channel_std_floor > 0.0:
            std_floor_penalty = F.relu(self.latent_channel_std_floor - channel_std).pow(2).mean()
        else:
            std_floor_penalty = latent_tensor.new_tensor(0.0)

        return {
            "channel_std": channel_std,
            "latent_batch_channel_std_mean": std_mean,
            "latent_batch_channel_std_min": std_min,
            "latent_batch_channel_std_max": std_max,
            "latent_batch_channel_std_cv": std_cv,
            "latent_utilized_channel_count": utilized_count,
            "latent_utilized_channel_fraction": utilized_fraction,
            "latent_std_floor_penalty": std_floor_penalty,
        }

    def forward(self, rgb_target, rgb_recon, posterior, latent_for_health=None):
        rgb_l1 = F.l1_loss(rgb_recon, rgb_target)

        if self.lpips is None:
            rgb_lpips = rgb_recon.new_tensor(0.0)
        else:
            rgb_lpips = self.lpips(rgb_recon.contiguous(), rgb_target.contiguous()).mean()

        kl_loss = posterior.kl().mean()

        if latent_for_health is None:
            latent_for_health = posterior.mode()
        latent_metrics = self._latent_health_metrics(latent_for_health)

        total_loss = (
            self.rgb_l1_weight * rgb_l1
            + self.rgb_lpips_weight * rgb_lpips
            + self.kl_weight * kl_loss
            + self.latent_channel_std_floor_weight * latent_metrics["latent_std_floor_penalty"]
        )

        loss_dict = {
            "rgb_l1": rgb_l1,
            "rgb_lpips": rgb_lpips,
            "kl": kl_loss,
            "latent_std_floor_penalty": latent_metrics["latent_std_floor_penalty"],
            "latent_batch_channel_std_mean": latent_metrics["latent_batch_channel_std_mean"],
            "latent_batch_channel_std_min": latent_metrics["latent_batch_channel_std_min"],
            "latent_batch_channel_std_max": latent_metrics["latent_batch_channel_std_max"],
            "latent_batch_channel_std_cv": latent_metrics["latent_batch_channel_std_cv"],
            "latent_utilized_channel_count": latent_metrics["latent_utilized_channel_count"],
            "latent_utilized_channel_fraction": latent_metrics["latent_utilized_channel_fraction"],
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
        discriminator_config=None,
        generator_adversarial_weight=0.0,
        discriminator_start_step=0,
        discriminator_learning_rate=None,
        discriminator_beta1=0.5,
        discriminator_beta2=0.9,
    ):
        super().__init__()
        ignore_keys = [] if ignore_keys is None else list(ignore_keys)

        self.rgb_channels = int(rgb_channels)
        self.num_classes = 0
        self.sample_posterior = bool(sample_posterior)
        self.image_key = image_key
        self.embed_dim = int(embed_dim)
        self.generator_adversarial_weight = float(generator_adversarial_weight)
        self.discriminator_start_step = int(discriminator_start_step)
        self.discriminator_learning_rate = (
            None if discriminator_learning_rate is None else float(discriminator_learning_rate)
        )
        self.discriminator_beta1 = float(discriminator_beta1)
        self.discriminator_beta2 = float(discriminator_beta2)

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

        if getattr(self.decoder, "give_pre_end", False):
            # The project-layer RGB head consumes decoder pre-end features,
            # so the vendor decoder tail stays unused during training.
            for module in (self.decoder.norm_out, self.decoder.conv_out):
                for param in module.parameters():
                    param.requires_grad = False

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

        self.discriminator = None
        self.use_discriminator = discriminator_config is not None and self.generator_adversarial_weight > 0.0
        if self.use_discriminator:
            self.discriminator = instantiate_from_config(discriminator_config)
            self.automatic_optimization = False

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

    def _adversarial_factor(self):
        if not self.use_discriminator:
            return 0.0
        return 0.0 if int(self.global_step) < self.discriminator_start_step else 1.0

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
        latent_mode = posterior.mode()
        if sample_posterior is None:
            sample_posterior = self.sample_posterior
        z = posterior.sample() if sample_posterior else latent_mode
        rgb_recon = self.decode(z)
        total_loss, loss_dict = self.loss(
            image,
            rgb_recon,
            posterior,
            latent_for_health=latent_mode,
        )
        return {
            "image": image,
            "z": z,
            "latent_mode": latent_mode,
            "posterior": posterior,
            "rgb_recon": rgb_recon,
            "total_loss": total_loss,
            "loss_dict": loss_dict,
        }

    def encode_batch(self, batch, sample_posterior=None):
        image = self.get_input(batch)
        posterior = self.encode(image)
        latent_mode = posterior.mode()
        if sample_posterior is None:
            sample_posterior = self.sample_posterior
        z = posterior.sample() if sample_posterior else latent_mode
        return {
            "z": z,
            "latent_mode": latent_mode,
            "posterior": posterior,
            "image": image,
        }

    def decode_latents(self, z):
        rgb_recon = self.decode(z)
        return {
            "rgb_recon": rgb_recon,
        }

    def _log_reconstruction_metrics(self, batch, outputs, split):
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
        self.log(
            f"{split}/recon_total_loss",
            total_loss.detach(),
            prog_bar=False,
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

    def _evaluate_adversarial_terms(self, image, rgb_recon):
        generator_adv_loss = rgb_recon.new_tensor(0.0)
        discriminator_loss = rgb_recon.new_tensor(0.0)
        logits_real_mean = rgb_recon.new_tensor(0.0)
        logits_fake_mean = rgb_recon.new_tensor(0.0)
        disc_factor = self._adversarial_factor()

        if self.use_discriminator:
            logits_fake = self.discriminator(rgb_recon)
            logits_real = self.discriminator(image)
            generator_adv_loss = _hinge_generator_loss(logits_fake)
            discriminator_loss = _hinge_discriminator_loss(logits_real, logits_fake)
            logits_real_mean = logits_real.mean()
            logits_fake_mean = logits_fake.mean()

        return {
            "generator_adversarial_loss": generator_adv_loss,
            "discriminator_loss": discriminator_loss,
            "discriminator_factor": rgb_recon.new_tensor(float(disc_factor)),
            "discriminator_logits_real_mean": logits_real_mean,
            "discriminator_logits_fake_mean": logits_fake_mean,
        }

    def _log_adversarial_metrics(self, split, batch_size, metrics):
        sync_dist = self._should_sync_dist()
        for name, value in metrics.items():
            self.log(
                f"{split}/{name}",
                value.detach(),
                prog_bar=False,
                logger=True,
                on_step=(split == "train"),
                on_epoch=True,
                batch_size=batch_size,
                sync_dist=sync_dist,
            )

    def shared_step(self, batch, split):
        outputs = self(batch)
        self._log_reconstruction_metrics(batch, outputs, split=split)
        if self.use_discriminator:
            batch_size = self._get_log_batch_size(batch)
            with torch.no_grad():
                adv_metrics = self._evaluate_adversarial_terms(outputs["image"], outputs["rgb_recon"])
            self._log_adversarial_metrics(split, batch_size=batch_size, metrics=adv_metrics)
        return outputs["total_loss"], self._detach_scalars(outputs["loss_dict"])

    def training_step(self, batch, batch_idx):
        if not self.use_discriminator:
            total_loss, _ = self.shared_step(batch, split="train")
            return total_loss

        outputs = self(batch)
        batch_size = self._get_log_batch_size(batch)
        self._log_reconstruction_metrics(batch, outputs, split="train")

        ae_optimizer, disc_optimizer = self.optimizers()
        disc_factor = self._adversarial_factor()

        generator_adv_loss = outputs["rgb_recon"].new_tensor(0.0)
        generator_total_loss = outputs["total_loss"]
        self.toggle_optimizer(ae_optimizer)
        ae_optimizer.zero_grad()
        logits_fake = self.discriminator(outputs["rgb_recon"])
        generator_adv_loss = _hinge_generator_loss(logits_fake)
        generator_total_loss = generator_total_loss + (
            float(disc_factor) * self.generator_adversarial_weight * generator_adv_loss
        )
        self.manual_backward(generator_total_loss)
        ae_optimizer.step()
        self.untoggle_optimizer(ae_optimizer)

        discriminator_loss = outputs["rgb_recon"].new_tensor(0.0)
        logits_real_mean = outputs["rgb_recon"].new_tensor(0.0)
        logits_fake_mean = outputs["rgb_recon"].new_tensor(0.0)
        self.toggle_optimizer(disc_optimizer)
        disc_optimizer.zero_grad()
        logits_real = self.discriminator(outputs["image"].detach())
        logits_fake = self.discriminator(outputs["rgb_recon"].detach())
        discriminator_loss = _hinge_discriminator_loss(logits_real, logits_fake)
        self.manual_backward(float(disc_factor) * discriminator_loss)
        disc_optimizer.step()
        self.untoggle_optimizer(disc_optimizer)
        logits_real_mean = logits_real.mean().detach()
        logits_fake_mean = logits_fake.mean().detach()

        adv_metrics = {
            "generator_total_loss": generator_total_loss.detach(),
            "generator_adversarial_loss": generator_adv_loss.detach(),
            "discriminator_loss": discriminator_loss.detach(),
            "discriminator_factor": outputs["rgb_recon"].new_tensor(float(disc_factor)),
            "discriminator_logits_real_mean": logits_real_mean,
            "discriminator_logits_fake_mean": logits_fake_mean,
        }
        self._log_adversarial_metrics("train", batch_size=batch_size, metrics=adv_metrics)
        return generator_total_loss.detach()

    def validation_step(self, batch, batch_idx):
        _, detached_loss_dict = self.shared_step(batch, split="val")
        return detached_loss_dict["total_loss"]

    def configure_optimizers(self):
        lr = float(getattr(self, "learning_rate", 1.0e-4))
        autoencoder_parameters = [
            *list(self.encoder.parameters()),
            *list(self.decoder.parameters()),
            *list(self.quant_conv.parameters()),
            *list(self.post_quant_conv.parameters()),
            *list(self.rgb_head.parameters()),
        ]
        autoencoder_optimizer = torch.optim.Adam(
            [param for param in autoencoder_parameters if param.requires_grad],
            lr=lr,
            betas=(0.5, 0.9),
        )
        if not self.use_discriminator:
            return autoencoder_optimizer

        discriminator_optimizer = torch.optim.Adam(
            [param for param in self.discriminator.parameters() if param.requires_grad],
            lr=lr if self.discriminator_learning_rate is None else float(self.discriminator_learning_rate),
            betas=(self.discriminator_beta1, self.discriminator_beta2),
        )
        return [autoencoder_optimizer, discriminator_optimizer]

    @torch.no_grad()
    def log_images(self, batch, sample_posterior=False, **kwargs):
        outputs = self(batch, sample_posterior=sample_posterior)
        image = self.get_input(batch)
        return {
            "inputs_image": image.to(self.device),
            "reconstructions_image": outputs["rgb_recon"],
        }
