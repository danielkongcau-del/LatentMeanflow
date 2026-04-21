import importlib.util
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from latent_meanflow.losses.semantic_structure import (
    boundary_bce_loss,
    build_valid_mask,
    mask_index_to_boundary_target,
    semantic_probs_to_soft_boundary,
)
from latent_meanflow.utils.palette import build_default_palette


@lru_cache(maxsize=1)
def _load_maskgit_vqgan_module():
    module_path = (
        Path(__file__).resolve().parents[2]
        / "third_party"
        / "maskgit-pytorch"
        / "models"
        / "stage1"
        / "maskgit"
        / "maskgit_vqgan.py"
    )
    if not module_path.exists():
        raise FileNotFoundError(f"maskgit-pytorch VQGAN source not found: {module_path}")

    spec = importlib.util.spec_from_file_location("latent_meanflow_vendor_maskgit_vqgan", module_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"Could not load spec for {module_path}")
    spec.loader.exec_module(module)
    return module


class MaskGitPaletteVQTokenizer(pl.LightningModule):
    def __init__(
        self,
        pretrained_weight_path=None,
        monitor="val/semantic_ce",
        resolution=256,
        num_classes=7,
        ignore_index=-1,
        hidden_channels=128,
        channel_mult=(1, 1, 2, 2, 4),
        num_res_blocks=2,
        z_channels=256,
        codebook_size=1024,
        codebook_dim=256,
        commitment_cost=0.25,
        rgb_l1_weight=1.0,
        semantic_ce_weight=1.0,
        semantic_dice_weight=0.5,
        boundary_loss_weight=0.25,
        vq_loss_weight=1.0,
        palette_logit_scale=64.0,
        weight_decay=0.01,
        optimizer_beta1=0.9,
        optimizer_beta2=0.95,
        mask_index_key="mask_index",
        palette_image_key="palette_image",
    ):
        super().__init__()
        self.save_hyperparameters()

        self.monitor = str(monitor)
        self.learning_rate = 1.0e-4
        self.num_classes = int(num_classes)
        self.ignore_index = None if ignore_index is None else int(ignore_index)
        self.weight_decay = float(weight_decay)
        self.optimizer_betas = (float(optimizer_beta1), float(optimizer_beta2))
        self.mask_index_key = str(mask_index_key)
        self.palette_image_key = str(palette_image_key)

        self.rgb_l1_weight = float(rgb_l1_weight)
        self.semantic_ce_weight = float(semantic_ce_weight)
        self.semantic_dice_weight = float(semantic_dice_weight)
        self.boundary_loss_weight = float(boundary_loss_weight)
        self.vq_loss_weight = float(vq_loss_weight)
        self.palette_logit_scale = float(palette_logit_scale)

        config = SimpleNamespace(
            channel_mult=list(channel_mult),
            num_resolutions=len(tuple(channel_mult)),
            dropout=0.0,
            hidden_channels=int(hidden_channels),
            num_channels=3,
            num_res_blocks=int(num_res_blocks),
            resolution=int(resolution),
            z_channels=int(z_channels),
        )
        vendor = _load_maskgit_vqgan_module()
        self.encoder = vendor.Encoder(config)
        self.decoder = vendor.Decoder(config)
        self.quantize = vendor.VectorQuantizer(
            num_embeddings=int(codebook_size),
            embedding_dim=int(codebook_dim),
            commitment_cost=float(commitment_cost),
        )

        self.pretrained_weight_path = None if pretrained_weight_path in {None, ""} else str(pretrained_weight_path)
        if self.pretrained_weight_path is not None:
            self._load_pretrained_weights(self.pretrained_weight_path)

        palette_spec = build_default_palette(self.num_classes, ignore_index=self.ignore_index)
        palette = torch.from_numpy(palette_spec["palette"]).to(dtype=torch.float32) / 255.0
        self.register_buffer("palette_rgb", palette, persistent=False)
        ignore_color = torch.as_tensor(palette_spec["ignore_color"], dtype=torch.float32) / 255.0
        self.register_buffer("ignore_color_rgb", ignore_color, persistent=False)

    def _load_pretrained_weights(self, pretrained_weight_path):
        ckpt_path = Path(pretrained_weight_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Pretrained MaskGIT VQ tokenizer weights not found: {ckpt_path}")
        payload = torch.load(ckpt_path, map_location="cpu")
        if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
            state_dict = payload["state_dict"]
        elif isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], dict):
            state_dict = payload["model"]
        elif isinstance(payload, dict):
            state_dict = payload
        else:
            raise ValueError(f"Unsupported pretrained payload type: {type(payload).__name__}")
        self.load_state_dict(state_dict, strict=True)

    def _prepare_mask_index(self, mask_index):
        if not isinstance(mask_index, torch.Tensor):
            mask_index = torch.as_tensor(mask_index)
        if mask_index.ndim == 4 and int(mask_index.shape[1]) == 1:
            mask_index = mask_index[:, 0]
        if mask_index.ndim != 3:
            raise ValueError(
                f"MaskGitPaletteVQTokenizer expects mask_index with shape [B, H, W], got {tuple(mask_index.shape)}"
            )
        return mask_index.long().contiguous()

    def _prepare_palette_image(self, palette_image):
        if not isinstance(palette_image, torch.Tensor):
            palette_image = torch.as_tensor(palette_image)
        if palette_image.ndim == 3:
            palette_image = palette_image.unsqueeze(0)
        if palette_image.ndim != 4 or int(palette_image.shape[1]) != 3:
            raise ValueError(
                "MaskGitPaletteVQTokenizer expects palette_image with shape [B, 3, H, W], "
                f"got {tuple(palette_image.shape)}"
            )
        return palette_image.to(dtype=torch.float32).contiguous()

    def _cross_entropy_loss(self, mask_logits, mask_index):
        valid_mask = build_valid_mask(mask_index, ignore_index=self.ignore_index)
        if not bool(torch.any(valid_mask)):
            return mask_logits.new_tensor(0.0)
        safe_targets = mask_index.clone()
        safe_targets[~valid_mask] = 0
        loss = F.cross_entropy(mask_logits, safe_targets, reduction="none")
        valid_weight = valid_mask.to(dtype=loss.dtype)
        return (loss * valid_weight).sum() / valid_weight.sum().clamp_min(1.0)

    def _dice_loss(self, mask_logits, mask_index):
        valid_mask = build_valid_mask(mask_index, ignore_index=self.ignore_index)
        if not bool(torch.any(valid_mask)):
            return mask_logits.new_tensor(0.0)
        safe_targets = mask_index.clone()
        safe_targets[~valid_mask] = 0
        probs = torch.softmax(mask_logits, dim=1)
        target = F.one_hot(safe_targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        valid_mask = valid_mask.unsqueeze(1)
        probs = probs * valid_mask
        target = target * valid_mask
        intersection = (probs * target).sum(dim=(0, 2, 3))
        denominator = probs.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3))
        dice = (2.0 * intersection + 1.0e-6) / (denominator + 1.0e-6)
        return 1.0 - dice.mean()

    def _compute_mask_metrics(self, mask_index, mask_logits):
        valid_mask = build_valid_mask(mask_index, ignore_index=self.ignore_index)
        if not bool(torch.any(valid_mask)):
            zero = mask_logits.new_tensor(0.0)
            return {"pixel_accuracy": zero, "miou": zero}
        pred = torch.argmax(mask_logits, dim=1)
        correct = (pred == mask_index) & valid_mask
        pixel_accuracy = correct.to(dtype=torch.float32).sum() / valid_mask.to(dtype=torch.float32).sum().clamp_min(1.0)
        ious = []
        for class_idx in range(self.num_classes):
            pred_mask = pred == class_idx
            target_mask = mask_index == class_idx
            intersection = (pred_mask & target_mask & valid_mask).sum().to(dtype=torch.float32)
            union = ((pred_mask | target_mask) & valid_mask).sum().to(dtype=torch.float32)
            if float(union.item()) > 0.0:
                ious.append(intersection / union)
        miou = torch.stack(ious).mean() if ious else mask_logits.new_tensor(0.0)
        return {"pixel_accuracy": pixel_accuracy, "miou": miou}

    def _image_to_palette_logits(self, palette_image):
        palette = self.palette_rgb.to(device=palette_image.device, dtype=palette_image.dtype)
        distances = (palette_image.unsqueeze(1) - palette.view(1, self.num_classes, 3, 1, 1)).pow(2).sum(dim=2)
        return -float(self.palette_logit_scale) * distances

    def forward(self, batch):
        palette_image = self._prepare_palette_image(batch[self.palette_image_key])
        mask_index = self._prepare_mask_index(batch[self.mask_index_key]).to(device=palette_image.device)

        hidden_states = self.encoder(palette_image)
        quantized_states, codebook_indices, vq_loss = self.quantize(hidden_states, return_loss=True)
        reconstructed = torch.clamp(self.decoder(quantized_states), 0.0, 1.0)
        mask_logits = self._image_to_palette_logits(reconstructed)
        mask_probs = torch.softmax(mask_logits, dim=1)
        valid_mask = build_valid_mask(mask_index, ignore_index=self.ignore_index)
        boundary_target = mask_index_to_boundary_target(mask_index, ignore_index=self.ignore_index)
        boundary_pred = semantic_probs_to_soft_boundary(mask_probs, valid_mask=valid_mask)

        rgb_l1 = F.l1_loss(reconstructed, palette_image)
        semantic_ce = self._cross_entropy_loss(mask_logits, mask_index)
        semantic_dice = self._dice_loss(mask_logits, mask_index)
        boundary_loss = boundary_bce_loss(
            boundary_pred,
            boundary_target,
            valid_mask=valid_mask,
        )
        if vq_loss is None:
            vq_loss = reconstructed.new_tensor(0.0)

        total_loss = (
            self.rgb_l1_weight * rgb_l1
            + self.semantic_ce_weight * semantic_ce
            + self.semantic_dice_weight * semantic_dice
            + self.boundary_loss_weight * boundary_loss
            + self.vq_loss_weight * vq_loss
        )

        with torch.no_grad():
            metrics = self._compute_mask_metrics(mask_index, mask_logits)

        return {
            "loss": total_loss,
            "loss_dict": {
                "rgb_l1": rgb_l1.detach(),
                "semantic_ce": semantic_ce.detach(),
                "semantic_dice": semantic_dice.detach(),
                "boundary_loss": boundary_loss.detach(),
                "vq_loss": vq_loss.detach(),
                "total_loss": total_loss.detach(),
            },
            "metrics": {
                "pixel_accuracy": metrics["pixel_accuracy"].detach(),
                "miou": metrics["miou"].detach(),
            },
            "palette_image": palette_image,
            "reconstructed_image": reconstructed,
            "mask_index": mask_index,
            "pred_mask_index": torch.argmax(mask_logits, dim=1),
            "boundary_target": boundary_target.detach(),
            "boundary_pred": boundary_pred.detach(),
            "code_indices": codebook_indices.detach(),
        }

    def _log_scalars(self, split, scalars):
        return {f"{split}/{name}": value for name, value in scalars.items()}

    def shared_step(self, batch, split):
        outputs = self(batch)
        batch_size = int(outputs["palette_image"].shape[0])
        metrics = self._log_scalars(split, outputs["loss_dict"])
        metrics.update(self._log_scalars(split, outputs["metrics"]))
        self.log(
            f"{split}/loss",
            outputs["loss"].detach(),
            prog_bar=True,
            logger=True,
            on_step=(split == "train"),
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log_dict(
            metrics,
            prog_bar=False,
            logger=True,
            on_step=(split == "train"),
            on_epoch=True,
            batch_size=batch_size,
        )
        return outputs["loss"], outputs

    def training_step(self, batch, batch_idx):
        del batch_idx
        loss, _ = self.shared_step(batch, split="train")
        return loss

    def validation_step(self, batch, batch_idx):
        del batch_idx
        _, outputs = self.shared_step(batch, split="val")
        return outputs["loss"].detach()

    def log_images(self, batch, split="train", **kwargs):
        del split, kwargs
        outputs = self(batch)
        return {
            "input_image": outputs["palette_image"] * 2.0 - 1.0,
            "recon_image": outputs["reconstructed_image"] * 2.0 - 1.0,
            "target_mask_index": outputs["mask_index"].unsqueeze(1).to(dtype=torch.float32),
            "recon_mask_index": outputs["pred_mask_index"].unsqueeze(1).to(dtype=torch.float32),
        }

    def configure_optimizers(self):
        lr = float(getattr(self, "learning_rate", 1.0e-4))
        return torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            betas=self.optimizer_betas,
            weight_decay=self.weight_decay,
        )

    def on_fit_start(self):
        super().on_fit_start()
        if getattr(self.trainer, "is_global_zero", True):
            print(
                "[MaskGitPaletteVQTokenizer] "
                f"pretrained_weight_path={self.pretrained_weight_path}, "
                f"num_classes={self.num_classes}, ignore_index={self.ignore_index}, "
                f"rgb_l1_weight={self.rgb_l1_weight:.3f}, semantic_ce_weight={self.semantic_ce_weight:.3f}, "
                f"semantic_dice_weight={self.semantic_dice_weight:.3f}, boundary_loss_weight={self.boundary_loss_weight:.3f}, "
                f"vq_loss_weight={self.vq_loss_weight:.3f}, palette_logit_scale={self.palette_logit_scale:.3f}"
            )
