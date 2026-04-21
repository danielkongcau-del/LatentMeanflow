import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[3]
repo_root_str = str(REPO_ROOT)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

from latent_meanflow.losses.semantic_structure import (  # noqa: E402
    adjacency_l1_loss,
    area_ratio_l1_loss,
    boundary_bce_loss,
    build_valid_mask,
    mask_index_to_boundary_target,
    same_region_consistency_l1_loss,
    semantic_probs_to_soft_boundary,
)
from latent_meanflow.utils.palette import build_default_palette  # noqa: E402


class SemanticMaskAuxiliaryLoss(nn.Module):
    def __init__(
            self,
            *,
            num_classes: int,
            ignore_index: int = None,
            supervision_size: int = None,
            palette_logit_scale: float = 64.0,
            semantic_ce_weight: float = 1.0,
            semantic_dice_weight: float = 0.25,
            boundary_loss_weight: float = 0.10,
            area_ratio_loss_weight: float = 0.10,
            adjacency_loss_weight: float = 0.10,
            same_region_consistency_weight: float = 0.10,
            presence_loss_weight: float = 0.10,
            richness_loss_weight: float = 0.05,
            use_class_weights: bool = True,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.ignore_index = None if ignore_index is None else int(ignore_index)
        self.supervision_size = None if supervision_size is None else int(supervision_size)
        self.palette_logit_scale = float(palette_logit_scale)
        self.semantic_ce_weight = float(semantic_ce_weight)
        self.semantic_dice_weight = float(semantic_dice_weight)
        self.boundary_loss_weight = float(boundary_loss_weight)
        self.area_ratio_loss_weight = float(area_ratio_loss_weight)
        self.adjacency_loss_weight = float(adjacency_loss_weight)
        self.same_region_consistency_weight = float(same_region_consistency_weight)
        self.presence_loss_weight = float(presence_loss_weight)
        self.richness_loss_weight = float(richness_loss_weight)
        self.use_class_weights = bool(use_class_weights)

        palette_spec = build_default_palette(self.num_classes, ignore_index=self.ignore_index)
        palette = torch.from_numpy(palette_spec["palette"]).to(dtype=torch.float32) / 255.0
        self.register_buffer("palette_rgb", palette, persistent=False)

    @property
    def enabled(self):
        return any(
            weight > 0.0
            for weight in (
                self.semantic_ce_weight,
                self.semantic_dice_weight,
                self.boundary_loss_weight,
                self.area_ratio_loss_weight,
                self.adjacency_loss_weight,
                self.same_region_consistency_weight,
                self.presence_loss_weight,
                self.richness_loss_weight,
            )
        )

    def image_norm_to_unit(self, image):
        image = torch.as_tensor(image)
        if not image.is_floating_point():
            image = image.to(dtype=torch.float32)
        if image.ndim != 4 or int(image.shape[1]) != 3:
            raise ValueError(f"Expected image with shape [B, 3, H, W], got {tuple(image.shape)}")
        if float(image.min().detach().item()) < 0.0:
            image = (image + 1.0) / 2.0
        return image.clamp(0.0, 1.0)

    def image_to_mask_logits(self, image):
        image = self.image_norm_to_unit(image)
        palette = self.palette_rgb.to(device=image.device, dtype=image.dtype)
        distances = (image.unsqueeze(1) - palette.view(1, self.num_classes, 3, 1, 1)).pow(2).sum(dim=2)
        return -self.palette_logit_scale * distances

    def image_to_mask_index(self, image):
        logits = self.image_to_mask_logits(image)
        return torch.argmax(logits, dim=1)

    def prepare_supervision_inputs(self, pred_image, target_mask_index):
        pred_image = self.image_norm_to_unit(pred_image)
        target_mask_index = torch.as_tensor(target_mask_index, device=pred_image.device)
        if target_mask_index.ndim != 3:
            raise ValueError(
                f"Expected target_mask_index with shape [B, H, W], got {tuple(target_mask_index.shape)}"
            )

        if self.supervision_size is None:
            return pred_image, target_mask_index.long()

        current_size = int(pred_image.shape[-1])
        if current_size == self.supervision_size and int(pred_image.shape[-2]) == self.supervision_size:
            return pred_image, target_mask_index.long()

        pred_image = F.interpolate(
            pred_image,
            size=(self.supervision_size, self.supervision_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        target_mask_index = F.interpolate(
            target_mask_index.unsqueeze(1).to(dtype=torch.float32),
            size=(self.supervision_size, self.supervision_size),
            mode="nearest",
        ).squeeze(1).long()
        return pred_image, target_mask_index

    def mask_index_to_onehot(self, mask_index):
        mask_index = torch.as_tensor(mask_index)
        valid_mask = build_valid_mask(mask_index, ignore_index=self.ignore_index)
        safe_targets = mask_index.clone()
        safe_targets[~valid_mask] = 0
        onehot = F.one_hot(safe_targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        return onehot * valid_mask.unsqueeze(1).to(dtype=onehot.dtype)

    def _pixel_class_weights(self, safe_targets, valid_mask):
        if not self.use_class_weights:
            return torch.ones_like(safe_targets, dtype=torch.float32)
        valid_targets = safe_targets[valid_mask]
        if valid_targets.numel() == 0:
            return torch.ones_like(safe_targets, dtype=torch.float32)
        counts = torch.bincount(valid_targets.reshape(-1), minlength=self.num_classes).to(dtype=torch.float32)
        present = counts > 0
        weights = torch.ones_like(counts)
        weights[present] = counts[present].clamp_min(1.0).rsqrt()
        weights = weights / weights[present].mean().clamp_min(1.0e-8)
        return weights[safe_targets]

    def semantic_ce_loss(self, mask_logits, mask_index):
        valid_mask = build_valid_mask(mask_index, ignore_index=self.ignore_index)
        if not torch.any(valid_mask):
            return mask_logits.new_tensor(0.0)
        safe_targets = mask_index.clone()
        safe_targets[~valid_mask] = 0
        per_pixel = F.cross_entropy(mask_logits, safe_targets, reduction="none")
        weights = self._pixel_class_weights(safe_targets, valid_mask).to(device=mask_logits.device, dtype=per_pixel.dtype)
        valid_weight = valid_mask.to(dtype=per_pixel.dtype)
        return (per_pixel * weights * valid_weight).sum() / valid_weight.sum().clamp_min(1.0)

    def semantic_dice_loss(self, mask_logits, mask_index):
        valid_mask = build_valid_mask(mask_index, ignore_index=self.ignore_index)
        if not torch.any(valid_mask):
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

    def presence_losses(self, mask_probs, mask_onehot):
        pred_presence = mask_probs.amax(dim=(2, 3)).clamp(1.0e-6, 1.0 - 1.0e-6)
        target_presence = mask_onehot.amax(dim=(2, 3))
        presence_loss = F.binary_cross_entropy(pred_presence, target_presence)
        richness_loss = torch.abs(pred_presence.sum(dim=1) - target_presence.sum(dim=1)).mean()
        return {
            "presence_loss": presence_loss,
            "richness_loss": richness_loss,
            "pred_presence": pred_presence.detach(),
            "target_presence": target_presence.detach(),
        }

    def compute_mask_metrics(self, mask_logits, mask_index):
        valid_mask = build_valid_mask(mask_index, ignore_index=self.ignore_index)
        if not bool(torch.any(valid_mask)):
            zero = mask_logits.new_tensor(0.0)
            return {
                "semantic_pixel_accuracy": zero,
                "semantic_miou": zero,
                "pred_richness_mean": zero,
                "target_richness_mean": zero,
                "richness_gap": zero,
            }
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

        pred_presence = F.one_hot(pred.clamp_min(0), num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        pred_presence = pred_presence.amax(dim=(2, 3))
        target_onehot = self.mask_index_to_onehot(mask_index)
        target_presence = target_onehot.amax(dim=(2, 3))
        pred_richness = pred_presence.sum(dim=1)
        target_richness = target_presence.sum(dim=1)
        return {
            "semantic_pixel_accuracy": pixel_accuracy.detach(),
            "semantic_miou": miou.detach(),
            "pred_richness_mean": pred_richness.mean().detach(),
            "target_richness_mean": target_richness.mean().detach(),
            "richness_gap": torch.abs(pred_richness - target_richness).mean().detach(),
        }

    def forward(self, *, pred_image, target_mask_index):
        pred_image, target_mask_index = self.prepare_supervision_inputs(pred_image, target_mask_index)
        mask_logits = self.image_to_mask_logits(pred_image)
        mask_probs = torch.softmax(mask_logits, dim=1)
        mask_onehot = self.mask_index_to_onehot(target_mask_index)
        valid_mask = build_valid_mask(target_mask_index, ignore_index=self.ignore_index)

        semantic_ce = self.semantic_ce_loss(mask_logits, target_mask_index)
        semantic_dice = self.semantic_dice_loss(mask_logits, target_mask_index)
        boundary_target = mask_index_to_boundary_target(target_mask_index, ignore_index=self.ignore_index)
        boundary_pred = semantic_probs_to_soft_boundary(mask_probs, valid_mask=valid_mask)
        boundary_loss = boundary_bce_loss(boundary_pred, boundary_target, valid_mask=valid_mask)
        area_ratio_loss, pred_area_ratio, target_area_ratio = area_ratio_l1_loss(
            mask_probs,
            mask_onehot,
            valid_mask=valid_mask,
        )
        adjacency_loss, pred_adjacency, target_adjacency = adjacency_l1_loss(
            mask_probs,
            mask_onehot,
            valid_mask=valid_mask,
        )
        same_region_consistency_loss = same_region_consistency_l1_loss(
            mask_probs,
            target_mask_index,
            valid_mask=valid_mask,
        )
        presence_outputs = self.presence_losses(mask_probs, mask_onehot)

        semantic_aux_total = (
            self.semantic_ce_weight * semantic_ce
            + self.semantic_dice_weight * semantic_dice
            + self.boundary_loss_weight * boundary_loss
            + self.area_ratio_loss_weight * area_ratio_loss
            + self.adjacency_loss_weight * adjacency_loss
            + self.same_region_consistency_weight * same_region_consistency_loss
            + self.presence_loss_weight * presence_outputs["presence_loss"]
            + self.richness_loss_weight * presence_outputs["richness_loss"]
        )
        with torch.no_grad():
            metrics = self.compute_mask_metrics(mask_logits, target_mask_index)

        return {
            "semantic_aux_total": semantic_aux_total,
            "mask_logits": mask_logits,
            "mask_probs": mask_probs,
            "loss_dict": {
                "semantic_ce": semantic_ce.detach(),
                "semantic_dice": semantic_dice.detach(),
                "boundary_loss": boundary_loss.detach(),
                "area_ratio_loss": area_ratio_loss.detach(),
                "adjacency_loss": adjacency_loss.detach(),
                "same_region_consistency_loss": same_region_consistency_loss.detach(),
                "presence_loss": presence_outputs["presence_loss"].detach(),
                "richness_loss": presence_outputs["richness_loss"].detach(),
                "semantic_aux_total": semantic_aux_total.detach(),
            },
            "metrics": metrics,
        }
