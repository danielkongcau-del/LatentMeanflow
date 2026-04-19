import os

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback

from latent_meanflow.utils.palette import build_default_palette, colorize_mask_index


class SemanticPairImageLogger(Callback):
    def __init__(
        self,
        batch_frequency,
        max_images=8,
        clamp=True,
        increase_log_steps=True,
        rescale=True,
        disabled=False,
        log_on_batch_idx=False,
        log_first_step=False,
        log_images_kwargs=None,
        latest_only=False,
        ignore_index=None,
        log_train=True,
        log_validation=True,
    ):
        super().__init__()
        self.batch_freq = int(batch_frequency)
        self.max_images = int(max_images)
        self.clamp = bool(clamp)
        self.rescale = bool(rescale)
        self.disabled = bool(disabled)
        self.log_on_batch_idx = bool(log_on_batch_idx)
        self.log_first_step = bool(log_first_step)
        self.log_images_kwargs = dict(log_images_kwargs or {})
        self.latest_only = bool(latest_only)
        self.ignore_index = None if ignore_index is None else int(ignore_index)
        self.log_train = bool(log_train)
        self.log_validation = bool(log_validation)

        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]

    def _is_writer_rank(self, trainer):
        is_global_zero = getattr(trainer, "is_global_zero", None)
        if is_global_zero is not None:
            return bool(is_global_zero)
        global_rank = getattr(trainer, "global_rank", None)
        if global_rank is None:
            return True
        return int(global_rank) == 0

    def _check_frequency(self, check_idx):
        if check_idx == 0 and self.log_first_step:
            return True
        if check_idx in self.log_steps:
            return True
        return check_idx > 0 and self.batch_freq > 0 and (check_idx % self.batch_freq == 0)

    def _resolve_save_dir(self, trainer, pl_module):
        if getattr(pl_module, "logger", None) is not None and getattr(pl_module.logger, "save_dir", None):
            return pl_module.logger.save_dir
        if getattr(trainer, "log_dir", None):
            return trainer.log_dir
        if getattr(trainer, "default_root_dir", None):
            return trainer.default_root_dir
        return "."

    def _make_image_grid(self, tensor):
        tensor = tensor.detach().cpu()
        if self.clamp:
            tensor = torch.clamp(tensor, -1.0 if self.rescale else 0.0, 1.0)
        grid = torchvision.utils.make_grid(tensor, nrow=min(4, tensor.shape[0]))
        if self.rescale:
            grid = (grid + 1.0) / 2.0
        grid = torch.clamp(grid, 0.0, 1.0)
        grid = grid.permute(1, 2, 0).numpy()
        return (grid * 255.0).astype(np.uint8)

    def _make_mask_grid(self, tensor, num_classes):
        tensor = tensor.detach().cpu()
        if tensor.dim() == 4 and tensor.shape[1] == 1:
            tensor = tensor[:, 0]
        elif tensor.dim() != 3:
            raise ValueError(f"Expected mask tensor with shape [B, 1, H, W] or [B, H, W], got {tuple(tensor.shape)}")

        palette = build_default_palette(num_classes, ignore_index=self.ignore_index)
        colored = []
        for sample in tensor:
            mask_index = torch.round(sample).to(torch.int64).numpy()
            colored_mask = colorize_mask_index(
                mask_index,
                num_classes=num_classes,
                palette_spec=palette,
                ignore_index=self.ignore_index,
            )
            colored.append(torch.from_numpy(colored_mask).permute(2, 0, 1).float() / 255.0)

        colored = torch.stack(colored, dim=0)
        grid = torchvision.utils.make_grid(colored, nrow=min(4, colored.shape[0]))
        grid = torch.clamp(grid, 0.0, 1.0)
        grid = grid.permute(1, 2, 0).numpy()
        return (grid * 255.0).astype(np.uint8)

    def save_local(self, save_dir, split, images, global_step, current_epoch, batch_idx, num_classes):
        root = os.path.join(save_dir, "semantic_images", split)
        os.makedirs(root, exist_ok=True)

        for key, value in images.items():
            if not isinstance(value, torch.Tensor):
                continue
            if value.dim() not in {3, 4}:
                continue

            if value.dim() == 3:
                value = value.unsqueeze(1)

            if key.endswith("_image"):
                output = self._make_image_grid(value)
            elif key.endswith("_mask_index"):
                output = self._make_mask_grid(value, num_classes=num_classes)
            else:
                continue

            if self.latest_only:
                filename = f"{key}.png"
            else:
                filename = f"{key}_gs-{global_step:06}_e-{current_epoch:06}_b-{batch_idx:06}.png"
            Image.fromarray(output).save(os.path.join(root, filename))

    def log_img(self, trainer, pl_module, batch, batch_idx, split="train"):
        # In distributed training, only rank 0 should write local preview images.
        # This avoids duplicated I/O and last-writer-wins races on shared paths.
        if not self._is_writer_rank(trainer):
            return
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if self.disabled or not self._check_frequency(int(check_idx)) or self.max_images <= 0:
            return
        if not hasattr(pl_module, "log_images") or not callable(pl_module.log_images):
            return

        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        with torch.no_grad():
            images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

        if is_train:
            pl_module.train()

        trimmed = {}
        for key, value in images.items():
            if isinstance(value, torch.Tensor):
                trimmed[key] = value[: self.max_images]
            else:
                trimmed[key] = value

        num_classes = int(getattr(pl_module, "num_classes", 1))
        save_dir = self._resolve_save_dir(trainer, pl_module)
        self.save_local(
            save_dir=save_dir,
            split=split,
            images=trimmed,
            global_step=int(pl_module.global_step),
            current_epoch=int(pl_module.current_epoch),
            batch_idx=int(batch_idx),
            num_classes=num_classes,
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.log_train:
            return
        self.log_img(trainer, pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not self.log_validation:
            return
        self.log_img(trainer, pl_module, batch, batch_idx, split="val")
