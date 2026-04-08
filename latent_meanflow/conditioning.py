from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ConditionSliceError(TypeError):
    message: str

    def __str__(self):
        return self.message


class LatentConditioning(dict):
    """Dictionary-like conditioning container that supports batch slicing.

    Objectives such as AlphaFlow slice `condition` with boolean masks during
    branch routing. Using a project-layer container keeps the math unchanged
    while making class and spatial conditions slice together.
    """

    def __init__(self, *, class_label=None, spatial=None, spatial_fullres=None, **extras):
        super().__init__()
        if class_label is not None:
            self["class_label"] = class_label
        if spatial is not None:
            self["spatial"] = spatial
        if spatial_fullres is not None:
            self["spatial_fullres"] = spatial_fullres
        for name, value in extras.items():
            if value is not None:
                self[name] = value

    def __getitem__(self, key):
        if isinstance(key, str):
            return super().__getitem__(key)
        sliced = LatentConditioning()
        for name, value in self.items():
            if value is None:
                continue
            if not isinstance(value, torch.Tensor):
                raise ConditionSliceError(
                    f"LatentConditioning only supports slicing tensor values, got key '{name}' "
                    f"with value type {type(value)}."
                )
            sliced[name] = value[key]
        return sliced

    def to(self, *, device=None, dtype=None):
        moved = LatentConditioning()
        for name, value in self.items():
            if not isinstance(value, torch.Tensor):
                moved[name] = value
                continue
            target_dtype = dtype
            if name == "class_label":
                target_dtype = torch.long
            moved[name] = value.to(device=device, dtype=target_dtype)
        return moved
