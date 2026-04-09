import json
from pathlib import Path

import torch
import torch.nn as nn


LATENT_NORMALIZATION_MODES = {"none", "global_affine", "per_channel_affine"}


def _sorted_channel_items(per_channel_stats):
    def _channel_index(name):
        try:
            return int(str(name).split("_")[-1])
        except (TypeError, ValueError):
            return str(name)

    return sorted(dict(per_channel_stats or {}).items(), key=lambda item: _channel_index(item[0]))


def build_latent_stats_payload_from_summary(summary):
    channel_items = _sorted_channel_items(summary.get("per_channel_stats", {}))
    per_channel_mean = [float((stats or {}).get("mean", 0.0) or 0.0) for _, stats in channel_items]
    per_channel_std = [float((stats or {}).get("std", 0.0) or 0.0) for _, stats in channel_items]

    return {
        "format_version": 1,
        "source": {
            "name": summary.get("name"),
            "config": summary.get("config"),
            "checkpoint": summary.get("checkpoint"),
            "split": summary.get("split"),
        },
        "latent_shape": list(summary.get("latent_shape", [])),
        "latent_spatial_shape": list(summary.get("latent_spatial_shape", [])),
        "stats": {
            "global": {
                "mean": float(summary.get("latent_mean", 0.0) or 0.0),
                "std": float(summary.get("latent_std", 0.0) or 0.0),
            },
            "per_channel": {
                "mean": per_channel_mean,
                "std": per_channel_std,
            },
        },
        "channel_collapse": summary.get("channel_collapse"),
        "config_metadata": summary.get("config_metadata"),
    }


def write_latent_stats_json(path, summary):
    path = Path(path)
    payload = build_latent_stats_payload_from_summary(summary)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _resolve_summary_from_payload(payload, *, summary_name=None):
    if "format_version" in payload and "stats" in payload:
        return payload

    candidate_collections = [
        payload.get("summaries"),
        payload.get("ranking"),
        payload.get("measured_candidates"),
    ]
    for collection in candidate_collections:
        if not isinstance(collection, list) or not collection:
            continue
        if summary_name is None:
            selected = collection[0]
        else:
            selected = next((item for item in collection if item.get("name") == summary_name), None)
            if selected is None:
                continue
        return build_latent_stats_payload_from_summary(selected)

    if "per_channel_stats" in payload and "latent_mean" in payload:
        return build_latent_stats_payload_from_summary(payload)

    raise ValueError("Could not resolve latent stats from the provided payload.")


def load_latent_stats_payload(stats_path, *, summary_name=None):
    stats_path = Path(stats_path).resolve()
    if not stats_path.exists():
        raise FileNotFoundError(f"Latent stats file not found: {stats_path}")
    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    resolved = _resolve_summary_from_payload(payload, summary_name=summary_name)
    resolved["resolved_from"] = str(stats_path)
    resolved["requested_summary_name"] = summary_name
    return resolved


class LatentAffineNormalizer(nn.Module):
    def __init__(
        self,
        *,
        mode="none",
        latent_channels,
        stats_path=None,
        summary_name=None,
        global_mean=None,
        global_std=None,
        per_channel_mean=None,
        per_channel_std=None,
        std_floor=1.0e-6,
    ):
        super().__init__()
        self.mode = str(mode)
        if self.mode not in LATENT_NORMALIZATION_MODES:
            raise ValueError(
                f"Unsupported latent normalization mode={self.mode!r}. "
                f"Expected one of {sorted(LATENT_NORMALIZATION_MODES)}."
            )

        self.latent_channels = int(latent_channels)
        self.stats_path = None if stats_path is None else str(Path(stats_path).resolve())
        self.summary_name = None if summary_name is None else str(summary_name)
        self.std_floor = float(std_floor)
        if self.std_floor <= 0.0:
            raise ValueError(f"std_floor must be positive, got {self.std_floor}")

        resolved_payload = None
        if self.mode != "none" and stats_path is not None:
            resolved_payload = load_latent_stats_payload(stats_path, summary_name=summary_name)
            stats = resolved_payload["stats"]
            if global_mean is None:
                global_mean = stats["global"]["mean"]
            if global_std is None:
                global_std = stats["global"]["std"]
            if per_channel_mean is None:
                per_channel_mean = stats["per_channel"]["mean"]
            if per_channel_std is None:
                per_channel_std = stats["per_channel"]["std"]

        if self.mode == "none":
            mean_tensor = torch.zeros(1, self.latent_channels, 1, 1, dtype=torch.float32)
            std_tensor = torch.ones(1, self.latent_channels, 1, 1, dtype=torch.float32)
            raw_std_tensor = std_tensor.clone()
        elif self.mode == "global_affine":
            mean_value = 0.0 if global_mean is None else float(global_mean)
            std_value = 1.0 if global_std is None else float(global_std)
            raw_std_tensor = torch.full((1, self.latent_channels, 1, 1), std_value, dtype=torch.float32)
            mean_tensor = torch.full((1, self.latent_channels, 1, 1), mean_value, dtype=torch.float32)
            std_tensor = raw_std_tensor.clamp(min=self.std_floor)
        else:
            if per_channel_mean is None or per_channel_std is None:
                raise ValueError(
                    "per_channel_affine mode requires per_channel_mean and per_channel_std "
                    "or a stats_path that provides them."
                )
            if len(per_channel_mean) != self.latent_channels or len(per_channel_std) != self.latent_channels:
                raise ValueError(
                    "Per-channel latent stats shape mismatch: "
                    f"expected {self.latent_channels} channels, got "
                    f"{len(per_channel_mean)} means and {len(per_channel_std)} stds."
                )
            mean_tensor = torch.tensor(per_channel_mean, dtype=torch.float32).view(1, self.latent_channels, 1, 1)
            raw_std_tensor = torch.tensor(per_channel_std, dtype=torch.float32).view(
                1, self.latent_channels, 1, 1
            )
            std_tensor = raw_std_tensor.clamp(min=self.std_floor)

        self.register_buffer("mean", mean_tensor, persistent=False)
        self.register_buffer("std", std_tensor, persistent=False)
        self.register_buffer("raw_std", raw_std_tensor, persistent=False)

        clamped_channels = int((self.raw_std < self.std_floor).sum().item())
        self.metadata = {
            "mode": self.mode,
            "enabled": bool(self.mode != "none"),
            "stats_path": self.stats_path,
            "summary_name": self.summary_name,
            "std_floor": self.std_floor,
            "clamped_channel_count": clamped_channels,
            "effective_std_min": float(self.std.min().item()),
            "effective_std_max": float(self.std.max().item()),
            "raw_std_min": float(self.raw_std.min().item()),
            "raw_std_max": float(self.raw_std.max().item()),
            "resolved_source": None if resolved_payload is None else resolved_payload.get("resolved_from"),
        }

    @property
    def enabled(self):
        return bool(self.mode != "none")

    def describe(self):
        return dict(self.metadata)

    def normalize(self, z):
        if not self.enabled:
            return z
        return (z - self.mean.to(device=z.device, dtype=z.dtype)) / self.std.to(device=z.device, dtype=z.dtype)

    def denormalize(self, z):
        if not self.enabled:
            return z
        return z * self.std.to(device=z.device, dtype=z.dtype) + self.mean.to(device=z.device, dtype=z.dtype)


def build_latent_normalizer(config, *, latent_channels):
    config = dict(config or {})
    mode = str(config.get("mode", "none"))
    if mode == "none":
        return LatentAffineNormalizer(mode="none", latent_channels=latent_channels)
    return LatentAffineNormalizer(
        mode=mode,
        latent_channels=latent_channels,
        stats_path=config.get("stats_path"),
        summary_name=config.get("summary_name"),
        global_mean=config.get("global_mean"),
        global_std=config.get("global_std"),
        per_channel_mean=config.get("per_channel_mean"),
        per_channel_std=config.get("per_channel_std"),
        std_floor=config.get("std_floor", 1.0e-6),
    )
