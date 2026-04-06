import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

UNDEFINED_CLASS_ID = -10_000


def _to_plain_container(value):
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    return OmegaConf.to_container(value, resolve=True)


def _load_spec_from_path(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Label spec file not found: {path}")
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def resolve_gray_to_class_id(gray_to_class_id, ignore_index=None):
    if gray_to_class_id is None:
        raise ValueError("gray_to_class_id must be provided")

    spec = gray_to_class_id
    if isinstance(spec, (str, Path)):
        spec = _load_spec_from_path(spec)
    spec = _to_plain_container(spec)

    if not isinstance(spec, dict):
        raise TypeError("gray_to_class_id must be a mapping or a path to a JSON/YAML mapping file")

    if "gray_to_class_id" in spec:
        resolved_ignore_index = ignore_index if ignore_index is not None else spec.get("ignore_index")
        spec = spec["gray_to_class_id"]
    else:
        resolved_ignore_index = ignore_index

    if not isinstance(spec, dict) or not spec:
        raise ValueError("gray_to_class_id must contain at least one gray-to-class mapping")

    resolved = {}
    for raw_gray, class_id in spec.items():
        gray_value = int(raw_gray)
        class_index = int(class_id)
        if not 0 <= gray_value <= 255:
            raise ValueError(f"Gray value must be in [0, 255], got {gray_value}")
        resolved[gray_value] = class_index

    if resolved_ignore_index is not None:
        resolved_ignore_index = int(resolved_ignore_index)

    valid_class_ids = sorted(
        class_id for class_id in set(resolved.values()) if class_id != resolved_ignore_index
    )
    if not valid_class_ids:
        raise ValueError("gray_to_class_id must define at least one non-ignored class")
    if min(valid_class_ids) < 0:
        raise ValueError("Non-ignored class ids must be non-negative")

    return resolved, resolved_ignore_index


def build_lookup_table(gray_to_class_id, undefined_value=UNDEFINED_CLASS_ID):
    lookup = np.full(256, undefined_value, dtype=np.int64)
    for gray_value, class_id in gray_to_class_id.items():
        lookup[int(gray_value)] = int(class_id)
    return lookup


def infer_num_classes(gray_to_class_id, ignore_index=None):
    valid_class_ids = [class_id for class_id in gray_to_class_id.values() if class_id != ignore_index]
    if not valid_class_ids:
        raise ValueError("Could not infer num_classes from gray_to_class_id")
    return max(valid_class_ids) + 1


def build_gray_value_stats(gray_value_counts):
    total_pixels = int(sum(int(count) for count in gray_value_counts.values()))
    stats = []
    for gray_value in sorted(int(value) for value in gray_value_counts.keys()):
        pixel_count = int(gray_value_counts[gray_value])
        pixel_ratio = 0.0 if total_pixels == 0 else pixel_count / total_pixels
        stats.append(
            {
                "gray_value": int(gray_value),
                "pixel_count": pixel_count,
                "pixel_ratio": float(pixel_ratio),
            }
        )
    return stats, total_pixels


def build_mapping_template(gray_values, ignore_index=None):
    ordered_values = [int(value) for value in sorted(set(gray_values))]
    template = {
        "gray_to_class_id": {int(gray_value): idx for idx, gray_value in enumerate(ordered_values)},
        "ignore_index": None if ignore_index is None else int(ignore_index),
    }
    return template


def render_mapping_template(template, output_format="yaml"):
    output_format = output_format.lower()
    if output_format == "json":
        return json.dumps(template, indent=2, sort_keys=True)
    if output_format not in {"yaml", "yml"}:
        raise ValueError(f"Unsupported output format: {output_format}")
    return OmegaConf.to_yaml(OmegaConf.create(template), resolve=True)
