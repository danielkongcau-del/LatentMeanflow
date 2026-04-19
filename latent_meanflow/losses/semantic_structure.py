import torch
import torch.nn.functional as F


def build_valid_mask(mask_index, ignore_index=None):
    mask_index = torch.as_tensor(mask_index)
    if mask_index.ndim != 3:
        raise ValueError(f"Expected mask_index with shape [B, H, W], got {tuple(mask_index.shape)}")
    if mask_index.dtype == torch.bool and ignore_index is None:
        return mask_index
    valid_mask = torch.ones_like(mask_index, dtype=torch.bool)
    if ignore_index is not None:
        valid_mask &= mask_index != int(ignore_index)
    return valid_mask


def mask_index_to_boundary_target(mask_index, ignore_index=None):
    mask_index = torch.as_tensor(mask_index)
    if mask_index.ndim != 3:
        raise ValueError(f"Expected mask_index with shape [B, H, W], got {tuple(mask_index.shape)}")

    valid_mask = build_valid_mask(mask_index, ignore_index=ignore_index)
    boundary = torch.zeros(
        (int(mask_index.shape[0]), 1, int(mask_index.shape[1]), int(mask_index.shape[2])),
        device=mask_index.device,
        dtype=torch.float32,
    )

    if int(mask_index.shape[-1]) > 1:
        horizontal = valid_mask[:, :, :-1] & valid_mask[:, :, 1:] & (mask_index[:, :, :-1] != mask_index[:, :, 1:])
        horizontal = horizontal.unsqueeze(1).to(dtype=boundary.dtype)
        boundary[:, :, :, :-1] = torch.maximum(boundary[:, :, :, :-1], horizontal)
        boundary[:, :, :, 1:] = torch.maximum(boundary[:, :, :, 1:], horizontal)

    if int(mask_index.shape[-2]) > 1:
        vertical = valid_mask[:, :-1, :] & valid_mask[:, 1:, :] & (mask_index[:, :-1, :] != mask_index[:, 1:, :])
        vertical = vertical.unsqueeze(1).to(dtype=boundary.dtype)
        boundary[:, :, :-1, :] = torch.maximum(boundary[:, :, :-1, :], vertical)
        boundary[:, :, 1:, :] = torch.maximum(boundary[:, :, 1:, :], vertical)

    boundary = boundary * valid_mask.unsqueeze(1).to(dtype=boundary.dtype)
    return boundary


def semantic_probs_to_soft_boundary(mask_probs, valid_mask=None):
    mask_probs = torch.as_tensor(mask_probs)
    if mask_probs.ndim != 4:
        raise ValueError(
            f"Expected semantic probabilities with shape [B, K, H, W], got {tuple(mask_probs.shape)}"
        )
    if valid_mask is None:
        valid_mask = torch.ones(
            (int(mask_probs.shape[0]), int(mask_probs.shape[2]), int(mask_probs.shape[3])),
            device=mask_probs.device,
            dtype=torch.bool,
        )
    else:
        valid_mask = build_valid_mask(valid_mask, ignore_index=None)

    boundary = torch.zeros(
        (int(mask_probs.shape[0]), 1, int(mask_probs.shape[2]), int(mask_probs.shape[3])),
        device=mask_probs.device,
        dtype=mask_probs.dtype,
    )

    if int(mask_probs.shape[-1]) > 1:
        horizontal_valid = (valid_mask[:, :, :-1] & valid_mask[:, :, 1:]).unsqueeze(1).to(dtype=mask_probs.dtype)
        same_prob = (mask_probs[:, :, :, :-1] * mask_probs[:, :, :, 1:]).sum(dim=1, keepdim=True)
        edge_prob = (1.0 - same_prob).clamp(min=0.0, max=1.0) * horizontal_valid
        boundary = 1.0 - (1.0 - boundary) * (1.0 - F.pad(edge_prob, (0, 1, 0, 0)))
        boundary = 1.0 - (1.0 - boundary) * (1.0 - F.pad(edge_prob, (1, 0, 0, 0)))

    if int(mask_probs.shape[-2]) > 1:
        vertical_valid = (valid_mask[:, :-1, :] & valid_mask[:, 1:, :]).unsqueeze(1).to(dtype=mask_probs.dtype)
        same_prob = (mask_probs[:, :, :-1, :] * mask_probs[:, :, 1:, :]).sum(dim=1, keepdim=True)
        edge_prob = (1.0 - same_prob).clamp(min=0.0, max=1.0) * vertical_valid
        boundary = 1.0 - (1.0 - boundary) * (1.0 - F.pad(edge_prob, (0, 0, 0, 1)))
        boundary = 1.0 - (1.0 - boundary) * (1.0 - F.pad(edge_prob, (0, 0, 1, 0)))

    return boundary * valid_mask.unsqueeze(1).to(dtype=mask_probs.dtype)


def boundary_bce_loss(pred_boundary, target_boundary, valid_mask=None):
    pred_boundary = torch.as_tensor(pred_boundary)
    target_boundary = torch.as_tensor(target_boundary, device=pred_boundary.device, dtype=pred_boundary.dtype)
    if pred_boundary.shape != target_boundary.shape:
        raise ValueError(
            f"Boundary prediction/target shape mismatch: {tuple(pred_boundary.shape)} vs {tuple(target_boundary.shape)}"
        )
    if pred_boundary.ndim != 4 or int(pred_boundary.shape[1]) != 1:
        raise ValueError(
            f"Expected boundary maps with shape [B, 1, H, W], got {tuple(pred_boundary.shape)}"
        )

    if valid_mask is None:
        weight = torch.ones_like(pred_boundary, dtype=pred_boundary.dtype)
    else:
        weight = build_valid_mask(valid_mask, ignore_index=None).unsqueeze(1).to(
            device=pred_boundary.device,
            dtype=pred_boundary.dtype,
        )

    loss = F.binary_cross_entropy(
        pred_boundary.clamp(min=1.0e-6, max=1.0 - 1.0e-6),
        target_boundary.clamp(min=0.0, max=1.0),
        reduction="none",
    )
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)


def compute_class_area_ratios(mask_distribution, valid_mask=None):
    mask_distribution = torch.as_tensor(mask_distribution)
    if mask_distribution.ndim != 4:
        raise ValueError(
            f"Expected mask distribution with shape [B, K, H, W], got {tuple(mask_distribution.shape)}"
        )

    if valid_mask is None:
        weight = torch.ones(
            (int(mask_distribution.shape[0]), 1, int(mask_distribution.shape[2]), int(mask_distribution.shape[3])),
            device=mask_distribution.device,
            dtype=mask_distribution.dtype,
        )
    else:
        weight = build_valid_mask(valid_mask, ignore_index=None).unsqueeze(1).to(
            device=mask_distribution.device,
            dtype=mask_distribution.dtype,
        )

    weighted_distribution = mask_distribution * weight
    area = weighted_distribution.sum(dim=(2, 3))
    denom = weight.sum(dim=(2, 3)).clamp_min(1.0)
    ratios = area / denom
    ratio_mass = ratios.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
    return ratios / ratio_mass


def area_ratio_l1_loss(pred_distribution, target_distribution, valid_mask=None):
    pred_ratios = compute_class_area_ratios(pred_distribution, valid_mask=valid_mask)
    target_ratios = compute_class_area_ratios(target_distribution, valid_mask=valid_mask)
    loss = torch.abs(pred_ratios - target_ratios).mean()
    return loss, pred_ratios, target_ratios


def compute_class_adjacency_matrix(mask_distribution, valid_mask=None):
    mask_distribution = torch.as_tensor(mask_distribution)
    if mask_distribution.ndim != 4:
        raise ValueError(
            f"Expected mask distribution with shape [B, K, H, W], got {tuple(mask_distribution.shape)}"
        )

    batch_size = int(mask_distribution.shape[0])
    num_classes = int(mask_distribution.shape[1])
    height = int(mask_distribution.shape[2])
    width = int(mask_distribution.shape[3])

    if valid_mask is None:
        valid_mask = torch.ones((batch_size, height, width), device=mask_distribution.device, dtype=torch.bool)
    else:
        valid_mask = build_valid_mask(valid_mask, ignore_index=None)

    adjacency = torch.zeros(
        (batch_size, num_classes, num_classes),
        device=mask_distribution.device,
        dtype=mask_distribution.dtype,
    )

    if width > 1:
        horizontal_valid = (valid_mask[:, :, :-1] & valid_mask[:, :, 1:]).unsqueeze(1).to(
            dtype=mask_distribution.dtype
        )
        left = mask_distribution[:, :, :, :-1] * horizontal_valid
        right = mask_distribution[:, :, :, 1:]
        adjacency = adjacency + torch.einsum("bkhw,blhw->bkl", left, right)

    if height > 1:
        vertical_valid = (valid_mask[:, :-1, :] & valid_mask[:, 1:, :]).unsqueeze(1).to(
            dtype=mask_distribution.dtype
        )
        top = mask_distribution[:, :, :-1, :] * vertical_valid
        bottom = mask_distribution[:, :, 1:, :]
        adjacency = adjacency + torch.einsum("bkhw,blhw->bkl", top, bottom)

    adjacency = 0.5 * (adjacency + adjacency.transpose(1, 2))
    off_diagonal_mask = 1.0 - torch.eye(num_classes, device=adjacency.device, dtype=adjacency.dtype).unsqueeze(0)
    adjacency = adjacency * off_diagonal_mask
    denom = adjacency.sum(dim=(1, 2), keepdim=True).clamp_min(1.0e-8)
    return adjacency / denom


def adjacency_l1_loss(pred_distribution, target_distribution, valid_mask=None):
    pred_adjacency = compute_class_adjacency_matrix(pred_distribution, valid_mask=valid_mask)
    target_adjacency = compute_class_adjacency_matrix(target_distribution, valid_mask=valid_mask)
    loss = torch.abs(pred_adjacency - target_adjacency).mean()
    return loss, pred_adjacency, target_adjacency
