import math
import torch
import torch.nn.functional as F


def normalize_map(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    xmin = x.min()
    xmax = x.max()
    return (x - xmin) / (xmax - xmin + eps)


def _infer_grid(hw_tokens: int):
    side = int(round(math.sqrt(hw_tokens)))
    if side * side == hw_tokens:
        return side, side
    for h in range(side, 0, -1):
        if hw_tokens % h == 0:
            return h, hw_tokens // h
    return 1, hw_tokens


def resize_token_map(token_map: torch.Tensor, image_hw, patch_size: int = 16, has_cls_token: bool = True) -> torch.Tensor:
    if token_map.ndim != 1:
        raise ValueError('token_map must be 1D')

    h, w = image_hw
    vec = token_map[1:] if has_cls_token and token_map.numel() > 1 else token_map

    gh = max(h // patch_size, 1)
    gw = max(w // patch_size, 1)
    if gh * gw != vec.numel():
        gh, gw = _infer_grid(vec.numel())

    grid = vec.reshape(1, 1, gh, gw)
    up = F.interpolate(grid, size=(h, w), mode='bilinear', align_corners=False)[0, 0]
    return normalize_map(up)


def attention_diffuseness(attn_map: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    vec = attn_map.reshape(-1)
    vec = vec.clamp_min(0)
    if vec.sum() <= 0:
        return torch.tensor(1.0, device=vec.device)
    p = vec / (vec.sum() + eps)
    entropy = -(p * (p + eps).log()).sum()
    max_entropy = math.log(float(vec.numel()) + eps)
    return (entropy / (max_entropy + eps)).clamp(0, 1)


def confidence_summary(confidences: torch.Tensor):
    if confidences is None or confidences.numel() == 0:
        return {'mean': 0.0, 'min': 0.0}
    return {
        'mean': float(confidences.mean().item()),
        'min': float(confidences.min().item()),
    }
