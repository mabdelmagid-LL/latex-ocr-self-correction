import cv2
import numpy as np
import torch
from typing import Sequence

from .trace import resize_token_map


CRITICAL_TOKEN_KEYS = ('\\frac', '\\sqrt', '^', '_', '\\left', '\\right', '{', '}', '(', ')', '[', ']')


def _connected_components(image_tensor: torch.Tensor, min_component_area: int = 8, component_padding: int = 0):
    img = image_tensor.detach().cpu()
    if img.ndim == 3:
        img = img[0]
    arr = img.numpy()
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()
    arr = (arr * 255).astype(np.uint8)

    _, bw = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if bw.mean() > 127:
        bw = 255 - bw
    n, labels, stats, _ = cv2.connectedComponentsWithStats((bw > 0).astype(np.uint8), connectivity=8)
    comps = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_component_area:
            continue
        mask = (labels == i)
        if component_padding > 0:
            kernel_size = 2 * int(component_padding) + 1
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        comps.append(mask)
    return comps


def _iou(mask_a, mask_b):
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def _dice(mask_a, mask_b):
    inter = np.logical_and(mask_a, mask_b).sum()
    denom = mask_a.sum() + mask_b.sum()
    if denom == 0:
        return 0.0
    return float((2.0 * inter) / denom)


def _overlap(mask_a, mask_b, mode: str = 'iou'):
    if mode == 'dice':
        return _dice(mask_a, mask_b)
    return _iou(mask_a, mask_b)


def attribution_consistency_score(
    image_tensor: torch.Tensor,
    token_ids,
    token_maps: torch.Tensor,
    tokenizer,
    patch_size: int = 16,
    top_percent: float = 0.15,
    min_component_area: int = 8,
    component_padding: int = 0,
    critical_token_keys: Sequence[str] | None = CRITICAL_TOKEN_KEYS,
    overlap_mode: str = 'iou',
):
    if token_maps is None or not torch.is_tensor(token_maps) or token_maps.numel() == 0:
        return 0.0

    top_percent = float(np.clip(top_percent, 0.01, 0.99))
    comps = _connected_components(
        image_tensor,
        min_component_area=min_component_area,
        component_padding=component_padding,
    )
    if len(comps) == 0:
        return 0.0

    if token_maps.ndim == 3:
        token_maps = token_maps[0]

    h, w = comps[0].shape
    scores = []
    max_tokens = min(len(token_ids), token_maps.shape[0])
    for idx in range(max_tokens):
        tok = tokenizer.convert_ids_to_tokens(int(token_ids[idx]))
        tok = '' if tok is None else tok.replace('Ġ', ' ').strip()
        if critical_token_keys and not any(k in tok for k in critical_token_keys):
            continue

        amap = resize_token_map(token_maps[idx], (h, w), patch_size=patch_size)
        a = amap.detach().cpu().numpy()
        thr = np.quantile(a, 1 - top_percent)
        amask = a >= thr

        best = 0.0
        for c in comps:
            best = max(best, _overlap(amask, c, mode=overlap_mode))
        scores.append(best)

    if len(scores) == 0:
        return 0.0
    return float(np.mean(scores))
