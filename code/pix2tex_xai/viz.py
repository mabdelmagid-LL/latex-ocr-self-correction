import json
import os
import re

import cv2
import numpy as np
import torch

from .trace import resize_token_map


CRITICAL_TOKEN_PATTERNS = [
    r'\\frac', r'\\sqrt', r'\^', r'_', r'\\left', r'\\right', r'\\overline', r'\\underline', r'[\{\}\(\)\[\]]'
]


def _sanitize(s: str):
    return re.sub(r'[^a-zA-Z0-9_\-]+', '_', s)[:40]


def _to_uint8_image(img_tensor: torch.Tensor):
    img = img_tensor.detach().cpu()
    if img.ndim == 3:
        if img.shape[0] in (1, 3):
            img = img[0]
        else:
            img = img.squeeze(0)
    if img.ndim != 2:
        img = img.squeeze()
    arr = img.numpy()
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()
    arr = (arr * 255).astype(np.uint8)
    return arr


def _is_critical_token(tok: str):
    return any(re.search(p, tok or '') is not None for p in CRITICAL_TOKEN_PATTERNS)


def _select_indices(tokens, max_tokens=8):
    critical = [i for i, tok in enumerate(tokens) if _is_critical_token(tok)]
    if len(critical) >= max_tokens:
        return critical[:max_tokens]
    rest = [i for i in range(len(tokens)) if i not in critical]
    return (critical + rest)[:max_tokens]


def save_attention_overlays(image_tensor: torch.Tensor, trace: dict, tokenizer, output_dir: str, patch_size: int = 16, max_tokens: int = 8):
    os.makedirs(output_dir, exist_ok=True)
    print(f'[XAI-VIZ] creating/using dir: {os.path.abspath(output_dir)}')

    tokens = trace.get('tokens', None)
    if tokens is None:
        print('[XAI-VIZ] no tokens in trace')
        return

    if tokens.ndim == 2:
        token_ids = tokens[0].detach().cpu().tolist()
    else:
        token_ids = tokens.detach().cpu().tolist()

    cross_attn = trace.get('cross_attentions', None)
    grad_attr = trace.get('grad_attributions', None)
    ig_attr = trace.get('ig_attributions', None)
    print(f'[XAI-VIZ] cross_attn={cross_attn is not None}, grad_attr={grad_attr is not None}, ig_attr={ig_attr is not None}')

    token_maps = None
    if torch.is_tensor(cross_attn) and cross_attn.numel() > 0:
        token_maps = cross_attn[0] if cross_attn.ndim == 3 else cross_attn

    grad_maps = None
    if torch.is_tensor(grad_attr) and grad_attr.numel() > 0:
        grad_maps = grad_attr[0] if grad_attr.ndim == 3 else grad_attr

    ig_maps = None
    if torch.is_tensor(ig_attr) and ig_attr.numel() > 0:
        ig_maps = ig_attr[0] if ig_attr.ndim == 3 else ig_attr

    if token_maps is None and grad_maps is None and ig_maps is None:
        print('[XAI-VIZ] no attention/grad maps - returning early')
        return
    print(f'[XAI-VIZ] token_maps shape: {token_maps.shape if token_maps is not None else None}')

    img_gray = _to_uint8_image(image_tensor)
    base_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    h, w = img_gray.shape

    decoded_tokens = []
    available_counts = [m.shape[0] for m in (token_maps, grad_maps, ig_maps) if m is not None]
    max_count = max(available_counts) if len(available_counts) > 0 else 0
    for tid in token_ids[:max_count]:
        tok = tokenizer.convert_ids_to_tokens(int(tid))
        tok = '' if tok is None else tok.replace('Ġ', ' ').strip()
        decoded_tokens.append(tok)

    selected = _select_indices(decoded_tokens, max_tokens=max_tokens)

    saved = []
    for idx in selected:
        tok = decoded_tokens[idx] if idx < len(decoded_tokens) else f'tok_{idx}'
        if token_maps is not None and idx < token_maps.shape[0]:
            try:
                attn_map = resize_token_map(token_maps[idx], (h, w), patch_size=patch_size)
                heat = (attn_map.detach().cpu().numpy() * 255).astype(np.uint8)
                heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(base_rgb, 0.55, heat, 0.45, 0)
                fname = f'attn_{idx:03d}_{_sanitize(tok)}.png'
                fpath = os.path.join(output_dir, fname)
                cv2.imwrite(fpath, overlay)
                saved.append({'index': idx, 'token': tok, 'type': 'attention', 'file': fname})
                print(f'[XAI-VIZ] saved {fname}')
            except Exception as e:
                print(f'[XAI-VIZ] error saving attn_{idx}: {e}')

        if grad_maps is not None and idx < grad_maps.shape[0]:
            try:
                gmap = resize_token_map(grad_maps[idx], (h, w), patch_size=patch_size)
                heat_g = (gmap.detach().cpu().numpy() * 255).astype(np.uint8)
                heat_g = cv2.applyColorMap(heat_g, cv2.COLORMAP_TURBO)
                overlay_g = cv2.addWeighted(base_rgb, 0.55, heat_g, 0.45, 0)
                fname_g = f'grad_{idx:03d}_{_sanitize(tok)}.png'
                fpath_g = os.path.join(output_dir, fname_g)
                cv2.imwrite(fpath_g, overlay_g)
                saved.append({'index': idx, 'token': tok, 'type': 'gradcam', 'file': fname_g})
                print(f'[XAI-VIZ] saved {fname_g}')
            except Exception as e:
                print(f'[XAI-VIZ] error saving grad_{idx}: {e}')

        if ig_maps is not None and idx < ig_maps.shape[0]:
            try:
                igmap = resize_token_map(ig_maps[idx], (h, w), patch_size=patch_size)
                heat_ig = (igmap.detach().cpu().numpy() * 255).astype(np.uint8)
                heat_ig = cv2.applyColorMap(heat_ig, cv2.COLORMAP_INFERNO)
                overlay_ig = cv2.addWeighted(base_rgb, 0.55, heat_ig, 0.45, 0)
                fname_ig = f'ig_{idx:03d}_{_sanitize(tok)}.png'
                fpath_ig = os.path.join(output_dir, fname_ig)
                cv2.imwrite(fpath_ig, overlay_ig)
                saved.append({'index': idx, 'token': tok, 'type': 'integrated_gradients', 'file': fname_ig})
                print(f'[XAI-VIZ] saved {fname_ig}')
            except Exception as e:
                print(f'[XAI-VIZ] error saving ig_{idx}: {e}')

    summary = {
        'tokens': decoded_tokens,
        'saved_maps': saved,
        'quality': trace.get('quality', {}),
    }
    with open(os.path.join(output_dir, 'xai_trace.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
