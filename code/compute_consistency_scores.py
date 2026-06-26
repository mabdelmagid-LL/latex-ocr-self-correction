from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from munch import Munch

from pix2tex.cli import LatexOCR, minmax_size
from pix2tex.utils import pad
from pix2tex.dataset.transforms import test_transform
from pix2tex.xai.consistency import attribution_consistency_score, CRITICAL_TOKEN_KEYS
from pix2tex.xai.gradcam import add_gradcam_to_trace
from pix2tex.xai.integrated_gradients import add_integrated_gradients_to_trace
from pix2tex.xai.trace import attention_diffuseness

IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'}


def _list_images(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def _load_processed(path: Path, gradcam: bool, ig: bool) -> dict[str, dict[str, str]]:
    """Map filename -> stored output columns for resume mode."""
    if not path.exists():
        return {}
    out: dict[str, dict[str, str]] = {}
    with path.open('r', newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            name = (row.get('image') or '').strip()
            if not name:
                continue
            a = (row.get('attn_consistency') or row.get('consistency') or '').strip()
            g = (row.get('gradcam_consistency') or '').strip()
            i = (row.get('ig_consistency') or '').strip()
            d = (row.get('diffuseness') or '').strip()
            out[name] = {'consistency': a, 'gradcam': g, 'ig': i, 'diffuseness': d}
    return out


def _should_skip(processed: dict, name: str, gradcam: bool, ig: bool) -> bool:
    if name not in processed:
        return False
    row = processed[name]
    a, g, i, d = row.get('consistency', ''), row.get('gradcam', ''), row.get('ig', ''), row.get('diffuseness', '')
    if not a:
        return False
    if gradcam and not g:
        return False
    if ig and not i:
        return False
    if not d:
        return False
    return True


def _mean_attention_diffuseness(attn: torch.Tensor | None) -> float:
    """Match cli.py: average entropy-based diffuseness over cross-attention steps."""
    if not torch.is_tensor(attn) or attn.numel() == 0:
        return 1.0
    diffs: list[float] = []
    steps = attn.shape[1] if attn.ndim >= 3 else 0
    for t in range(steps):
        diffs.append(attention_diffuseness(attn[0, t]).item())
    return float(sum(diffs) / max(len(diffs), 1))


def main() -> int:
    parser = argparse.ArgumentParser(description='Compute attribution-consistency scores for images.')
    parser.add_argument('--data-dir', type=Path, default=Path('data/formulae_extracted_full/test'), help='Folder of images')
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output CSV (default: xai_outputs/test_consistency_scores.csv or _with_gradcam.csv)',
    )
    parser.add_argument('--gradcam', action='store_true', help='Also compute Grad-CAM maps and grad consistency')
    parser.add_argument('--xai-max-tokens', type=int, default=12, help='Max tokens for Grad-CAM (when --gradcam)')
    parser.add_argument('--ig', action='store_true', help='Also compute Integrated Gradients maps and IG consistency')
    parser.add_argument('--ig-steps', type=int, default=16, help='Integration steps for Integrated Gradients')
    parser.add_argument('--top-percent', type=float, default=0.15, help='Top fraction of attribution kept per token')
    parser.add_argument('--min-component-area', type=int, default=8, help='Ignore connected components smaller than this')
    parser.add_argument('--component-padding', type=int, default=0, help='Dilate connected components by this many pixels')
    parser.add_argument('--all-tokens', action='store_true', help='Score all tokens instead of only critical formula tokens')
    parser.add_argument('--overlap-mode', choices=['iou', 'dice'], default='iou', help='Overlap metric used for scoring')
    parser.add_argument('--no-resume', action='store_true', help='Ignore existing CSV and recompute all')
    parser.add_argument('--no-cuda', action='store_true', help='Force CPU')
    parser.add_argument(
        '--resize',
        action='store_true',
        help='Use learned image resizer if available (default: off, matches batch scripts)',
    )
    parser.add_argument('--limit', type=int, default=None, help='Process only first N images (debug)')
    parser.add_argument('--temperature', type=float, default=0.333)
    parser.add_argument('--progress-every', type=int, default=10, help='Print progress every N images')
    parser.add_argument('--pix2tex-model-dir', type=str, default='',
                        help='Path to pix2tex model dir (contains checkpoints/ and settings/). '
                             'Defaults to the pix2tex package install. '
                             'Use models/pix2tex_baseline if pix2tex is not installed.')
    args = parser.parse_args()

    if args.pix2tex_model_dir:
        import contextlib, os, pix2tex.cli as _cli, pix2tex.utils as _utils
        _dir = str(Path(args.pix2tex_model_dir).resolve())
        @contextlib.contextmanager
        def _patched():
            saved = os.getcwd(); os.chdir(_dir)
            try: yield
            finally: os.chdir(saved)
        _utils.in_model_path = _patched
        _cli.in_model_path = _patched

    data_dir = args.data_dir
    if not data_dir.is_dir():
        print(f'Not a directory: {data_dir}', file=sys.stderr)
        return 1

    out_csv = args.output
    if out_csv is None:
        if args.gradcam and args.ig:
            out_csv = Path('xai_outputs/test_consistency_scores_with_gradcam_ig.csv')
        elif args.gradcam:
            out_csv = Path('xai_outputs/test_consistency_scores_with_gradcam.csv')
        elif args.ig:
            out_csv = Path('xai_outputs/test_consistency_scores_with_ig.csv')
        else:
            out_csv = Path('xai_outputs/test_consistency_scores.csv')
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    files = _list_images(data_dir)
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        print(f'No images in {data_dir}', file=sys.stderr)
        return 1

    processed: dict[str, dict[str, str]] = {}
    if not args.no_resume:
        processed = _load_processed(out_csv, args.gradcam, args.ig)

    args_ocr = Munch({
        'config': 'settings/config.yaml',
        'checkpoint': 'checkpoints/weights.pth',
        'no_cuda': args.no_cuda,
        'no_resize': not args.resize,
        'temperature': args.temperature,
        'explain': True,
        'gradcam': False,
    })
    ocr = LatexOCR(args_ocr)

    attn_scores: list[float] = []
    grad_scores: list[float] = []
    ig_scores: list[float] = []
    diff_scores: list[float] = []
    for v in processed.values():
        a, g, i, d = v.get('consistency', ''), v.get('gradcam', ''), v.get('ig', ''), v.get('diffuseness', '')
        if a:
            try:
                attn_scores.append(float(a))
            except ValueError:
                pass
        if args.gradcam and g:
            try:
                grad_scores.append(float(g))
            except ValueError:
                pass
        if args.ig and i:
            try:
                ig_scores.append(float(i))
            except ValueError:
                pass
        if d:
            try:
                diff_scores.append(float(d))
            except ValueError:
                pass

    total = len(files)
    with out_csv.open('w', newline='') as f:
        w = csv.writer(f)
        header = ['image', 'attn_consistency']
        if args.gradcam:
            header.append('gradcam_consistency')
        if args.ig:
            header.append('ig_consistency')
        header.append('diffuseness')
        w.writerow(header)

        done = 0
        for fp in files:
            if not args.no_resume and _should_skip(processed, fp.name, args.gradcam, args.ig):
                row = processed[fp.name]
                out_row = [fp.name, row['consistency']]
                if args.gradcam:
                    out_row.append(row['gradcam'])
                if args.ig:
                    out_row.append(row['ig'])
                out_row.append(row['diffuseness'])
                w.writerow(out_row)
                done += 1
                if done % args.progress_every == 0:
                    f.flush()
                    am = float(np.mean(attn_scores)) if attn_scores else float('nan')
                dm = float(np.mean(diff_scores)) if diff_scores else float('nan')
                if args.gradcam or args.ig:
                    metrics = [f'attn_mean={am:.6f}']
                    gm = float(np.mean(grad_scores)) if grad_scores else float('nan')
                    im = float(np.mean(ig_scores)) if ig_scores else float('nan')
                    if args.gradcam:
                        metrics.append(f'grad_mean={gm:.6f}')
                    if args.ig:
                        metrics.append(f'ig_mean={im:.6f}')
                    metrics.append(f'diff_mean={dm:.6f}')
                    print(f"processed {done}/{total} (resume) {' '.join(metrics)}", flush=True)
                else:
                    print(f'processed {done}/{total} (resume) cons_mean={am:.6f} diff_mean={dm:.6f}', flush=True)
                continue

            try:
                img = Image.open(fp)
                img = minmax_size(pad(img), ocr.args.max_dimensions, ocr.args.min_dimensions)
                arr = np.array(pad(img).convert('RGB'))
                t = test_transform(image=arr)['image'][:1].unsqueeze(0)
                im = t.to(ocr.args.device)

                with torch.no_grad():
                    trace = ocr.model.generate_with_trace(im, temperature=ocr.args.get('temperature', 0.333))

                if args.gradcam:
                    trace = add_gradcam_to_trace(ocr.model, im, trace, max_tokens=args.xai_max_tokens)
                if args.ig:
                    trace = add_integrated_gradients_to_trace(
                        ocr.model,
                        im,
                        trace,
                        max_tokens=args.xai_max_tokens,
                        steps=args.ig_steps,
                    )

                tokens = trace['tokens']
                token_ids = tokens[0].detach().cpu().tolist() if tokens.ndim == 2 else tokens.detach().cpu().tolist()

                attn_cons = attribution_consistency_score(
                    image_tensor=im[0].detach().cpu(),
                    token_ids=token_ids,
                    token_maps=trace.get('cross_attentions', None),
                    tokenizer=ocr.tokenizer,
                    patch_size=ocr.args.get('patch_size', 16),
                    top_percent=args.top_percent,
                    min_component_area=args.min_component_area,
                    component_padding=args.component_padding,
                    critical_token_keys=() if args.all_tokens else CRITICAL_TOKEN_KEYS,
                    overlap_mode=args.overlap_mode,
                )
                attn_scores.append(attn_cons)
                diff = _mean_attention_diffuseness(trace.get('cross_attentions', None))
                diff_scores.append(diff)

                grad_cons = None
                ig_cons = None

                if args.gradcam:
                    grad_cons = attribution_consistency_score(
                        image_tensor=im[0].detach().cpu(),
                        token_ids=token_ids,
                        token_maps=trace.get('grad_attributions', None),
                        tokenizer=ocr.tokenizer,
                        patch_size=ocr.args.get('patch_size', 16),
                        top_percent=args.top_percent,
                        min_component_area=args.min_component_area,
                        component_padding=args.component_padding,
                        critical_token_keys=() if args.all_tokens else CRITICAL_TOKEN_KEYS,
                        overlap_mode=args.overlap_mode,
                    )
                    grad_scores.append(grad_cons)

                if args.ig:
                    ig_cons = attribution_consistency_score(
                        image_tensor=im[0].detach().cpu(),
                        token_ids=token_ids,
                        token_maps=trace.get('ig_attributions', None),
                        tokenizer=ocr.tokenizer,
                        patch_size=ocr.args.get('patch_size', 16),
                        top_percent=args.top_percent,
                        min_component_area=args.min_component_area,
                        component_padding=args.component_padding,
                        critical_token_keys=() if args.all_tokens else CRITICAL_TOKEN_KEYS,
                        overlap_mode=args.overlap_mode,
                    )
                    ig_scores.append(ig_cons)

                out_row = [fp.name, f'{attn_cons:.10f}']
                if args.gradcam:
                    out_row.append(f'{grad_cons:.10f}' if grad_cons is not None else '')
                if args.ig:
                    out_row.append(f'{ig_cons:.10f}' if ig_cons is not None else '')
                out_row.append(f'{diff:.10f}')
                w.writerow(out_row)
            except Exception as e:
                out_row = [fp.name, '']
                if args.gradcam:
                    out_row.append('')
                if args.ig:
                    out_row.append('')
                out_row.append('')
                w.writerow(out_row)
                print(f'ERROR {fp.name}: {type(e).__name__}: {e}', flush=True)

            done += 1
            if done % args.progress_every == 0:
                f.flush()
                am = float(np.mean(attn_scores)) if attn_scores else float('nan')
                dm = float(np.mean(diff_scores)) if diff_scores else float('nan')
                if args.gradcam or args.ig:
                    metrics = [f'attn_mean={am:.6f}']
                    gm = float(np.mean(grad_scores)) if grad_scores else float('nan')
                    im = float(np.mean(ig_scores)) if ig_scores else float('nan')
                    if args.gradcam:
                        metrics.append(f'grad_mean={gm:.6f}')
                    if args.ig:
                        metrics.append(f'ig_mean={im:.6f}')
                    metrics.append(f'diff_mean={dm:.6f}')
                    print(f"processed {done}/{total} {' '.join(metrics)}", flush=True)
                else:
                    print(f'processed {done}/{total} cons_mean={am:.6f} diff_mean={dm:.6f}', flush=True)

    if attn_scores:
        arr = np.array(attn_scores, dtype=float)
        print('DONE attention consistency', flush=True)
        print('  count', len(arr), 'mean', float(arr.mean()), 'median', float(np.median(arr)), 'min', float(arr.min()), 'max', float(arr.max()), flush=True)
    if diff_scores:
        arrd = np.array(diff_scores, dtype=float)
        print('DONE attention diffuseness', flush=True)
        print('  count', len(arrd), 'mean', float(arrd.mean()), 'median', float(np.median(arrd)), 'min', float(arrd.min()), 'max', float(arrd.max()), flush=True)
    if args.gradcam and grad_scores:
        arrg = np.array(grad_scores, dtype=float)
        print('DONE gradcam consistency', flush=True)
        print('  count', len(arrg), 'mean', float(arrg.mean()), 'median', float(np.median(arrg)), 'min', float(arrg.min()), 'max', float(arrg.max()), flush=True)
    if args.ig and ig_scores:
        arri = np.array(ig_scores, dtype=float)
        print('DONE integrated-gradients consistency', flush=True)
        print('  count', len(arri), 'mean', float(arri.mean()), 'median', float(np.median(arri)), 'min', float(arri.min()), 'max', float(arri.max()), flush=True)
    print('csv', str(out_csv.resolve()), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
