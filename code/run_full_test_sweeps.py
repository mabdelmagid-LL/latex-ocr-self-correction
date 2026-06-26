from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from munch import Munch

from pix2tex.cli import LatexOCR, minmax_size
from pix2tex.dataset.transforms import test_transform
from pix2tex.utils import pad, post_process
from pix2tex.xai.consistency import attribution_consistency_score, CRITICAL_TOKEN_KEYS
from pix2tex.xai.gradcam import add_gradcam_to_trace
from pix2tex.xai.integrated_gradients import add_integrated_gradients_to_trace

IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'}

QUALITY_PRESETS = [
    {
        'preset': 'baseline',
        'quality_gate': False,
        'quality_confidence_threshold': 0.995,
        'quality_diffuseness_threshold': 0.81,
        'quality_consistency_threshold': 0.04,
        'quality_max_retries': 2,
        'quality_retry_temperatures': '0.12,0.22,0.30',
    },
    {
        'preset': 'balanced',
        'quality_gate': True,
        'quality_confidence_threshold': 0.995,
        'quality_diffuseness_threshold': 0.81,
        'quality_consistency_threshold': 0.04,
        'quality_max_retries': 2,
        'quality_retry_temperatures': '0.12,0.22,0.30',
    },
    {
        'preset': 'strict_conf',
        'quality_gate': True,
        'quality_confidence_threshold': 0.992,
        'quality_diffuseness_threshold': 0.81,
        'quality_consistency_threshold': 0.04,
        'quality_max_retries': 2,
        'quality_retry_temperatures': '0.12,0.22,0.30',
    },
    {
        'preset': 'diff_sensitive',
        'quality_gate': True,
        'quality_confidence_threshold': 0.995,
        'quality_diffuseness_threshold': 0.78,
        'quality_consistency_threshold': 0.04,
        'quality_max_retries': 2,
        'quality_retry_temperatures': '0.12,0.22,0.30',
    },
    {
        'preset': 'cons_sensitive',
        'quality_gate': True,
        'quality_confidence_threshold': 0.995,
        'quality_diffuseness_threshold': 0.81,
        'quality_consistency_threshold': 0.06,
        'quality_max_retries': 2,
        'quality_retry_temperatures': '0.12,0.22,0.30',
    },
    {
        'preset': 'more_retries',
        'quality_gate': True,
        'quality_confidence_threshold': 0.995,
        'quality_diffuseness_threshold': 0.81,
        'quality_consistency_threshold': 0.04,
        'quality_max_retries': 3,
        'quality_retry_temperatures': '0.08,0.16,0.24,0.32',
    },
    {
        'preset': 'strict_all',
        'quality_gate': True,
        'quality_confidence_threshold': 0.998,
        'quality_diffuseness_threshold': 0.76,
        'quality_consistency_threshold': 0.08,
        'quality_max_retries': 3,
        'quality_retry_temperatures': '0.08,0.14,0.20,0.28',
    },
]

CONSISTENCY_CONFIGS = [
    {
        'config': 'default',
        'top_percent': 0.15,
        'min_component_area': 8,
        'component_padding': 0,
        'critical_token_keys': CRITICAL_TOKEN_KEYS,
        'overlap_mode': 'iou',
    },
    {
        'config': 'top25',
        'top_percent': 0.25,
        'min_component_area': 8,
        'component_padding': 0,
        'critical_token_keys': CRITICAL_TOKEN_KEYS,
        'overlap_mode': 'iou',
    },
    {
        'config': 'top35',
        'top_percent': 0.35,
        'min_component_area': 8,
        'component_padding': 0,
        'critical_token_keys': CRITICAL_TOKEN_KEYS,
        'overlap_mode': 'iou',
    },
    {
        'config': 'pad2',
        'top_percent': 0.25,
        'min_component_area': 8,
        'component_padding': 2,
        'critical_token_keys': CRITICAL_TOKEN_KEYS,
        'overlap_mode': 'iou',
    },
    {
        'config': 'dice',
        'top_percent': 0.25,
        'min_component_area': 8,
        'component_padding': 2,
        'critical_token_keys': CRITICAL_TOKEN_KEYS,
        'overlap_mode': 'dice',
    },
    {
        'config': 'all_tokens',
        'top_percent': 0.25,
        'min_component_area': 8,
        'component_padding': 2,
        'critical_token_keys': (),
        'overlap_mode': 'dice',
    },
]


def _list_images(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open('r', newline='') as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _configure_ocr(no_cuda: bool, resize: bool, temperature: float) -> LatexOCR:
    args = Munch({
        'config': 'settings/config.yaml',
        'checkpoint': 'checkpoints/weights.pth',
        'no_cuda': no_cuda,
        'no_resize': not resize,
        'temperature': temperature,
        'explain': True,
        'gradcam': False,
    })
    return LatexOCR(args)


def _extract_trace_quality(trace: dict | None) -> dict[str, float | bool | str]:
    quality = trace.get('quality', {}) if isinstance(trace, dict) else {}
    return {
        'confidence_mean': float(quality.get('confidence_mean', 0.0) or 0.0),
        'confidence_min': float(quality.get('confidence_min', 0.0) or 0.0),
        'diffuseness': float(quality.get('diffuseness', 1.0) or 1.0),
        'consistency': float(quality.get('consistency', 0.0) or 0.0),
        'score': float(quality.get('score', -1.0) or -1.0),
        'redecode_triggered': bool(quality.get('redecode_triggered', False)),
        'redecode_used': bool(quality.get('redecode_used', False)),
        'selected_temperature': float(quality.get('selected_temperature', 0.0) or 0.0),
    }


def _set_quality_args(ocr: LatexOCR, preset: dict) -> None:
    for key, value in preset.items():
        if key == 'preset':
            continue
        setattr(ocr.args, key, value)


def run_quality_sweep(ocr: LatexOCR, images: list[Path], labels: list[str], output_dir: Path, limit: int | None, progress_every: int, preset_filter: str | None = None) -> tuple[Path, Path]:
    per_image_csv = output_dir / 'quality_gate_per_image.csv'
    summary_csv = output_dir / 'quality_gate_summary.csv'

    if limit is not None:
        images = images[:limit]

    processed = {(row.get('image', ''), row.get('preset', '')) for row in _load_csv_rows(per_image_csv)}
    rows = _load_csv_rows(per_image_csv)

    presets_to_run = QUALITY_PRESETS
    if preset_filter:
        presets_to_run = [p for p in QUALITY_PRESETS if p['preset'] == preset_filter]
        if not presets_to_run:
            print(f"Warning: preset '{preset_filter}' not found. Available presets: {[p['preset'] for p in QUALITY_PRESETS]}")
            return per_image_csv, summary_csv

    for i, fp in enumerate(images, start=1):
        try:
            idx = int(fp.stem)
        except ValueError:
            continue
        if idx < 0 or idx >= len(labels):
            continue

        truth = post_process(labels[idx].strip())
        if not truth:
            continue

        try:
            img = Image.open(fp)
            img = minmax_size(pad(img), ocr.args.max_dimensions, ocr.args.min_dimensions)
            arr = np.array(pad(img).convert('RGB'))
            t = test_transform(image=arr)['image'][:1].unsqueeze(0)
            im = t.to(ocr.args.device)
        except Exception as e:
            print(f'Warning: Skipping corrupted image {fp.name}: {e}', flush=True)
            continue

        for preset in presets_to_run:
            key = (fp.name, preset['preset'])
            if key in processed:
                continue
            _set_quality_args(ocr, preset)
            pred, trace = ocr(img)
            quality = _extract_trace_quality(trace)
            exact = post_process(pred.strip()) == truth
            rows.append({
                'image': fp.name,
                'preset': preset['preset'],
                'truth': truth,
                'prediction': post_process(pred.strip()),
                'exact_match': int(exact),
                **quality,
            })
            processed.add(key)

        if i % progress_every == 0:
            print(f'quality sweep processed {i}/{len(images)}', flush=True)
            _write_csv(per_image_csv, rows, list(rows[0].keys()))

    if rows:
        _write_csv(per_image_csv, rows, list(rows[0].keys()))

    df = pd.DataFrame(rows)
    for col in ['exact_match', 'confidence_mean', 'confidence_min', 'diffuseness', 'consistency', 'score', 'redecode_triggered', 'redecode_used', 'selected_temperature']:
        if col in df.columns:
            if col in {'redecode_triggered', 'redecode_used'}:
                df[col] = df[col].map(lambda value: 1 if str(value).lower() == 'true' else 0)
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    summary = (
        df.groupby('preset', as_index=False)
        .agg(
            evaluated=('image', 'count'),
            exact_match_rate=('exact_match', 'mean'),
            exact_match_count=('exact_match', 'sum'),
            trigger_rate=('redecode_triggered', 'mean'),
            retry_used_rate=('redecode_used', 'mean'),
            mean_confidence=('confidence_mean', 'mean'),
            mean_diffuseness=('diffuseness', 'mean'),
            mean_consistency=('consistency', 'mean'),
        )
        .sort_values('preset')
    )
    summary.to_csv(summary_csv, index=False)
    return per_image_csv, summary_csv


def run_consistency_sweep(ocr: LatexOCR, images: list[Path], output_dir: Path, limit: int | None, progress_every: int) -> tuple[Path, Path]:
    per_image_csv = output_dir / 'consistency_per_image.csv'
    summary_csv = output_dir / 'consistency_summary.csv'

    if limit is not None:
        images = images[:limit]

    processed = {row.get('image', '') for row in _load_csv_rows(per_image_csv)}
    rows = _load_csv_rows(per_image_csv)

    for i, fp in enumerate(images, start=1):
        if fp.name in processed:
            continue

        img = Image.open(fp)
        img = minmax_size(pad(img), ocr.args.max_dimensions, ocr.args.min_dimensions)
        arr = np.array(pad(img).convert('RGB'))
        t = test_transform(image=arr)['image'][:1].unsqueeze(0)
        im = t.to(ocr.args.device)

        with torch.no_grad():
            trace = ocr.model.generate_with_trace(im, temperature=ocr.args.get('temperature', 0.333))
        trace = add_gradcam_to_trace(ocr.model, im, trace, max_tokens=12)
        trace = add_integrated_gradients_to_trace(ocr.model, im, trace, max_tokens=12, steps=16)

        tokens = trace['tokens']
        token_ids = tokens[0].detach().cpu().tolist() if tokens.ndim == 2 else tokens.detach().cpu().tolist()

        row = {'image': fp.name}
        for cfg in CONSISTENCY_CONFIGS:
            attn = attribution_consistency_score(
                image_tensor=im[0].detach().cpu(),
                token_ids=token_ids,
                token_maps=trace.get('cross_attentions', None),
                tokenizer=ocr.tokenizer,
                patch_size=ocr.args.get('patch_size', 16),
                top_percent=cfg['top_percent'],
                min_component_area=cfg['min_component_area'],
                component_padding=cfg['component_padding'],
                critical_token_keys=cfg['critical_token_keys'],
                overlap_mode=cfg['overlap_mode'],
            )
            grad = attribution_consistency_score(
                image_tensor=im[0].detach().cpu(),
                token_ids=token_ids,
                token_maps=trace.get('grad_attributions', None),
                tokenizer=ocr.tokenizer,
                patch_size=ocr.args.get('patch_size', 16),
                top_percent=cfg['top_percent'],
                min_component_area=cfg['min_component_area'],
                component_padding=cfg['component_padding'],
                critical_token_keys=cfg['critical_token_keys'],
                overlap_mode=cfg['overlap_mode'],
            )
            ig = attribution_consistency_score(
                image_tensor=im[0].detach().cpu(),
                token_ids=token_ids,
                token_maps=trace.get('ig_attributions', None),
                tokenizer=ocr.tokenizer,
                patch_size=ocr.args.get('patch_size', 16),
                top_percent=cfg['top_percent'],
                min_component_area=cfg['min_component_area'],
                component_padding=cfg['component_padding'],
                critical_token_keys=cfg['critical_token_keys'],
                overlap_mode=cfg['overlap_mode'],
            )
            row[f"{cfg['config']}_attn"] = float(attn)
            row[f"{cfg['config']}_grad"] = float(grad)
            row[f"{cfg['config']}_ig"] = float(ig)

        rows.append(row)
        processed.add(fp.name)

        if i % progress_every == 0:
            print(f'consistency sweep processed {i}/{len(images)}', flush=True)
            _write_csv(per_image_csv, rows, list(rows[0].keys()))

    if rows:
        _write_csv(per_image_csv, rows, list(rows[0].keys()))

    df = pd.DataFrame(rows)
    for cfg in CONSISTENCY_CONFIGS:
        for suffix in ['attn', 'grad', 'ig']:
            col = f"{cfg['config']}_{suffix}"
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    summary_rows = []
    for cfg in CONSISTENCY_CONFIGS:
        prefix = cfg['config']
        attn_col = f'{prefix}_attn'
        grad_col = f'{prefix}_grad'
        ig_col = f'{prefix}_ig'
        summary_rows.append({
            'config': prefix,
            'evaluated': int(len(df)),
            'attn_mean': float(pd.to_numeric(df[attn_col], errors='coerce').mean()),
            'attn_std': float(pd.to_numeric(df[attn_col], errors='coerce').std(ddof=1)),
            'attn_median': float(pd.to_numeric(df[attn_col], errors='coerce').median()),
            'grad_mean': float(pd.to_numeric(df[grad_col], errors='coerce').mean()),
            'grad_std': float(pd.to_numeric(df[grad_col], errors='coerce').std(ddof=1)),
            'grad_median': float(pd.to_numeric(df[grad_col], errors='coerce').median()),
            'ig_mean': float(pd.to_numeric(df[ig_col], errors='coerce').mean()),
            'ig_std': float(pd.to_numeric(df[ig_col], errors='coerce').std(ddof=1)),
            'ig_median': float(pd.to_numeric(df[ig_col], errors='coerce').median()),
        })
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    return per_image_csv, summary_csv


def plot_quality(summary_csv: Path, output_dir: Path) -> Path:
    df = pd.read_csv(summary_csv)
    df = df.sort_values('exact_match_rate', ascending=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df))
    ax.bar(x, df['exact_match_rate'], color='steelblue', edgecolor='black', label='Exact match rate')
    ax.plot(x, df['trigger_rate'], color='darkorange', marker='o', linewidth=2, label='Trigger rate')
    ax.plot(x, df['retry_used_rate'], color='seagreen', marker='o', linewidth=2, label='Retry used rate')
    ax.set_xticks(x)
    ax.set_xticklabels(df['preset'], rotation=25, ha='right')
    ax.set_ylim(0, max(0.5, float(df[['exact_match_rate', 'trigger_rate', 'retry_used_rate']].to_numpy().max()) * 1.15))
    ax.set_ylabel('Rate')
    ax.set_title('Full test-set quality gate sweep')
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path = output_dir / 'quality_gate_summary.png'
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_consistency(summary_csv: Path, output_dir: Path) -> Path:
    df = pd.read_csv(summary_csv)
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df))
    width = 0.25
    ax.bar(x - width, df['attn_mean'], width=width, label='Attention', color='steelblue', edgecolor='black')
    ax.bar(x, df['grad_mean'], width=width, label='Grad-CAM', color='coral', edgecolor='black')
    ax.bar(x + width, df['ig_mean'], width=width, label='IG', color='seagreen', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(df['config'], rotation=20, ha='right')
    ax.set_ylabel('Mean consistency')
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path = output_dir / 'consistency_summary.png'
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description='Run full test-set sweeps and plots.')
    parser.add_argument('--data-dir', type=Path, default=Path('data/formulae_extracted_full/test'))
    parser.add_argument('--labels', type=Path, default=Path('data/math.txt'))
    parser.add_argument('--output-dir', type=Path, default=Path('xai_outputs/full_test_sweeps'))
    parser.add_argument('--limit', type=int, default=None, help='Debug limit for a small subset')
    parser.add_argument('--no-cuda', action='store_true', help='Force CPU')
    parser.add_argument('--resize', action='store_true', help='Use learned image resizer if available')
    parser.add_argument('--temperature', type=float, default=0.333)
    parser.add_argument('--progress-every', type=int, default=100, help='Print progress every N images')
    parser.add_argument('--preset', type=str, default=None, help='Run only a specific preset (e.g., cons_sensitive)')
    parser.add_argument('--quality-only', action='store_true')
    parser.add_argument('--consistency-only', action='store_true')
    parser.add_argument('--no-plots', action='store_true')
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

    if not args.data_dir.is_dir():
        print(f'Not a directory: {args.data_dir}')
        return 1
    if not args.labels.is_file():
        print(f'Labels file not found: {args.labels}')
        return 1

    images = _list_images(args.data_dir)
    if not images:
        print(f'No images found in {args.data_dir}')
        return 1

    labels = args.labels.read_text(encoding='utf-8', errors='replace').splitlines()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ocr = _configure_ocr(args.no_cuda, args.resize, args.temperature)

    quality_csv = None
    quality_summary_csv = None
    consistency_csv = None
    consistency_summary_csv = None

    if not args.consistency_only:
        quality_csv, quality_summary_csv = run_quality_sweep(ocr, images, labels, args.output_dir, args.limit, args.progress_every, args.preset)
        print(f'quality per-image CSV: {quality_csv}')
        print(f'quality summary CSV:  {quality_summary_csv}')

    if not args.quality_only:
        consistency_csv, consistency_summary_csv = run_consistency_sweep(ocr, images, args.output_dir, args.limit, args.progress_every)
        print(f'consistency per-image CSV: {consistency_csv}')
        print(f'consistency summary CSV:  {consistency_summary_csv}')

    if not args.no_plots:
        if quality_summary_csv is not None:
            q_plot = plot_quality(quality_summary_csv, args.output_dir)
            print(f'quality plot: {q_plot}')
        if consistency_summary_csv is not None:
            c_plot = plot_consistency(consistency_summary_csv, args.output_dir)
            print(f'consistency plot: {c_plot}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
