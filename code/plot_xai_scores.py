from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=float)
    s = pd.to_numeric(df[col], errors='coerce')
    return s.dropna()


def main() -> int:
    p = argparse.ArgumentParser(description='Plot XAI score CSVs (consistency, diffuseness, optional Grad-CAM).')
    p.add_argument('--csv', type=Path, default=Path('xai_outputs/test_consistency_scores.csv'))
    p.add_argument('--out-dir', type=Path, default=None, help='Output directory for PNGs (default: CSV parent)')
    p.add_argument('--show', action='store_true', help='Also display figures interactively')
    args = p.parse_args()

    csv_path = args.csv
    if not csv_path.is_file():
        print(f'File not found: {csv_path}', flush=True)
        return 1

    out_dir = args.out_dir if args.out_dir is not None else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = csv_path.stem

    df = pd.read_csv(csv_path)
    has_diff = 'diffuseness' in df.columns
    has_grad = 'gradcam_consistency' in df.columns and 'attn_consistency' in df.columns
    has_cons = 'consistency' in df.columns

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    if has_grad:
        a = _numeric_series(df, 'attn_consistency')
        g = _numeric_series(df, 'gradcam_consistency')
        ax1.hist(a, bins=40, alpha=0.55, label='Attention consistency', color='steelblue', edgecolor='black')
        ax1.hist(g, bins=40, alpha=0.55, label='Grad-CAM consistency', color='coral', edgecolor='black')
        ax1.set_title('Consistency: attention vs Grad-CAM')
    elif has_cons:
        c = _numeric_series(df, 'consistency')
        ax1.hist(c, bins=40, alpha=0.75, color='steelblue', edgecolor='black')
        ax1.set_title('Attention consistency')
    else:
        print('No consistency columns found (expected consistency or attn_consistency).', flush=True)
        return 1
    ax1.set_xlabel('Score')
    ax1.set_ylabel('Count')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    p1 = out_dir / f'{stem}_consistency.png'
    fig1.savefig(p1, dpi=150)
    print('saved', p1.resolve(), flush=True)
    if args.show:
        plt.show()
    plt.close(fig1)

    if has_diff:
        d = _numeric_series(df, 'diffuseness')
        if len(d) > 0:
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            ax2.hist(d, bins=40, alpha=0.75, color='seagreen', edgecolor='black')
            ax2.set_xlabel('Mean attention diffuseness')
            ax2.set_ylabel('Count')
            ax2.set_title('Cross-attention diffuseness (higher = more spread)')
            ax2.grid(True, alpha=0.3)
            fig2.tight_layout()
            p2 = out_dir / f'{stem}_diffuseness.png'
            fig2.savefig(p2, dpi=150)
            print('saved', p2.resolve(), flush=True)
            if args.show:
                plt.show()
            plt.close(fig2)

            if has_grad:
                x = pd.to_numeric(df['attn_consistency'], errors='coerce')
            else:
                x = pd.to_numeric(df['consistency'], errors='coerce')
            y = pd.to_numeric(df['diffuseness'], errors='coerce')
            mask = x.notna() & y.notna()
            x, y = x[mask], y[mask]
            if len(x) > 5:
                fig3, ax3 = plt.subplots(figsize=(8, 7))
                ax3.scatter(x, y, alpha=0.25, s=12, c='purple')
                ax3.set_xlabel('Consistency')
                ax3.set_ylabel('Diffuseness')
                ax3.set_title('Consistency vs diffuseness (per image)')
                ax3.grid(True, alpha=0.3)
                fig3.tight_layout()
                p3 = out_dir / f'{stem}_scatter_consistency_diffuseness.png'
                fig3.savefig(p3, dpi=150)
                print('saved', p3.resolve(), flush=True)
                if args.show:
                    plt.show()
                plt.close(fig3)

    parts = []
    if has_grad:
        a = _numeric_series(df, 'attn_consistency')
        g = _numeric_series(df, 'gradcam_consistency')
        if len(a):
            parts.append(f'attn_consistency n={len(a)} mean={a.mean():.4f}')
        if len(g):
            parts.append(f'grad_consistency n={len(g)} mean={g.mean():.4f}')
    elif has_cons:
        c = _numeric_series(df, 'consistency')
        if len(c):
            parts.append(f'consistency n={len(c)} mean={c.mean():.4f}')
    if has_diff:
        d = _numeric_series(df, 'diffuseness')
        if len(d):
            parts.append(f'diffuseness n={len(d)} mean={d.mean():.4f} median={d.median():.4f}')
    if parts:
        print('summary:', ' | '.join(parts), flush=True)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
