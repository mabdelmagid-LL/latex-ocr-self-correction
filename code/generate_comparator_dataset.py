"""
generate_comparator_dataset.py
===============================
Pre-renders all training/val pairs for the comparator and saves them to disk.
Training then just loads images â€” no on-the-fly LaTeX rendering.

Output structure:
  <out_dir>/
    train/
      pairs.csv          (input_path, render_path, label, pair_type)
      renders/           (pre-rendered formula PNGs)
    val/
      pairs.csv
      renders/

Usage:
  python generate_comparator_dataset.py --out-dir comparator_dataset
  python generate_comparator_dataset.py --out-dir comparator_dataset --pairs-per-image 4 --hard-negative-prob 1.0
"""

import argparse
import csv
import random
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from train_comparator_hf import (
    FormulaRenderer,
    collect_split_samples,
    filter_renderable_samples,
    load_formulas,
    mutate_formula,
    preprocess_input_image,
    seed_everything,
)


def make_pairs_for_image(
    row,
    all_rows,
    renderer: FormulaRenderer,
    pairs_per_image: int,
    hard_negative_prob: float,
    rng: random.Random,
    renders_dir: Path,
    by_len: dict,
    min_render_ink: int,
) -> list:
    """Generate pairs for one input image. Returns list of dicts."""
    img_id, img_path, formula = row
    results = []

    try:
        ref_render = renderer.render(formula)
    except Exception:
        return results  # skip unrenderable image

    for pair_idx in range(pairs_per_image):
        is_pos = (pair_idx % 2 == 0)

        if is_pos:
            target_formula = formula
            label = 1
            pair_type = "positive"
        else:
            target_formula = formula
            pair_type = "negative_random"

            for _ in range(32):
                if rng.random() < hard_negative_prob:
                    cand = mutate_formula(formula, rng)
                    cand_type = "negative_mutation"
                else:
                    bucket = min(len(formula) // 10, 50)
                    candidates = by_len.get(bucket, [])
                    if len(candidates) >= 2:
                        cand = all_rows[rng.choice(candidates)][2]
                    else:
                        cand = rng.choice(all_rows)[2]
                    cand_type = "negative_nearby"

                if cand == formula:
                    continue
                if renderer.ink_pixels(cand) < min_render_ink:
                    continue

                # Accept mutations regardless of MSE (they are the hard negatives)
                if cand_type == "negative_mutation":
                    target_formula = cand
                    pair_type = cand_type
                    break

                # For easy negatives, keep MSE filter
                cand_render = renderer.render(cand)
                mse = float(np.mean((ref_render - cand_render) ** 2))
                if mse >= 0.005:
                    target_formula = cand
                    pair_type = cand_type
                    break

            label = 0

        # Render target formula
        render_arr = renderer.render(target_formula)

        # Save render to disk as PNG
        render_fname = f"{img_id}_{pair_idx}_{pair_type}.png"
        render_path = renders_dir / render_fname
        render_img = Image.fromarray((render_arr * 255).astype(np.uint8))
        try:
            render_img.save(render_path)
        except Exception:
            continue  # skip pair if PNG save fails

        results.append({
            "input_path": str(img_path),
            "render_path": str(render_path),
            "label": label,
            "pair_type": pair_type,
            "formula": formula,
            "target_formula": target_formula,
        })

    return results


def generate_split(
    split: str,
    rows,
    out_dir: Path,
    pairs_per_image: int,
    hard_negative_prob: float,
    min_render_ink: int,
    seed: int,
    renderer: FormulaRenderer,
    append: bool = False,
):
    split_dir = out_dir / split
    renders_dir = split_dir / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    # Build length-bucket index for nearby negatives
    by_len = {}
    for idx, (_, _, formula) in enumerate(rows):
        bucket = min(len(formula) // 10, 50)
        by_len.setdefault(bucket, []).append(idx)

    all_pairs = []
    print(f"\n[{split}] Generating {len(rows)} أ— {pairs_per_image} pairs "
          f"({hard_negative_prob:.0%} hard negatives)...")

    for row in tqdm(rows, desc=split):
        pairs = make_pairs_for_image(
            row=row,
            all_rows=rows,
            renderer=renderer,
            pairs_per_image=pairs_per_image,
            hard_negative_prob=hard_negative_prob,
            rng=rng,
            renders_dir=renders_dir,
            by_len=by_len,
            min_render_ink=min_render_ink,
        )
        all_pairs.extend(pairs)

    # Write CSV (append mode for chunked generation)
    csv_path = split_dir / "pairs.csv"
    fieldnames = ["input_path", "render_path", "label", "pair_type", "formula", "target_formula"]
    mode = "a" if append and csv_path.exists() else "w"
    with open(csv_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()
        writer.writerows(all_pairs)

    pos = sum(1 for p in all_pairs if p["label"] == 1)
    neg = sum(1 for p in all_pairs if p["label"] == 0)
    mut = sum(1 for p in all_pairs if p["pair_type"] == "negative_mutation")
    print(f"  Saved {len(all_pairs)} pairs -> {csv_path}")
    print(f"  Positive: {pos} | Negative: {neg} (mutations: {mut})")
    return csv_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--extracted-root", default="data/formulae_extracted_full")
    parser.add_argument("--math-txt", default="data/math.txt")
    parser.add_argument("--out-dir", default="comparator_dataset")
    parser.add_argument("--pairs-per-image", type=int, default=4,
                        help="Pairs to generate per training image (half pos, half neg)")
    parser.add_argument("--hard-negative-prob", type=float, default=1.0,
                        help="Fraction of negatives that are hard mutations (1.0 = all)")
    parser.add_argument("--train-samples", type=int, default=0,
                        help="Max training images (0 = all)")
    parser.add_argument("--val-samples", type=int, default=0,
                        help="Max val images (0 = all)")
    parser.add_argument("--start-idx", type=int, default=0,
                        help="Start index into train_rows (for chunked generation)")
    parser.add_argument("--skip-val", action="store_true",
                        help="Skip val split generation (for train-only chunks)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip train split generation")
    parser.add_argument("--min-render-ink", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ws = Path(args.workspace)
    out_dir = ws / args.out_dir
    seed_everything(args.seed)

    formulas = load_formulas(ws / args.math_txt)
    extracted_root = ws / args.extracted_root

    renderer = FormulaRenderer(height=96, width=384, dpi=140, cache_max_entries=50000)

    train_rows = collect_split_samples(extracted_root, "train", formulas, 0, args.seed)
    val_rows   = collect_split_samples(extracted_root, "val",   formulas, args.val_samples, args.seed + 1)

    # Slice train chunk
    if args.start_idx > 0 or args.train_samples > 0:
        end_idx = args.start_idx + args.train_samples if args.train_samples > 0 else len(train_rows)
        train_rows = train_rows[args.start_idx:end_idx]

    print(f"Train images: {len(train_rows)} (chunk start={args.start_idx}) | Val images: {len(val_rows)}")
    print(f"Pairs per image: {args.pairs_per_image} | Hard-neg prob: {args.hard_negative_prob:.0%}")

    append = args.start_idx > 0
    if not args.skip_train:
        generate_split("train", train_rows, out_dir, args.pairs_per_image,
                       args.hard_negative_prob, args.min_render_ink, args.seed, renderer,
                       append=append)
    if not args.skip_val:
        generate_split("val", val_rows, out_dir, args.pairs_per_image,
                       args.hard_negative_prob, args.min_render_ink, args.seed + 1, renderer,
                       append=append)

    print(f"\nDataset saved to: {out_dir}")
    print(f"Use --dataset-dir {out_dir} when training.")


if __name__ == "__main__":
    main()
