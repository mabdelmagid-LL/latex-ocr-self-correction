"""
evaluate_pipeline_from_existing.py
====================================
Load already-decoded pix2tex predictions (results_pix2tex_gd_full_fixed128)
and run comparator scoring + feedback loop on top. No re-decoding needed.

Usage:
  python -u evaluate_pipeline_from_existing.py --arch v2a --feedback-samples 1000
  python -u evaluate_pipeline_from_existing.py --arch v2b --feedback-samples 1000
"""

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from jiwer import cer
from sacrebleu import corpus_bleu

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from train_comparator_hf import (
    HFComparatorModel,
    FormulaRenderer,
    apply_lora_to_linear_layers,
    preprocess_input_image,
    seed_everything,
)
from evaluate_full_pipeline_test import (
    load_comparator,
    score_batch,
    feedback_loop,
    compute_nlp_metrics,
    remove_spaces,
    _img_raw_width,
    plot_pipeline,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--pix2tex-model-dir", default="",
                        help="Path to pix2tex model dir (contains checkpoints/ and settings/). "
                             "Defaults to the pix2tex package install. "
                             "Use models/pix2tex_baseline if pix2tex is not installed.")
    parser.add_argument("--predictions-csv",
                        default="results_pix2tex_gd_full_fixed128/pix2tex_gd_test_predictions.csv")
    parser.add_argument("--extracted-root", default="data/formulae_extracted_full")
    parser.add_argument("--comparator-checkpoint",
                        default="results_m2_v2a/comparator_v2.pt")
    parser.add_argument("--arch", default="v2a",
                        choices=["baseline", "v2a", "v2b", "v2c", "v2d"])
    parser.add_argument("--backbone-type", default="pix2tex_encoder")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--sample-size", type=int, default=1000,
                        help="Random sample of predictions to evaluate (0=all)")
    parser.add_argument("--feedback-samples", type=int, default=0,
                        help="Further limit feedback loop to N below-tau samples (0=all below-tau)")
    parser.add_argument("--max-iters", type=int, default=2)
    parser.add_argument("--pix2tex-batch-size", type=int, default=26)
    parser.add_argument("--comparator-batch-size", type=int, default=64)
    parser.add_argument("--max-seq-len", type=int, default=192)
    parser.add_argument("--output-dir", default="results_m2_pipeline_existing")
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fresh-decode", action="store_true",
                        help="Re-decode baseline with pix2tex (width-sorted) instead of loading from CSV")
    args = parser.parse_args()

    if args.pix2tex_model_dir:
        import contextlib, pix2tex.cli as _cli, pix2tex.utils as _utils
        _dir = str(Path(args.pix2tex_model_dir).resolve())
        @contextlib.contextmanager
        def _patched():
            saved = os.getcwd(); os.chdir(_dir)
            try: yield
            finally: os.chdir(saved)
        _utils.in_model_path = _patched
        _cli.in_model_path = _patched

    ws = Path(args.workspace)
    out_dir = ws / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    print(f"\n{'='*60}")
    print(f"Pipeline Eval (existing predictions + comparator)")
    print(f"{'='*60}")
    print(f"Device : {device} | Arch : {args.arch} | tau : {args.tau}")

    # ---- Load existing predictions ------------------------------------------
    print(f"\n[1/4] Loading existing predictions...")
    pred_csv = ws / args.predictions_csv
    df_preds = pd.read_csv(pred_csv)
    print(f"  Loaded {len(df_preds)} predictions from {pred_csv.name}")

    # Map id â†’ image path
    extracted_root = ws / args.extracted_root / "test"
    id_to_path = {}
    for p in sorted(extracted_root.glob("*.png")):
        try:
            id_to_path[int(p.stem)] = p
        except Exception:
            pass

    # Filter to rows with existing image files
    df_preds = df_preds[df_preds["id"].isin(id_to_path)].copy().reset_index(drop=True)
    print(f"  {len(df_preds)} rows have matching image files")

    # Sample 1000 (or --sample-size) rows
    if 0 < args.sample_size < len(df_preds):
        df_preds = df_preds.sample(n=args.sample_size, random_state=args.seed).reset_index(drop=True)
        print(f"  Sampled {len(df_preds)} rows (seed={args.seed})")

    ids   = df_preds["id"].tolist()
    refs  = df_preds["reference"].tolist()
    paths = [id_to_path[i] for i in ids]

    def _load_pix2tex(dev):
        from self_correcting_render_compare import FastPix2Tex
        ocr = FastPix2Tex(device=dev, max_seq_len=args.max_seq_len, temperature=0.25)
        if dev == "cuda":
            if hasattr(ocr, "image_resizer") and ocr.image_resizer is not None:
                ocr.image_resizer = ocr.image_resizer.to(dev)
            elif hasattr(ocr, "model") and hasattr(ocr.model, "image_resizer"):
                ocr.model.image_resizer = ocr.model.image_resizer.to(dev)
        return ocr

    # ---- Fresh baseline decode (optional) -----------------------------------
    # Phase 1: pix2tex alone on GPU â†’ decode â†’ release GPU fully
    if args.fresh_decode:
        from evaluate_full_pipeline_test import decode_images
        print(f"\n[1b/4] Fresh pix2tex baseline decode (width-sorted, batch={args.pix2tex_batch_size})...")
        if device == "cuda":
            torch.cuda.empty_cache()
        ocr_base = _load_pix2tex(device)
        t0 = time.perf_counter()
        preds = decode_images(ocr_base, paths, args.pix2tex_batch_size, device)
        print(f"  Decoded {len(preds)} images in {time.perf_counter()-t0:.1f}s")
        del ocr_base; gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    else:
        preds = df_preds["prediction"].fillna("").tolist()

    # ---- Load comparator on GPU, score, then move to CPU -------------------
    # Phase 2: comparator alone on GPU â†’ score â†’ move to CPU to free GPU for pix2tex
    print(f"\n[2/4] Loading comparator [{args.arch}]...")
    ckpt_path = ws / args.comparator_checkpoint
    comparator = load_comparator(
        ckpt_path, args.backbone_type, args.lora_rank,
        args.lora_alpha, args.lora_dropout, device, arch=args.arch,
    )
    renderer = FormulaRenderer(height=96, width=384, dpi=140, cache_max_entries=35000)

    # Score on GPU with small batch to avoid ResNetV2 stem OOM
    gpu_cmp_batch = min(args.comparator_batch_size, 4)
    print(f"\n[3/4] Scoring {len(preds)} predictions with comparator (batch={gpu_cmp_batch}, device={device})...")
    t0 = time.perf_counter()
    baseline_scores = score_batch(comparator, renderer, paths, preds, device, gpu_cmp_batch)
    print(f"  Scored in {time.perf_counter()-t0:.1f}s")

    # Move comparator to CPU so pix2tex can use the full GPU
    if device == "cuda":
        comparator.cpu()
        torch.cuda.empty_cache()
        print(f"  Comparator moved to CPU (GPU free for pix2tex)")

    baseline_metrics = compute_nlp_metrics(refs, preds)
    print(f"  Baseline EM={baseline_metrics['exact_match']:.4f} "
          f"CER={baseline_metrics['cer']:.4f} BLEU={baseline_metrics['bleu']:.4f}")

    accepted_b = sum(s >= args.tau for s in baseline_scores)
    below_tau  = [i for i, s in enumerate(baseline_scores) if s < args.tau]
    print(f"  Accepted (score>={args.tau}): {accepted_b}/{len(preds)} ({accepted_b/len(preds):.1%})")
    print(f"  Below tau: {len(below_tau)} samples")

    # ---- Feedback loop on subset -------------------------------------------
    final_preds  = list(preds)
    final_scores = list(baseline_scores)
    n_improved   = 0

    if args.max_iters > 0 and below_tau:
        print(f"\n[4/4] Loading pix2tex for feedback loop...")
        if device == "cuda":
            torch.cuda.empty_cache()
        ocr = _load_pix2tex(device)

        # Select feedback subset
        if args.feedback_samples > 0 and len(below_tau) > args.feedback_samples:
            random.seed(args.seed)
            fb_indices = random.sample(below_tau, args.feedback_samples)
        else:
            fb_indices = below_tau
        print(f"  Feedback loop on {len(fb_indices)} samples "
              f"({args.max_iters} iters, tau={args.tau})")

        fb_paths  = [paths[i]  for i in fb_indices]
        fb_preds  = [preds[i]  for i in fb_indices]
        fb_scores = [baseline_scores[i] for i in fb_indices]

        fb_preds_out, fb_scores_out, n_improved = feedback_loop(
            ocr=ocr, model=comparator, renderer=renderer,
            paths=fb_paths, best_preds=fb_preds, best_scores=fb_scores,
            tau=args.tau, max_iters=args.max_iters,
            batch_size=args.pix2tex_batch_size, device=device,
        )
        for local_i, orig_i in enumerate(fb_indices):
            final_preds[orig_i]  = fb_preds_out[local_i]
            final_scores[orig_i] = fb_scores_out[local_i]

        del ocr; gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    # ---- Metrics & save -----------------------------------------------------
    final_metrics = compute_nlp_metrics(refs, final_preds)
    refs_ns       = [remove_spaces(r) for r in refs]
    preds_ns_b    = [remove_spaces(p) for p in preds]
    preds_ns_f    = [remove_spaces(p) for p in final_preds]
    baseline_correct = [int(r == p) for r, p in zip(refs_ns, preds_ns_b)]
    final_correct    = [int(r == p) for r, p in zip(refs_ns, preds_ns_f)]
    baseline_cer_s   = [float(cer([r], [p])) for r, p in zip(refs_ns, preds_ns_b)]
    final_cer_s      = [float(cer([r], [p])) for r, p in zip(refs_ns, preds_ns_f)]

    df_out = pd.DataFrame({
        "id": ids, "reference": refs,
        "baseline_pred": preds, "final_pred": final_preds,
        "baseline_score": baseline_scores, "final_score": final_scores,
        "baseline_correct": baseline_correct, "final_correct": final_correct,
        "baseline_cer_sample": baseline_cer_s, "final_cer_sample": final_cer_s,
        "formula_len": [len(r) for r in refs],
        "score_improved": [int(fs > bs + 0.01) for bs, fs in zip(baseline_scores, final_scores)],
    })
    df_out.to_csv(out_dir / "pipeline_predictions.csv", index=False)

    accepted_f = sum(s >= args.tau for s in final_scores)
    r_val = float(np.corrcoef(baseline_scores, baseline_correct)[0, 1])
    metrics_out = {
        "samples": len(df_preds),
        "device": device,
        "arch": args.arch,
        "tau": args.tau,
        "feedback_samples": len(fb_indices) if args.max_iters > 0 and below_tau else 0,
        "feedback_improved": int(n_improved),
        "accepted_baseline": int(accepted_b),
        "accepted_final": int(accepted_f),
        "baseline": baseline_metrics,
        "final": final_metrics,
        "delta": {k: round(final_metrics[k] - baseline_metrics[k], 6) for k in baseline_metrics},
        "mean_baseline_score": round(float(np.mean(baseline_scores)), 4),
        "mean_final_score": round(float(np.mean(final_scores)), 4),
        "score_vs_exact_pearson_r": round(r_val, 4),
    }
    with open(out_dir / "pipeline_metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    plot_pipeline(df_out, out_dir)

    print(f"\n{'='*60}")
    print(f"RESULTS ({len(df_preds)} samples, arch={args.arch})")
    print(f"{'='*60}")
    print(f"  {'Metric':<20} {'Baseline':>10} {'Pipeline':>10} {'Delta':>8}")
    print(f"  {'-'*50}")
    for k in ["exact_match", "cer", "bleu"]:
        b, f = baseline_metrics[k], final_metrics[k]
        d = f - b
        print(f"  {k:<20} {b:>10.4f} {f:>10.4f} {('+' if d>=0 else '')+str(round(d,4)):>8}")
    print(f"\n  Accepted: {accepted_b}/{len(preds)} -> {accepted_f}/{len(preds)}")
    print(f"  Score <-> Correctness Pearson r: {r_val:.4f}")
    print(f"  Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
