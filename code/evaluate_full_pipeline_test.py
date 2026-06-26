"""
evaluate_full_pipeline_test.py
==============================
End-to-end evaluation of the full architecture:
    pix2tex (OCR backbone) + HF comparator (re-ranking / quality filter)

Pipeline
--------
1. Load pix2tex and the best comparator checkpoint.
2. Sample N images from the GD held-out test split.
3. Run pix2tex to get a baseline prediction per image.
4. Score each (image, rendered_prediction) pair with the comparator.
5. Optionally run the iterative feedback loop (re-decode low-score samples).
6. Report and compare:
     Baseline pix2tex  vs  pix2tex + comparator filter
   Metrics: Exact-Match (no-space), CER, BLEU, Acceptance Rate

Advanced analyses
-----------------
- Comparator score vs OCR correctness (calibration of the score as a quality signal)
- Per-score-bucket accuracy / CER
- Confusion: how often does the feedback loop help vs hurt?
- Improvement distribution (خ” score, خ”BLEU per sample)

All outputs land in --output-dir (default: results_m2_pipeline/).
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from jiwer import cer
from sacrebleu import corpus_bleu

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from train_comparator_hf import (
    HFComparatorModel,
    FormulaRenderer,
    apply_lora_to_linear_layers,
    collect_split_samples,
    filter_renderable_samples,
    load_formulas,
    preprocess_input_image,
    seed_everything,
)


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def load_comparator(
    checkpoint_pt: Path,
    backbone_type: str,
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float,
    device: str,
    arch: str = "baseline",
):
    if arch == "baseline":
        model = HFComparatorModel(pretrained=False, backbone_type=backbone_type)
        encoder_module = model.backbone.encoder
    else:
        from train_comparator_hf_v2 import HFComparatorModelV2
        model = HFComparatorModelV2(arch=arch, backbone_type=backbone_type, pretrained=False)
        encoder_module = getattr(model.backbone, "encoder", None)
    if lora_rank > 0 and encoder_module is not None:
        n = apply_lora_to_linear_layers(
            encoder_module, rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout,
        )
        print(f"  LoRA applied to {n} layers (rank={lora_rank})")
    ckpt = torch.load(checkpoint_pt, map_location="cpu")
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.backbone.load_state_dict(sd, strict=True)
    model.eval().to(device)
    print(f"  Comparator loaded [{arch}]: {checkpoint_pt.parent.name}/{checkpoint_pt.name}")
    return model


def load_pix2tex(device: str, max_seq_len: int = 192, temperature: float = 0.25):
    from self_correcting_render_compare import FastPix2Tex
    print(f"  Loading pix2tex (device={device}, max_seq_len={max_seq_len})...")
    ocr = FastPix2Tex(device=device, max_seq_len=max_seq_len, temperature=temperature)
    # Move image_resizer to GPU if available
    if device == "cuda":
        if hasattr(ocr, "image_resizer") and ocr.image_resizer is not None:
            ocr.image_resizer = ocr.image_resizer.to(device)
        elif hasattr(ocr, "model") and hasattr(ocr.model, "image_resizer") and ocr.model.image_resizer is not None:
            ocr.model.image_resizer = ocr.model.image_resizer.to(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
    print("  pix2tex loaded.")
    return ocr


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def score_batch(
    model,
    renderer: FormulaRenderer,
    paths: list,
    preds: list,
    device: str,
    batch_size: int = 128,
) -> list:
    """Score (input_image, rendered_prediction) pairs. Loads images on-the-fly."""
    scores = []
    bs = max(1, batch_size)
    for s in range(0, len(preds), bs):
        e = min(len(preds), s + bs)
        xs = []
        for i in range(s, e):
            inp = preprocess_input_image(paths[i], 96, 384)
            rnd = renderer.render(preds[i])
            diff = np.abs(inp - rnd)
            mul = inp * rnd
            x = np.stack([inp, rnd, diff, mul], axis=0)
            xs.append(x)
        xb = torch.from_numpy(np.stack(xs)).float().to(device)
        with torch.inference_mode():
            logits = model(input_tensor=xb)["logits"]
            sc = torch.sigmoid(logits).cpu().numpy()
        scores.extend(float(v) for v in sc)
    return scores


# ---------------------------------------------------------------------------
# Batch decode helpers
# ---------------------------------------------------------------------------

def _img_raw_width(p) -> int:
    """Return pixel width of an image file without decoding full content."""
    try:
        from PIL import Image
        with Image.open(p) as im:
            return im.size[0]
    except Exception:
        return 0


def decode_images(
    ocr,
    paths: list,
    batch_size: int,
    device: str,
) -> list:
    """Run pix2tex on a list of image paths, return predictions.

    Images are sorted by raw pixel width before batching so that each batch
    contains images of similar width, minimising collate_with_padding waste
    and restoring pix2tex accuracy to the expected ~32% exact-match level.
    """
    from evaluate_pix2tex_gd import build_input_tensor, collate_with_padding
    from pix2tex.utils import post_process, token2str
    from PIL import Image

    # Sort indices by raw image width â€” same-width images batch together
    order = sorted(range(len(paths)), key=lambda i: _img_raw_width(paths[i]))
    sorted_paths = [paths[i] for i in order]

    preds_sorted = [""] * len(paths)
    bs = max(1, batch_size)
    use_amp = device == "cuda"
    total = len(sorted_paths)

    for s in range(0, total, bs):
        e = min(total, s + bs)
        tensors, valid_positions = [], []
        for pos in range(s, e):
            try:
                img = Image.open(sorted_paths[pos]).convert("RGB")
                t = build_input_tensor(ocr, img, resize=True)
                if t is not None:
                    tensors.append(t)
                    valid_positions.append(pos)
            except Exception:
                pass
        if not tensors:
            continue
        batch = collate_with_padding(tensors)
        if batch is None:
            continue
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                dec = ocr.model.generate(batch.to(device), temperature=ocr.args.temperature)
        batch_preds = [post_process(x) for x in token2str(dec, ocr.tokenizer)]
        for pos, pred in zip(valid_positions, batch_preds):
            preds_sorted[pos] = pred
        if (s // bs) % 20 == 0:
            print(f"    decoded {e}/{total}")

    # Restore original order
    preds = [""] * len(paths)
    for sort_pos, orig_idx in enumerate(order):
        preds[orig_idx] = preds_sorted[sort_pos]
    return preds


# ---------------------------------------------------------------------------
# Feedback loop: re-decode low-confidence samples
# ---------------------------------------------------------------------------

def feedback_loop(
    ocr,
    model,
    renderer: FormulaRenderer,
    paths: list,
    best_preds: list,
    best_scores: list,
    tau: float,
    max_iters: int,
    batch_size: int,
    device: str,
) -> tuple:
    """Iteratively re-decode samples with score < tau and keep improvements."""
    from evaluate_pix2tex_gd import build_input_tensor, collate_with_padding
    from pix2tex.utils import post_process, token2str
    from PIL import Image

    alt_configs = [
        (0.18, int(ocr.args.max_seq_len)),
        (0.35, int(ocr.args.max_seq_len)),
        (0.25, min(512, int(ocr.args.max_seq_len) + 128)),
    ]
    best_preds = list(best_preds)
    best_scores = list(best_scores)
    n_improved = 0
    bs = max(1, batch_size)

    for it, (temp, seq_len) in enumerate(alt_configs[:max_iters]):
        active = [i for i, sc in enumerate(best_scores) if sc < tau]
        if not active:
            print(f"  Iter {it+1}: all samples above tau={tau}, stopping early.")
            break
        print(f"  Iter {it+1}: {len(active)} samples below tau={tau} (temp={temp}, seq={seq_len})")

        old_seq = ocr.args.max_seq_len
        old_temp = ocr.args.temperature
        ocr.args.max_seq_len = seq_len
        ocr.args.temperature = temp

        # Sort active indices by image width for efficient batching
        active_sorted = sorted(active, key=lambda i: _img_raw_width(paths[i]))
        use_amp = device == "cuda"
        cand_preds = {}
        for s in range(0, len(active_sorted), bs):
            idxs = active_sorted[s:s + bs]
            tensors = []
            valid_local = []
            for i in idxs:
                try:
                    img = Image.open(paths[i]).convert("RGB")
                    t = build_input_tensor(ocr, img, resize=True)
                    if t is not None:
                        tensors.append(t)
                        valid_local.append(i)
                except Exception:
                    pass
            if not tensors:
                continue
            batch = collate_with_padding(tensors)
            if batch is None:
                continue
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    dec = ocr.model.generate(batch.to(device), temperature=ocr.args.temperature)
            batch_p = [post_process(x) for x in token2str(dec, ocr.tokenizer)]
            for idx, pred in zip(valid_local, batch_p):
                cand_preds[idx] = pred

        ocr.args.max_seq_len = old_seq
        ocr.args.temperature = old_temp

        if not cand_preds:
            continue

        cand_idxs = list(cand_preds.keys())
        cand_list = [cand_preds[i] for i in cand_idxs]
        cand_paths = [paths[i] for i in cand_idxs]
        # Score on whatever device the model is currently on
        score_device = next(model.parameters()).device.type
        cand_scores_new = score_batch(model, renderer, cand_paths, cand_list, score_device, bs)

        for i, pred, score in zip(cand_idxs, cand_list, cand_scores_new):
            if score > best_scores[i]:
                if best_scores[i] < tau <= score:
                    n_improved += 1
                best_scores[i] = score
                best_preds[i] = pred

    return best_preds, best_scores, n_improved


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def remove_spaces(s: str) -> str:
    import re
    return re.sub(r"\s+", "", str(s) if s is not None else "")


def compute_nlp_metrics(refs: list, preds: list) -> dict:
    refs_ns = [remove_spaces(r) for r in refs]
    preds_ns = [remove_spaces(p) for p in preds]
    exact = float(np.mean([r == p for r, p in zip(refs_ns, preds_ns)]))
    cer_val = float(cer(refs_ns, preds_ns)) if refs_ns else 0.0
    bleu_val = float(corpus_bleu(preds_ns, [refs_ns], force=True).score / 100.0) if refs_ns else 0.0
    return {"exact_match": exact, "cer": cer_val, "bleu": bleu_val}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_pipeline(df: pd.DataFrame, out_dir: Path) -> None:
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # ---- Score distribution: correct vs incorrect predictions ---------------
    correct_mask = df["baseline_correct"].astype(bool)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df.loc[correct_mask, "baseline_score"], bins=40, alpha=0.7,
            color="#16A34A", density=True, label=f"Correct (n={correct_mask.sum()})")
    ax.hist(df.loc[~correct_mask, "baseline_score"], bins=40, alpha=0.7,
            color="#DC2626", density=True, label=f"Incorrect (n={(~correct_mask).sum()})")
    ax.axvline(0.5, color="black", linestyle="--", lw=1.5, label="tau=0.5")
    ax.set_xlabel("Comparator Score")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution: Correct vs Incorrect Predictions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "score_vs_correctness.png", dpi=150)
    plt.close(fig)

    # ---- Score bucket vs accuracy (does higher score = better prediction?) --
    df2 = df.copy()
    df2["score_bucket"] = (df2["baseline_score"] * 10).astype(int) / 10.0
    bucket_stats = df2.groupby("score_bucket").agg(
        accuracy=("baseline_correct", "mean"),
        count=("baseline_correct", "count"),
        mean_cer=("baseline_cer_sample", "mean"),
    ).reset_index()

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()
    ax1.plot(bucket_stats["score_bucket"], bucket_stats["accuracy"],
             "o-", color="#2563EB", lw=2, label="Exact Match Rate")
    ax1.plot(bucket_stats["score_bucket"], 1 - bucket_stats["mean_cer"],
             "s--", color="#7C3AED", lw=1.5, label="1 - CER")
    ax2.bar(bucket_stats["score_bucket"], bucket_stats["count"],
            width=0.08, alpha=0.3, color="#94A3B8")
    ax1.set_xlabel("Comparator Score Bucket")
    ax1.set_ylabel("Quality Metric")
    ax2.set_ylabel("Count")
    ax1.set_title("Score vs OCR Quality â€” Is the Score a Good Quality Signal?")
    ax1.set_ylim(0, 1)
    ax1.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(plots / "score_vs_quality.png", dpi=150)
    plt.close(fig)

    # ---- Delta score from feedback loop ------------------------------------
    if "final_score" in df.columns:
        delta = df["final_score"] - df["baseline_score"]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(delta, bins=40, color="#2563EB", alpha=0.8)
        ax.axvline(0, color="black", linestyle="--", lw=1.5)
        improved = (delta > 0.01).sum()
        degraded = (delta < -0.01).sum()
        ax.set_xlabel("Delta Score (final âˆ’ baseline)")
        ax.set_ylabel("Count")
        ax.set_title(f"Score Change from Feedback Loop\n"
                     f"Improved: {improved} | Degraded: {degraded} | Unchanged: {len(delta)-improved-degraded}")
        fig.tight_layout()
        fig.savefig(plots / "feedback_score_delta.png", dpi=150)
        plt.close(fig)

    # ---- Baseline vs final: metrics comparison bar chart -------------------
    if "final_exact" in df.columns:
        categories = ["Exact Match", "1 âˆ’ CER", "BLEU"]
        baseline_v = [
            df["baseline_correct"].mean(),
            1 - df["baseline_cer_sample"].mean(),
            df["baseline_bleu_sample"].mean() if "baseline_bleu_sample" in df else 0,
        ]
        final_v = [
            df["final_correct"].mean() if "final_correct" in df else baseline_v[0],
            1 - df["final_cer_sample"].mean() if "final_cer_sample" in df else baseline_v[1],
            df["final_bleu_sample"].mean() if "final_bleu_sample" in df else baseline_v[2],
        ]
        x = np.arange(len(categories))
        w = 0.35
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(x - w / 2, baseline_v, w, label="pix2tex baseline", color="#94A3B8")
        ax.bar(x + w / 2, final_v, w, label="+ Comparator", color="#2563EB")
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        ax.set_title("Baseline pix2tex vs Full Pipeline")
        ax.legend()
        for i, (b, f) in enumerate(zip(baseline_v, final_v)):
            delta = f - b
            sign = "+" if delta >= 0 else ""
            ax.text(i + w / 2, f + 0.01, f"{sign}{delta:.3f}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        fig.savefig(plots / "baseline_vs_pipeline.png", dpi=150)
        plt.close(fig)

    print(f"  Plots saved to {plots}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end pix2tex + comparator pipeline evaluation"
    )
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--pix2tex-model-dir", default="",
                        help="Path to pix2tex model dir (contains checkpoints/ and settings/). "
                             "Defaults to the pix2tex package install. "
                             "Use models/pix2tex_baseline if pix2tex is not installed.")
    parser.add_argument(
        "--comparator-checkpoint",
        default="results_comparator_hf_pix2tex_lora_main/render_compare_comparator_hf.pt",
    )
    parser.add_argument("--backbone-type", default="pix2tex_encoder",
                        choices=["mobilenet_v3_small", "pix2tex_encoder"])
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--extracted-root", default="data/formulae_extracted_full")
    parser.add_argument("--math-txt", default="data/math.txt")
    parser.add_argument("--test-samples", type=int, default=0,
                        help="Images to evaluate (0 = full test set)")
    parser.add_argument("--max-seq-len", type=int, default=192)
    parser.add_argument("--pix2tex-batch-size", type=int, default=26,
                        help="Batch size for pix2tex decode")
    parser.add_argument("--comparator-batch-size", type=int, default=64)
    parser.add_argument("--tau", type=float, default=0.5,
                        help="Comparator acceptance threshold")
    parser.add_argument("--max-iters", type=int, default=2,
                        help="Feedback loop iterations (0 = no loop)")
    parser.add_argument("--max-feedback-samples", type=int, default=0,
                        help="Random subset for feedback loop (0 = all below-tau samples)")
    parser.add_argument("--min-render-ink", type=int, default=0,
                        help="Skip render filter if 0 (fast); set >0 to pre-filter blank renders")
    parser.add_argument("--output-dir", default="results_m2_pipeline")
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--arch", default="baseline",
                        choices=["baseline", "v2a", "v2b", "v2c", "v2d"])
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
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"\n{'='*60}")
    print(f"Full Pipeline Test Evaluation")
    print(f"{'='*60}")
    print(f"Device  : {device}")
    print(f"Arch    : {args.arch}")
    print(f"Samples : {args.test_samples}")
    print(f"Output  : {out_dir}")

    # ---- Load models --------------------------------------------------------
    print(f"\n[1/5] Loading models...")
    ckpt_path = ws / args.comparator_checkpoint
    comparator = load_comparator(
        ckpt_path, args.backbone_type, args.lora_rank,
        args.lora_alpha, args.lora_dropout, device, arch=args.arch,
    )
    ocr = load_pix2tex(device, max_seq_len=args.max_seq_len)

    # ---- Build test split ---------------------------------------------------
    print(f"\n[2/5] Building test split...")
    formulas = load_formulas(ws / args.math_txt)
    extracted_root = ws / args.extracted_root
    renderer = FormulaRenderer(height=96, width=384, dpi=140, cache_max_entries=35000)

    rows = collect_split_samples(
        extracted_root=extracted_root,
        split="test",
        formulas=formulas,
        sample_count=args.test_samples,
        seed=args.seed,
    )
    if args.min_render_ink > 0:
        rows = filter_renderable_samples(rows, renderer, args.min_render_ink)
        print(f"  {len(rows)} test images after render filter (ink>={args.min_render_ink})")
    else:
        print(f"  {len(rows)} test images (render filter skipped)")

    ids = [r[0] for r in rows]
    paths = [r[1] for r in rows]
    refs = [r[2] for r in rows]

    # ---- pix2tex baseline decode (with checkpoint) --------------------------
    decode_ckpt = out_dir / "decode_checkpoint.json"
    print(f"\n[3/5] Running pix2tex baseline decode (batch={args.pix2tex_batch_size})...")
    if decode_ckpt.exists():
        print(f"  Loading cached decode from {decode_ckpt.name} (delete to re-run)")
        with open(decode_ckpt) as f:
            ckpt_data = json.load(f)
        baseline_preds = ckpt_data["preds"]
        decode_time = ckpt_data.get("decode_time_seconds", 0.0)
        print(f"  Loaded {len(baseline_preds)} cached predictions.")
    else:
        t0 = time.perf_counter()
        baseline_preds = decode_images(ocr, paths, args.pix2tex_batch_size, device)
        decode_time = time.perf_counter() - t0
        print(f"  Decoded {len(baseline_preds)} images in {decode_time:.1f}s "
              f"({decode_time / max(1, len(paths)):.2f}s/img)")
        with open(decode_ckpt, "w") as f:
            json.dump({"preds": baseline_preds, "decode_time_seconds": round(decode_time, 2)}, f)
        print(f"  Decode checkpoint saved to {decode_ckpt.name}")

    # Score baseline predictions
    print("  Scoring baseline predictions with comparator...")
    baseline_scores = score_batch(
        comparator, renderer, paths, baseline_preds, device,
        args.comparator_batch_size,
    )

    # Baseline NLP metrics
    baseline_metrics = compute_nlp_metrics(refs, baseline_preds)
    print(f"  Baseline  EM={baseline_metrics['exact_match']:.4f} "
          f"CER={baseline_metrics['cer']:.4f} BLEU={baseline_metrics['bleu']:.4f}")

    # Per-sample baseline metrics
    refs_ns = [remove_spaces(r) for r in refs]
    preds_ns_base = [remove_spaces(p) for p in baseline_preds]
    baseline_correct = [int(r == p) for r, p in zip(refs_ns, preds_ns_base)]
    baseline_cer_sample = [float(cer([r], [p])) for r, p in zip(refs_ns, preds_ns_base)]

    # ---- Feedback loop ------------------------------------------------------
    final_preds = list(baseline_preds)
    final_scores = list(baseline_scores)
    n_feedback_improved = 0

    if args.max_iters > 0:
        # Optionally restrict feedback loop to a random subset of below-tau samples
        feedback_paths = paths
        feedback_indices = None
        if args.max_feedback_samples > 0:
            import random as _random
            below_tau = [i for i, s in enumerate(baseline_scores) if s < args.tau]
            if len(below_tau) > args.max_feedback_samples:
                _random.seed(args.seed)
                feedback_indices = _random.sample(below_tau, args.max_feedback_samples)
                feedback_indices_set = set(feedback_indices)
                # Build subset lists aligned by position
                feedback_paths = [paths[i] for i in feedback_indices]
                _fb_preds = [baseline_preds[i] for i in feedback_indices]
                _fb_scores = [baseline_scores[i] for i in feedback_indices]
                print(f"\n[4/5] Feedback loop on {len(feedback_indices)}/{len(below_tau)} "
                      f"below-tau samples (--max-feedback-samples={args.max_feedback_samples})")
            else:
                feedback_indices = None
        if feedback_indices is None:
            print(f"\n[4/5] Running feedback loop (max_iters={args.max_iters}, tau={args.tau})...")

        t1 = time.perf_counter()
        if feedback_indices is not None:
            _fb_preds_out, _fb_scores_out, n_feedback_improved = feedback_loop(
                ocr=ocr, model=comparator, renderer=renderer,
                paths=feedback_paths, best_preds=_fb_preds, best_scores=_fb_scores,
                tau=args.tau, max_iters=args.max_iters,
                batch_size=args.pix2tex_batch_size, device=device,
            )
            final_preds = list(baseline_preds)
            final_scores = list(baseline_scores)
            for local_i, orig_i in enumerate(feedback_indices):
                final_preds[orig_i] = _fb_preds_out[local_i]
                final_scores[orig_i] = _fb_scores_out[local_i]
        else:
            final_preds, final_scores, n_feedback_improved = feedback_loop(
                ocr=ocr, model=comparator, renderer=renderer,
                paths=paths, best_preds=baseline_preds, best_scores=baseline_scores,
                tau=args.tau, max_iters=args.max_iters,
                batch_size=args.pix2tex_batch_size, device=device,
            )
        loop_time = time.perf_counter() - t1
        print(f"  Feedback loop done in {loop_time:.1f}s, "
              f"{n_feedback_improved} samples crossed tau threshold")
    else:
        print(f"\n[4/5] Skipping feedback loop (--max-iters=0)")

    # Final NLP metrics
    final_metrics = compute_nlp_metrics(refs, final_preds)
    preds_ns_final = [remove_spaces(p) for p in final_preds]
    final_correct = [int(r == p) for r, p in zip(refs_ns, preds_ns_final)]
    final_cer_sample = [float(cer([r], [p])) for r, p in zip(refs_ns, preds_ns_final)]

    # ---- Build output DataFrame + plots -------------------------------------
    print(f"\n[5/5] Saving results and generating plots...")
    df = pd.DataFrame({
        "id": ids,
        "reference": refs,
        "baseline_pred": baseline_preds,
        "final_pred": final_preds,
        "baseline_score": baseline_scores,
        "final_score": final_scores,
        "baseline_correct": baseline_correct,
        "final_correct": final_correct,
        "baseline_cer_sample": baseline_cer_sample,
        "final_cer_sample": final_cer_sample,
        "formula_len": [len(r) for r in refs],
        "score_improved": [int(fs > bs + 0.01) for bs, fs in zip(baseline_scores, final_scores)],
    })
    df.to_csv(out_dir / "pipeline_predictions.csv", index=False)

    metrics_out = {
        "samples": len(rows),
        "device": device,
        "arch": args.arch,
        "tau": args.tau,
        "max_iters": args.max_iters,
        "feedback_improved": int(n_feedback_improved),
        "accepted_baseline": int(sum(s >= args.tau for s in baseline_scores)),
        "accepted_final": int(sum(s >= args.tau for s in final_scores)),
        "baseline": baseline_metrics,
        "final": final_metrics,
        "delta": {
            k: round(final_metrics[k] - baseline_metrics[k], 6)
            for k in baseline_metrics
        },
        "decode_time_seconds": round(decode_time, 2),
        "mean_baseline_score": round(float(np.mean(baseline_scores)), 4),
        "mean_final_score": round(float(np.mean(final_scores)), 4),
        "score_vs_exact_pearson_r": round(
            float(np.corrcoef(baseline_scores, baseline_correct)[0, 1]), 4
        ),
    }
    with open(out_dir / "pipeline_metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    plot_pipeline(df, out_dir)

    # ---- Print summary ------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"FULL PIPELINE TEST RESULTS ({len(rows)} samples)")
    print(f"{'='*60}")
    print(f"{'Metric':<22} {'Baseline':>10} {'Pipeline':>10} {'Delta':>8}")
    print(f"{'-'*50}")
    for k in ["exact_match", "cer", "bleu"]:
        b = baseline_metrics[k]
        f = final_metrics[k]
        d = f - b
        sign = "+" if d >= 0 else ""
        print(f"  {k:<20} {b:>10.4f} {f:>10.4f} {sign+str(round(d,4)):>8}")

    accepted_b = sum(s >= args.tau for s in baseline_scores)
    accepted_f = sum(s >= args.tau for s in final_scores)
    print(f"\n  Accepted by comparator (tau={args.tau}):")
    print(f"    Baseline : {accepted_b}/{len(rows)} ({accepted_b/len(rows):.1%})")
    print(f"    After loop: {accepted_f}/{len(rows)} ({accepted_f/len(rows):.1%})")
    print(f"\n  Score <-> Correctness correlation (Pearson r): "
          f"{metrics_out['score_vs_exact_pearson_r']:.4f}")
    print(f"\n  Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
