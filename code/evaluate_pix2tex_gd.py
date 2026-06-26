import argparse
import hashlib
import io
import json
import os
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from jiwer import cer
from PIL import Image
from pix2tex.cli import LatexOCR, minmax_size, pad, test_transform
from pix2tex.utils import post_process, token2str
from sacrebleu import corpus_bleu
from tqdm.auto import tqdm


def remove_spaces(text: str) -> str:
    return re.sub(r"\s+", "", str(text) if text is not None else "")


def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split_entries(formulae_zip: Path, split: str):
    with zipfile.ZipFile(formulae_zip, "r") as zf:
        names = [
            n for n in zf.namelist() if n.startswith(f"{split}/") and n.endswith(".png")
        ]
    return sorted(names)


def load_split_entries_from_dir(images_root: Path):
    return sorted(str(p) for p in images_root.glob("*.png"))


def extract_split_to_dir(formulae_zip: Path, split: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(formulae_zip, "r") as zf:
        members = [
            n for n in zf.namelist() if n.startswith(f"{split}/") and n.endswith(".png")
        ]
        for name in tqdm(members, desc=f"extract {split}", unit="img"):
            target = out_dir / Path(name).name
            if target.exists():
                continue
            with zf.open(name, "r") as src, target.open("wb") as dst:
                dst.write(src.read())
    return load_split_entries_from_dir(out_dir)


def sort_entries_by_width(entries, cache_path: Path):
    width_map = {}
    if cache_path.exists():
        try:
            width_map = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            width_map = {}

    updated = False
    for p in entries:
        key = str(Path(p).name)
        if key in width_map:
            continue
        try:
            with Image.open(p) as im:
                width_map[key] = int(im.size[0])
                updated = True
        except Exception:
            width_map[key] = 0
            updated = True

    if updated:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(width_map), encoding="utf-8")

    return sorted(entries, key=lambda x: width_map.get(str(Path(x).name), 0))


def load_labels(math_txt: Path):
    return math_txt.read_text(encoding="utf-8").splitlines()


def build_input_tensor(ocr: LatexOCR, img, resize: bool = True) -> torch.Tensor:
    img = cast(Any, img)
    img = minmax_size(pad(img), ocr.args.max_dimensions, ocr.args.min_dimensions)  # type: ignore[arg-type]

    if (ocr.image_resizer is not None and not ocr.args.no_resize) and resize:
        with torch.no_grad():
            input_image = img.convert("RGB").copy()  # type: ignore[attr-defined]
            ratio, current_w, current_h = 1.0, input_image.size[0], input_image.size[1]
            for _ in range(10):
                current_h = int(current_h * ratio)
                resized = input_image.resize(
                    (current_w, current_h),
                    (
                        Image.Resampling.BILINEAR
                        if ratio > 1
                        else Image.Resampling.LANCZOS
                    ),
                )
                img = pad(minmax_size(resized, ocr.args.max_dimensions, ocr.args.min_dimensions))  # type: ignore[arg-type]
                t = test_transform(image=np.array(img.convert("RGB")))["image"][
                    :1
                ].unsqueeze(0)
                target_w = (
                    ocr.image_resizer(t.to(ocr.args.device)).argmax(-1).item() + 1
                ) * 32
                if target_w == img.size[0]:  # type: ignore[attr-defined]
                    break
                ratio = target_w / img.size[0]  # type: ignore[attr-defined]
    else:
        np_img = np.array(pad(img).convert("RGB"))  # type: ignore[arg-type]
        t = test_transform(image=np_img)["image"][:1].unsqueeze(0)

    return t


def collate_with_padding(tensors):
    if not tensors:
        return None
    max_h = max(t.shape[2] for t in tensors)
    max_w = max(t.shape[3] for t in tensors)
    padded = []
    for t in tensors:
        pad_h = max_h - t.shape[2]
        pad_w = max_w - t.shape[3]
        padded.append(F.pad(t, (0, pad_w, 0, pad_h), mode="constant", value=1.0))
    return torch.cat(padded, dim=0)


def load_and_prepare_sample(name, labels, use_extracted, zf, model, use_resizer):
    file_id = int(Path(name).stem)
    if file_id >= len(labels):
        return None, None, None, {"name": name, "reason": "label_index_out_of_range"}

    try:
        if use_extracted:
            img = Image.open(name).convert("RGB")
        else:
            img_bytes = zf.read(name)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        t = build_input_tensor(model, img, resize=use_resizer)
        return file_id, labels[file_id], t, None
    except Exception as e:
        return None, None, None, {"name": name, "reason": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pix2tex on Google Drive formulae.zip split"
    )
    parser.add_argument(
        "--workspace", type=str, default="."
    )
    parser.add_argument("--gd-dir", type=str, default="data")
    parser.add_argument("--pix2tex-model-dir", type=str, default="",
                        help="Path to pix2tex model dir (contains checkpoints/ and settings/). "
                             "Defaults to the pix2tex package install. "
                             "Use models/pix2tex_baseline if pix2tex is not installed.")
    parser.add_argument(
        "--split", type=str, default="test", choices=["train", "val", "test"]
    )
    parser.add_argument("--output-dir", type=str, default="results_pix2tex_gd")
    parser.add_argument("--limit", type=int, default=0, help="0 means full split")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Initial batch size (recommended 16 or higher)",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=32,
        help="Upper bound for adaptive batch growth",
    )
    parser.add_argument("--min-batch-size", type=int, default=1)
    parser.add_argument(
        "--no-adaptive-batch",
        action="store_true",
        help="Keep batch size fixed (no dynamic growth/shrink)",
    )
    parser.add_argument(
        "--fixed-batch-size",
        type=int,
        default=0,
        help="If >0, overrides --batch-size with a constant batch size",
    )
    parser.add_argument("--disable-resizer", action="store_true")
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cuda", "cpu"]
    )
    parser.add_argument("--no-mixed-precision", action="store_true")
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=256,
        help="Lower for faster decoding, higher for better long-formula accuracy",
    )
    parser.add_argument(
        "--prefer-extracted",
        action="store_true",
        help="Use extracted split images from disk instead of ZIP reads",
    )
    parser.add_argument(
        "--prepare-extracted",
        action="store_true",
        help="Extract the selected split from ZIP before evaluation",
    )
    parser.add_argument(
        "--extracted-root", type=str, default="data/formulae_extracted_full"
    )
    parser.add_argument(
        "--bucket-by-width",
        action="store_true",
        help="Sort extracted images by width to reduce padding waste in batches",
    )
    parser.add_argument(
        "--loader-workers",
        type=int,
        default=6,
        help="Parallel workers for CPU image load/preprocess in extracted mode",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1000,
        help="Write progress checkpoint every N processed entries",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous progress checkpoint in output dir",
    )
    parser.add_argument(
        "--ignore-resume-signature",
        action="store_true",
        help="Allow resume even when checkpoint config signature does not match",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.25,
        help="Decoding temperature passed to pix2tex generate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling",
    )
    parser.add_argument(
        "--reported-case-preset",
        action="store_true",
        help=(
            "Apply strict settings closer to repo-reported evaluation "
            "(resizer on, seq len 512, temp 0.333, fixed batch 10, no AMP, no adaptive batch, no bucketing)"
        ),
    )
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

    if args.reported_case_preset:
        args.disable_resizer = False
        args.max_seq_len = 512
        args.temperature = 0.333
        args.fixed_batch_size = 10
        args.no_adaptive_batch = True
        args.no_mixed_precision = True
        args.bucket_by_width = False

    seed_everything(int(args.seed))

    workspace = Path(args.workspace)
    gd_dir = workspace / args.gd_dir
    out_dir = workspace / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Keep model/cache artifacts inside workspace as requested.
    cache_root = workspace / "model_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_root / "hf")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_root / "hf" / "transformers")
    os.environ["XDG_CACHE_HOME"] = str(cache_root / "xdg")

    formulae_zip = gd_dir / "formulae.zip"
    math_txt = gd_dir / "math.txt"

    if not formulae_zip.exists() or not math_txt.exists():
        raise FileNotFoundError("Expected data/formulae.zip and data/math.txt")

    labels = load_labels(math_txt)
    extracted_split_dir = workspace / args.extracted_root / args.split
    use_extracted = False

    if args.prepare_extracted:
        entries = extract_split_to_dir(formulae_zip, args.split, extracted_split_dir)
        use_extracted = True
    elif args.prefer_extracted and extracted_split_dir.exists():
        entries = load_split_entries_from_dir(extracted_split_dir)
        use_extracted = True
    else:
        entries = load_split_entries(formulae_zip, args.split)

    if args.limit > 0:
        entries = entries[: args.limit]

    if use_extracted and args.bucket_by_width:
        width_cache = extracted_split_dir / "_width_cache.json"
        entries = sort_entries_by_width(entries, width_cache)

    print(f"Loaded labels: {len(labels)}")
    print(f"Evaluating split '{args.split}' with {len(entries)} images")

    model = LatexOCR()
    use_resizer = not args.disable_resizer
    if args.device == "auto":
        run_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        run_device = args.device

    # pix2tex defaults to CPU; move model explicitly for GPU throughput.
    model.args.device = run_device
    model.args.max_seq_len = int(args.max_seq_len)
    model.args.temperature = float(args.temperature)
    model.model = model.model.to(run_device)
    if model.image_resizer is not None:
        model.image_resizer = model.image_resizer.to(run_device)

    use_amp = run_device == "cuda" and not args.no_mixed_precision
    amp_dtype = (
        torch.bfloat16
        if (run_device == "cuda" and torch.cuda.is_bf16_supported())
        else torch.float16
    )
    print(
        f"input_mode={'extracted_dir' if use_extracted else 'zip_stream'} | max_seq_len={model.args.max_seq_len} | temp={float(model.args.temperature):.3f} | seed={args.seed} | device={run_device}"
    )

    refs_raw = []
    preds_raw = []
    ids = []
    processed_names = []
    failed = []
    oom_retries = 0
    initial_batch_size = (
        max(args.fixed_batch_size, 1)
        if args.fixed_batch_size > 0
        else max(args.batch_size, 1)
    )
    current_batch_size = initial_batch_size
    max_batch_size = max(args.max_batch_size, current_batch_size)
    stable_steps = 0
    cpu_prepare_seconds = 0.0
    gpu_generate_seconds = 0.0

    if run_device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    progress_path = out_dir / f"pix2tex_gd_{args.split}_progress.json"
    pred_ckpt_path = out_dir / f"pix2tex_gd_{args.split}_predictions_checkpoint.csv"
    failed_ckpt_path = out_dir / f"pix2tex_gd_{args.split}_failed_checkpoint.json"
    checkpoint_saved_count = 0
    # Signature covers evaluation-semantic settings only.
    # Performance knobs (batching/AMP/worker count) are intentionally excluded
    # so a run can resume faster without forcing a restart.
    run_signature = {
        "split": args.split,
        "input_mode": "extracted_dir" if use_extracted else "zip_stream",
        "disable_resizer": bool(args.disable_resizer),
        "max_seq_len": int(args.max_seq_len),
        "temperature": float(args.temperature),
    }
    run_signature_hash = hashlib.sha256(
        json.dumps(run_signature, sort_keys=True).encode("utf-8")
    ).hexdigest()

    def save_checkpoint(current_idx: int):
        nonlocal checkpoint_saved_count
        ckpt = {
            "idx": int(current_idx),
            "processed_count": len(ids),
            "failed_count": len(failed),
            "runtime_seconds": time.perf_counter() - t0,
            "device": run_device,
            "batch_size": current_batch_size,
            "oom_retries": oom_retries,
            "run_signature": run_signature,
            "run_signature_hash": run_signature_hash,
        }
        progress_path.write_text(json.dumps(ckpt, indent=2), encoding="utf-8")

        # Append only newly processed rows to avoid O(n) full rewrites each checkpoint.
        if len(ids) > checkpoint_saved_count:
            delta_df = pd.DataFrame(
                {
                    "name": processed_names[checkpoint_saved_count : len(ids)],
                    "id": ids[checkpoint_saved_count : len(ids)],
                    "reference": refs_raw[checkpoint_saved_count : len(ids)],
                    "prediction": preds_raw[checkpoint_saved_count : len(ids)],
                }
            )
            if pred_ckpt_path.exists() and checkpoint_saved_count > 0:
                delta_df.to_csv(pred_ckpt_path, mode="a", header=False, index=False)
            else:
                delta_df.to_csv(pred_ckpt_path, index=False)
            checkpoint_saved_count = len(ids)

        failed_ckpt_path.write_text(json.dumps(failed, indent=2), encoding="utf-8")

    start_idx = 0
    if args.resume and progress_path.exists() and pred_ckpt_path.exists():
        try:
            state = json.loads(progress_path.read_text(encoding="utf-8"))
            state_sig = state.get("run_signature_hash")
            if state_sig and state_sig != run_signature_hash:
                if not args.ignore_resume_signature:
                    raise RuntimeError(
                        "Checkpoint config mismatch. Use --ignore-resume-signature to force resume."
                    )
                print("Warning: checkpoint signature mismatch ignored by user request")
            start_idx = int(state.get("idx", 0))
            pred_ckpt = pd.read_csv(pred_ckpt_path).fillna("")
            if len(pred_ckpt) > 0:
                processed_names = pred_ckpt["name"].astype(str).tolist()
                ids = pred_ckpt["id"].astype(int).tolist()
                refs_raw = pred_ckpt["reference"].astype(str).tolist()
                preds_raw = pred_ckpt["prediction"].astype(str).tolist()
                checkpoint_saved_count = len(ids)
            if failed_ckpt_path.exists():
                failed = json.loads(failed_ckpt_path.read_text(encoding="utf-8"))
            print(
                f"Resuming from idx={start_idx}, restored {len(ids)} predictions and {len(failed)} failures"
            )
        except Exception as e:
            print(f"Resume requested but checkpoint restore failed: {e}")
            start_idx = 0

    with zipfile.ZipFile(formulae_zip, "r") as zf:
        idx = start_idx
        next_checkpoint = (
            ((start_idx // args.checkpoint_every) + 1) * args.checkpoint_every
            if args.checkpoint_every > 0
            else 0
        )
        pbar = tqdm(total=len(entries), desc=f"pix2tex {args.split}", unit="img")
        if start_idx > 0:
            pbar.update(start_idx)
        while idx < len(entries):
            bs = min(current_batch_size, len(entries) - idx)
            batch_names = entries[idx : idx + bs]

            batch_ids = []
            batch_refs = []
            batch_tensors = []
            prep_t0 = time.perf_counter()
            if use_extracted and args.loader_workers > 1:
                with ThreadPoolExecutor(max_workers=args.loader_workers) as ex:
                    results = list(
                        ex.map(
                            lambda n: load_and_prepare_sample(
                                n, labels, use_extracted, zf, model, use_resizer
                            ),
                            batch_names,
                        )
                    )
            else:
                results = [
                    load_and_prepare_sample(
                        name, labels, use_extracted, zf, model, use_resizer
                    )
                    for name in batch_names
                ]

            batch_success_names = []
            for name, result in zip(batch_names, results):
                file_id, ref, t, err = result
                if err is not None:
                    failed.append(err)
                    continue
                batch_ids.append(file_id)
                batch_refs.append(ref)
                batch_tensors.append(t)
                batch_success_names.append(name)
            cpu_prepare_seconds += time.perf_counter() - prep_t0

            if not batch_tensors:
                idx += bs
                pbar.update(bs)
                continue

            im = collate_with_padding(batch_tensors)
            if im is None:
                idx += bs
                pbar.update(bs)
                continue
            try:
                gen_t0 = time.perf_counter()
                with torch.inference_mode():
                    temp = float(model.args.get("temperature", 0.25) or 0.25)
                    batch_input = im.to(model.args.device, non_blocking=True)
                    if use_amp:
                        with torch.autocast(device_type="cuda", dtype=amp_dtype):
                            dec = model.model.generate(
                                batch_input,
                                temperature=temp,
                            )
                    else:
                        dec = model.model.generate(
                            batch_input,
                            temperature=temp,
                        )
                gpu_generate_seconds += time.perf_counter() - gen_t0
                batch_preds = [post_process(x) for x in token2str(dec, model.tokenizer)]
            except RuntimeError as e:
                if (
                    "out of memory" in str(e).lower()
                    and (not args.no_adaptive_batch)
                    and bs > args.min_batch_size
                ):
                    oom_retries += 1
                    current_batch_size = max(args.min_batch_size, bs // 2)
                    stable_steps = 0
                    if run_device == "cuda":
                        torch.cuda.empty_cache()
                    pbar.set_postfix(
                        bs=current_batch_size,
                        oom=oom_retries,
                        eval=len(refs_raw),
                        failed=len(failed),
                    )
                    continue
                if "out of memory" in str(e).lower():
                    oom_retries += 1
                    for name in batch_names:
                        failed.append({"name": name, "reason": "gpu_oom"})
                    idx += bs
                    stable_steps = 0
                    if run_device == "cuda":
                        torch.cuda.empty_cache()
                    pbar.update(bs)
                    pbar.set_postfix(
                        bs=current_batch_size,
                        oom=oom_retries,
                        eval=len(refs_raw),
                        failed=len(failed),
                    )
                    continue
                raise

            ids.extend(batch_ids)
            refs_raw.extend(batch_refs)
            preds_raw.extend(batch_preds)
            processed_names.extend(batch_success_names[: len(batch_preds)])
            idx += bs
            stable_steps += 1

            if (
                (not args.no_adaptive_batch)
                and stable_steps >= 20
                and current_batch_size < max_batch_size
            ):
                current_batch_size = min(max_batch_size, current_batch_size + 2)
                stable_steps = 0

            pbar.update(bs)
            pbar.set_postfix(
                bs=current_batch_size,
                oom=oom_retries,
                eval=len(refs_raw),
                failed=len(failed),
            )

            if args.checkpoint_every > 0 and idx >= next_checkpoint:
                save_checkpoint(idx)
                while next_checkpoint <= idx:
                    next_checkpoint += args.checkpoint_every

        pbar.close()

    save_checkpoint(len(entries))

    t1 = time.perf_counter()

    refs_nospace = [remove_spaces(x) for x in refs_raw]
    preds_nospace = [remove_spaces(x) for x in preds_raw]

    tokenizer = model.tokenizer
    refs_tok = [tokenizer.tokenize(x) for x in refs_nospace]
    preds_tok = [tokenizer.tokenize(x) for x in preds_nospace]
    refs_tok_text = [" ".join(x) for x in refs_tok]
    preds_tok_text = [" ".join(x) for x in preds_tok]

    n = len(refs_raw)
    exact_raw = sum(int(p == r) for p, r in zip(preds_raw, refs_raw)) / max(n, 1)
    exact_nospace = sum(int(p == r) for p, r in zip(preds_nospace, refs_nospace)) / max(
        n, 1
    )
    exact_tok = sum(int(p == r) for p, r in zip(preds_tok, refs_tok)) / max(n, 1)

    metrics = {
        "model": "lukas-blecher/LaTeX-OCR (pix2tex)",
        "dataset_source": "Google Drive folder 13CA4vAmOmD_I_dSbvLp-Lf0s6KiaNfuO",
        "data_files": {
            "formulae_zip": str(formulae_zip),
            "math_txt": str(math_txt),
        },
        "split": args.split,
        "split_images_requested": len(entries),
        "samples_evaluated": n,
        "failed_samples": len(failed),
        "limit": args.limit,
        "runtime": {
            "device": run_device,
            "total_inference_seconds": t1 - t0,
            "samples_per_second": n / max(t1 - t0, 1e-8),
            "cpu_prepare_seconds": cpu_prepare_seconds,
            "gpu_generate_seconds": gpu_generate_seconds,
            "cpu_prepare_pct": 100.0 * cpu_prepare_seconds / max(t1 - t0, 1e-8),
            "gpu_generate_pct": 100.0 * gpu_generate_seconds / max(t1 - t0, 1e-8),
            "peak_gpu_memory_mb": (
                float(torch.cuda.max_memory_allocated() / (1024**2))
                if run_device == "cuda"
                else None
            ),
            "initial_batch_size": initial_batch_size,
            "final_batch_size": current_batch_size,
            "max_batch_size": max_batch_size,
            "oom_retries": oom_retries,
            "mixed_precision": use_amp,
            "amp_dtype": str(amp_dtype) if use_amp else None,
            "adaptive_batch": not args.no_adaptive_batch,
            "fixed_batch_size": initial_batch_size if args.no_adaptive_batch else None,
        },
        "evaluation_protocol": {
            "label_mapping": "label = math.txt[int(filename_stem)]",
            "text_preprocessing": "remove all spaces for no-space metrics",
            "tokenization": "pix2tex tokenizer on no-space strings",
            "input_mode": "extracted_dir" if use_extracted else "zip_stream",
            "max_seq_len": int(model.args.max_seq_len),
            "temperature": float(model.args.temperature),
            "seed": int(args.seed),
            "reported_case_preset": bool(args.reported_case_preset),
            "bucket_by_width": bool(use_extracted and args.bucket_by_width),
        },
        "metrics": {
            "exact_match_raw": float(exact_raw),
            "exact_match_no_space": float(exact_nospace),
            "exact_match_tokenized": float(exact_tok),
            "cer_no_space": float(cer(refs_nospace, preds_nospace)) if n else 0.0,
            "bleu_no_space": (
                float(
                    corpus_bleu(preds_nospace, [refs_nospace], force=True).score / 100.0
                )
                if n
                else 0.0
            ),
            "cer_tokenized": float(cer(refs_tok_text, preds_tok_text)) if n else 0.0,
            "bleu_tokenized": (
                float(
                    corpus_bleu(preds_tok_text, [refs_tok_text], force=True).score
                    / 100.0
                )
                if n
                else 0.0
            ),
        },
        "repo_reported_reference": {
            "bleu": 0.88,
            "cer": 0.10,
            "token_exact_match": 0.60,
        },
    }

    pred_df = pd.DataFrame(
        {
            "id": ids,
            "reference": refs_raw,
            "prediction": preds_raw,
            "reference_no_space": refs_nospace,
            "prediction_no_space": preds_nospace,
        }
    )

    pred_path = out_dir / f"pix2tex_gd_{args.split}_predictions.csv"
    metrics_path = out_dir / f"pix2tex_gd_{args.split}_metrics.json"
    failed_path = out_dir / f"pix2tex_gd_{args.split}_failed.json"

    pred_df.to_csv(pred_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    failed_path.write_text(json.dumps(failed, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"Saved predictions: {pred_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved failures: {failed_path}")


if __name__ == "__main__":
    main()
