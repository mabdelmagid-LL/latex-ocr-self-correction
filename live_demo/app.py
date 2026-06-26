"""
live_demo/app.py — XAI Quality Gate Live Demo
Upload a formula image, watch pix2tex decode it, then see the XAI analysis step by step.
"""

from __future__ import annotations

import base64
import gc
import io
import json
import os
import sys
import tempfile
import uuid
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
PROJ_ROOT = ROOT.parent
XAI_PKG   = PROJ_ROOT / "project_xai" / "project_xai"
sys.path.insert(0, str(XAI_PKG))
sys.path.insert(0, str(PROJ_ROOT))

UPLOAD_DIR = ROOT / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

# ── lazy model loading ─────────────────────────────────────────────────────────
_ocr = None

def get_ocr():
    global _ocr
    if _ocr is None:
        from munch import Munch
        from pix2tex.cli import LatexOCR
        args = Munch({
            "config":      "settings/config.yaml",
            "checkpoint":  "checkpoints/weights.pth",
            "no_cuda":     not torch.cuda.is_available(),
            "no_resize":   True,
            "temperature": 0.333,
            "explain":     True,
            "gradcam":     False,
        })
        _ocr = LatexOCR(args)
    return _ocr


# ── helpers ───────────────────────────────────────────────────────────────────

def _tensor_to_b64(img_tensor: torch.Tensor, h: int, w: int) -> str:
    """Return a base64 PNG of an attention/grad map overlaid on a grayscale image."""
    import cv2
    from pix2tex.xai.trace import resize_token_map

    img = img_tensor.detach().cpu()
    if img.ndim == 3:
        img = img[0]
    arr = img.numpy()
    arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)
    base_rgb = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

    return base64.b64encode(cv2.imencode(".png", base_rgb)[1]).decode()


def _make_overlay_b64(image_tensor: torch.Tensor, map_1d: torch.Tensor,
                       image_hw, patch_size: int = 16,
                       colormap: int = None) -> str:
    import cv2
    from pix2tex.xai.trace import resize_token_map

    import cv2 as _cv2
    if colormap is None:
        colormap = _cv2.COLORMAP_JET

    img = image_tensor.detach().cpu()
    if img.ndim == 3:
        img = img[0]
    arr = img.numpy()
    arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)
    base_rgb = _cv2.cvtColor(arr, _cv2.COLOR_GRAY2BGR)

    h, w = image_hw
    resized = resize_token_map(map_1d, (h, w), patch_size=patch_size)
    heat = (resized.detach().cpu().numpy() * 255).astype(np.uint8)
    heat_color = _cv2.applyColorMap(heat, colormap)
    overlay = _cv2.addWeighted(base_rgb, 0.55, heat_color, 0.45, 0)

    _, buf = _cv2.imencode(".png", overlay)
    return base64.b64encode(buf).decode()


def _mean_diffuseness(cross_attn: torch.Tensor | None) -> float:
    from pix2tex.xai.trace import attention_diffuseness
    if not torch.is_tensor(cross_attn) or cross_attn.numel() == 0:
        return 1.0
    steps = cross_attn.shape[1] if cross_attn.ndim >= 3 else 0
    vals = [attention_diffuseness(cross_attn[0, t]).item() for t in range(steps)]
    return float(sum(vals) / max(len(vals), 1))


def _build_overlays(image_tensor, token_maps, token_ids, tokenizer, patch_size,
                    max_tokens=6, colormap=None):
    import cv2
    if not torch.is_tensor(token_maps) or token_maps.numel() == 0:
        return []
    maps = token_maps[0] if token_maps.ndim == 3 else token_maps
    h, w = image_tensor.shape[-2], image_tensor.shape[-1]

    overlays = []
    for i in range(min(max_tokens, maps.shape[0])):
        tok_id = token_ids[i] if i < len(token_ids) else 0
        tok = tokenizer.convert_ids_to_tokens(int(tok_id))
        tok = (tok or "").replace("Ġ", " ").strip()
        b64 = _make_overlay_b64(image_tensor[0], maps[i], (h, w),
                                 patch_size=patch_size,
                                 colormap=colormap or cv2.COLORMAP_JET)
        overlays.append({"token": tok, "index": i, "img": b64})
    return overlays


# ── LaTeX renderer ───────────────────────────────────────────────────────────

def render_latex_b64(latex: str) -> str | None:
    """Render a LaTeX string to a base64 PNG using matplotlib."""
    try:
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.patch.set_facecolor("#1a1f2e")
        text = fig.text(
            0, 0, f"${latex}$",
            fontsize=22,
            color="#e2e8f0",
            family="serif",
        )
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150,
                    bbox_inches=text.get_window_extent(
                        renderer=fig.canvas.get_renderer()
                    ).transformed(fig.dpi_scale_trans.inverted()).expanded(1.15, 1.4),
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()
    except Exception:
        plt.close("all")
        # Fallback: simple centered text
        try:
            fig, ax = plt.subplots(figsize=(8, 1.5))
            ax.axis("off")
            ax.text(0.5, 0.5, f"${latex}$", ha="center", va="center",
                    fontsize=20, color="#e2e8f0", family="serif",
                    transform=ax.transAxes)
            fig.patch.set_facecolor("#1a1f2e")
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight",
                        facecolor=fig.get_facecolor(), dpi=150)
            plt.close(fig)
            buf.seek(0)
            return base64.b64encode(buf.read()).decode()
        except Exception:
            plt.close("all")
            return None


# ── quality gate ──────────────────────────────────────────────────────────────

PRESETS = {
    "balanced":        {"min_confidence": 0.995, "max_diffuseness": 0.81, "min_consistency": 0.05},
    "strict_conf":     {"min_confidence": 0.70,  "max_diffuseness": 0.90, "min_consistency": 0.10},
    "diff_sensitive":  {"min_confidence": 0.50,  "max_diffuseness": 0.75, "min_consistency": 0.12},
    "lenient":         {"min_confidence": 0.40,  "max_diffuseness": 0.95, "min_consistency": 0.05},
    "cons_sensitive":  {"min_confidence": 0.995, "max_diffuseness": 0.81, "min_consistency": 0.06},
}


def quality_gate_check(attn_cons: float, diffuseness: float, preset_name: str = "balanced") -> dict:
    p = PRESETS.get(preset_name, PRESETS["balanced"])
    fails = []
    if diffuseness > p["max_diffuseness"]:
        fails.append(f"diffuseness {diffuseness:.3f} > {p['max_diffuseness']}")
    if attn_cons < p["min_consistency"]:
        fails.append(f"attn_consistency {attn_cons:.3f} < {p['min_consistency']}")
    passed = len(fails) == 0
    return {"passed": passed, "fails": fails, "preset": preset_name, "thresholds": p}


# ── main analysis endpoint ─────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", presets=list(PRESETS.keys()))


@app.route("/analyze", methods=["POST"])
def analyze():
    import cv2

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    preset = request.form.get("preset", "balanced")
    max_iters = int(request.form.get("max_iters", 2))
    do_gradcam = request.form.get("gradcam", "false").lower() == "true"

    # Save upload
    uid = uuid.uuid4().hex[:8]
    ext = Path(file.filename).suffix or ".png"
    upload_path = UPLOAD_DIR / f"{uid}{ext}"
    file.save(str(upload_path))

    steps = []

    try:
        from munch import Munch
        from pix2tex.cli import minmax_size
        from pix2tex.dataset.transforms import test_transform
        from pix2tex.utils import pad, post_process, token2str
        from pix2tex.xai.consistency import CRITICAL_TOKEN_KEYS, attribution_consistency_score
        from pix2tex.xai.trace import attention_diffuseness

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        ocr = get_ocr()

        # ── Step 1: load & preprocess image ───────────────────────────────────
        pil_img = Image.open(upload_path)
        pil_img = minmax_size(pad(pil_img), ocr.args.max_dimensions, ocr.args.min_dimensions)
        arr = np.array(pad(pil_img).convert("RGB"))
        t = test_transform(image=arr)["image"][:1].unsqueeze(0)
        im = t.to(ocr.args.device)
        h, w = im.shape[-2], im.shape[-1]
        patch_size = ocr.args.get("patch_size", 16)

        # Save preprocessed image as b64
        img_np = im[0, 0].detach().cpu().numpy()
        img_np = ((img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8) * 255).astype(np.uint8)
        _, buf = cv2.imencode(".png", img_np)
        input_b64 = base64.b64encode(buf).decode()

        steps.append({
            "step": 1,
            "title": "Input Image Preprocessed",
            "body": f"Image resized to {w}×{h} for pix2tex encoder.",
            "image": input_b64,
        })

        # ── Step 2: pix2tex decode ─────────────────────────────────────────────
        with torch.no_grad():
            trace = ocr.model.generate_with_trace(im, temperature=ocr.args.get("temperature", 0.333))

        tokens = trace["tokens"]
        token_ids = tokens[0].detach().cpu().tolist() if tokens.ndim == 2 else tokens.detach().cpu().tolist()
        latex_pred = post_process(token2str(tokens, ocr.tokenizer)[0])

        # Per-token confidence scores
        conf_tensor = trace.get("confidences", None)
        if torch.is_tensor(conf_tensor) and conf_tensor.numel() > 0:
            conf_list = (conf_tensor[0] if conf_tensor.ndim == 2 else conf_tensor).detach().cpu().tolist()
        else:
            conf_list = [1.0] * len(token_ids)

        # Build list of (token_string, confidence) pairs
        token_strings = [
            (ocr.tokenizer.convert_ids_to_tokens(int(tid)) or "").replace("Ġ", " ")
            for tid in token_ids
        ]
        token_conf_pairs = [
            {"token": tok, "conf": round(float(c), 4)}
            for tok, c in zip(token_strings, conf_list)
        ]
        conf_threshold = 0.90

        steps.append({
            "step": 2,
            "title": "pix2tex Decoding",
            "body": f"Decoded <strong>{len(token_ids)}</strong> tokens. "
                    f"Tokens below {int(conf_threshold*100)}% confidence are highlighted.",
            "latex": latex_pred,
            "token_conf_pairs": token_conf_pairs,
            "conf_threshold": conf_threshold,
        })

        # ── Step 3: attention consistency + diffuseness ────────────────────────
        cross_attn = trace.get("cross_attentions", None)

        attn_cons = attribution_consistency_score(
            image_tensor=im[0].detach().cpu(),
            token_ids=token_ids,
            token_maps=cross_attn,
            tokenizer=ocr.tokenizer,
            patch_size=patch_size,
            top_percent=0.15,
            critical_token_keys=CRITICAL_TOKEN_KEYS,
            overlap_mode="iou",
        )
        diffuseness = _mean_diffuseness(cross_attn)

        attn_overlays = _build_overlays(im, cross_attn, token_ids,
                                         ocr.tokenizer, patch_size,
                                         max_tokens=6,
                                         colormap=cv2.COLORMAP_JET)

        steps.append({
            "step": 3,
            "title": "Attention Consistency Analysis",
            "body": (
                f"Attention consistency (IoU across critical tokens): "
                f"<strong>{attn_cons:.4f}</strong><br>"
                f"Attention diffuseness (normalized entropy): "
                f"<strong>{diffuseness:.4f}</strong>"
            ),
            "overlays": attn_overlays,
            "scores": {"attn_consistency": round(attn_cons, 4),
                       "diffuseness": round(diffuseness, 4)},
        })

        # ── Step 4: optional Grad-CAM ──────────────────────────────────────────
        grad_cons = None
        if do_gradcam:
            from pix2tex.xai.gradcam import add_gradcam_to_trace
            trace = add_gradcam_to_trace(ocr.model, im, trace, max_tokens=8)
            grad_attr = trace.get("grad_attributions", None)
            grad_cons = attribution_consistency_score(
                image_tensor=im[0].detach().cpu(),
                token_ids=token_ids,
                token_maps=grad_attr,
                tokenizer=ocr.tokenizer,
                patch_size=patch_size,
                top_percent=0.15,
                critical_token_keys=CRITICAL_TOKEN_KEYS,
                overlap_mode="iou",
            )
            grad_overlays = _build_overlays(im, grad_attr, token_ids,
                                             ocr.tokenizer, patch_size,
                                             max_tokens=6,
                                             colormap=cv2.COLORMAP_TURBO)
            steps.append({
                "step": 4,
                "title": "Grad-CAM Attribution",
                "body": f"Grad-CAM consistency: <strong>{grad_cons:.4f}</strong>",
                "overlays": grad_overlays,
                "scores": {"gradcam_consistency": round(grad_cons, 4)},
            })

        # ── Step 5: quality gate check ─────────────────────────────────────────
        gate = quality_gate_check(attn_cons, diffuseness, preset)

        steps.append({
            "step": 5,
            "title": "Quality Gate Decision",
            "body": (
                f"Preset: <strong>{preset}</strong><br>"
                + (
                    "<span class='pass'>✔ PASSED — prediction accepted</span>"
                    if gate["passed"]
                    else "<span class='fail'>✘ FAILED — triggering re-decode</span><br>"
                         + "<br>".join(f"• {f}" for f in gate["fails"])
                )
            ),
            "gate": gate,
        })

        # ── Step 6: feedback loop (if gate failed) ─────────────────────────────
        final_latex = latex_pred
        redecode_steps = []

        if not gate["passed"] and max_iters > 0:
            for iteration in range(1, max_iters + 1):
                with torch.no_grad():
                    trace2 = ocr.model.generate_with_trace(im, temperature=ocr.args.get("temperature", 0.333) + 0.05 * iteration)

                tokens2 = trace2["tokens"]
                ids2 = tokens2[0].detach().cpu().tolist() if tokens2.ndim == 2 else tokens2.detach().cpu().tolist()
                latex2 = post_process(token2str(tokens2, ocr.tokenizer)[0])

                conf2_raw = trace2.get("confidences", None)
                if torch.is_tensor(conf2_raw) and conf2_raw.numel() > 0:
                    conf2_list = (conf2_raw[0] if conf2_raw.ndim == 2 else conf2_raw).detach().cpu().tolist()
                else:
                    conf2_list = [1.0] * len(ids2)

                tok_strings2 = [
                    (ocr.tokenizer.convert_ids_to_tokens(int(tid)) or "").replace("Ġ", " ")
                    for tid in ids2
                ]
                token_conf2 = [{"token": tok, "conf": round(float(c), 4)}
                               for tok, c in zip(tok_strings2, conf2_list)]

                cross2 = trace2.get("cross_attentions", None)
                cons2 = attribution_consistency_score(
                    image_tensor=im[0].detach().cpu(),
                    token_ids=ids2,
                    token_maps=cross2,
                    tokenizer=ocr.tokenizer,
                    patch_size=patch_size,
                    top_percent=0.15,
                    critical_token_keys=CRITICAL_TOKEN_KEYS,
                    overlap_mode="iou",
                )
                diff2 = _mean_diffuseness(cross2)
                gate2 = quality_gate_check(cons2, diff2, preset)

                redecode_steps.append({
                    "iteration": iteration,
                    "latex": latex2,
                    "attn_consistency": round(cons2, 4),
                    "diffuseness": round(diff2, 4),
                    "passed": gate2["passed"],
                    "token_conf_pairs": token_conf2,
                    "conf_threshold": conf_threshold,
                })

                if gate2["passed"]:
                    final_latex = latex2
                    break
                if iteration == max_iters:
                    # Keep best (highest consistency)
                    all_candidates = [{"latex": latex_pred, "cons": attn_cons}] + [
                        {"latex": r["latex"], "cons": r["attn_consistency"]} for r in redecode_steps
                    ]
                    best = max(all_candidates, key=lambda x: x["cons"])
                    final_latex = best["latex"]

            body = f"Ran <strong>{len(redecode_steps)}</strong> re-decode iteration(s)."
            if redecode_steps[-1]["passed"]:
                body += " <span class='pass'>Quality gate passed on retry.</span>"
            else:
                body += " <span class='warn'>Gate still failing — kept best candidate.</span>"

            steps.append({
                "step": 6,
                "title": "Re-decode Feedback Loop",
                "body": body,
                "redecode_steps": redecode_steps,
                "final_latex": final_latex,
            })

        return jsonify({"steps": steps, "final_latex": final_latex, "input_b64": input_b64})

    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        return jsonify({"error": str(exc), "traceback": tb}), 500
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@app.route("/static/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(str(UPLOAD_DIR), filename)


if __name__ == "__main__":
    app.run(debug=False, port=5000)
