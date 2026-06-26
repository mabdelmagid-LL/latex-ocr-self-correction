import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from jiwer import cer
from matplotlib.mathtext import MathTextParser
from PIL import Image, ImageFilter

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

from pix2tex.cli import (
    Munch,
    PreTrainedTokenizerFast,
    ResNetV2,
    StdConv2dSame,
    download_checkpoints,
    get_model,
    in_model_path,
    parse_args,
    yaml,
)
from pix2tex.utils import post_process, token2str
from sacrebleu import corpus_bleu
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


def _patch_pix2tex_model_dir(model_dir: str):
    """Redirect pix2tex's in_model_path() to a custom directory (e.g. models/pix2tex_baseline)."""
    import contextlib
    import pix2tex.cli as _cli
    import pix2tex.utils as _utils

    @contextlib.contextmanager
    def _patched():
        saved = os.getcwd()
        os.chdir(model_dir)
        try:
            yield
        finally:
            os.chdir(saved)

    _utils.in_model_path = _patched
    _cli.in_model_path = _patched
    global in_model_path
    in_model_path = _patched


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def remove_spaces(text: str) -> str:
    return re.sub(r"\s+", "", str(text) if text is not None else "")


@dataclass
class HardwareConfig:
    device: str
    vram_gb: float
    cpu_cores: int
    train_samples: int
    val_samples: int
    batch_size: int
    workers: int
    epochs: int


def detect_hardware(force_device: str = "auto") -> HardwareConfig:
    cuda_ok = torch.cuda.is_available()
    if force_device == "cuda":
        device = "cuda"
    elif force_device == "cpu":
        device = "cpu"
    else:
        device = "cuda" if cuda_ok else "cpu"

    vram_gb = 0.0
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        vram_gb = float(props.total_memory) / (1024**3)

    cpu_cores = os.cpu_count() or 4

    if device == "cuda" and vram_gb >= 10:
        return HardwareConfig(
            device, vram_gb, cpu_cores, 20000, 2400, 128, min(10, cpu_cores), 3
        )
    if device == "cuda" and vram_gb >= 6:
        return HardwareConfig(
            device, vram_gb, cpu_cores, 6000, 900, 48, min(6, cpu_cores), 2
        )
    if device == "cuda":
        return HardwareConfig(
            device, vram_gb, cpu_cores, 5000, 800, 32, min(6, cpu_cores), 2
        )
    return HardwareConfig(
        device, vram_gb, cpu_cores, 2500, 500, 24, min(6, cpu_cores), 2
    )


class FormulaRenderer:
    def __init__(
        self,
        height: int = 96,
        width: int = 384,
        dpi: int = 140,
        cache_max_entries: int = 20000,
    ):
        self.height = int(height)
        self.width = int(width)
        self.dpi = int(dpi)
        self.cache_max_entries = max(0, int(cache_max_entries))
        self.parser = MathTextParser("agg")
        self.cache: Dict[str, np.ndarray] = {}

    def _cache_set(self, key: str, value: np.ndarray) -> None:
        if self.cache_max_entries <= 0:
            return
        if key in self.cache:
            self.cache[key] = value
            return
        if len(self.cache) >= self.cache_max_entries:
            # Keep memory bounded when training with many synthetic negatives.
            self.cache.clear()
        self.cache[key] = value

    def render(self, latex: str) -> np.ndarray:
        key = str(latex)
        if key in self.cache:
            return self.cache[key]

        canvas = np.ones((self.height, self.width), dtype=np.float32)
        txt = key.strip()
        if not txt:
            self._cache_set(key, canvas)
            return canvas

        try:
            parsed = self.parser.parse(txt, dpi=self.dpi)
            arr = np.array(parsed.image, dtype=np.uint8)
            if arr.ndim != 2:
                arr = arr[..., 0]
            # mathtext image uses black=0, white=255. Keep white background as 1.0.
            fg = 1.0 - (arr.astype(np.float32) / 255.0)
            h, w = fg.shape
            if h == 0 or w == 0:
                self._cache_set(key, canvas)
                return canvas

            scale = min(self.height / max(h, 1), self.width / max(w, 1), 1.0)
            if scale < 1.0:
                nh = max(1, int(round(h * scale)))
                nw = max(1, int(round(w * scale)))
                fg = (
                    np.array(
                        Image.fromarray((fg * 255).astype(np.uint8)).resize(
                            (nw, nh), Image.Resampling.BILINEAR
                        ),
                        dtype=np.float32,
                    )
                    / 255.0
                )
                h, w = fg.shape

            y0 = (self.height - h) // 2
            x0 = (self.width - w) // 2
            canvas[y0 : y0 + h, x0 : x0 + w] = np.minimum(
                canvas[y0 : y0 + h, x0 : x0 + w], fg
            )
        except Exception:
            pass

        self._cache_set(key, canvas)
        return canvas

    def ink_pixels(self, latex: str, threshold: float = 0.98) -> int:
        arr = self.render(latex)
        return int(np.count_nonzero(arr < float(threshold)))


class FastPix2Tex:
    """Direct pix2tex model wrapper using architecture+weights without LatexOCR API."""

    def __init__(self, device: str, max_seq_len: int, temperature: float):
        requested_max_seq_len = int(max_seq_len)
        with in_model_path():
            cfg = Munch(
                {
                    "config": "settings/config.yaml",
                    "checkpoint": "checkpoints/weights.pth",
                    "no_cuda": (device != "cuda"),
                    "no_resize": False,
                }
            )
            with open(cfg.config, "r", encoding="utf-8") as f:
                params = yaml.load(f, Loader=yaml.FullLoader)

            args = parse_args(Munch(params))
            args.update(**vars(cfg))
            args.wandb = False
            args.device = device
            args.temperature = float(temperature)

            if not os.path.exists(args.checkpoint):
                download_checkpoints()

            self.model = get_model(args)
            self.model.load_state_dict(
                torch.load(args.checkpoint, map_location=args.device)
            )
            self.model = self.model.to(args.device).eval()

            native_max_seq_len = int(args.max_seq_len)

            self.image_resizer = None
            ckpt_dir = os.path.dirname(args.checkpoint)
            rs_path = os.path.join(ckpt_dir, "image_resizer.pth")
            if os.path.exists(rs_path):
                self.image_resizer = ResNetV2(
                    layers=[2, 3, 3],
                    num_classes=max(args.max_dimensions) // 32,
                    global_pool="avg",
                    in_chans=1,
                    drop_rate=0.05,
                    preact=True,
                    stem_type="same",
                    conv_layer=StdConv2dSame,
                ).to(args.device)
                self.image_resizer.load_state_dict(
                    torch.load(rs_path, map_location=args.device)
                )
                self.image_resizer.eval()

            self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer)
            args.max_seq_len = max(16, min(requested_max_seq_len, native_max_seq_len))
            self.args = args


def preprocess_input_image(path: Path, height: int, width: int) -> np.ndarray:
    try:
        img = Image.open(path).convert("L")
        arr = np.array(img, dtype=np.float32) / 255.0
    except Exception:
        return np.ones((height, width), dtype=np.float32)
    # Keep formulas dark over bright background.
    arr = np.clip(arr, 0.0, 1.0)

    h, w = arr.shape
    scale = min(height / max(h, 1), width / max(w, 1), 1.0)
    if scale < 1.0:
        nh = max(1, int(round(h * scale)))
        nw = max(1, int(round(w * scale)))
        arr = (
            np.array(
                Image.fromarray((arr * 255).astype(np.uint8)).resize(
                    (nw, nh), Image.Resampling.BILINEAR
                ),
                dtype=np.float32,
            )
            / 255.0
        )
        h, w = arr.shape

    canvas = np.ones((height, width), dtype=np.float32)
    y0 = (height - h) // 2
    x0 = (width - w) // 2
    canvas[y0 : y0 + h, x0 : x0 + w] = arr
    return canvas


def augment_pair(inp: np.ndarray, rnd: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Lightweight augmentations to improve robustness to rendering and compression changes.
    if random.random() < 0.35:
        noise = np.random.normal(0, 0.03, inp.shape).astype(np.float32)
        inp = np.clip(inp + noise, 0.0, 1.0)

    if random.random() < 0.25:
        img = Image.fromarray((inp * 255).astype(np.uint8))
        img = img.filter(ImageFilter.GaussianBlur(radius=0.6))
        inp = np.array(img, dtype=np.float32) / 255.0

    if random.random() < 0.20:
        alpha = random.uniform(0.85, 1.15)
        inp = np.clip((inp - 0.5) * alpha + 0.5, 0.0, 1.0)

    if random.random() < 0.20:
        beta = random.uniform(0.90, 1.10)
        rnd = np.clip((rnd - 0.5) * beta + 0.5, 0.0, 1.0)

    return inp, rnd


def mutate_formula(formula: str) -> str:
    s = str(formula)
    if not s:
        return s

    # Hard-negative style edits: small local changes that preserve visual similarity.
    ops = ["flip_sign", "drop_token", "brace", "swap_pow_sub", "digit"]
    op = random.choice(ops)

    if op == "flip_sign":
        if "+" in s:
            return s.replace("+", "-", 1)
        if "-" in s:
            return s.replace("-", "+", 1)

    if op == "drop_token" and len(s) > 4:
        i = random.randrange(0, len(s))
        return s[:i] + s[i + 1 :]

    if op == "brace":
        if "{" in s:
            return s.replace("{", "", 1)
        if "}" in s:
            return s.replace("}", "", 1)
        return "{" + s + "}"

    if op == "swap_pow_sub":
        if "^" in s:
            return s.replace("^", "_", 1)
        if "_" in s:
            return s.replace("_", "^", 1)

    if op == "digit":
        digits = [i for i, ch in enumerate(s) if ch.isdigit()]
        if digits:
            idx = random.choice(digits)
            new_d = str((int(s[idx]) + random.randint(1, 8)) % 10)
            return s[:idx] + new_d + s[idx + 1 :]

    if len(s) >= 2:
        i = random.randrange(0, len(s) - 1)
        return s[:i] + s[i + 1] + s[i] + s[i + 2 :]
    return s


class PairDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Tuple[int, Path, str]],
        renderer: FormulaRenderer,
        train_mode: bool,
        cache_inputs: bool,
        hard_negative_prob: float,
        min_render_ink: int,
        pair_repeat_factor: int,
    ):
        self.samples = list(samples)
        self.renderer = renderer
        self.train_mode = bool(train_mode)
        self.cache_inputs = bool(cache_inputs)
        self.hard_negative_prob = float(hard_negative_prob)
        self.min_render_ink = int(min_render_ink)
        self.pair_repeat_factor = max(1, int(pair_repeat_factor))
        self.input_cache: List[np.ndarray | None] = [None] * len(self.samples)

        if self.cache_inputs:
            for idx, (_, path, _) in enumerate(
                tqdm(self.samples, desc="cache inputs", unit="img")
            ):
                self.input_cache[idx] = preprocess_input_image(
                    path, self.renderer.height, self.renderer.width
                )

        # Build quick index by formula length for harder random negatives.
        self.by_len: Dict[int, List[int]] = {}
        for idx, (_, _, formula) in enumerate(self.samples):
            bucket = min(len(formula) // 10, 50)
            self.by_len.setdefault(bucket, []).append(idx)

    def __len__(self) -> int:
        # Each sample contributes one positive and one negative pair.
        return len(self.samples) * 2 * self.pair_repeat_factor

    def _pick_nearby_formula(self, formula: str) -> str:
        bucket = min(len(formula) // 10, 50)
        candidates = self.by_len.get(bucket, [])
        if len(candidates) >= 2:
            j = random.choice(candidates)
            return self.samples[j][2]
        return random.choice(self.samples)[2]

    def __getitem__(self, idx: int):
        base_idx = (idx // 2) % len(self.samples)
        _, path, formula = self.samples[base_idx]

        if self.cache_inputs and self.input_cache[base_idx] is not None:
            inp = np.array(self.input_cache[base_idx], copy=True)
        else:
            inp = preprocess_input_image(
                path, self.renderer.height, self.renderer.width
            )

        if idx % 2 == 0:
            target_formula = formula
            label = 1.0
        else:
            # Generate a negative pair that is visually dissimilar.
            target_formula = formula
            ref_render = self.renderer.render(formula)
            min_mse_threshold = 0.005  # Minimum visual dissimilarity.
            for attempt in range(12):
                if random.random() < self.hard_negative_prob:
                    cand = mutate_formula(formula)
                else:
                    cand = self._pick_nearby_formula(formula)
                    # Enhance: pick truly random formula to increase dissimilarity.
                    if random.random() < 0.3:
                        cand = random.choice(self.samples)[2]

                if (
                    cand != formula
                    and self.renderer.ink_pixels(cand) >= self.min_render_ink
                ):
                    cand_render = self.renderer.render(cand)
                    mse = float(np.mean((ref_render - cand_render) ** 2))
                    # Accept only if visually dissimilar enough.
                    if mse >= min_mse_threshold:
                        target_formula = cand
                        break
            if target_formula == formula:
                # Fallback: keep trying mutations until we get something different.
                for _ in range(8):
                    cand = mutate_formula(target_formula)
                    if cand != target_formula:
                        target_formula = cand
                        break
            label = 0.0

        rnd = self.renderer.render(target_formula)

        if self.train_mode:
            inp, rnd = augment_pair(inp, rnd)

        diff = np.abs(inp - rnd)
        mul = inp * rnd
        x = np.stack([inp, rnd, diff, mul], axis=0)
        x = torch.from_numpy(x).float()
        y = torch.tensor(label, dtype=torch.float32)
        return x, y


class ComparatorNet(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = None
        if pretrained:
            try:
                weights = tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            except Exception:
                weights = None

        try:
            backbone = tvm.mobilenet_v3_small(weights=weights)
        except Exception:
            backbone = tvm.mobilenet_v3_small(weights=None)

        self.encoder = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # MobileNetV3-Small embedding width before classifier.
        emb_dim = int(backbone.classifier[0].in_features)
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def _encode_gray(self, img_1ch: torch.Tensor) -> torch.Tensor:
        img_3ch = img_1ch.repeat(1, 3, 1, 1)
        feat = self.encoder(img_3ch)
        feat = self.pool(feat).flatten(1)
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x[:, 0:1]
        rnd = x[:, 1:2]
        f_inp = self._encode_gray(inp)
        f_rnd = self._encode_gray(rnd)
        diff = torch.abs(f_inp - f_rnd)
        mul = f_inp * f_rnd
        z = torch.cat([f_inp, f_rnd, diff, mul], dim=1)
        return self.classifier(z).squeeze(-1)


def load_formulas(math_txt: Path) -> List[str]:
    return math_txt.read_text(encoding="utf-8").splitlines()


def collect_split_samples(
    extracted_root: Path,
    split: str,
    formulas: Sequence[str],
    sample_count: int,
    seed: int,
) -> List[Tuple[int, Path, str]]:
    split_dir = extracted_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing split dir: {split_dir}")

    rows: List[Tuple[int, Path, str]] = []
    for p in sorted(split_dir.glob("*.png")):
        try:
            fid = int(p.stem)
        except Exception:
            continue
        if 0 <= fid < len(formulas):
            rows.append((fid, p, formulas[fid]))

    if sample_count > 0 and sample_count < len(rows):
        rng = random.Random(seed)
        rows = rng.sample(rows, sample_count)

    return rows


def filter_renderable_samples(
    rows: Sequence[Tuple[int, Path, str]],
    renderer: FormulaRenderer,
    min_render_ink: int,
) -> List[Tuple[int, Path, str]]:
    kept: List[Tuple[int, Path, str]] = []
    for row in tqdm(rows, desc="filter renderable", unit="img"):
        formula = str(row[2])
        if renderer.ink_pixels(formula) >= int(min_render_ink):
            kept.append(row)
    return kept


def train_comparator(
    model: ComparatorNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    epochs: int,
    lr: float,
    use_amp: bool,
    patience: int,
    min_delta: float,
    freeze_encoder: bool = True,
    unfreeze_last_blocks: int = 0,
    backbone_lr_scale: float = 0.1,
    log_every: int = 100,
) -> Dict[str, float]:
    model.to(device)

    enc_blocks = list(model.encoder.children())
    total_blocks = len(enc_blocks)

    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    unfreeze_n = 0
    if not freeze_encoder:
        unfreeze_n = total_blocks
    else:
        unfreeze_n = max(0, min(int(unfreeze_last_blocks), total_blocks))

    if unfreeze_n > 0:
        for block in enc_blocks[-unfreeze_n:]:
            for param in block.parameters():
                param.requires_grad = True

    param_groups = [
        {
            "params": [p for p in model.classifier.parameters() if p.requires_grad],
            "lr": float(lr),
        }
    ]
    backbone_params = [p for p in model.encoder.parameters() if p.requires_grad]
    if backbone_params:
        param_groups.append(
            {
                "params": backbone_params,
                "lr": float(lr) * float(backbone_lr_scale),
            }
        )

    trainable_count = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    total_count = sum(int(p.numel()) for p in model.parameters())
    print(
        "fine-tune setup: "
        f"unfreeze_last_blocks={unfreeze_n}/{total_blocks} "
        f"backbone_lr_scale={float(backbone_lr_scale):.4f} "
        f"trainable_params={trainable_count}/{total_count}"
    )

    opt = torch.optim.AdamW(param_groups, lr=float(lr), weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    bce = nn.BCEWithLogitsLoss()
    use_amp = bool(use_amp and device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_acc = -1.0
    best_state = None
    history = {}

    no_improve = 0
    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        tr_n = 0

        train_iter = tqdm(train_loader, desc=f"cmp train ep{ep}", unit="batch")
        for step, (xb, yb) in enumerate(train_iter, start=1):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=use_amp
            ):
                logits = model(xb)
                loss = bce(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            tr_loss += float(loss.item()) * xb.shape[0]
            tr_n += int(xb.shape[0])
            running_train_loss = tr_loss / max(tr_n, 1)
            train_iter.set_postfix(
                loss=f"{running_train_loss:.4f}", lr=f"{opt.param_groups[0]['lr']:.2e}"
            )
            if int(log_every) > 0 and (step % int(log_every) == 0):
                print(
                    f"epoch={ep} step={step}/{len(train_loader)} "
                    f"train_loss={running_train_loss:.4f} lr={opt.param_groups[0]['lr']:.2e}"
                )

        model.eval()
        val_loss = 0.0
        val_n = 0
        val_correct = 0

        with torch.inference_mode():
            val_iter = tqdm(val_loader, desc=f"cmp val ep{ep}", unit="batch")
            for xb, yb in val_iter:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                with torch.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=use_amp
                ):
                    logits = model(xb)
                    loss = bce(logits, yb)

                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                val_correct += int((preds == yb).sum().item())
                val_loss += float(loss.item()) * xb.shape[0]
                val_n += int(xb.shape[0])
                running_val_loss = val_loss / max(val_n, 1)
                running_val_acc = val_correct / max(val_n, 1)
                val_iter.set_postfix(
                    loss=f"{running_val_loss:.4f}", acc=f"{running_val_acc:.4f}"
                )

        sched.step()

        tr_loss /= max(tr_n, 1)
        val_loss /= max(val_n, 1)
        val_acc = val_correct / max(val_n, 1)

        history[f"epoch_{ep}"] = {
            "train_loss": tr_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        print(
            f"ep={ep} train_loss={tr_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > (best_val_acc + float(min_delta)):
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= max(1, int(patience)):
            print(
                f"early stop at epoch {ep}: no val_acc improvement for {patience} epoch(s)"
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "best_val_acc": best_val_acc,
        "history": history,
        "epochs_trained": len(history),
    }


def build_pair_tensor(
    inp_path: Path,
    latex: str,
    renderer: FormulaRenderer,
) -> torch.Tensor:
    inp = preprocess_input_image(inp_path, renderer.height, renderer.width)
    rnd = renderer.render(latex)
    diff = np.abs(inp - rnd)
    mul = inp * rnd
    x = np.stack([inp, rnd, diff, mul], axis=0)
    return torch.from_numpy(x).float().unsqueeze(0)


def build_pair_tensor_from_input(
    inp: np.ndarray,
    latex: str,
    renderer: FormulaRenderer,
) -> torch.Tensor:
    rnd = renderer.render(latex)
    diff = np.abs(inp - rnd)
    mul = inp * rnd
    x = np.stack([inp, rnd, diff, mul], axis=0)
    return torch.from_numpy(x).float()


def fix_brackets(text: str) -> str:
    s = str(text)
    pairs = [("{", "}"), ("(", ")"), ("[", "]")]
    for op, cl in pairs:
        balance = 0
        out = []
        for ch in s:
            if ch == op:
                balance += 1
                out.append(ch)
            elif ch == cl:
                if balance > 0:
                    balance -= 1
                    out.append(ch)
            else:
                out.append(ch)
        s = "".join(out) + (cl * balance)
    return s


def build_decode_tensor(
    ocr: FastPix2Tex, path: Path, use_resizer: bool
) -> torch.Tensor | None:
    from evaluate_pix2tex_gd import build_input_tensor

    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return None
    return build_input_tensor(ocr, img, resize=use_resizer)


def decode_with_settings(
    ocr: FastPix2Tex,
    decode_tensor: torch.Tensor | None,
    decode_kwargs: Dict,
) -> str:
    if decode_tensor is None:
        return ""

    with torch.inference_mode():
        dec = ocr.model.generate(decode_tensor.to(ocr.args.device), **decode_kwargs)
    pred = post_process(token2str(dec, ocr.tokenizer)[0])
    return pred


def decode_batch_with_settings(
    ocr: FastPix2Tex,
    decode_tensors: Sequence[torch.Tensor | None],
    decode_kwargs: Dict,
) -> List[str]:
    from evaluate_pix2tex_gd import collate_with_padding

    out = ["" for _ in range(len(decode_tensors))]
    valid_idx = [i for i, t in enumerate(decode_tensors) if t is not None]
    if not valid_idx:
        return out

    batch_tensors = [
        decode_tensors[i] for i in valid_idx if decode_tensors[i] is not None
    ]
    im = collate_with_padding(batch_tensors)
    if im is None:
        return out

    with torch.inference_mode():
        dec = ocr.model.generate(im.to(ocr.args.device), **decode_kwargs)
    preds = [post_process(x) for x in token2str(dec, ocr.tokenizer)]
    for i, p in zip(valid_idx, preds):
        out[i] = p
    return out


def score_predictions_batch(
    cmp_model: ComparatorNet,
    renderer: FormulaRenderer,
    input_arrays: Sequence[np.ndarray],
    preds: Sequence[str],
    device: str,
    batch_size: int,
) -> List[float]:
    scores: List[float] = []
    for s in range(0, len(preds), max(1, batch_size)):
        e = min(len(preds), s + max(1, batch_size))
        xb = torch.stack(
            [
                build_pair_tensor_from_input(input_arrays[i], preds[i], renderer)
                for i in range(s, e)
            ],
            dim=0,
        ).to(device)
        with torch.inference_mode():
            sc = torch.sigmoid(cmp_model(xb)).detach().cpu().numpy().tolist()
        scores.extend(float(x) for x in sc)
    return scores


def evaluate_feedback_loop(
    ocr: FastPix2Tex,
    cmp_model: ComparatorNet,
    renderer: FormulaRenderer,
    val_rows: Sequence[Tuple[int, Path, str]],
    tau: float,
    max_iters: int,
    feedback_batch_size: int,
    out_csv: Path,
    out_metrics: Path,
) -> None:
    cmp_model.eval()
    device = ocr.args.device

    decode_plan = [
        {"temperature": 0.25, "seq_len": int(ocr.args.max_seq_len)},
        {"temperature": 0.18, "seq_len": int(ocr.args.max_seq_len)},
        {"temperature": 0.35, "seq_len": int(ocr.args.max_seq_len)},
        {"temperature": 0.25, "seq_len": min(768, int(ocr.args.max_seq_len) + 128)},
    ]

    ids = [int(r[0]) for r in val_rows]
    paths = [r[1] for r in val_rows]
    refs = [str(r[2]) for r in val_rows]

    t0 = time.perf_counter()
    decode_tensors = [
        build_decode_tensor(ocr, p, use_resizer=True)
        for p in tqdm(paths, desc="prep decode", unit="img")
    ]
    input_arrays = [
        preprocess_input_image(p, renderer.height, renderer.width)
        for p in tqdm(paths, desc="prep input", unit="img")
    ]

    base_cfg = dict(decode_plan[0])
    ocr.args.max_seq_len = int(base_cfg.pop("seq_len", ocr.args.max_seq_len))
    baseline_preds = ["" for _ in range(len(val_rows))]
    bs = max(1, int(feedback_batch_size))
    for s in tqdm(range(0, len(val_rows), bs), desc="decode baseline", unit="batch"):
        e = min(len(val_rows), s + bs)
        baseline_preds[s:e] = decode_batch_with_settings(
            ocr, decode_tensors[s:e], base_cfg
        )

    baseline_scores = score_predictions_batch(
        cmp_model=cmp_model,
        renderer=renderer,
        input_arrays=input_arrays,
        preds=baseline_preds,
        device=device,
        batch_size=bs,
    )

    best_preds = list(baseline_preds)
    best_scores = list(baseline_scores)

    candidate_steps = decode_plan[1 : 1 + max(0, int(max_iters))]
    active = [i for i, sc in enumerate(best_scores) if sc < tau]
    for cfg in candidate_steps:
        if not active:
            break

        cfg_local = dict(cfg)
        ocr.args.max_seq_len = int(cfg_local.pop("seq_len", ocr.args.max_seq_len))

        cand_preds: Dict[int, str] = {}
        for s in range(0, len(active), bs):
            idxs = active[s : s + bs]
            part_tensors = [decode_tensors[i] for i in idxs]
            part_preds = decode_batch_with_settings(ocr, part_tensors, cfg_local)
            for i_local, pred in zip(idxs, part_preds):
                cand_preds[i_local] = fix_brackets(pred)

        idxs_ordered = [i for i in active if i in cand_preds]
        cand_pred_list = [cand_preds[i] for i in idxs_ordered]
        cand_input_list = [input_arrays[i] for i in idxs_ordered]
        cand_scores = score_predictions_batch(
            cmp_model=cmp_model,
            renderer=renderer,
            input_arrays=cand_input_list,
            preds=cand_pred_list,
            device=device,
            batch_size=bs,
        )

        for i, pred, score in zip(idxs_ordered, cand_pred_list, cand_scores):
            if float(score) > float(best_scores[i]):
                best_scores[i] = float(score)
                best_preds[i] = pred

        active = [i for i in active if best_scores[i] < tau]

    rows = []
    improved = 0
    baseline_refs = [remove_spaces(x) for x in refs]
    final_refs = [remove_spaces(x) for x in refs]
    final_preds = [remove_spaces(x) for x in best_preds]
    baseline_preds_nospace = [remove_spaces(x) for x in baseline_preds]

    for i in range(len(val_rows)):
        if (
            final_preds[i] == baseline_refs[i]
            and baseline_preds_nospace[i] != baseline_refs[i]
        ):
            improved += 1
        rows.append(
            {
                "id": ids[i],
                "reference": refs[i],
                "baseline_prediction": baseline_preds[i],
                "final_prediction": best_preds[i],
                "baseline_score": float(baseline_scores[i]),
                "final_score": float(best_scores[i]),
                "accepted": int(float(best_scores[i]) >= tau),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    baseline_exact = (
        float(np.mean([a == b for a, b in zip(baseline_refs, baseline_preds_nospace)]))
        if baseline_refs
        else 0.0
    )
    final_exact = (
        float(np.mean([a == b for a, b in zip(final_refs, final_preds)]))
        if final_refs
        else 0.0
    )

    baseline_cer = (
        float(cer(baseline_refs, baseline_preds_nospace)) if baseline_refs else 0.0
    )
    final_cer = float(cer(final_refs, final_preds)) if final_refs else 0.0

    baseline_bleu = (
        float(
            corpus_bleu(baseline_preds_nospace, [baseline_refs], force=True).score
            / 100.0
        )
        if baseline_refs
        else 0.0
    )
    final_bleu = (
        float(corpus_bleu(final_preds, [final_refs], force=True).score / 100.0)
        if final_refs
        else 0.0
    )

    metrics = {
        "samples": int(len(val_rows)),
        "tau": float(tau),
        "max_iters": int(max_iters),
        "baseline_exact_no_space": baseline_exact,
        "final_exact_no_space": final_exact,
        "baseline_cer_no_space": baseline_cer,
        "final_cer_no_space": final_cer,
        "baseline_bleu_no_space": baseline_bleu,
        "final_bleu_no_space": final_bleu,
        "improved_exact_count": int(improved),
        "accepted_count": int((df["accepted"] == 1).sum()),
        "runtime_seconds": float(time.perf_counter() - t0),
    }
    out_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a learned render-compare network and run a self-correcting feedback loop."
    )
    parser.add_argument(
        "--workspace", type=str, default="."
    )
    parser.add_argument("--gd-dir", type=str, default="data")
    parser.add_argument(
        "--extracted-root", type=str, default="data/formulae_extracted_full"
    )
    parser.add_argument("--pix2tex-model-dir", type=str, default="",
                        help="Path to pix2tex model dir (contains checkpoints/ and settings/). "
                             "Defaults to the pix2tex package install. "
                             "Use models/pix2tex_baseline if pix2tex is not installed.")
    parser.add_argument("--output-dir", type=str, default="results_self_correcting")
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cuda", "cpu"]
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--quick-mode",
        action="store_true",
        help="Use aggressive fast settings for iteration/debug runs.",
    )

    parser.add_argument(
        "--train-samples", type=int, default=0, help="0 = auto from hardware"
    )
    parser.add_argument(
        "--val-samples", type=int, default=0, help="0 = auto from hardware"
    )
    parser.add_argument(
        "--batch-size", type=int, default=0, help="0 = auto from hardware"
    )
    parser.add_argument("--workers", type=int, default=0, help="0 = auto from hardware")
    parser.add_argument("--epochs", type=int, default=0, help="0 = auto from hardware")
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--no-pretrained-comparator", action="store_true")
    parser.add_argument("--hard-negative-prob", type=float, default=0.45)
    parser.add_argument("--min-render-ink", type=int, default=24)
    parser.add_argument("--renderer-cache-max", type=int, default=20000)
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Print train logs every N steps (0 disables step logging)",
    )
    parser.add_argument("--no-train-amp", action="store_true")
    parser.add_argument(
        "--unfreeze-encoder",
        action="store_true",
        help="Full training (default: fine-tune frozen encoder)",
    )
    parser.add_argument(
        "--unfreeze-last-blocks",
        type=int,
        default=2,
        help="Number of final encoder blocks to unfreeze during fine-tuning",
    )
    parser.add_argument(
        "--backbone-lr-scale",
        type=float,
        default=0.05,
        help="Backbone LR multiplier relative to --lr",
    )
    parser.add_argument(
        "--target-train-pairs",
        type=int,
        default=0,
        help="0 = no virtual expansion, else minimum effective train pair count",
    )
    parser.add_argument(
        "--target-val-pairs",
        type=int,
        default=0,
        help="0 = no virtual expansion, else minimum effective val pair count",
    )
    parser.add_argument("--comparator-checkpoint", type=str, default="")
    parser.add_argument("--finetune-comparator", action="store_true")
    parser.add_argument("--comparator-only", action="store_true")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--feedback-batch-size", type=int, default=8)

    parser.add_argument("--tau", type=float, default=0.70)
    parser.add_argument("--max-iters", type=int, default=3)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.25)
    args = parser.parse_args()

    if args.pix2tex_model_dir:
        _patch_pix2tex_model_dir(str(Path(args.pix2tex_model_dir).resolve()))

    seed_everything(int(args.seed))

    workspace = Path(args.workspace)
    out_dir = workspace / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    hw = detect_hardware(args.device)
    train_samples = (
        int(args.train_samples) if int(args.train_samples) > 0 else hw.train_samples
    )
    val_samples = int(args.val_samples) if int(args.val_samples) > 0 else hw.val_samples
    batch_size = int(args.batch_size) if int(args.batch_size) > 0 else hw.batch_size
    workers = int(args.workers) if int(args.workers) > 0 else hw.workers
    epochs = int(args.epochs) if int(args.epochs) > 0 else hw.epochs

    if os.name == "nt":
        workers = 0

    if args.quick_mode:
        train_samples = min(train_samples, 1500)
        val_samples = min(val_samples, 200)
        batch_size = min(batch_size, 32)
        workers = 0
        epochs = min(epochs, 1)
        args.max_seq_len = min(int(args.max_seq_len), 256)
        args.max_iters = min(int(args.max_iters), 1)
        args.feedback_batch_size = max(4, min(int(args.feedback_batch_size), 16))
        args.min_render_ink = max(int(args.min_render_ink), 12)

    print(
        f"hardware: device={hw.device} vram_gb={hw.vram_gb:.2f} cpu_cores={hw.cpu_cores} | "
        f"train_samples={train_samples} val_samples={val_samples} batch={batch_size} workers={workers} epochs={epochs}"
    )

    formulas = load_formulas(workspace / args.gd_dir / "math.txt")
    extracted_root = workspace / args.extracted_root

    renderer = FormulaRenderer(
        height=96,
        width=384,
        dpi=140,
        cache_max_entries=int(args.renderer_cache_max),
    )

    train_rows_raw = collect_split_samples(
        extracted_root, "train", formulas, train_samples, args.seed
    )
    val_rows_raw = collect_split_samples(
        extracted_root, "val", formulas, val_samples, args.seed + 1
    )

    train_rows = filter_renderable_samples(
        train_rows_raw, renderer=renderer, min_render_ink=int(args.min_render_ink)
    )
    val_rows = filter_renderable_samples(
        val_rows_raw, renderer=renderer, min_render_ink=int(args.min_render_ink)
    )

    print(
        f"train rows used: {len(train_rows)} / {len(train_rows_raw)} | "
        f"val rows used: {len(val_rows)} / {len(val_rows_raw)} | "
        f"min_render_ink={int(args.min_render_ink)}"
    )

    if len(train_rows) < 64 or len(val_rows) < 32:
        raise RuntimeError(
            "Too few renderable samples after filtering. Lower --min-render-ink or increase sampling."
        )

    train_base_pairs = len(train_rows) * 2
    val_base_pairs = len(val_rows) * 2
    train_repeat_factor = 1
    val_repeat_factor = 1
    if int(args.target_train_pairs) > 0:
        train_repeat_factor = max(
            1, int(np.ceil(int(args.target_train_pairs) / max(train_base_pairs, 1)))
        )
    if int(args.target_val_pairs) > 0:
        val_repeat_factor = max(
            1, int(np.ceil(int(args.target_val_pairs) / max(val_base_pairs, 1)))
        )

    print(
        f"effective pairs: train={train_base_pairs * train_repeat_factor} "
        f"(base={train_base_pairs}, repeat={train_repeat_factor}) | "
        f"val={val_base_pairs * val_repeat_factor} "
        f"(base={val_base_pairs}, repeat={val_repeat_factor})"
    )

    train_ds = PairDataset(
        train_rows,
        renderer,
        train_mode=True,
        cache_inputs=True,
        hard_negative_prob=float(args.hard_negative_prob),
        min_render_ink=int(args.min_render_ink),
        pair_repeat_factor=train_repeat_factor,
    )
    val_ds = PairDataset(
        val_rows,
        renderer,
        train_mode=False,
        cache_inputs=True,
        hard_negative_prob=float(args.hard_negative_prob),
        min_render_ink=int(args.min_render_ink),
        pair_repeat_factor=val_repeat_factor,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=(hw.device == "cuda"),
        persistent_workers=(workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, workers // 2),
        pin_memory=(hw.device == "cuda"),
        persistent_workers=(workers > 1),
    )

    cmp_model = ComparatorNet(pretrained=(not args.no_pretrained_comparator))
    if args.finetune_comparator and str(args.comparator_checkpoint).strip():
        ckpt_path = Path(args.comparator_checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = workspace / ckpt_path
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state = ckpt.get("state_dict", ckpt)
            cmp_model.load_state_dict(state, strict=False)
            print(f"Loaded comparator checkpoint: {ckpt_path}")
        else:
            print(
                f"Comparator checkpoint not found, training from scratch: {ckpt_path}"
            )

    train_info = train_comparator(
        cmp_model,
        train_loader,
        val_loader,
        hw.device,
        epochs,
        float(args.lr),
        use_amp=(not args.no_train_amp),
        patience=int(args.patience),
        min_delta=float(args.min_delta),
        freeze_encoder=(not args.unfreeze_encoder),
        unfreeze_last_blocks=int(args.unfreeze_last_blocks),
        backbone_lr_scale=float(args.backbone_lr_scale),
        log_every=int(args.log_every),
    )

    model_path = out_dir / "render_compare_comparator.pt"
    torch.save(
        {"state_dict": cmp_model.state_dict(), "train_info": train_info}, model_path
    )

    if args.comparator_only:
        summary = {
            "hardware": {
                "device": hw.device,
                "vram_gb": hw.vram_gb,
                "cpu_cores": hw.cpu_cores,
            },
            "run_config": {
                "train_samples": len(train_rows),
                "val_samples": len(val_rows),
                "batch_size": batch_size,
                "workers": workers,
                "epochs": epochs,
                "hard_negative_prob": float(args.hard_negative_prob),
                "min_render_ink": int(args.min_render_ink),
                "renderer_cache_max": int(args.renderer_cache_max),
                "log_every": int(args.log_every),
                "unfreeze_encoder": bool(args.unfreeze_encoder),
                "unfreeze_last_blocks": int(args.unfreeze_last_blocks),
                "backbone_lr_scale": float(args.backbone_lr_scale),
                "target_train_pairs": int(args.target_train_pairs),
                "target_val_pairs": int(args.target_val_pairs),
                "effective_train_pairs": int(len(train_ds)),
                "effective_val_pairs": int(len(val_ds)),
                "finetune_comparator": bool(args.finetune_comparator),
                "pretrained_comparator": bool(not args.no_pretrained_comparator),
                "comparator_only": True,
                "patience": int(args.patience),
                "min_delta": float(args.min_delta),
            },
            "comparator": train_info,
            "artifacts": {
                "model": str(model_path),
            },
        }
        (out_dir / "run_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(f"Saved comparator model: {model_path}")
        return

    # Prepare pix2tex model for feedback-loop inference on validation samples.
    cache_root = workspace / "model_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_root / "hf")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_root / "hf" / "transformers")
    os.environ["XDG_CACHE_HOME"] = str(cache_root / "xdg")

    ocr = FastPix2Tex(
        device=hw.device,
        max_seq_len=int(args.max_seq_len),
        temperature=float(args.temperature),
    )

    pred_csv = out_dir / "self_correcting_val_predictions.csv"
    met_json = out_dir / "self_correcting_val_metrics.json"

    evaluate_feedback_loop(
        ocr=ocr,
        cmp_model=cmp_model.to(hw.device),
        renderer=renderer,
        val_rows=val_rows,
        tau=float(args.tau),
        max_iters=int(args.max_iters),
        feedback_batch_size=int(args.feedback_batch_size),
        out_csv=pred_csv,
        out_metrics=met_json,
    )

    summary = {
        "hardware": {
            "device": hw.device,
            "vram_gb": hw.vram_gb,
            "cpu_cores": hw.cpu_cores,
        },
        "run_config": {
            "train_samples": len(train_rows),
            "val_samples": len(val_rows),
            "batch_size": batch_size,
            "workers": workers,
            "epochs": epochs,
            "tau": float(args.tau),
            "max_iters": int(args.max_iters),
            "feedback_batch_size": int(args.feedback_batch_size),
            "hard_negative_prob": float(args.hard_negative_prob),
            "min_render_ink": int(args.min_render_ink),
            "renderer_cache_max": int(args.renderer_cache_max),
            "log_every": int(args.log_every),
            "unfreeze_encoder": bool(args.unfreeze_encoder),
            "unfreeze_last_blocks": int(args.unfreeze_last_blocks),
            "backbone_lr_scale": float(args.backbone_lr_scale),
            "target_train_pairs": int(args.target_train_pairs),
            "target_val_pairs": int(args.target_val_pairs),
            "effective_train_pairs": int(len(train_ds)),
            "effective_val_pairs": int(len(val_ds)),
            "finetune_comparator": bool(args.finetune_comparator),
            "pretrained_comparator": bool(not args.no_pretrained_comparator),
        },
        "comparator": train_info,
        "artifacts": {
            "model": str(model_path),
            "predictions_csv": str(pred_csv),
            "metrics_json": str(met_json),
        },
    }
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Saved comparator model: {model_path}")
    print(f"Saved validation predictions: {pred_csv}")
    print(f"Saved validation metrics: {met_json}")


if __name__ == "__main__":
    main()
