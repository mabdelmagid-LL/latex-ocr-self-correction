import argparse
import gc
import json
import os
import random
import re
from pathlib import Path
from collections import OrderedDict, defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from matplotlib.mathtext import MathTextParser
from PIL import Image
from torch.utils.data import Dataset
from pix2tex.cli import (
    Munch,
    download_checkpoints,
    get_model,
    in_model_path,
    parse_args,
    yaml,
)

# Use PyTorch backend only for Hugging Face to reduce startup overhead/noise.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")


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

from transformers import Trainer, TrainerCallback, TrainingArguments


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def preprocess_input_image(path: Path, height: int, width: int) -> np.ndarray:
    try:
        img = Image.open(path).convert("L")
    except Exception:
        return np.ones((height, width), dtype=np.float32)

    arr = np.array(img, dtype=np.float32) / 255.0
    if arr.ndim != 2:
        arr = arr[..., 0]

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


def mutate_formula(formula: str, rng: random.Random | None = None) -> str:
    r = rng if rng is not None else random
    s = str(formula)
    if not s:
        return s

    ops = [
        "flip_sign",
        "drop_token",
        "brace",
        "swap_pow_sub",
        "digit",
        "op_substitute",
        "inject_letter",
        "cmd_swap",
    ]
    op = r.choice(ops)

    if op == "flip_sign":
        if "+" in s:
            return s.replace("+", "-", 1)
        if "-" in s:
            return s.replace("-", "+", 1)

    if op == "drop_token" and len(s) > 4:
        i = r.randrange(0, len(s))
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
            idx = r.choice(digits)
            new_d = str((int(s[idx]) + r.randint(1, 8)) % 10)
            return s[:idx] + new_d + s[idx + 1 :]

    if op == "op_substitute":
        # Hard visual negatives: common operator substitutions.
        subs = [
            ("+", r"\\times"),
            ("+", r"\\cdot"),
            ("-", "+"),
            ("=", r"\\approx"),
            (r"\\times", "+"),
            (r"\\cdot", "+"),
            (r"\\leq", r"\\lt"),
            (r"\\geq", r"\\gt"),
        ]
        r.shuffle(subs)
        for a, b in subs:
            if a in s:
                return s.replace(a, b, 1)

    if op == "inject_letter":
        letters = [i for i, ch in enumerate(s) if ch.isalpha()]
        if letters:
            idx = r.choice(letters)
            add = r.choice(["a", "b", "x", "y"])
            # Example: n -> na for challenging near-miss symbols.
            return s[: idx + 1] + add + s[idx + 1 :]

    if op == "cmd_swap":
        cmd_pairs = [
            (r"\\sin", r"\\cos"),
            (r"\\cos", r"\\sin"),
            (r"\\tan", r"\\sin"),
            (r"\\log", r"\\ln"),
            (r"\\ln", r"\\log"),
        ]
        r.shuffle(cmd_pairs)
        for a, b in cmd_pairs:
            if a in s:
                return s.replace(a, b, 1)

    if len(s) >= 2:
        i = r.randrange(0, len(s) - 1)
        return s[:i] + s[i + 1] + s[i] + s[i + 2 :]
    return s


class FormulaRenderer:
    def __init__(
        self,
        height: int = 96,
        width: int = 384,
        dpi: int = 140,
        cache_max_entries: int = 12000,
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

    def clear_cache(self) -> None:
        self.cache.clear()


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be > 0")

        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0.0 else nn.Identity()

        for p in self.base.parameters():
            p.requires_grad = False

        self.lora_A = nn.Linear(self.base.in_features, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, self.base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=np.sqrt(5.0))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.lora_B(self.dropout(self.lora_A(x))) * self.scaling
        return base_out + lora_out


def apply_lora_to_linear_layers(
    module: nn.Module,
    rank: int,
    alpha: float,
    dropout: float,
) -> int:
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(
                module,
                name,
                LoRALinear(base=child, rank=rank, alpha=alpha, dropout=dropout),
            )
            replaced += 1
        else:
            replaced += apply_lora_to_linear_layers(child, rank, alpha, dropout)
    return replaced


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
        emb_dim = int(backbone.classifier[0].in_features)
        self.emb_dim = emb_dim
        feat_dim = emb_dim * 4 + 2
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.Sigmoid(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
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
        cos = F.cosine_similarity(f_inp, f_rnd, dim=1).unsqueeze(1)
        l2 = torch.norm(f_inp - f_rnd, p=2, dim=1).unsqueeze(1)
        z = torch.cat([f_inp, f_rnd, diff, mul, cos, l2], dim=1)
        return self.classifier(z).squeeze(-1)


def load_pix2tex_encoder_module() -> nn.Module:
    # Some Windows CPU builds fail in timm/pix2tex model init with MKLDNN primitives.
    # Disable MKLDNN for this initialization path to make runs stable.
    if hasattr(torch.backends, "mkldnn"):
        torch.backends.mkldnn.enabled = False

    cfg = Munch(
        {
            "config": "settings/config.yaml",
            "checkpoint": "checkpoints/weights.pth",
            "no_cuda": True,
            "no_resize": False,
        }
    )
    with in_model_path():
        with open(cfg.config, "r", encoding="utf-8") as f:
            params = yaml.load(f, Loader=yaml.FullLoader)

        args = parse_args(Munch(params))
        args.update(**vars(cfg))
        args.wandb = False
        args.device = "cpu"

        if not os.path.exists(args.checkpoint):
            download_checkpoints()

        model = get_model(args)
        state_dict = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state_dict)
        encoder = model.encoder
    return encoder


class Pix2TexEncoderComparatorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = load_pix2tex_encoder_module()
        # pix2tex encoder emits [B, T, C], with C=256 for current checkpoint.
        emb_dim = 256
        self.emb_dim = emb_dim
        feat_dim = emb_dim * 4 + 2
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.Sigmoid(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def _encode_gray(self, img_1ch: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(img_1ch)
        if feat.ndim == 3:
            return feat.mean(dim=1)
        if feat.ndim == 4:
            return feat.flatten(2).mean(dim=2)
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x[:, 0:1]
        rnd = x[:, 1:2]
        f_inp = self._encode_gray(inp)
        f_rnd = self._encode_gray(rnd)
        diff = torch.abs(f_inp - f_rnd)
        mul = f_inp * f_rnd
        cos = F.cosine_similarity(f_inp, f_rnd, dim=1).unsqueeze(1)
        l2 = torch.norm(f_inp - f_rnd, p=2, dim=1).unsqueeze(1)
        z = torch.cat([f_inp, f_rnd, diff, mul, cos, l2], dim=1)
        return self.classifier(z).squeeze(-1)


class HFComparatorModel(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        backbone_type: str = "mobilenet_v3_small",
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        focal_weight: float = 0.35,
    ):
        super().__init__()
        if str(backbone_type) == "pix2tex_encoder":
            self.backbone = Pix2TexEncoderComparatorNet()
        else:
            self.backbone = ComparatorNet(pretrained=pretrained)
        self.focal_gamma = float(focal_gamma)
        self.focal_alpha = float(focal_alpha)
        self.focal_weight = float(focal_weight)

    def forward(self, input_tensor: torch.Tensor, labels: torch.Tensor | None = None):
        logits = self.backbone(input_tensor)
        out = {"logits": logits}
        if labels is not None:
            y = labels.float()
            bce_raw = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
            pt = torch.exp(-bce_raw)
            alpha_t = self.focal_alpha * y + (1.0 - self.focal_alpha) * (1.0 - y)
            focal_raw = alpha_t * ((1.0 - pt) ** self.focal_gamma) * bce_raw
            bce_loss = bce_raw.mean()
            focal_loss = focal_raw.mean()
            fw = min(max(self.focal_weight, 0.0), 1.0)
            loss = (1.0 - fw) * bce_loss + fw * focal_loss
            out["loss"] = loss
        return out


class HardNegativeCurriculumCallback(TrainerCallback):
    def __init__(self, start_prob: float, end_prob: float, num_epochs: int):
        self.start_prob = float(start_prob)
        self.end_prob = float(end_prob)
        self.num_epochs = max(1, int(num_epochs))

    def on_epoch_begin(self, args, state, control, **kwargs):
        train_dataset = (
            kwargs.get("train_dataloader").dataset
            if kwargs.get("train_dataloader") is not None
            else None
        )
        if train_dataset is None or not hasattr(train_dataset, "hard_negative_prob"):
            return
        epoch_idx = 0.0 if state.epoch is None else float(state.epoch)
        if self.num_epochs <= 1:
            t = 1.0
        else:
            t = min(max(epoch_idx / (self.num_epochs - 1), 0.0), 1.0)
        new_prob = self.start_prob + t * (self.end_prob - self.start_prob)
        train_dataset.hard_negative_prob = float(new_prob)
        print(
            f"hard-negative curriculum: epoch={epoch_idx:.2f} hard_negative_prob={new_prob:.4f}"
        )


class PairResampleMemoryCallback(TrainerCallback):
    def __init__(
        self,
        resample_pairs_each_epoch: bool,
        clear_render_cache_each_epoch: bool,
        clear_input_cache_each_epoch: bool,
        clear_cuda_cache_each_epoch: bool,
        log_memory_each_epoch: bool,
    ):
        self.resample_pairs_each_epoch = bool(resample_pairs_each_epoch)
        self.clear_render_cache_each_epoch = bool(clear_render_cache_each_epoch)
        self.clear_input_cache_each_epoch = bool(clear_input_cache_each_epoch)
        self.clear_cuda_cache_each_epoch = bool(clear_cuda_cache_each_epoch)
        self.log_memory_each_epoch = bool(log_memory_each_epoch)

    def _cleanup(self, dataset, tag: str) -> None:
        if dataset is not None and self.clear_render_cache_each_epoch and hasattr(dataset, "renderer") and dataset.renderer is not None:
            dataset.renderer.clear_cache()
        if dataset is not None and self.clear_input_cache_each_epoch and hasattr(dataset, "clear_input_cache"):
            dataset.clear_input_cache()
        gc.collect()
        if torch.cuda.is_available() and self.clear_cuda_cache_each_epoch:
            torch.cuda.empty_cache()
        if self.log_memory_each_epoch:
            if torch.cuda.is_available():
                alloc_mb = torch.cuda.memory_allocated() / (1024 ** 2)
                reserv_mb = torch.cuda.memory_reserved() / (1024 ** 2)
                print(
                    f"memory[{tag}]: cuda_alloc_mb={alloc_mb:.1f} cuda_reserved_mb={reserv_mb:.1f}"
                )
            if dataset is not None and hasattr(dataset, "renderer") and dataset.renderer is not None:
                print(f"memory[{tag}]: renderer_cache_entries={len(dataset.renderer.cache)}")
                if hasattr(dataset, "input_cache_size"):
                    print(f"memory[{tag}]: input_cache_entries={dataset.input_cache_size()}")

    def on_epoch_begin(self, args, state, control, **kwargs):
        train_dl = kwargs.get("train_dataloader")
        train_dataset = train_dl.dataset if train_dl is not None else None
        if self.resample_pairs_each_epoch and train_dataset is not None and hasattr(train_dataset, "set_epoch"):
            epoch_idx = 0 if state.epoch is None else int(float(state.epoch))
            train_dataset.set_epoch(epoch_idx)
            print(f"pair-resample: train epoch={epoch_idx}")
        self._cleanup(train_dataset, "epoch_begin")

    def on_evaluate(self, args, state, control, **kwargs):
        eval_dl = kwargs.get("eval_dataloader")
        eval_dataset = eval_dl.dataset if eval_dl is not None else None
        if self.resample_pairs_each_epoch and eval_dataset is not None and hasattr(eval_dataset, "set_epoch"):
            epoch_idx = 0 if state.epoch is None else int(float(state.epoch))
            eval_dataset.set_epoch(100000 + epoch_idx)
            print(f"pair-resample: val epoch={epoch_idx}")
        self._cleanup(eval_dataset, "evaluate")


class HFPairDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[Tuple[int, Path, str]],
        renderer: FormulaRenderer,
        target_pairs: int,
        hard_negative_prob: float,
        min_render_ink: int,
        train_mode: bool,
        base_seed: int,
        input_cache_max_entries: int,
    ):
        self.rows = list(rows)
        self.renderer = renderer
        self.target_pairs = max(2, int(target_pairs))
        self.hard_negative_prob = float(hard_negative_prob)
        self.min_render_ink = int(min_render_ink)
        self.train_mode = bool(train_mode)
        self.base_seed = int(base_seed)
        self.epoch = 0
        self.rng = random.Random(self.base_seed)
        self.input_cache_max_entries = max(0, int(input_cache_max_entries))
        self.input_cache: "OrderedDict[int, np.ndarray]" = OrderedDict()

        self.by_len: Dict[int, List[int]] = {}
        for idx, (_, _, formula) in enumerate(self.rows):
            bucket = min(len(formula) // 10, 50)
            self.by_len.setdefault(bucket, []).append(idx)

        self.index_order = list(range(len(self.rows)))
        self.set_epoch(0)

    def __len__(self) -> int:
        return self.target_pairs

    def _pick_nearby_formula(self, formula: str) -> str:
        bucket = min(len(formula) // 10, 50)
        candidates = self.by_len.get(bucket, [])
        if len(candidates) >= 2:
            j = self.rng.choice(candidates)
            return self.rows[j][2]
        return self.rng.choice(self.rows)[2]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.rng = random.Random(self.base_seed + self.epoch)
        self.index_order = list(range(len(self.rows)))
        self.rng.shuffle(self.index_order)

    def _get_input(self, row_idx: int) -> np.ndarray:
        if row_idx in self.input_cache:
            arr = self.input_cache.pop(row_idx)
            self.input_cache[row_idx] = arr
            return arr

        _, path, _ = self.rows[row_idx]
        arr = preprocess_input_image(path, self.renderer.height, self.renderer.width)

        if self.input_cache_max_entries > 0:
            while len(self.input_cache) >= self.input_cache_max_entries:
                self.input_cache.popitem(last=False)
            self.input_cache[row_idx] = arr
        return arr

    def clear_input_cache(self) -> None:
        self.input_cache.clear()

    def input_cache_size(self) -> int:
        return len(self.input_cache)

    def __getitem__(self, idx: int):
        x, label, _ = self.make_pair(idx)
        return {
            "input_tensor": torch.from_numpy(x).float(),
            "labels": torch.tensor(label, dtype=torch.float32),
        }

    def make_pair(self, idx: int) -> Tuple[np.ndarray, float, str]:
        base_idx = self.index_order[idx % len(self.rows)]
        _, _, formula = self.rows[base_idx]
        inp = np.array(self._get_input(base_idx), copy=True)

        is_pos = idx % 2 == 0
        if is_pos:
            target_formula = formula
            label = 1.0
            pair_type = "type_1_positive_exact_render"
        else:
            target_formula = formula
            ref_render = self.renderer.render(formula)
            pair_type = "type_4_negative_random_formula"
            for _ in range(24):
                if self.rng.random() < self.hard_negative_prob:
                    cand = mutate_formula(formula, self.rng)
                    cand_source = "type_2_negative_mutation"
                else:
                    cand = self._pick_nearby_formula(formula)
                    cand_source = "type_3_negative_nearby_bucket"
                    if self.rng.random() < 0.3:
                        cand = self.rng.choice(self.rows)[2]
                        cand_source = "type_4_negative_random_formula"

                if (
                    cand != formula
                    and self.renderer.ink_pixels(cand) >= self.min_render_ink
                ):
                    # For mutations (hard negatives) accept regardless of MSE â€”
                    # visually similar mutations are the most valuable training signal.
                    # Only apply the MSE floor to easy random/nearby negatives.
                    if cand_source == "type_2_negative_mutation":
                        target_formula = cand
                        pair_type = cand_source
                        break
                    cand_render = self.renderer.render(cand)
                    mse = float(np.mean((ref_render - cand_render) ** 2))
                    if mse >= 0.005:
                        target_formula = cand
                        pair_type = cand_source
                        break
            if target_formula == formula:
                # Final fallback still counts as type_4 and only enforces different formula.
                for _ in range(128):
                    cand = self.rng.choice(self.rows)[2]
                    if cand != formula:
                        target_formula = cand
                        pair_type = "type_4_negative_random_formula"
                        break
            label = 0.0

        rnd = self.renderer.render(target_formula)

        diff = np.abs(inp - rnd)
        mul = inp * rnd
        # Retry once on MemoryError â€” Windows heap fragmentation can cause
        # spurious allocation failures for tiny arrays; gc clears cyclic refs.
        try:
            x = np.stack([inp, rnd, diff, mul], axis=0)
        except MemoryError:
            import gc
            gc.collect()
            x = np.stack([inp, rnd, diff, mul], axis=0)
        return x, label, pair_type


def evaluate_by_pair_type(
    model: HFComparatorModel,
    dataset: HFPairDataset,
    device: str,
    batch_size: int,
    seed: int,
) -> Dict[str, object]:
    random.seed(int(seed))
    np.random.seed(int(seed))

    model.eval()
    type_total: Dict[str, int] = defaultdict(int)
    type_correct: Dict[str, int] = defaultdict(int)
    all_correct = 0
    total_pairs = int(len(dataset))
    bs = max(1, int(batch_size))

    for s in range(0, total_pairs, bs):
        chunk_x: List[np.ndarray] = []
        chunk_y: List[float] = []
        chunk_t: List[str] = []
        end = min(s + bs, total_pairs)
        for i in range(s, end):
            x, y, t = dataset.make_pair(i)
            chunk_x.append(x)
            chunk_y.append(float(y))
            chunk_t.append(t)

        xb = torch.from_numpy(np.stack(chunk_x, axis=0)).float().to(device)
        with torch.inference_mode():
            logits = model(input_tensor=xb)["logits"]
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float().cpu().numpy()

        ys = np.array(chunk_y, dtype=np.float32)
        correct = (preds == ys).astype(np.int32)
        all_correct += int(correct.sum())

        for j, t in enumerate(chunk_t):
            type_total[t] += 1
            type_correct[t] += int(correct[j])

    per_type = {}
    for t in sorted(type_total.keys()):
        n = int(type_total[t])
        c = int(type_correct[t])
        per_type[t] = {
            "count": n,
            "correct": c,
            "accuracy": float(c / max(1, n)),
        }

    overall_acc = float(all_correct / max(1, total_pairs))
    return {
        "overall_accuracy": overall_acc,
        "pairs": int(total_pairs),
        "pair_type_accuracy": per_type,
    }


def data_collator(features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "input_tensor": torch.stack([f["input_tensor"] for f in features], dim=0),
        "labels": torch.stack([f["labels"] for f in features], dim=0),
    }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(np.float32)
    acc = float((preds == labels).mean())
    return {"accuracy": acc}


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
    kept = []
    for row in rows:
        if renderer.ink_pixels(row[2]) >= int(min_render_ink):
            kept.append(row)
    return kept


def configure_finetune(
    model: HFComparatorModel,
    unfreeze_last_blocks: int,
    full_unfreeze: bool,
    head_only: bool,
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float,
) -> int:
    backbone = model.backbone
    enc_blocks = list(backbone.encoder.children())

    lora_layers = 0
    if int(lora_rank) > 0:
        lora_layers = apply_lora_to_linear_layers(
            backbone.encoder,
            rank=int(lora_rank),
            alpha=float(lora_alpha),
            dropout=float(lora_dropout),
        )
        print(f"lora: enabled on encoder linear layers count={lora_layers} rank={lora_rank}")

    for p in backbone.encoder.parameters():
        p.requires_grad = False
    for p in backbone.classifier.parameters():
        p.requires_grad = True

    # Re-enable LoRA adapter parameters after global encoder freeze.
    if lora_layers > 0:
        for m in backbone.encoder.modules():
            if isinstance(m, LoRALinear):
                for p in m.lora_A.parameters():
                    p.requires_grad = True
                for p in m.lora_B.parameters():
                    p.requires_grad = True

    if bool(head_only):
        return sum(int(p.numel()) for p in model.parameters() if p.requires_grad)

    if full_unfreeze:
        for p in backbone.encoder.parameters():
            p.requires_grad = True
    else:
        n = max(0, min(int(unfreeze_last_blocks), len(enc_blocks)))
        if n > 0:
            for block in enc_blocks[-n:]:
                for p in block.parameters():
                    p.requires_grad = True

    return sum(int(p.numel()) for p in model.parameters() if p.requires_grad)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HF Trainer comparator training with adaptive LR scheduling"
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
    parser.add_argument("--output-dir", type=str, default="results_comparator_hf")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--backbone-type",
        type=str,
        default="mobilenet_v3_small",
        choices=["mobilenet_v3_small", "pix2tex_encoder"],
    )

    parser.add_argument("--train-samples", type=int, default=12000)
    parser.add_argument("--val-samples", type=int, default=2000)
    parser.add_argument("--target-train-pairs", type=int, default=500000)
    parser.add_argument("--target-val-pairs", type=int, default=60000)
    parser.add_argument(
        "--pair-target-mode",
        type=str,
        default="total",
        choices=["total", "per_epoch"],
        help="Interpret target pair counts as total across run or per epoch",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--auto-find-batch-size",
        action="store_true",
        help="Let HF reduce batch size automatically on OOM",
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument(
        "--lr-scheduler-type",
        type=str,
        default="reduce_lr_on_plateau",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
            "inverse_sqrt",
            "reduce_lr_on_plateau",
        ],
    )
    parser.add_argument("--hard-negative-prob", type=float, default=0.4)
    parser.add_argument(
        "--hard-negative-prob-end",
        type=float,
        default=-1.0,
        help="If >=0, use curriculum from --hard-negative-prob to this value",
    )
    parser.add_argument("--min-render-ink", type=int, default=24)
    parser.add_argument("--renderer-cache-max", type=int, default=12000)
    parser.add_argument(
        "--input-cache-max",
        type=int,
        default=3000,
        help="Max number of preprocessed input images cached in RAM per dataset",
    )
    parser.add_argument(
        "--resample-pairs-each-epoch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reshuffle/reseed pair generation every epoch for train and validation",
    )
    parser.add_argument(
        "--clear-render-cache-each-epoch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clear renderer cache at epoch boundaries to control memory",
    )
    parser.add_argument(
        "--clear-cuda-cache-each-epoch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Call torch.cuda.empty_cache at epoch boundaries",
    )
    parser.add_argument(
        "--clear-input-cache-each-epoch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clear preprocessed input-image cache at epoch boundaries",
    )
    parser.add_argument(
        "--log-memory-each-epoch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print memory/cache stats at epoch boundaries",
    )
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--focal-weight", type=float, default=0.35)
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=0,
        help="LoRA rank for encoder linear layers (0 disables LoRA)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=float,
        default=16.0,
        help="LoRA alpha scaling",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout on adapter branch",
    )

    parser.add_argument("--unfreeze-last-blocks", type=int, default=1)
    parser.add_argument("--full-unfreeze", action="store_true")
    parser.add_argument(
        "--head-only",
        action="store_true",
        help="Train only classifier head (freeze full encoder)",
    )
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default="",
        help="Path to HF checkpoint directory to resume training from",
    )
    args = parser.parse_args()

    if args.pix2tex_model_dir:
        _patch_pix2tex_model_dir(str(Path(args.pix2tex_model_dir).resolve()))

    seed_everything(int(args.seed))

    if os.name == "nt" and int(args.dataloader_num_workers) > 0:
        print(
            "Windows detected: forcing dataloader-num-workers=0 to avoid spawn memory issues"
        )
        args.dataloader_num_workers = 0

    workspace = Path(args.workspace)
    out_dir = workspace / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    formulas = load_formulas(workspace / args.gd_dir / "math.txt")
    extracted_root = workspace / args.extracted_root
    renderer = FormulaRenderer(cache_max_entries=int(args.renderer_cache_max))

    train_rows_raw = collect_split_samples(
        extracted_root, "train", formulas, int(args.train_samples), int(args.seed)
    )
    val_rows_raw = collect_split_samples(
        extracted_root, "val", formulas, int(args.val_samples), int(args.seed) + 1
    )

    train_rows = filter_renderable_samples(
        train_rows_raw, renderer, int(args.min_render_ink)
    )
    val_rows = filter_renderable_samples(
        val_rows_raw, renderer, int(args.min_render_ink)
    )

    if len(train_rows) < 64 or len(val_rows) < 32:
        raise RuntimeError("Too few renderable rows for stable training")

    if str(args.pair_target_mode) == "total":
        target_train_pairs = max(
            2, int(np.ceil(int(args.target_train_pairs) / max(1, int(args.epochs))))
        )
        target_val_pairs = max(
            2, int(np.ceil(int(args.target_val_pairs) / max(1, int(args.epochs))))
        )
    else:
        target_train_pairs = max(2, int(args.target_train_pairs))
        target_val_pairs = max(2, int(args.target_val_pairs))

    print(
        f"pair target mode={args.pair_target_mode} | "
        f"train_pairs_per_epoch={target_train_pairs} val_pairs_per_eval={target_val_pairs}"
    )

    train_ds = HFPairDataset(
        rows=train_rows,
        renderer=renderer,
        target_pairs=target_train_pairs,
        hard_negative_prob=float(args.hard_negative_prob),
        min_render_ink=int(args.min_render_ink),
        train_mode=True,
        base_seed=int(args.seed) + 101,
        input_cache_max_entries=int(args.input_cache_max),
    )
    val_ds = HFPairDataset(
        rows=val_rows,
        renderer=renderer,
        target_pairs=target_val_pairs,
        hard_negative_prob=float(args.hard_negative_prob),
        min_render_ink=int(args.min_render_ink),
        train_mode=False,
        base_seed=int(args.seed) + 202,
        input_cache_max_entries=int(args.input_cache_max),
    )

    model = HFComparatorModel(
        pretrained=True,
        backbone_type=str(args.backbone_type),
        focal_gamma=float(args.focal_gamma),
        focal_alpha=float(args.focal_alpha),
        focal_weight=float(args.focal_weight),
    )
    trainable = configure_finetune(
        model,
        int(args.unfreeze_last_blocks),
        bool(args.full_unfreeze),
        bool(args.head_only),
        int(args.lora_rank),
        float(args.lora_alpha),
        float(args.lora_dropout),
    )
    total = sum(int(p.numel()) for p in model.parameters())
    print(f"trainable_params={trainable}/{total}")

    training_args = TrainingArguments(
        output_dir=str(out_dir / "hf_ckpts"),
        per_device_train_batch_size=int(args.batch_size),
        per_device_eval_batch_size=int(args.batch_size),
        auto_find_batch_size=bool(args.auto_find_batch_size),
        gradient_accumulation_steps=max(1, int(args.gradient_accumulation_steps)),
        num_train_epochs=float(args.epochs),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        warmup_ratio=float(args.warmup_ratio),
        lr_scheduler_type=str(args.lr_scheduler_type),
        logging_strategy="steps",
        logging_steps=int(args.logging_steps),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=bool(args.fp16),
        report_to=[],
        dataloader_num_workers=max(0, int(args.dataloader_num_workers)),
        dataloader_pin_memory=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    if float(args.hard_negative_prob_end) >= 0.0:
        trainer.add_callback(
            HardNegativeCurriculumCallback(
                start_prob=float(args.hard_negative_prob),
                end_prob=float(args.hard_negative_prob_end),
                num_epochs=int(args.epochs),
            )
        )

    trainer.add_callback(
        PairResampleMemoryCallback(
            resample_pairs_each_epoch=bool(args.resample_pairs_each_epoch),
            clear_render_cache_each_epoch=bool(args.clear_render_cache_each_epoch),
            clear_input_cache_each_epoch=bool(args.clear_input_cache_each_epoch),
            clear_cuda_cache_each_epoch=bool(args.clear_cuda_cache_each_epoch),
            log_memory_each_epoch=bool(args.log_memory_each_epoch),
        )
    )

    resume_ckpt = str(args.resume_from_checkpoint).strip()
    if resume_ckpt:
        trainer.train(resume_from_checkpoint=resume_ckpt)
    else:
        trainer.train()
    metrics = trainer.evaluate()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    type_eval = evaluate_by_pair_type(
        model=model,
        dataset=val_ds,
        device=device,
        batch_size=max(1, int(args.batch_size)),
        seed=int(args.seed) + 123,
    )

    pair_type_eval_path = out_dir / "pair_type_eval.json"
    pair_type_eval_path.write_text(json.dumps(type_eval, indent=2), encoding="utf-8")

    model_path = out_dir / "render_compare_comparator_hf.pt"
    torch.save(
        {"state_dict": model.backbone.state_dict(), "metrics": metrics}, model_path
    )

    summary = {
        "trainable_params": int(trainable),
        "total_params": int(total),
        "train_rows": int(len(train_rows)),
        "val_rows": int(len(val_rows)),
        "target_train_pairs": int(args.target_train_pairs),
        "target_val_pairs": int(args.target_val_pairs),
        "pair_target_mode": str(args.pair_target_mode),
        "backbone_type": str(args.backbone_type),
        "head_only": bool(args.head_only),
        "effective_train_pairs_per_epoch": int(target_train_pairs),
        "effective_val_pairs_per_eval": int(target_val_pairs),
        "batch_size": int(args.batch_size),
        "auto_find_batch_size": bool(args.auto_find_batch_size),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "dataloader_num_workers": int(args.dataloader_num_workers),
        "epochs": int(args.epochs),
        "resume_from_checkpoint": str(args.resume_from_checkpoint),
        "lr_scheduler_type": str(args.lr_scheduler_type),
        "hard_negative_prob": float(args.hard_negative_prob),
        "hard_negative_prob_end": float(args.hard_negative_prob_end),
        "resample_pairs_each_epoch": bool(args.resample_pairs_each_epoch),
        "clear_render_cache_each_epoch": bool(args.clear_render_cache_each_epoch),
        "clear_input_cache_each_epoch": bool(args.clear_input_cache_each_epoch),
        "clear_cuda_cache_each_epoch": bool(args.clear_cuda_cache_each_epoch),
        "log_memory_each_epoch": bool(args.log_memory_each_epoch),
        "focal_gamma": float(args.focal_gamma),
        "focal_alpha": float(args.focal_alpha),
        "focal_weight": float(args.focal_weight),
        "lora_rank": int(args.lora_rank),
        "lora_alpha": float(args.lora_alpha),
        "lora_dropout": float(args.lora_dropout),
        "metrics": metrics,
        "pair_type_eval": type_eval,
        "artifacts": {
            "model": str(model_path),
            "pair_type_eval": str(pair_type_eval_path),
        },
    }
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
