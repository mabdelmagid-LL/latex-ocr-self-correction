"""
train_comparator_hf_v2.py
=========================
Three improved comparator architectures for Milestone 2.
All other training infrastructure (dataset, loss, HF Trainer loop, CLI) is
inherited from train_comparator_hf.py â€” only the model backbone changes.

Architecture variants
---------------------
v2a  ResidualHead
     Drop-in replacement for the original flat MLP classifier head.
     Adds skip-connections and BatchNorm between layers so gradients flow
     better for deeper feature integration.
     Hypothesis: richer non-linear fusion of the four feature groups
     (f_inp, f_rnd, diff, mul) improves discrimination without adding
     parameters that would be prone to overfit.

v2b  CrossAttentionComparatorNet
     Instead of concatenating [f_inp, f_rnd, diff, mul] and feeding a MLP,
     we treat f_inp and f_rnd as queries/keys and compute cross-attention
     over their token-level representations.  This lets the model attend to
     the specific spatial positions where the two formula images disagree
     rather than reasoning purely on pooled global vectors.
     Works only with the pix2tex_encoder backbone (emits [B, T, 256] tokens
     before pooling).

v2c  ContrastiveComparatorNet
     Re-frames the problem as metric learning instead of binary
     classification.  A projection MLP maps each image into a unit-sphere
     embedding; the InfoNCE/NT-Xent loss pulls positive pairs together and
     repels negatives.  At inference, cosine similarity replaces the MLP
     head for scoring â€” making the scoring function directly interpretable
     and threshold-free.

Usage
-----
python train_comparator_hf_v2.py \\
    --arch v2a \\
    --output-dir results_m2_v2a \\
    [all other args same as train_comparator_hf.py]

python train_comparator_hf_v2.py \\
    --arch v2b \\
    --output-dir results_m2_v2b \\
    --backbone-type pix2tex_encoder \\
    --head-only

python train_comparator_hf_v2.py \\
    --arch v2c \\
    --output-dir results_m2_v2c \\
    --backbone-type pix2tex_encoder \\
    --head-only \\
    --contrastive-temp 0.07
"""

import argparse
import gc
import json
import math
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# Re-use everything that hasn't changed from v1
from train_comparator_hf import (
    HFPairDataset,
    FormulaRenderer,
    LoRALinear,
    apply_lora_to_linear_layers,
    collect_split_samples,
    configure_finetune,
    data_collator,
    evaluate_by_pair_type,
    filter_renderable_samples,
    load_formulas,
    load_pix2tex_encoder_module,
    preprocess_input_image,
    seed_everything,
    HardNegativeCurriculumCallback,
    PairResampleMemoryCallback,
)

import csv

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")


def _patch_pix2tex_model_dir(model_dir: str):
    """Redirect pix2tex's in_model_path() to a custom directory (e.g. models/pix2tex_baseline)."""
    import contextlib
    import pix2tex.cli as _cli
    import pix2tex.utils as _utils
    import train_comparator_hf as _hf

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
    _hf.in_model_path = _patched
from transformers import Trainer, TrainingArguments


class PregenDataset(Dataset):
    """Load pre-generated pairs from generate_comparator_dataset.py output."""

    def __init__(self, split_dir: Path, img_h: int = 96, img_w: int = 384):
        self.img_h = img_h
        self.img_w = img_w
        csv_path = split_dir / "pairs.csv"
        self.rows = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                self.rows.append((r["input_path"], r["render_path"], int(r["label"])))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        inp_path, rnd_path, label = self.rows[idx]
        inp = preprocess_input_image(Path(inp_path), self.img_h, self.img_w)
        rnd_img = np.array(
            __import__("PIL").Image.open(rnd_path).convert("L").resize(
                (self.img_w, self.img_h), __import__("PIL").Image.LANCZOS
            ),
            dtype=np.float32,
        ) / 255.0
        x = np.stack([inp, rnd_img], axis=0)  # (2, H, W)
        return {
            "input_tensor": torch.from_numpy(x).float(),
            "labels": torch.tensor(label, dtype=torch.float32),
        }


# ============================================================================
# v2a â€” ResidualHead
# ============================================================================

class ResidualBlock(nn.Module):
    """Two-layer residual block with BatchNorm and optional projection."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class ResidualHead(nn.Module):
    """
    Replacement for the original flat MLP head.

    Input  : concatenated feature vector [f_inp, f_rnd, diff, mul, cos, l2]
    Output : scalar logit (use with BCEWithLogitsLoss)

    Changes vs baseline
    -------------------
    1. Input projection to a wider 768-dim hidden space (vs 512 baseline).
    2. Three residual blocks with BatchNorm â€” each preserves gradient flow.
    3. Final bottleneck 768 â†’ 256 â†’ 1 with dropout.
    4. No sigmoid mid-network (the original had Sigmoid after the first
       Linear(512,512) which hard-saturates gradients near آ±1).
    """

    def __init__(self, feat_dim: int, hidden: int = 768, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(
            ResidualBlock(hidden, dropout=dropout),
            ResidualBlock(hidden, dropout=dropout),
            ResidualBlock(hidden, dropout=dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.proj(z)
        h = self.blocks(h)
        return self.head(h).squeeze(-1)


# ============================================================================
# v2b â€” CrossAttentionComparatorNet
# ============================================================================

class MultiHeadCrossAttention(nn.Module):
    """
    Cross-attention between two token sequences.

    query  : [B, T_q, D]
    key    : [B, T_k, D]
    value  : [B, T_k, D]
    output : [B, T_q, D]  (same shape as query)
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        B, T_q, D = query.shape
        _, T_k, _ = key.shape
        H, Dh = self.num_heads, self.head_dim

        Q = self.q_proj(query).reshape(B, T_q, H, Dh).transpose(1, 2)  # [B, H, T_q, Dh]
        K = self.k_proj(key).reshape(B, T_k, H, Dh).transpose(1, 2)
        V = self.v_proj(value).reshape(B, T_k, H, Dh).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) * self.scale  # [B, H, T_q, T_k]
        attn = self.dropout(torch.softmax(attn, dim=-1))
        out = (attn @ V).transpose(1, 2).reshape(B, T_q, D)  # [B, T_q, D]
        return self.out_proj(out)


class CrossAttentionFusionBlock(nn.Module):
    """
    Bidirectional cross-attention fusion between two token sequences.
    Query: token sequence of image A attending to image B, and vice versa.
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn_a2b = MultiHeadCrossAttention(dim, num_heads, dropout)
        self.attn_b2a = MultiHeadCrossAttention(dim, num_heads, dropout)
        self.norm_a = nn.LayerNorm(dim)
        self.norm_b = nn.LayerNorm(dim)
        self.ffn_a = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.ffn_b = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.norm_a2 = nn.LayerNorm(dim)
        self.norm_b2 = nn.LayerNorm(dim)

    def forward(
        self, tok_a: torch.Tensor, tok_b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Cross-attention: A reads B, B reads A
        a_ctx = self.norm_a(tok_a + self.attn_a2b(tok_a, tok_b, tok_b))
        b_ctx = self.norm_b(tok_b + self.attn_b2a(tok_b, tok_a, tok_a))
        # FFN
        a_out = self.norm_a2(a_ctx + self.ffn_a(a_ctx))
        b_out = self.norm_b2(b_ctx + self.ffn_b(b_ctx))
        return a_out, b_out


class CrossAttentionComparatorNet(nn.Module):
    """
    Comparator using the pix2tex encoder + cross-attention fusion.

    The pix2tex encoder produces token-level features [B, T, 256].
    Instead of pooling immediately and concatenating, we run one block of
    bidirectional cross-attention so each token can 'see' its counterpart
    before we pool and classify.

    Architecture
    ------------
    1. pix2tex encoder â†’ [B, T, 256] tokens for each image.
    2. Cross-attention fusion block: input tokens attend to rendered tokens.
    3. Mean-pool both attended sequences â†’ two 256-dim vectors.
    4. Concatenate [f_inp, f_rnd, diff, mul, cos, l2] â†’ ResidualHead.

    Why better than baseline
     ------------------------
    The baseline pools first, losing spatial/positional information.
    Cross-attention lets the model focus on specific token positions that
    differ (e.g. a subscript that changed, a missing fraction bar) before
    collapsing to a scalar.
    """

    def __init__(self):
        super().__init__()
        self.encoder = load_pix2tex_encoder_module()
        dim = 256  # pix2tex token dim
        self.emb_dim = dim
        self.fusion = CrossAttentionFusionBlock(dim, num_heads=4, dropout=0.1)
        feat_dim = dim * 4 + 2
        self.head = ResidualHead(feat_dim, hidden=512, dropout=0.1)

    def _encode(self, img_1ch: torch.Tensor) -> torch.Tensor:
        """Return token sequence [B, T, C] without pooling."""
        feat = self.encoder(img_1ch)
        if feat.ndim == 2:
            feat = feat.unsqueeze(1)   # [B, 1, C]
        elif feat.ndim == 4:
            B, C, H, W = feat.shape
            feat = feat.flatten(2).transpose(1, 2)  # [B, T, C]
        # feat is now [B, T, C]
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x[:, 0:1]   # [B, 1, H, W]
        rnd = x[:, 1:2]

        tok_inp = self._encode(inp)   # [B, T, 256]
        tok_rnd = self._encode(rnd)

        # Bidirectional cross-attention
        tok_inp_ctx, tok_rnd_ctx = self.fusion(tok_inp, tok_rnd)

        f_inp = tok_inp_ctx.mean(dim=1)   # [B, 256]
        f_rnd = tok_rnd_ctx.mean(dim=1)

        diff = torch.abs(f_inp - f_rnd)
        mul = f_inp * f_rnd
        cos = F.cosine_similarity(f_inp, f_rnd, dim=1).unsqueeze(1)
        l2 = torch.norm(f_inp - f_rnd, p=2, dim=1).unsqueeze(1)
        z = torch.cat([f_inp, f_rnd, diff, mul, cos, l2], dim=1)
        return self.head(z)


# ============================================================================
# v2c â€” ContrastiveComparatorNet  (InfoNCE / NT-Xent)
# ============================================================================

class ProjectionMLP(nn.Module):
    """Two-layer MLP projection head for contrastive learning."""

    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, dim=-1)  # unit sphere


class ContrastiveComparatorNet(nn.Module):
    """
    Contrastive comparator using InfoNCE loss.

    At TRAINING time:
        Positive pair â†’ embeddings should be close (high cosine sim).
        Negative pair â†’ embeddings should be far apart.
        Loss: InfoNCE (NT-Xent style) over the batch.

    At INFERENCE time:
        Score = cosine_similarity(z_inp, z_rnd) âˆˆ [-1, 1].
        Rescaled to [0, 1] as (1 + cosine_sim) / 2 for threshold-compatible
        comparison with the baseline (د„=0.5 â†’ cosine_sim=0).

    Why this is interesting
    -----------------------
    The binary BCE formulation trains the network to output a scalar for
    each pair independently, which can overfit to stylistic rendering artefacts.
    Contrastive metric learning instead builds a unified embedding space where
    "matching" is defined geometrically â€” making the score more transferable
    to unseen formula styles and model-generated predictions.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.encoder = load_pix2tex_encoder_module()
        self.emb_dim = 256
        self.temperature = float(temperature)
        self.projector = ProjectionMLP(in_dim=self.emb_dim, hidden_dim=256, out_dim=128)

    def _encode(self, img_1ch: torch.Tensor) -> torch.Tensor:
        """Mean-pooled encoder output â†’ [B, 256]."""
        feat = self.encoder(img_1ch)
        if feat.ndim == 3:
            return feat.mean(dim=1)
        if feat.ndim == 4:
            return feat.flatten(2).mean(dim=2)
        return feat

    def embed(self, img_1ch: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised 128-dim embedding for a grayscale image."""
        feat = self._encode(img_1ch)
        return self.projector(feat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return cosine similarity âˆˆ [-1,1] rescaled to [0,1].
        This matches the BCEWithLogitsLoss convention at inference time
        while still being meaningful as a metric distance.
        """
        inp = x[:, 0:1]
        rnd = x[:, 1:2]
        z_inp = self.embed(inp)
        z_rnd = self.embed(rnd)
        cos_sim = F.cosine_similarity(z_inp, z_rnd, dim=1)   # âˆˆ [-1, 1]
        # Rescale to (0, 1): 0 â†’ -1 (opposite), 0.5 â†’ 0 (orthogonal), 1 â†’ +1 (identical)
        return (cos_sim + 1.0) / 2.0 - 0.5  # logit-like: positive = match


class InfoNCELoss(nn.Module):
    """
    NT-Xent loss for contrastive learning.

    Within a batch, positive pairs are those with label==1.
    For each positive pair (i, j), all other items in the batch are negatives.

    Note: this loss works best with large batch sizes (â‰¥128) so all
    negative pairs are represented.  With small batches, the in-batch
    negatives may not cover the full difficulty spectrum â€” combine with
    the pre-mined HFPairDataset negatives by treating the BCE term as an
    auxiliary loss.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = float(temperature)

    def forward(
        self,
        z_inp: torch.Tensor,   # [B, D]  L2-normalised
        z_rnd: torch.Tensor,   # [B, D]  L2-normalised
        labels: torch.Tensor,  # [B]     1.0=positive, 0.0=negative
    ) -> torch.Tensor:
        B = z_inp.shape[0]
        device = z_inp.device

        # All-pairs cosine similarity matrix [B, B]
        sim_ii = torch.mm(z_inp, z_rnd.t()) / self.temperature  # inp_i vs rnd_j
        sim_jj = torch.mm(z_rnd, z_inp.t()) / self.temperature  # rnd_i vs inp_j

        # Positive mask: diagonal entries (each sample vs its own pair partner)
        pos_mask = torch.eye(B, device=device, dtype=torch.bool)

        # NT-Xent loss: for each row, the diagonal is the positive, others are negatives
        loss_a = F.cross_entropy(sim_ii, torch.arange(B, device=device))
        loss_b = F.cross_entropy(sim_jj, torch.arange(B, device=device))
        return (loss_a + loss_b) / 2.0


# ============================================================================
# Unified HFComparatorModelV2 wrapper (mirrors HFComparatorModel API)
# ============================================================================

class HFComparatorModelV2(nn.Module):
    """
    Unified wrapper that uses one of the three new backbone architectures
    but exposes exactly the same interface as HFComparatorModel so the HF
    Trainer loop in train_comparator_hf.py can drive it without changes.

    The forward() method returns {"logits": ..., "loss": ...} matching the
    HF Trainer expectations.
    """

    ARCH_CHOICES = ("v2a", "v2b", "v2c", "v2d")

    def __init__(
        self,
        arch: str = "v2a",
        backbone_type: str = "pix2tex_encoder",  # used only for v2a
        pretrained: bool = True,
        # BCE+focal loss params (same defaults as HFComparatorModel)
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        focal_weight: float = 0.35,
        # Contrastive-specific (v2c only)
        contrastive_temp: float = 0.07,
        contrastive_bce_weight: float = 0.3,
        # Hard-negative margin loss (v2a / v2b)
        margin: float = 0.0,        # 0 = disabled; e.g. 0.3 adds margin penalty
        margin_weight: float = 1.0, # weight of margin loss term
    ):
        super().__init__()
        if arch not in self.ARCH_CHOICES:
            raise ValueError(f"arch must be one of {self.ARCH_CHOICES}, got {arch!r}")

        self.arch = arch
        self.focal_gamma = float(focal_gamma)
        self.focal_alpha = float(focal_alpha)
        self.focal_weight = float(focal_weight)
        self.contrastive_bce_weight = float(contrastive_bce_weight)
        self.margin = float(margin)
        self.margin_weight = float(margin_weight)

        if arch == "v2a":
            # ResidualHead on top of pix2tex_encoder or mobilenet_v3_small
            if backbone_type == "pix2tex_encoder":
                encoder = load_pix2tex_encoder_module()
                emb_dim = 256

                class _Enc(nn.Module):
                    def __init__(self, enc, dim):
                        super().__init__()
                        self.encoder = enc
                        self.emb_dim = dim
                        feat_dim = dim * 4 + 2
                        self.classifier = ResidualHead(feat_dim, hidden=768, dropout=0.1)

                    def _encode_gray(self, img):
                        feat = self.encoder(img)
                        if feat.ndim == 3:
                            return feat.mean(dim=1)
                        if feat.ndim == 4:
                            return feat.flatten(2).mean(dim=2)
                        return feat

                    def forward(self, x):
                        f_inp = self._encode_gray(x[:, 0:1])
                        f_rnd = self._encode_gray(x[:, 1:2])
                        diff = torch.abs(f_inp - f_rnd)
                        mul = f_inp * f_rnd
                        cos = F.cosine_similarity(f_inp, f_rnd, dim=1).unsqueeze(1)
                        l2 = torch.norm(f_inp - f_rnd, p=2, dim=1).unsqueeze(1)
                        z = torch.cat([f_inp, f_rnd, diff, mul, cos, l2], dim=1)
                        return self.classifier(z)

                self.backbone = _Enc(encoder, emb_dim)
            else:
                import torchvision.models as tvm
                weights = None
                if pretrained:
                    try:
                        weights = tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1
                    except Exception:
                        pass
                try:
                    base = tvm.mobilenet_v3_small(weights=weights)
                except Exception:
                    base = tvm.mobilenet_v3_small(weights=None)
                enc = base.features
                pool = nn.AdaptiveAvgPool2d((1, 1))
                emb_dim = int(base.classifier[0].in_features)

                class _MobEnc(nn.Module):
                    def __init__(self, enc, pool, dim):
                        super().__init__()
                        self.encoder = enc
                        self.pool = pool
                        self.emb_dim = dim
                        feat_dim = dim * 4 + 2
                        self.classifier = ResidualHead(feat_dim, hidden=768, dropout=0.1)

                    def _encode_gray(self, img):
                        return self.pool(self.encoder(img.repeat(1, 3, 1, 1))).flatten(1)

                    def forward(self, x):
                        f_inp = self._encode_gray(x[:, 0:1])
                        f_rnd = self._encode_gray(x[:, 1:2])
                        diff = torch.abs(f_inp - f_rnd)
                        mul = f_inp * f_rnd
                        cos = F.cosine_similarity(f_inp, f_rnd, dim=1).unsqueeze(1)
                        l2 = torch.norm(f_inp - f_rnd, p=2, dim=1).unsqueeze(1)
                        z = torch.cat([f_inp, f_rnd, diff, mul, cos, l2], dim=1)
                        return self.classifier(z)

                self.backbone = _MobEnc(enc, pool, emb_dim)

        elif arch == "v2b":
            self.backbone = CrossAttentionComparatorNet()

        elif arch == "v2c":
            self.backbone = ContrastiveComparatorNet(temperature=contrastive_temp)

        elif arch == "v2d":
            # True Siamese network:
            # - Shared pix2tex encoder (updated via LoRA)
            # - Small projection MLP: 256 â†’ 128, L2-normalised onto unit sphere
            # - Loss: contrastive (Hinge) â€” pulls positives together, pushes
            #   negatives apart by at least `margin` in cosine-distance space
            # - Inference score = cosine_similarity(e_inp, e_rnd) in [âˆ’1, 1],
            #   shifted to [0, 1] so د„=0.5 â†” orthogonal embeddings
            encoder = load_pix2tex_encoder_module()
            emb_dim = 256
            proj_dim = 128

            class _SiameseNet(nn.Module):
                def __init__(self, enc, in_dim, proj_dim):
                    super().__init__()
                    self.encoder = enc
                    self.emb_dim = in_dim
                    # Two-layer projection MLP onto unit sphere
                    self.projector = nn.Sequential(
                        nn.Linear(in_dim, in_dim),
                        nn.BatchNorm1d(in_dim),
                        nn.GELU(),
                        nn.Linear(in_dim, proj_dim),
                    )
                def _encode_raw(self, img):
                    feat = self.encoder(img)
                    if feat.ndim == 3:
                        return feat.mean(dim=1)
                    if feat.ndim == 4:
                        return feat.flatten(2).mean(dim=2)
                    return feat

                def embed(self, img):
                    """Return L2-normalised embedding."""
                    return F.normalize(self.projector(self._encode_raw(img)), dim=-1)

                def forward(self, x):
                    """Return cosine similarity shifted to logit space for inference."""
                    e_inp = self.embed(x[:, 0:1])
                    e_rnd = self.embed(x[:, 1:2])
                    # cosine sim âˆˆ [âˆ’1, 1] â†’ centre at 0 so sigmoid(0) = 0.5 at threshold
                    cos_sim = F.cosine_similarity(e_inp, e_rnd, dim=1)
                    return cos_sim * 5.0  # scale: cos=0.6 â†’ logit=3 â†’ probâ‰ˆ0.95

            self.backbone = _SiameseNet(encoder, emb_dim, proj_dim)
            self.info_nce = InfoNCELoss(temperature=contrastive_temp)

    # -------------------------------------------------------------------------
    # Freeze helpers (mirrors configure_finetune for head-only / lora)
    # -------------------------------------------------------------------------
    def configure_training(
        self,
        head_only: bool = True,
        lora_rank: int = 0,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        full_unfreeze: bool = False,
        unfreeze_last_blocks: int = 1,
    ) -> int:
        backbone = self.backbone
        # Find the encoder attribute
        enc = getattr(backbone, "encoder", None)
        if enc is None:
            # For v2b, fusion layers are separate
            enc = getattr(backbone, "encoder", None)
        classifier = getattr(backbone, "classifier", None) or \
                     getattr(backbone, "head", None) or \
                     getattr(backbone, "projector", None)

        if lora_rank > 0 and enc is not None:
            n = apply_lora_to_linear_layers(enc, rank=lora_rank,
                                            alpha=lora_alpha, dropout=lora_dropout)
            print(f"  LoRA applied to {n} encoder layers (rank={lora_rank})")

        # Freeze encoder
        if enc is not None:
            for p in enc.parameters():
                p.requires_grad = False

        # Unfreeze LoRA adapters
        if lora_rank > 0 and enc is not None:
            for m in enc.modules():
                if isinstance(m, LoRALinear):
                    for p in m.lora_A.parameters():
                        p.requires_grad = True
                    for p in m.lora_B.parameters():
                        p.requires_grad = True

        # Unfreeze trainable head parts
        if self.arch == "v2b":
            # Always train the fusion block and head
            for p in backbone.fusion.parameters():
                p.requires_grad = True
            for p in backbone.head.parameters():
                p.requires_grad = True
        elif self.arch == "v2c":
            for p in backbone.projector.parameters():
                p.requires_grad = True
        elif self.arch == "v2d":
            for p in backbone.projector.parameters():
                p.requires_grad = True
        else:  # v2a
            if classifier is not None:
                for p in classifier.parameters():
                    p.requires_grad = True

        if not head_only and not full_unfreeze and enc is not None:
            blocks = list(enc.children())
            n = max(0, min(unfreeze_last_blocks, len(blocks)))
            for block in blocks[-n:]:
                for p in block.parameters():
                    p.requires_grad = True
        elif full_unfreeze and enc is not None:
            for p in enc.parameters():
                p.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
        return trainable

    # -------------------------------------------------------------------------
    # Forward â€” mirrors HFComparatorModel exactly
    # -------------------------------------------------------------------------
    def forward(self, input_tensor: torch.Tensor, labels: torch.Tensor | None = None):
        if self.arch == "v2c":
            return self._forward_contrastive(input_tensor, labels)
        if self.arch == "v2d":
            return self._forward_siamese(input_tensor, labels)
        return self._forward_discriminative(input_tensor, labels)

    def _forward_discriminative(self, x, labels):
        logits = self.backbone(x)
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

            # Margin loss on hard negatives: penalise any negative pair whose
            # predicted score exceeds (1 - margin), i.e. the model is too
            # confident it is a positive. Only active when margin > 0.
            if self.margin > 0.0:
                neg_mask = (y == 0.0)
                if neg_mask.any():
                    neg_probs = torch.sigmoid(logits[neg_mask])
                    # Push negatives below (1 - margin): relu(score - (1-margin))
                    margin_penalty = F.relu(neg_probs - (1.0 - self.margin)).pow(2)
                    loss = loss + self.margin_weight * margin_penalty.mean()

            out["loss"] = loss
        return out

    def _forward_contrastive(self, x, labels):
        inp = x[:, 0:1]
        rnd = x[:, 1:2]
        z_inp = self.backbone.embed(inp)   # [B, 128] L2-normalised
        z_rnd = self.backbone.embed(rnd)

        # Cosine sim â†’ rescaled logit
        cos_sim = F.cosine_similarity(z_inp, z_rnd, dim=1)
        logits = (cos_sim + 1.0) / 2.0 - 0.5   # centred around 0

        out = {"logits": logits}
        if labels is not None:
            # InfoNCE loss (unsupervised, all diagonals are positives)
            nce_loss = self.info_nce(z_inp, z_rnd, labels)

            # Auxiliary BCE for supervision signal
            y = labels.float()
            bce_loss = F.binary_cross_entropy_with_logits(logits, y)
            w = min(max(self.contrastive_bce_weight, 0.0), 1.0)
            out["loss"] = (1.0 - w) * nce_loss + w * bce_loss
        return out

    def _forward_siamese(self, x, labels):
        """
        True Siamese contrastive loss.

        For each pair:
          y=1 (positive/similar): loss = (1 - cos_sim)آ²       â†’ pull together
          y=0 (negative/different): loss = max(0, cos_sim - (1-margin))آ²  â†’ push apart

        Inference: logit = cos_sim * scale, so sigmoid gives similarity score.
        """
        e_inp = self.backbone.embed(x[:, 0:1])   # [B, 128] L2-normalised
        e_rnd = self.backbone.embed(x[:, 1:2])

        cos_sim = F.cosine_similarity(e_inp, e_rnd, dim=1)   # [B] âˆˆ [âˆ’1, 1]
        # Logit for accuracy metric: positive if cos_sim > 0
        logits = cos_sim * 5.0
        out = {"logits": logits}

        if labels is not None:
            y = labels.float()
            margin = self.margin if self.margin > 0.0 else 0.3  # default margin

            # Positive loss: (1 âˆ’ cos_sim)آ²  â€” want cos_sim â†’ 1
            pos_loss = (1.0 - cos_sim).pow(2)

            # Negative loss: hinge â€” penalise when cos_sim > (1 âˆ’ margin)
            neg_loss = F.relu(cos_sim - (1.0 - margin)).pow(2)

            loss = (y * pos_loss + (1.0 - y) * neg_loss).mean()
            out["loss"] = loss
        return out


# ============================================================================
# compute_metrics (same as v1, but works on the rescaled contrastive logits)
# ============================================================================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(np.float32)
    acc = float((preds == labels).mean())
    return {"accuracy": acc}


# ============================================================================
# Main training entry-point
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Milestone 2 â€” improved comparator architectures"
    )

    # Architecture selector (new)
    parser.add_argument(
        "--arch",
        type=str,
        default="v2a",
        choices=["v2a", "v2b", "v2c", "v2d"],
        help=(
            "v2a = ResidualHead backbone, "
            "v2b = CrossAttention fusion, "
            "v2c = Contrastive (InfoNCE), "
            "v2d = Single linear layer (Siamese)"
        ),
    )
    parser.add_argument(
        "--contrastive-temp",
        type=float,
        default=0.07,
        help="Temperature for InfoNCE loss (v2c only)",
    )
    parser.add_argument(
        "--contrastive-bce-weight",
        type=float,
        default=0.3,
        help="Weight of auxiliary BCE loss in v2c (0=pure InfoNCE, 1=pure BCE)",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.0,
        help="Hard-negative margin epsilon (0=disabled). Adds penalty when a negative "
             "pair scores above (1 - margin). Recommended: 0.3",
    )
    parser.add_argument(
        "--margin-weight",
        type=float,
        default=1.0,
        help="Weight of the margin penalty loss term.",
    )

    # --- Shared args (same as train_comparator_hf.py) ---
    parser.add_argument("--workspace", type=str, default=".")
    parser.add_argument("--gd-dir", type=str, default="data")
    parser.add_argument("--extracted-root", type=str,
                        default="data/formulae_extracted_full")
    parser.add_argument("--pix2tex-model-dir", type=str, default="",
                        help="Path to pix2tex model dir (contains checkpoints/ and settings/). "
                             "Defaults to the pix2tex package install. "
                             "Use models/pix2tex_baseline if pix2tex is not installed.")
    parser.add_argument("--output-dir", type=str, default="results_m2_v2a")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone-type", type=str, default="pix2tex_encoder",
                        choices=["mobilenet_v3_small", "pix2tex_encoder"])

    parser.add_argument("--dataset-dir", type=str, default="",
                        help="Path to pre-generated dataset (from generate_comparator_dataset.py). "
                             "If set, skips on-the-fly rendering entirely.")
    parser.add_argument("--train-samples", type=int, default=12000)
    parser.add_argument("--val-samples", type=int, default=2000)
    parser.add_argument("--target-train-pairs", type=int, default=500000)
    parser.add_argument("--target-val-pairs", type=int, default=60000)
    parser.add_argument("--pair-target-mode", type=str, default="per_epoch",
                        choices=["total", "per_epoch"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--auto-find-batch-size", action="store_true")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--finetune-from", type=str, default="",
                        help="Path to existing comparator_v2.pt to fine-tune from")
    parser.add_argument("--reinit-head", action="store_true", default=False,
                        help="Load encoder weights from --finetune-from but re-init the head randomly")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lr-scheduler-type", type=str,
                        default="reduce_lr_on_plateau")

    parser.add_argument("--hard-negative-prob", type=float, default=0.65)
    parser.add_argument("--hard-negative-prob-end", type=float, default=-1.0)
    parser.add_argument("--min-render-ink", type=int, default=24)
    parser.add_argument("--renderer-cache-max", type=int, default=12000)
    parser.add_argument("--input-cache-max", type=int, default=3000)

    parser.add_argument("--resample-pairs-each-epoch",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clear-render-cache-each-epoch",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clear-cuda-cache-each-epoch",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clear-input-cache-each-epoch",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-memory-each-epoch",
                        action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--head-only", action="store_true", default=True)
    parser.add_argument("--full-unfreeze", action="store_true")
    parser.add_argument("--unfreeze-last-blocks", type=int, default=1)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--fp16", action="store_true")

    args = parser.parse_args()

    if args.pix2tex_model_dir:
        _patch_pix2tex_model_dir(str(Path(args.pix2tex_model_dir).resolve()))

    seed_everything(args.seed)

    ws = Path(args.workspace)
    out_dir = ws / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"Milestone 2 Training  [{args.arch.upper()}]")
    print(f"{'='*60}")
    print(f"Device    : {device}")
    print(f"Arch      : {args.arch}")
    print(f"Backbone  : {args.backbone_type}")
    print(f"Output    : {out_dir}")

    # ---- Build model --------------------------------------------------------
    print(f"\n[1/4] Building model...")
    model = HFComparatorModelV2(
        arch=args.arch,
        backbone_type=args.backbone_type,
        pretrained=True,
        focal_gamma=2.0,
        focal_alpha=0.25,
        focal_weight=0.35,
        contrastive_temp=args.contrastive_temp,
        contrastive_bce_weight=args.contrastive_bce_weight,
        margin=args.margin,
        margin_weight=args.margin_weight,
    )
    trainable = model.configure_training(
        head_only=args.head_only,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        full_unfreeze=args.full_unfreeze,
        unfreeze_last_blocks=args.unfreeze_last_blocks,
    )
    # Load fine-tune checkpoint AFTER configure_training so LoRA layers exist
    if args.finetune_from:
        ckpt_path = Path(args.finetune_from)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

        if args.reinit_head:
            # Load only encoder weights; keep the head randomly initialised
            encoder_sd = {k: v for k, v in sd.items() if k.startswith("encoder.")}
            missing, unexpected = model.backbone.load_state_dict(encoder_sd, strict=False)
            print(f"  Loaded encoder weights from {ckpt_path} (reinit-head mode)")
            print(f"  Missing keys: {len(missing)}  Unexpected keys: {len(unexpected)}")
            # Re-initialise the head (classifier) with fresh random weights
            classifier = getattr(model.backbone, "classifier", None) or getattr(model.backbone, "head", None)
            if classifier is not None:
                for m in classifier.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                    elif isinstance(m, nn.BatchNorm1d):
                        nn.init.ones_(m.weight)
                        nn.init.zeros_(m.bias)
                print(f"  Head re-initialised with Xavier uniform weights")
        else:
            model.backbone.load_state_dict(sd, strict=True)
            print(f"  Loaded weights from {ckpt_path} for fine-tuning")
    total_params = sum(p.numel() for p in model.parameters())

    # ---- Build datasets -----------------------------------------------------
    print(f"\n[2/4] Loading data...")

    if args.dataset_dir:
        ds_root = Path(args.dataset_dir) if Path(args.dataset_dir).is_absolute() else ws / args.dataset_dir
        print(f"  Loading pre-generated dataset from {ds_root}")
        train_ds = PregenDataset(ds_root / "train")
        val_ds   = PregenDataset(ds_root / "val")
        renderer = None
        print(f"  Train: {len(train_ds)} pairs (pre-generated, no live rendering)")
        print(f"  Val  : {len(val_ds)} pairs (pre-generated, no live rendering)")
    else:
        math_txt = ws / args.gd_dir / "math.txt"
        formulas = load_formulas(math_txt)
        extracted_root = ws / args.extracted_root
        renderer = FormulaRenderer(
            height=96, width=384, dpi=140,
            cache_max_entries=args.renderer_cache_max,
        )

        def _pairs_for_mode(target, epochs):
            if args.pair_target_mode == "per_epoch":
                return target
            return max(1, target // max(1, epochs))

        train_rows = collect_split_samples(extracted_root, "train", formulas,
                                           args.train_samples, args.seed)
        val_rows = collect_split_samples(extracted_root, "val", formulas,
                                         args.val_samples, args.seed + 1)
        train_rows = filter_renderable_samples(train_rows, renderer, args.min_render_ink)
        val_rows = filter_renderable_samples(val_rows, renderer, args.min_render_ink)

        target_train = _pairs_for_mode(args.target_train_pairs, args.epochs)
        target_val = _pairs_for_mode(args.target_val_pairs, args.epochs)

        train_ds = HFPairDataset(
            rows=train_rows, renderer=renderer,
            target_pairs=target_train,
            hard_negative_prob=args.hard_negative_prob,
            min_render_ink=args.min_render_ink,
            train_mode=True, base_seed=args.seed,
            input_cache_max_entries=args.input_cache_max,
        )
        val_ds = HFPairDataset(
            rows=val_rows, renderer=renderer,
            target_pairs=target_val,
            hard_negative_prob=args.hard_negative_prob,
            min_render_ink=args.min_render_ink,
            train_mode=False, base_seed=args.seed + 100,
            input_cache_max_entries=args.input_cache_max,
        )
        print(f"  Train: {len(train_rows)} images -> {len(train_ds)} pairs/epoch")
        print(f"  Val  : {len(val_rows)} images -> {len(val_ds)} pairs/eval")

    # ---- HF Trainer ---------------------------------------------------------
    print(f"\n[3/4] Training...")
    total_steps = (len(train_ds) // args.batch_size) * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    training_args = TrainingArguments(
        output_dir=str(out_dir / "hf_ckpts"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        auto_find_batch_size=args.auto_find_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=args.fp16 and torch.cuda.is_available(),
        bf16=False,
        logging_steps=20,
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
    )

    callbacks = [
        PairResampleMemoryCallback(
            resample_pairs_each_epoch=args.resample_pairs_each_epoch,
            clear_render_cache_each_epoch=args.clear_render_cache_each_epoch,
            clear_input_cache_each_epoch=args.clear_input_cache_each_epoch,
            clear_cuda_cache_each_epoch=args.clear_cuda_cache_each_epoch,
            log_memory_each_epoch=args.log_memory_each_epoch,
        )
    ]
    if args.hard_negative_prob_end >= 0:
        callbacks.append(HardNegativeCurriculumCallback(
            start_prob=args.hard_negative_prob,
            end_prob=args.hard_negative_prob_end,
            num_epochs=args.epochs,
        ))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    trainer.train()
    metrics = trainer.evaluate()
    print(f"\nFinal eval: {metrics}")

    # ---- Save checkpoint + pair-type eval -----------------------------------
    print(f"\n[4/4] Saving artifacts...")
    torch.save(
        {"state_dict": model.backbone.state_dict(), "metrics": metrics},
        out_dir / "comparator_v2.pt",
    )

    if not args.dataset_dir:
        val_ds_pt = HFPairDataset(
            rows=val_rows, renderer=renderer,
            target_pairs=min(5000, target_val),
            hard_negative_prob=args.hard_negative_prob,
            min_render_ink=args.min_render_ink,
            train_mode=False, base_seed=999,
            input_cache_max_entries=args.input_cache_max,
        )
        pt_results = evaluate_by_pair_type(model, val_ds_pt, device, args.batch_size, 42)
        with open(out_dir / "pair_type_eval.json", "w") as f:
            json.dump(pt_results, f, indent=2)
        print(f"  Pair-type eval: {pt_results['overall_accuracy']:.4f}")

    summary = {
        "arch": args.arch,
        "backbone_type": args.backbone_type,
        "trainable_params": trainable,
        "total_params": total_params,
        "train_rows": len(train_ds),
        "val_rows": len(val_ds),
        "target_train_pairs": len(train_ds),
        "target_val_pairs": len(val_ds),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hard_negative_prob": args.hard_negative_prob,
        "lora_rank": args.lora_rank,
        "contrastive_temp": args.contrastive_temp if args.arch == "v2c" else None,
        "metrics": metrics,
        "pair_type_eval": pt_results if not args.dataset_dir else None,
    }
    with open(out_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Results in {out_dir}")
    print(f"  Val accuracy : {metrics.get('eval_accuracy', 0):.4f}")
    print(f"  Pair-type    : {pt_results['overall_accuracy']:.4f}")


if __name__ == "__main__":
    main()
