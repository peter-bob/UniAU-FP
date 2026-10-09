"""
clip_vit_ablation.py — Ablation Study Model Variants
=====================================================
Four progressively complete architectures for ablation experiments.

Config 1 — Baseline
    Pure CLIP ViT backbone, global CLS token → MLP regression.
    No anatomical region division, no language modality.

Config 2 — Baseline + MAD
    Adds SpatialMultiScaleAsymmetry on top of Config 1.
    Extracts upper/middle/lower L-R difference features;
    flattens and feeds into an MLP head.
    Verifies that refined spatial asymmetric feature extraction
    improves regression over a global token alone.

Config 3 — Baseline + MAD + CrossModal
    Adds CLIP text encoder + CrossModalEncoder on top of Config 2.
    Text AU descriptions serve as query vectors; local region features
    are the sole K/V memory (no global leakage).
    Verifies language-guided local feature alignment.

Config 4 — Full Model (= clip_vit_5)
    Adds auxiliary asymmetry prediction branch (with gradient scaling)
    on top of Config 3.
    Returns (logits, asym_pred) tuple for multi-task joint training.
    Verifies explicit asymmetry supervision's anti-overfitting effect.

Usage:
    from models.clip_vit_ablation import build_ablation_model
    model = build_ablation_model(config_id=1, args=args, clip_model=clip_model)
"""

from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# Shared config dataclass
# ===========================================================================

@dataclass
class ClipTrainConfig:
    freeze_clip: bool = True
    unfreeze_last_n: int = 2
    head_type: str = "regression"
    output_activation: str = "none"
    dropout: float = 0.2


# ===========================================================================
# Shared utilities
# ===========================================================================

class GradScalerLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None

def scale_gradient(x, scale=0.1):
    return GradScalerLayer.apply(x, scale)


class ResidualMLP(nn.Module):
    """Per-AU prediction head: in_dim → hidden_dim → 1."""
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, out_dim)
        self.shortcut = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()

    def forward(self, x):
        res = self.shortcut(x)
        x = self.fc1(x)
        x = self.ln(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.out_proj(self.act(x + res))


# ===========================================================================
# MAD — Spatial Multi-Scale Asymmetry (shared by Configs 2/3/4)
# ===========================================================================

class SpatialMultiScaleAsymmetry(nn.Module):
    """
    Extracts upper/middle/lower L-R asymmetry features from patch tokens.
    Input:  patch_tokens [B, 196, embed_dim]
    Output: region_features [B, 3, out_dim]  — mixed local features
            asym_features   [B, 3, out_dim]  — pure L-R difference features
    """
    def __init__(self, embed_dim=768, out_dim=256, dropout=0.1):
        super().__init__()
        self.REGION_NAMES = ["upper", "middle", "lower"]
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU()
        )
        self.channel_gate = nn.Sequential(
            nn.Linear(out_dim, out_dim // 4), nn.GELU(),
            nn.Linear(out_dim // 4, out_dim), nn.Sigmoid()
        )
        self.mix_gate = nn.ModuleDict({
            rn: nn.Sequential(nn.Linear(out_dim * 2, out_dim), nn.Sigmoid())
            for rn in self.REGION_NAMES
        })

    def _get_regions(self, tokens):
        B, N, D = tokens.shape
        grid = tokens.view(B, 14, 14, D)
        up   = grid[:, :6, :, :]
        mid  = grid[:, 4:10, :, :]
        down = grid[:, 8:, :, :]
        left  = {"upper": up[:, :, :7, :].mean(dim=(1,2)),
                 "middle": mid[:, :, :7, :].mean(dim=(1,2)),
                 "lower": down[:, :, :7, :].mean(dim=(1,2))}
        right = {"upper": up[:, :, 7:, :].flip(dims=[2]).mean(dim=(1,2)),
                 "middle": mid[:, :, 7:, :].flip(dims=[2]).mean(dim=(1,2)),
                 "lower": down[:, :, 7:, :].flip(dims=[2]).mean(dim=(1,2))}
        return left, right

    def forward(self, patch_tokens):
        patch_tokens = self.proj(patch_tokens)
        left_reg, right_reg = self._get_regions(patch_tokens)
        region_list, asym_list = [], []
        for rn in self.REGION_NAMES:
            L, R = left_reg[rn], right_reg[rn]
            diff = L - R
            gate = self.channel_gate(diff)
            asym_feat = diff * gate
            asym_list.append(asym_feat)
            abs_feat = L + R
            mix_w = self.mix_gate[rn](torch.cat([asym_feat, abs_feat], dim=-1))
            region_list.append(mix_w * asym_feat + (1 - mix_w) * abs_feat)
        return torch.stack(region_list, dim=1), torch.stack(asym_list, dim=1)


# ===========================================================================
# Cross-Modal Encoder (shared by Configs 3/4)
# ===========================================================================

class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model * 2, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, memory):
        q = self.norm1(query)
        attn_out, _ = self.cross_attn(q, memory, memory)
        query = query + self.dropout(attn_out)
        query = query + self.dropout(self.ffn(self.norm2(query)))
        return query


class CrossModalEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionLayer(d_model, nhead, dropout) for _ in range(num_layers)
        ])

    def forward(self, query, memory):
        for layer in self.layers:
            query = layer(query, memory)
        return query


# ===========================================================================
# Asymmetry Gaussian Head (used by Config 4 only)
# ===========================================================================

class AsymmetryGaussianHead(nn.Module):
    def __init__(self, d_model=256, num_pairs=5, num_bins=51, dropout=0.1):
        super().__init__()
        self.num_pairs = num_pairs
        self.num_bins = num_bins
        self.shared = nn.Sequential(
            nn.Linear(d_model * 3, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
        )
        self.reg_head = nn.Linear(128, num_pairs)
        self.dist_head = nn.Linear(128, num_pairs * num_bins)

    def forward(self, asym_features):
        B = asym_features.shape[0]
        feat = self.shared(asym_features.reshape(B, -1))
        return self.reg_head(feat), self.dist_head(feat).view(B, self.num_pairs, self.num_bins)


# ===========================================================================
# Shared CLIP visual extraction mixin
# ===========================================================================

class _CLIPVisualBase(nn.Module):
    """Common CLIP encoder + trainability config shared by all 4 configs."""

    def _build_cfg(self, args):
        return ClipTrainConfig(
            freeze_clip=bool(getattr(args, "freeze_clip", True)),
            unfreeze_last_n=int(getattr(args, "unfreeze_last_n", 2)),
            head_type=str(getattr(args, "head_type", "regression")),
            output_activation=str(getattr(args, "output_activation", "none")),
            dropout=float(getattr(args, "dropout", 0.2))
        )

    def _configure_clip_trainability(self):
        for p in self.parameters():
            p.requires_grad = False
        if not self.cfg.freeze_clip or self.cfg.unfreeze_last_n > 0:
            blocks = self.clip_visual.transformer.resblocks
            n = len(blocks) if not self.cfg.freeze_clip else self.cfg.unfreeze_last_n
            for b in blocks[-n:]:
                for p in b.parameters():
                    p.requires_grad = True
            for p in self.clip_visual.ln_post.parameters():
                p.requires_grad = True

    def _get_visual_features(self, x):
        ctx = (torch.no_grad()
               if self.cfg.freeze_clip and self.cfg.unfreeze_last_n <= 0
               else torch.enable_grad())
        with ctx:
            x = self.clip_visual.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
            cls_t = (self.clip_visual.class_embedding.to(x.dtype)
                     + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device))
            x = torch.cat([cls_t, x], dim=1) + self.clip_visual.positional_embedding.to(x.dtype)
            x = self.clip_visual.ln_pre(x).permute(1, 0, 2)
            x = self.clip_visual.transformer(x)
            x = x.permute(1, 0, 2)
            x = self.clip_visual.ln_post(x)
            cls_token = x[:, 0, :]
            patch_tokens = x[:, 1:, :]
            global_feat = (cls_token @ self.clip_visual.proj
                           if self.clip_visual.proj is not None else cls_token)
        return global_feat, patch_tokens

    def _apply_output_activation(self, y):
        act = self.cfg.output_activation.lower()
        if act == "sigmoid": return torch.sigmoid(y)
        if act == "tanh":    return torch.tanh(y)
        return y


# ===========================================================================
# Config 1 - C2 minus text stream (weakest reverse-ablation model)
# ===========================================================================

class CLIPVIT_C1_NoText(_CLIPVisualBase):
    """
    Reverse ablation Config 1.
    Removes the text stream after MAD has already been removed. To keep this
    as a clean weaker visual-only baseline, raw ViT patch tokens are mean-pooled
    and sent to a shared regression head. No text, no MAD, no aux.
    """
    def __init__(self, args, clip_model):
        super().__init__()
        self.cfg = self._build_cfg(args)
        self.num_classes = int(getattr(args, "classes", 10))
        self.clip_visual = clip_model.visual
        self.embed_dim = self.clip_visual.transformer.width
        self.d_model = 256
        self._configure_clip_trainability()

        drop = self.cfg.dropout
        self.visual_proj = nn.Sequential(
            nn.Linear(self.embed_dim, self.d_model), nn.LayerNorm(self.d_model), nn.GELU()
        )
        self.reg_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, 256),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(256, self.num_classes),
        )
        self.out_scale = nn.Parameter(torch.ones(self.num_classes))
        self.out_bias  = nn.Parameter(torch.zeros(self.num_classes))

    def forward(self, images, label_tokens=None):
        _, patch_tokens = self._get_visual_features(images)          # [B, 196, 768]
        patch_memory = self.visual_proj(patch_tokens)                # [B, 196, 256]
        pooled = patch_memory.mean(dim=1)                            # [B, 256]
        logits = self.reg_head(pooled)                               # [B, 10]
        logits = logits * self.out_scale + self.out_bias
        return self._apply_output_activation(logits)


# ===========================================================================
# Config 2 - C3 minus MAD
# ===========================================================================

class CLIPVIT_C2_NoMAD(_CLIPVisualBase):
    """
    Reverse ablation Config 2.
    Removes SpatialMultiScaleAsymmetry, but keeps and strengthens the original
    CLIP text stream. Text-guided AU queries attend to raw patch memory, then
    concatenate the attended AU feature with a global visual fallback. No MAD
    and no auxiliary asymmetry branch.
    """
    def __init__(self, args, clip_model):
        super().__init__()
        self.cfg = self._build_cfg(args)
        self.num_classes = int(getattr(args, "classes", 10))
        self.clip_visual = clip_model.visual
        self.clip_text   = clip_model.encode_text
        self.embed_dim = self.clip_visual.transformer.width
        self.d_model = 256
        self._configure_clip_trainability()

        drop = self.cfg.dropout
        self.visual_proj = nn.Sequential(
            nn.Linear(self.embed_dim, self.d_model), nn.LayerNorm(self.d_model), nn.GELU()
        )
        self.text_proj = nn.Sequential(
            nn.Linear(self.clip_visual.output_dim, self.d_model), nn.LayerNorm(self.d_model)
        )
        self.au_query_embed = nn.Parameter(
            torch.randn(1, self.num_classes, self.d_model) * 0.02
        )
        self.text_gate = nn.Parameter(torch.full((1, self.num_classes, 1), -0.5))
        self.cross_encoder = CrossModalEncoder(self.d_model, nhead=8, num_layers=2, dropout=drop)
        self.reg_head = ResidualMLP(self.d_model * 2, 256, 1, drop)
        self.out_scale = nn.Parameter(torch.ones(self.num_classes))
        self.out_bias  = nn.Parameter(torch.zeros(self.num_classes))

    def forward(self, images, label_tokens):
        B = images.shape[0]
        _, patch_tokens = self._get_visual_features(images)          # [B, 196, 768]
        patch_memory = self.visual_proj(patch_tokens)                # [B, 196, 256]

        with torch.no_grad():
            text_embeds = self.clip_text(label_tokens).float()
            text_embeds = F.normalize(text_embeds, dim=-1)
        text_queries = self.text_proj(text_embeds).unsqueeze(0).expand(B, -1, -1)
        text_weight = torch.sigmoid(self.text_gate).expand(B, -1, -1)
        au_queries = self.au_query_embed + text_weight * text_queries  # [B, 10, 256]

        f_au = self.cross_encoder(au_queries, patch_memory)          # [B, 10, 256]
        global_ctx = patch_memory.mean(dim=1, keepdim=True).expand(-1, self.num_classes, -1)
        f_au = torch.cat([f_au, global_ctx], dim=-1)                 # [B, 10, 512]
        logits = self.reg_head(f_au).squeeze(-1)                     # [B, 10]
        logits = logits * self.out_scale + self.out_bias
        return self._apply_output_activation(logits)


# ===========================================================================
# Config 3 - C4 minus auxiliary asymmetry branch
# ===========================================================================

class CLIPVIT_C3_NoAux(_CLIPVisualBase):
    """
    Reverse ablation Config 3.
    Keeps the full model's MAD + original CLIP text stream + cross-modal fusion,
    and removes only the auxiliary asymmetry prediction branch. This is the
    strongest single-task model among C1-C3.
    """
    def __init__(self, args, clip_model):
        super().__init__()
        self.cfg = self._build_cfg(args)
        self.num_classes = int(getattr(args, "classes", 10))
        self.clip_visual = clip_model.visual
        self.clip_text   = clip_model.encode_text
        self.embed_dim   = self.clip_visual.transformer.width
        self.d_model     = 256
        self._configure_clip_trainability()

        drop = self.cfg.dropout
        self.mad = SpatialMultiScaleAsymmetry(self.embed_dim, self.d_model, drop)
        self.text_proj = nn.Sequential(
            nn.Linear(self.clip_visual.output_dim, self.d_model), nn.LayerNorm(self.d_model)
        )
        self.au_query_embed = nn.Parameter(
            torch.randn(1, self.num_classes, self.d_model) * 0.02
        )
        self.cross_encoder = CrossModalEncoder(self.d_model, nhead=8, num_layers=2, dropout=drop)
        self.reg_head = ResidualMLP(self.d_model, 256, 1, drop)
        self.out_scale = nn.Parameter(torch.ones(self.num_classes))
        self.out_bias  = nn.Parameter(torch.zeros(self.num_classes))

    def forward(self, images, label_tokens):
        B = images.shape[0]
        _, patch_tokens = self._get_visual_features(images)         # [B, 196, 768]
        region_features, _ = self.mad(patch_tokens)                 # [B, 3, 256]

        with torch.no_grad():
            text_embeds = self.clip_text(label_tokens).float()
        text_queries = self.text_proj(text_embeds).unsqueeze(0).expand(B, -1, -1)
        au_queries   = text_queries + self.au_query_embed            # [B, 10, 256]

        f_au   = self.cross_encoder(au_queries, region_features)     # [B, 10, 256]
        logits = self.reg_head(f_au).squeeze(-1)                     # [B, 10]
        logits = logits * self.out_scale + self.out_bias
        return self._apply_output_activation(logits)


# ===========================================================================
# Config 4 — Full Model (= clip_vit_5)
# ===========================================================================

class CLIPVIT_C4_Full(_CLIPVisualBase):
    """
    Ablation Config 4 — Complete architecture.
    Adds auxiliary asymmetry prediction branch (with gradient scaling ×0.1)
    to Config 3. Multi-task joint training provides explicit asymmetry
    supervision and anti-overfitting regularization.
    Returns: (logits [B,10], asym_pred tuple)
    """
    def __init__(self, args, clip_model):
        super().__init__()
        self.cfg = self._build_cfg(args)
        self.num_classes  = int(getattr(args, "classes", 10))
        self.num_au_pairs = self.num_classes // 2
        self.clip_visual  = clip_model.visual
        self.clip_text    = clip_model.encode_text
        self.embed_dim    = self.clip_visual.transformer.width
        self.d_model      = 256
        self._configure_clip_trainability()

        drop = self.cfg.dropout
        self.mad = SpatialMultiScaleAsymmetry(self.embed_dim, self.d_model, drop)
        self.text_proj = nn.Sequential(
            nn.Linear(self.clip_visual.output_dim, self.d_model), nn.LayerNorm(self.d_model)
        )
        self.au_query_embed = nn.Parameter(
            torch.randn(1, self.num_classes, self.d_model) * 0.02
        )
        self.cross_encoder = CrossModalEncoder(self.d_model, nhead=8, num_layers=2, dropout=drop)
        self.reg_head = ResidualMLP(self.d_model, 256, 1, drop)
        self.out_scale = nn.Parameter(torch.ones(self.num_classes))
        self.out_bias  = nn.Parameter(torch.zeros(self.num_classes))

        asym_num_bins = int(getattr(args, "asym_num_bins", 51))
        self.asym_aux_head = AsymmetryGaussianHead(
            d_model=self.d_model, num_pairs=self.num_au_pairs,
            num_bins=asym_num_bins, dropout=drop
        )
        self._init_aux_weights()

    def _init_aux_weights(self):
        for m in self.asym_aux_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, images, label_tokens):
        B = images.shape[0]
        _, patch_tokens = self._get_visual_features(images)         # [B, 196, 768]
        region_features, asym_features = self.mad(patch_tokens)     # [B, 3, 256] each

        with torch.no_grad():
            text_embeds = self.clip_text(label_tokens).float()
        text_queries = self.text_proj(text_embeds).unsqueeze(0).expand(B, -1, -1)
        au_queries   = text_queries + self.au_query_embed            # [B, 10, 256]

        f_au   = self.cross_encoder(au_queries, region_features)     # [B, 10, 256]
        logits = self.reg_head(f_au).squeeze(-1)                     # [B, 10]
        logits = logits * self.out_scale + self.out_bias
        logits = self._apply_output_activation(logits)

        safe_asym = scale_gradient(asym_features, scale=0.1)
        asym_pred = self.asym_aux_head(safe_asym)
        return logits, asym_pred


# ===========================================================================
# Factory function
# ===========================================================================

_MODEL_MAP = {
    1: CLIPVIT_C1_NoText,
    2: CLIPVIT_C2_NoMAD,
    3: CLIPVIT_C3_NoAux,
    4: CLIPVIT_C4_Full,
}

_CONFIG_NAMES = {
    1: "C1_NoText_NoMAD_NoAux",
    2: "C2_NoMAD_NoAux",
    3: "C3_NoAux",
    4: "C4_Full",
}

def build_ablation_model(config_id: int, args, clip_model):
    """
    Build the ablation model for the given config ID (1–4).
    Returns a CLIPVIT model instance.
    """
    if config_id not in _MODEL_MAP:
        raise ValueError(f"config_id must be 1, 2, 3, or 4. Got: {config_id}")
    cls = _MODEL_MAP[config_id]
    print(f"[AblationModel] Building {_CONFIG_NAMES[config_id]} (Config {config_id})")
    return cls(args, clip_model)

def get_config_name(config_id: int) -> str:
    return _CONFIG_NAMES.get(config_id, f"Config{config_id}")


# ===========================================================================
# Optimizer param group helper — config-aware
# ===========================================================================

def get_head_params(model, config_id: int):
    """
    Returns the list of trainable head parameters for reverse ablation.
    Backbone params (unfrozen CLIP visual blocks) are everything else.
    """
    head_params = []

    if config_id == 1:
        head_params += list(model.visual_proj.parameters())
        head_params += list(model.reg_head.parameters())
        head_params += [model.out_scale, model.out_bias]

    elif config_id == 2:
        head_params += list(model.visual_proj.parameters())
        head_params += list(model.text_proj.parameters())
        head_params += [model.au_query_embed]
        head_params += [model.text_gate]
        head_params += list(model.cross_encoder.parameters())
        head_params += list(model.reg_head.parameters())
        head_params += [model.out_scale, model.out_bias]

    elif config_id == 3:
        head_params += list(model.mad.parameters())
        head_params += list(model.text_proj.parameters())
        head_params += [model.au_query_embed]
        head_params += list(model.cross_encoder.parameters())
        head_params += list(model.reg_head.parameters())
        head_params += [model.out_scale, model.out_bias]

    elif config_id == 4:
        head_params += list(model.mad.parameters())
        head_params += list(model.text_proj.parameters())
        head_params += [model.au_query_embed]
        head_params += list(model.cross_encoder.parameters())
        head_params += list(model.reg_head.parameters())
        head_params += [model.out_scale, model.out_bias]
        head_params += list(model.asym_aux_head.parameters())

    return head_params
