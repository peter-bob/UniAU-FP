"""Complete UniAU-FP model for unilateral AU intensity estimation."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ClipTrainConfig:
    freeze_clip: bool = True
    unfreeze_last_n: int = 2
    output_activation: str = "none"
    dropout: float = 0.2


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
        residual = self.shortcut(x)
        x = self.drop(self.act(self.ln(self.fc1(x))))
        x = self.fc2(x)
        return self.out_proj(self.act(x + residual))


class SpatialMultiScaleAsymmetry(nn.Module):
    """Extract upper, middle, and lower bilateral features from ViT patches."""

    REGION_NAMES = ("upper", "middle", "lower")

    def __init__(self, embed_dim=768, out_dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU()
        )
        self.channel_gate = nn.Sequential(
            nn.Linear(out_dim, out_dim // 4), nn.GELU(),
            nn.Linear(out_dim // 4, out_dim), nn.Sigmoid(),
        )
        self.mix_gate = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(out_dim * 2, out_dim), nn.Sigmoid())
            for name in self.REGION_NAMES
        })

    def _get_regions(self, tokens):
        batch, _, dim = tokens.shape
        grid = tokens.view(batch, 14, 14, dim)
        vertical_regions = {
            "upper": grid[:, :6],
            "middle": grid[:, 4:10],
            "lower": grid[:, 8:],
        }
        left, right = {}, {}
        for name, region in vertical_regions.items():
            left[name] = region[:, :, :7].mean(dim=(1, 2))
            right[name] = region[:, :, 7:].flip(dims=[2]).mean(dim=(1, 2))
        return left, right

    def forward(self, patch_tokens):
        patch_tokens = self.proj(patch_tokens)
        left, right = self._get_regions(patch_tokens)
        region_features, asym_features = [], []
        for name in self.REGION_NAMES:
            left_feature, right_feature = left[name], right[name]
            diff = left_feature - right_feature
            asym_feature = diff * self.channel_gate(diff)
            symmetric_feature = left_feature + right_feature
            mix = self.mix_gate[name](torch.cat([asym_feature, symmetric_feature], dim=-1))
            region_features.append(mix * asym_feature + (1 - mix) * symmetric_feature)
            asym_features.append(asym_feature)
        return torch.stack(region_features, dim=1), torch.stack(asym_features, dim=1)


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, memory):
        normalized = self.norm1(query)
        attended, _ = self.cross_attn(normalized, memory, memory)
        query = query + self.dropout(attended)
        return query + self.dropout(self.ffn(self.norm2(query)))


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


class AsymmetryGaussianHead(nn.Module):
    def __init__(self, d_model=256, num_pairs=5, num_bins=51, dropout=0.1):
        super().__init__()
        self.num_pairs = num_pairs
        self.num_bins = num_bins
        self.shared = nn.Sequential(
            nn.Linear(d_model * 3, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout)
        )
        self.reg_head = nn.Linear(128, num_pairs)
        self.dist_head = nn.Linear(128, num_pairs * num_bins)

    def forward(self, asym_features):
        batch = asym_features.shape[0]
        features = self.shared(asym_features.reshape(batch, -1))
        regression = self.reg_head(features)
        distribution = self.dist_head(features).view(batch, self.num_pairs, self.num_bins)
        return regression, distribution


class _CLIPVisualBase(nn.Module):
    def _build_cfg(self, args):
        return ClipTrainConfig(
            freeze_clip=bool(getattr(args, "freeze_clip", True)),
            unfreeze_last_n=int(getattr(args, "unfreeze_last_n", 2)),
            output_activation=str(getattr(args, "output_activation", "none")),
            dropout=float(getattr(args, "dropout", 0.2)),
        )

    def _configure_clip_trainability(self):
        for parameter in self.parameters():
            parameter.requires_grad = False
        if not self.cfg.freeze_clip or self.cfg.unfreeze_last_n > 0:
            blocks = self.clip_visual.transformer.resblocks
            count = len(blocks) if not self.cfg.freeze_clip else self.cfg.unfreeze_last_n
            for block in blocks[-count:] if count > 0 else ():
                for parameter in block.parameters():
                    parameter.requires_grad = True
            for parameter in self.clip_visual.ln_post.parameters():
                parameter.requires_grad = True

    def _get_visual_features(self, images):
        context = (
            torch.no_grad()
            if self.cfg.freeze_clip and self.cfg.unfreeze_last_n <= 0
            else nullcontext()
        )
        with context:
            x = self.clip_visual.conv1(images)
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
            cls_token = self.clip_visual.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
            )
            x = torch.cat([cls_token, x], dim=1)
            x = x + self.clip_visual.positional_embedding.to(x.dtype)
            x = self.clip_visual.ln_pre(x).permute(1, 0, 2)
            x = self.clip_visual.transformer(x).permute(1, 0, 2)
            x = self.clip_visual.ln_post(x)
            cls_features = x[:, 0]
            patch_features = x[:, 1:]
            global_features = (
                cls_features @ self.clip_visual.proj
                if self.clip_visual.proj is not None else cls_features
            )
        return global_features, patch_features

    def _apply_output_activation(self, values):
        activation = self.cfg.output_activation.lower()
        if activation == "sigmoid":
            return torch.sigmoid(values)
        if activation == "tanh":
            return torch.tanh(values)
        return values


class CLIPVIT(_CLIPVisualBase):
    """Full UniAU-FP architecture, including auxiliary asymmetry supervision."""

    def __init__(self, args, clip_model):
        super().__init__()
        self.cfg = self._build_cfg(args)
        self.num_classes = int(getattr(args, "classes", 10))
        self.num_au_pairs = self.num_classes // 2
        self.clip_visual = clip_model.visual
        self.clip_text = clip_model.encode_text
        self.embed_dim = self.clip_visual.transformer.width
        self.d_model = 256
        self._configure_clip_trainability()

        dropout = self.cfg.dropout
        self.mad = SpatialMultiScaleAsymmetry(self.embed_dim, self.d_model)
        self.text_proj = nn.Sequential(
            nn.Linear(self.clip_visual.output_dim, self.d_model), nn.LayerNorm(self.d_model)
        )
        self.au_query_embed = nn.Parameter(torch.randn(1, self.num_classes, self.d_model) * 0.02)
        self.cross_encoder = CrossModalEncoder(self.d_model, nhead=8, num_layers=2, dropout=dropout)
        self.reg_head = ResidualMLP(self.d_model, 256, 1, dropout)
        self.out_scale = nn.Parameter(torch.ones(self.num_classes))
        self.out_bias = nn.Parameter(torch.zeros(self.num_classes))
        self.asym_aux_head = AsymmetryGaussianHead(
            self.d_model,
            num_pairs=self.num_au_pairs,
            num_bins=int(getattr(args, "asym_num_bins", 51)),
            dropout=dropout,
        )
        self._init_aux_weights()

    def _init_aux_weights(self):
        for module in self.asym_aux_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, images, label_tokens):
        batch = images.shape[0]
        _, patch_tokens = self._get_visual_features(images)
        region_features, asym_features = self.mad(patch_tokens)

        with torch.no_grad():
            text_features = self.clip_text(label_tokens).float()
        text_queries = self.text_proj(text_features).unsqueeze(0).expand(batch, -1, -1)
        au_queries = text_queries + self.au_query_embed
        au_features = self.cross_encoder(au_queries, region_features)
        predictions = self.reg_head(au_features).squeeze(-1)
        predictions = self._apply_output_activation(predictions * self.out_scale + self.out_bias)

        aux_predictions = self.asym_aux_head(scale_gradient(asym_features, scale=0.1))
        return predictions, aux_predictions


# Kept as an alias for existing evaluation utilities that import this class name.
CLIPVIT_C4_Full = CLIPVIT
