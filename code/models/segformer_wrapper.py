"""
segformer_wrapper.py — builds a SegFormer-style model.

Two paths:
  1. If timm is installed, use MixTransformer from timm (mit_b2) with a
     custom lightweight decode head.
  2. If timm is not available, fall back to a fully self-contained
     SimpleMixTransformer + SimpleSegFormerHead implemented here.

The returned (backbone, head) tuple is compatible with RobustSegWrapper.
The backbone returns a 4-element list of feature maps at 1/4, 1/8, 1/16,
1/32 of the input resolution.  The head fuses all four scales and produces
(B, num_classes, H/4, W/4) logits (upsampled to full resolution by
RobustSegWrapper).
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_divisible(v: float, divisor: int = 8, min_value: int = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# ---------------------------------------------------------------------------
# Patch Embedding
# ---------------------------------------------------------------------------

class OverlapPatchEmbed(nn.Module):
    """Image or feature-map to patch embedding with overlapping patches."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int = 7,
        stride: int = 4,
        padding: int = 3,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=stride, padding=padding,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = self.proj(x)          # (B, C, H', W')
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        x = self.norm(x)
        return x, H, W


# ---------------------------------------------------------------------------
# Efficient Self-Attention (sequence-reduction)
# ---------------------------------------------------------------------------

class EfficientSelfAttention(nn.Module):
    """
    Multi-head self-attention with spatial sequence reduction (MixTransformer style).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        reduction_ratio: int = 1,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.reduction_ratio = reduction_ratio

        self.q  = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        if reduction_ratio > 1:
            self.sr   = nn.Conv2d(dim, dim, kernel_size=reduction_ratio, stride=reduction_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.reduction_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
        else:
            x_ = x

        kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x


# ---------------------------------------------------------------------------
# MLP block used inside transformer stage
# ---------------------------------------------------------------------------

class MixMLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.dw  = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)
        self.act  = nn.GELU()

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        x = self.act(self.fc1(x))
        # Depth-wise conv for local mixing
        x = x.transpose(1, 2).reshape(B, -1, H, W)
        x = self.dw(x).flatten(2).transpose(1, 2)
        x = self.drop(self.fc2(x))
        return x


# ---------------------------------------------------------------------------
# Single transformer block
# ---------------------------------------------------------------------------

class SimpleMixTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        reduction_ratio: int = 1,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = EfficientSelfAttention(
            dim, num_heads=num_heads,
            reduction_ratio=reduction_ratio,
            attn_drop=attn_drop, proj_drop=drop,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = MixMLP(dim, mlp_ratio=mlp_ratio, drop=drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x), H, W)
        return x


# ---------------------------------------------------------------------------
# One SegFormer stage (patch embed + N transformer blocks)
# ---------------------------------------------------------------------------

class MixTransformerStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int,
        stride: int,
        padding: int,
        depth: int,
        num_heads: int,
        reduction_ratio: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_embed = OverlapPatchEmbed(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            stride=stride,
            padding=padding,
        )
        self.blocks = nn.ModuleList([
            SimpleMixTransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                reduction_ratio=reduction_ratio,
                mlp_ratio=mlp_ratio,
                drop=drop,
                attn_drop=attn_drop,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x, H, W = self.patch_embed(x)
        for block in self.blocks:
            x = block(x, H, W)
        x = self.norm(x)
        return x, H, W


# ---------------------------------------------------------------------------
# SimpleMixTransformer backbone (MiT-B2 scale)
# ---------------------------------------------------------------------------

class SimpleMixTransformer(nn.Module):
    """
    Lightweight MixTransformer backbone inspired by MiT-B2 from SegFormer.
    Returns a list of 4 feature maps at 1/4, 1/8, 1/16, 1/32 resolution.

    Embed dims and depths are scaled down relative to MiT-B2 so the model
    runs without pretrained weights while still being structurally correct.
    """

    # (embed_dim, depth, num_heads, reduction_ratio)
    STAGE_CONFIGS = [
        (64,  3, 1, 8),   # stage 1 — 1/4  res
        (128, 4, 2, 4),   # stage 2 — 1/8  res
        (320, 6, 5, 2),   # stage 3 — 1/16 res
        (512, 3, 8, 1),   # stage 4 — 1/32 res
    ]

    def __init__(self, in_channels: int = 3, drop_rate: float = 0.0) -> None:
        super().__init__()
        self.stages = nn.ModuleList()

        prev_dim = in_channels
        patch_sizes   = [7, 3, 3, 3]
        strides       = [4, 2, 2, 2]
        paddings      = [3, 1, 1, 1]

        for i, (embed_dim, depth, num_heads, rr) in enumerate(self.STAGE_CONFIGS):
            stage = MixTransformerStage(
                in_channels=prev_dim,
                embed_dim=embed_dim,
                patch_size=patch_sizes[i],
                stride=strides[i],
                padding=paddings[i],
                depth=depth,
                num_heads=num_heads,
                reduction_ratio=rr,
                drop=drop_rate,
            )
            self.stages.append(stage)
            prev_dim = embed_dim

        self._embed_dims = [cfg[0] for cfg in self.STAGE_CONFIGS]

    @property
    def embed_dims(self) -> List[int]:
        return self._embed_dims

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outs = []
        for stage in self.stages:
            x_seq, H, W = stage(x)
            B, N, C = x_seq.shape
            # Reshape tokens back to spatial map (B, C, H, W)
            feat = x_seq.permute(0, 2, 1).reshape(B, C, H, W)
            outs.append(feat)
            x = feat
        return outs


# ---------------------------------------------------------------------------
# SimpleSegFormerHead
# ---------------------------------------------------------------------------

class SimpleSegFormerHead(nn.Module):
    """
    Lightweight decode head.

    1. Project each scale to a common embedding dimension via 1×1 conv.
    2. Upsample all scales to the coarsest scale (1/4 of input).
    3. Concatenate and fuse with a 1×1 conv + BN + ReLU.
    4. Classify with a final 1×1 conv.
    """

    def __init__(
        self,
        in_channels: List[int],
        embed_dim: int = 256,
        num_classes: int = 19,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # Linear (1×1 conv) projections for each scale
        self.linears = nn.ModuleList([
            nn.Conv2d(c, embed_dim, kernel_size=1) for c in in_channels
        ])

        # Fusion
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * len(in_channels), embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout)
        self.classifier = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        # Target spatial size = size of the first (coarsest/largest) feature map
        target_size = features[0].shape[-2:]

        projected = []
        for i, feat in enumerate(features):
            p = self.linears[i](feat)
            if p.shape[-2:] != target_size:
                p = F.interpolate(p, size=target_size, mode="bilinear", align_corners=False)
            projected.append(p)

        x = torch.cat(projected, dim=1)  # (B, embed_dim * num_scales, H', W')
        x = self.fuse(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# Timm-based wrapper (optional) — kept for backwards compatibility, but timm
# does not actually ship MiT/SegFormer models under any version; this path
# is unreachable via build_segformer() and pretrained loading goes through
# HFMiTBackbone below instead.
# ---------------------------------------------------------------------------

class TimmMiTBackbone(nn.Module):
    """Wraps a timm MiT encoder so it returns a list of 4 feature maps."""

    def __init__(self, timm_model) -> None:
        super().__init__()
        self.model = timm_model

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.model.forward_features(x)


# ---------------------------------------------------------------------------
# HuggingFace-based wrapper for the real ImageNet-pretrained MiT-B2 encoder
# ---------------------------------------------------------------------------

class HFMiTBackbone(nn.Module):
    """
    Wraps HuggingFace's `transformers.SegformerModel` (nvidia/mit-b2 weights)
    so it returns a list of 4 spatial feature maps, matching the interface
    SimpleMixTransformer exposes. HF's encoder already reshapes each stage's
    hidden state to (B, C, H, W) at 1/4, 1/8, 1/16, 1/32 resolution with
    channel dims [64, 128, 320, 512] — identical to SimpleMixTransformer's
    STAGE_CONFIGS, so both backbones are drop-in compatible with the same
    SimpleSegFormerHead.
    """

    def __init__(self, hf_model) -> None:
        super().__init__()
        self.model = hf_model

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        out = self.model(x, output_hidden_states=True)
        return list(out.hidden_states)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def build_segformer(
    pretrained: bool = False,
    num_classes: int = 19,
    embed_dim: int = 256,
    drop_rate: float = 0.0,
) -> Tuple[nn.Module, nn.Module]:
    """
    Build a SegFormer-style (backbone, head) pair.

    Uses HuggingFace's ImageNet-pretrained `nvidia/mit-b2` encoder when
    pretrained=True; otherwise falls back to the self-contained
    SimpleMixTransformer trained from random initialization. (timm does not
    ship MiT/SegFormer models under any version, despite the historical
    naming below — that path is dead and unreachable.)

    Args:
        pretrained: load ImageNet pretrained weights (HuggingFace path only).
        num_classes: number of segmentation classes.
        embed_dim: channel width of the decode head.
        drop_rate: dropout rate.

    Returns:
        (backbone, head) both as nn.Module.
    """
    if pretrained:
        from transformers import SegformerModel
        _hf_model = SegformerModel.from_pretrained("nvidia/mit-b2")
        backbone = HFMiTBackbone(_hf_model)
        in_channels = [64, 128, 320, 512]
    else:
        backbone = SimpleMixTransformer(in_channels=3, drop_rate=drop_rate)
        in_channels = backbone.embed_dims

    head = SimpleSegFormerHead(
        in_channels=in_channels,
        embed_dim=embed_dim,
        num_classes=num_classes,
        dropout=drop_rate,
    )

    return backbone, head
