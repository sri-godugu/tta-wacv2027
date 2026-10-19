#!/usr/bin/env python3
"""
visualize_comparison.py — qualitative vanilla vs PDA comparison figures.

For a handful of real (corrupted) images from a benchmark, renders a
5-panel figure per image:
    Input | Ground Truth | Vanilla Prediction | PDA Prediction | Correction Map

The correction map highlights, pixel-by-pixel (against ground truth,
ignoring label 255):
    green — vanilla was wrong, PDA fixed it
    red   — vanilla was right, PDA broke it
    gray  — unchanged (both right or both wrong)

This is the figure type that makes a quantitative mIoU delta legible —
useful for the paper's qualitative section, especially for the DeepLab/
Cityscapes-C high-severity case where the quantitative gap is largest.

Usage::

    python scripts/visualize_comparison.py \\
        --checkpoint runs/tcpr_deeplab_pretrained/best.pth --pretrained \\
        --backbone deeplabv3plus \\
        --dataset cityscapes_c --data_root datasets/cityscapes_c \\
        --cs_root datasets/cityscapes --corruption_type fog --severity 5 \\
        --norm_types bn --clean_final \\
        --num_samples 6 --output_dir figures/dl_cc_sev5
"""

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_ROOT  = os.path.dirname(_SCRIPT_DIR)
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.robustseg import RobustSegWrapper
from models.segformer_wrapper import build_segformer
from adaptation.pda import PerturbationDisagreementAdaptor
from data.datasets import ACDCDataset, CityscapesCDataset, FoggyCityscapesDataset, CityscapesDataset


CS19_PALETTE = np.array([
    [128, 64,  128], [244, 35,  232], [70,  70,  70 ], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30 ], [220, 220, 0  ],
    [107, 142, 35 ], [152, 251, 152], [70,  130, 180], [220, 20,  60 ],
    [255, 0,   0  ], [0,   0,   142], [0,   0,   70 ], [0,   60,  100],
    [0,   80,  100], [0,   0,   230], [119, 11,  32 ],
], dtype=np.uint8)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _colourise(pred: torch.Tensor) -> np.ndarray:
    pred_np = pred.cpu().numpy().astype(np.int64)
    out = np.zeros((*pred_np.shape, 3), dtype=np.uint8)
    for c in range(len(CS19_PALETTE)):
        out[pred_np == c] = CS19_PALETTE[c]
    out[pred_np == 255] = [0, 0, 0]
    return out


def _denormalize(img: torch.Tensor) -> np.ndarray:
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return (img.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def _correction_map(vanilla_pred, pda_pred, label) -> np.ndarray:
    valid = label != 255
    vanilla_correct = (vanilla_pred == label) & valid
    pda_correct     = (pda_pred == label) & valid

    h, w = label.shape
    out = np.full((h, w, 3), 200, dtype=np.uint8)  # gray = unchanged
    fixed  = (~vanilla_correct) & pda_correct       # green
    broken = vanilla_correct & (~pda_correct)       # red
    out[fixed.numpy()]  = [40, 180, 60]
    out[broken.numpy()] = [200, 40, 40]
    out[~valid.numpy()] = [0, 0, 0]
    return out


def build_dataset(args):
    crop_size = (args.crop_h, args.crop_w)
    if args.dataset == "cityscapes_c":
        return CityscapesCDataset(
            root=args.data_root, cs_root=args.cs_root,
            corruption_type=args.corruption_type, severity=args.severity,
            crop_size=crop_size,
        )
    elif args.dataset == "foggy":
        return FoggyCityscapesDataset(
            root=args.data_root, cs_root=args.cs_root,
            split=args.split, beta=args.fog_beta, crop_size=crop_size,
        )
    elif args.dataset == "acdc":
        return ACDCDataset(
            root=args.data_root, condition=args.condition,
            split=args.split, crop_size=crop_size,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--backbone", default="segformer", choices=["segformer", "deeplabv3plus"])
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--num_classes", type=int, default=19)

    p.add_argument("--dataset", default="cityscapes_c", choices=["cityscapes_c", "foggy", "acdc"])
    p.add_argument("--data_root", required=True)
    p.add_argument("--cs_root", default=None)
    p.add_argument("--corruption_type", default="fog")
    p.add_argument("--severity", type=int, default=5)
    p.add_argument("--fog_beta", default="0.02", choices=["0.005", "0.01", "0.02"])
    p.add_argument("--condition", default="fog")
    p.add_argument("--split", default="val")
    p.add_argument("--crop_h", type=int, default=512)
    p.add_argument("--crop_w", type=int, default=1024)

    p.add_argument("--norm_types", default="bn")
    p.add_argument("--clean_final", action="store_true")
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--lr_pda", type=float, default=1e-3)
    p.add_argument("--texture_threshold", type=float, default=0.45)

    p.add_argument("--num_samples", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    if args.backbone == "segformer":
        backbone, head = build_segformer(pretrained=args.pretrained, num_classes=args.num_classes)
    else:
        from models.deeplab_wrapper import build_deeplabv3plus
        backbone, head = build_deeplabv3plus(pretrained=args.pretrained, num_classes=args.num_classes)

    model = RobustSegWrapper(backbone, head)
    payload = torch.load(args.checkpoint, map_location="cpu")
    state = payload.get("model_state", payload)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    norm_types = frozenset(t.strip() for t in args.norm_types.split(",") if t.strip())

    ds = build_dataset(args)
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(ds), size=min(args.num_samples, len(ds)), replace=False)

    for i, idx in enumerate(indices):
        image, label = ds[idx]
        x = image.unsqueeze(0).to(device)

        with torch.no_grad():
            vanilla_logits = model(x)
            if vanilla_logits.shape[-2:] != label.shape:
                vanilla_logits = F.interpolate(vanilla_logits, size=label.shape, mode="bilinear", align_corners=False)
        vanilla_pred = vanilla_logits.argmax(dim=1)[0].cpu()

        adaptor = PerturbationDisagreementAdaptor(
            model=model, lr=args.lr_pda, K=args.K,
            texture_threshold=args.texture_threshold,
            norm_types=norm_types, clean_final=args.clean_final,
        )
        # Duplicate to a batch of 2 for the adaptation forward passes —
        # architectures with a global-average-pool branch (e.g. DeepLabV3's
        # ASPP) collapse to a 1x1 spatial map, and BatchNorm in train mode
        # (which PDA activates) can't compute variance from a single value
        # per channel at batch_size=1. Both rows are the same image, so we
        # just keep the first row of the result.
        x_batch = x.repeat(2, 1, 1, 1)
        with torch.set_grad_enabled(True):
            pda_logits = adaptor.adapt(x_batch)
        pda_logits = pda_logits[:1]
        if pda_logits.shape[-2:] != label.shape:
            pda_logits = F.interpolate(pda_logits, size=label.shape, mode="bilinear", align_corners=False)
        pda_pred = pda_logits.argmax(dim=1)[0].detach().cpu()
        adaptor.reset()

        img_np      = _denormalize(image)
        gt_colour   = _colourise(label)
        van_colour  = _colourise(vanilla_pred)
        pda_colour  = _colourise(pda_pred)
        corr_map    = _correction_map(vanilla_pred, pda_pred, label)

        valid = label != 255
        van_acc = ((vanilla_pred == label) & valid).float().sum() / valid.float().sum()
        pda_acc = ((pda_pred == label) & valid).float().sum() / valid.float().sum()

        fig, axes = plt.subplots(1, 5, figsize=(28, 6))
        panels = [
            (img_np, "Input"),
            (gt_colour, "Ground Truth"),
            (van_colour, f"Vanilla (pix.acc={van_acc:.3f})"),
            (pda_colour, f"PDA (pix.acc={pda_acc:.3f})"),
            (corr_map, "Correction map\n(green=fixed, red=broken)"),
        ]
        for ax, (arr, title) in zip(axes, panels):
            ax.imshow(arr)
            ax.set_title(title, fontsize=12)
            ax.axis("off")

        plt.tight_layout()
        out_path = os.path.join(args.output_dir, f"sample_{i:02d}_idx{idx}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[viz] saved {out_path}  (vanilla_acc={van_acc:.4f}, pda_acc={pda_acc:.4f})")


if __name__ == "__main__":
    main()
