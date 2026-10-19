#!/usr/bin/env python3
"""
diagnose_texture_channels.py — check what fraction of backbone channels
TextureChannelIdentifier actually flags as texture-dominant, on real
Cityscapes images, for whichever backbone/checkpoint is given.

This exists to check whether TCPR's texture-channel identification
(calibrated in the README around "~35% of channels in SegFormer-B2
stage 2" at threshold=0.45) transfers sensibly to a different backbone's
feature statistics, or whether the threshold (or the choice of which
layer to apply it to) needs to change per-backbone.

Usage::

    python scripts/diagnose_texture_channels.py \\
        --backbone deeplabv3plus --checkpoint runs/tcpr_deeplab_pretrained/best.pth \\
        --pretrained --data_root datasets/cityscapes --num_images 20
"""

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_ROOT  = os.path.dirname(_SCRIPT_DIR)
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

import torch

from models.robustseg import RobustSegWrapper
from models.segformer_wrapper import build_segformer
from models.texture_identifier import TextureChannelIdentifier
from data.datasets import CityscapesDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="segformer", choices=["segformer", "deeplabv3plus"])
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--num_classes", type=int, default=19)
    p.add_argument("--data_root", required=True)
    p.add_argument("--num_images", type=int, default=20)
    p.add_argument("--thresholds", default="0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60")
    p.add_argument("--device", default="")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.backbone == "segformer":
        backbone, head = build_segformer(pretrained=args.pretrained, num_classes=args.num_classes)
    else:
        from models.deeplab_wrapper import build_deeplabv3plus
        backbone, head = build_deeplabv3plus(pretrained=args.pretrained, num_classes=args.num_classes)

    model = RobustSegWrapper(backbone, head)
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location="cpu")
        state = payload.get("model_state", payload)
        model.load_state_dict(state, strict=True)
    model.to(device).eval()

    ds = CityscapesDataset(args.data_root, split="train", crop_size=(512, 1024))
    identifier = TextureChannelIdentifier()

    all_ratios = []
    n = min(args.num_images, len(ds))
    with torch.no_grad():
        for i in range(n):
            image, _ = ds[i]
            x = image.unsqueeze(0).to(device)
            primary = model.extract_features(x)
            ratios = identifier.compute_texture_ratios(primary)
            all_ratios.append(ratios.cpu())

    ratios_stacked = torch.stack(all_ratios, dim=0)  # (n_images, C)
    mean_ratios = ratios_stacked.mean(dim=0)         # (C,)

    C = mean_ratios.numel()
    print(f"[diag] backbone={args.backbone}  primary feature channels={C}")
    print(f"[diag] tau (texture ratio) stats: min={mean_ratios.min():.4f} "
          f"max={mean_ratios.max():.4f} mean={mean_ratios.mean():.4f} "
          f"median={mean_ratios.median():.4f}")

    thresholds = [float(t) for t in args.thresholds.split(",")]
    print(f"[diag] fraction of channels flagged texture-dominant at each threshold:")
    for t in thresholds:
        frac = (mean_ratios > t).float().mean().item()
        print(f"  threshold={t:.2f}  ->  {frac*100:.1f}% of {C} channels flagged")


if __name__ == "__main__":
    main()
