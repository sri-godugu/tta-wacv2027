#!/usr/bin/env python3
"""
train_robustnet.py — RobustNet (Instance-Selective Whitening) baseline
training script, structured identically to train.py so results are
directly comparable (same backbone, data, optimizer, schedule).

Example usage::

    python scripts/train_robustnet.py \\
        --data_root /data/cityscapes \\
        --epochs 80 \\
        --pretrained \\
        --lambda_isw 0.6 \\
        --checkpoint_dir runs/robustnet
"""

import argparse
import os
import random
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_ROOT  = os.path.dirname(_SCRIPT_DIR)
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

import torch
from torch.utils.data import DataLoader

from models.deeplab_wrapper import build_deeplabv3plus
from models.robustnet_wrapper import RobustNetWrapper
from training.trainer import Trainer, PolyLR
from data.datasets import CityscapesDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RobustNet (ISW) Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--num_classes",  type=int, default=19)
    parser.add_argument("--pretrained",   action="store_true",
                        help="Load ImageNet-pretrained ResNet-50 backbone.")
    parser.add_argument("--isw_layers",   type=str, default="layer1,layer2",
                        help="Comma-separated ResNet stage names to apply ISW to.")
    parser.add_argument("--lambda_isw",   type=float, default=0.6)
    parser.add_argument("--isw_margin_frac", type=float, default=0.33)

    parser.add_argument("--data_root",   type=str, required=True)
    parser.add_argument("--crop_h",      type=int, default=512)
    parser.add_argument("--crop_w",      type=int, default=1024)

    parser.add_argument("--epochs",      type=int,   default=80)
    parser.add_argument("--batch_size",  type=int,   default=4)
    parser.add_argument("--lr",          type=float, default=6e-5)
    parser.add_argument("--weight_decay",type=float, default=1e-2)
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--val_interval",type=int,   default=10)

    parser.add_argument("--checkpoint_dir", type=str, default="runs/robustnet")
    parser.add_argument("--log_dir",        type=str, default=None)
    parser.add_argument("--resume",         type=str, default=None)

    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--seed",   type=int, default=0)

    return parser.parse_args()


def main() -> None:
    args   = parse_args()
    set_seed(args.seed)
    device = (
        torch.device(args.device) if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[train] Using device: {device}  seed: {args.seed}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    log_dir = args.log_dir or os.path.join(args.checkpoint_dir, "logs")

    crop_size = (args.crop_h, args.crop_w)
    train_ds  = CityscapesDataset(args.data_root, split="train", crop_size=crop_size)
    val_ds    = CityscapesDataset(args.data_root, split="val",   crop_size=crop_size)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    print(f"[train] Train: {len(train_ds)} images | Val: {len(val_ds)} images")

    backbone, head = build_deeplabv3plus(pretrained=args.pretrained, num_classes=args.num_classes)
    isw_layers = [s.strip() for s in args.isw_layers.split(",") if s.strip()]
    model = RobustNetWrapper(
        backbone=backbone, head=head,
        isw_layer_names=isw_layers,
        lambda_isw=args.lambda_isw,
        isw_margin_frac=args.isw_margin_frac,
    )
    model.to(device)
    print(f"[train] Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"[train] ISW layers: {isw_layers}  lambda_isw: {args.lambda_isw}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_iters = len(train_loader) * args.epochs
    scheduler   = PolyLR(optimizer, total_iters=total_iters, power=0.9)

    trainer = Trainer(
        model=model, optimizer=optimizer, scheduler=scheduler, device=device,
        lambda_cons=args.lambda_isw,  # ISW weight, reusing the same Trainer slot
        log_dir=log_dir, num_classes=args.num_classes,
    )

    start_epoch = 0
    if args.resume:
        payload     = trainer.load_checkpoint(args.resume)
        start_epoch = payload.get("epoch", 0) + 1
        print(f"[train] Resuming from epoch {start_epoch}")

    best_miou      = trainer.best_miou
    best_ckpt_path = os.path.join(args.checkpoint_dir, "best.pth")

    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*60}")

        train_metrics = trainer.train_epoch(train_loader)
        print(
            f"  [train] loss={train_metrics['loss']:.4f}  "
            f"ce={train_metrics['loss_ce']:.4f}  "
            f"isw={train_metrics['loss_cons']:.4f}  "
            f"lr={train_metrics['lr']:.2e}  "
            f"time={train_metrics['time_s']:.1f}s"
        )

        if (epoch + 1) % args.val_interval == 0 or epoch == args.epochs - 1:
            val_metrics = trainer.validate(val_loader)
            miou    = val_metrics["miou"]
            pix_acc = val_metrics["pixel_acc"]
            print(f"  [val]   mIoU={miou:.4f}  pixel_acc={pix_acc:.4f}")

            if miou > best_miou:
                best_miou = miou
                trainer.best_miou = best_miou
                trainer.save_checkpoint(best_ckpt_path, extra={"epoch": epoch})
                print(f"  *** New best mIoU={best_miou:.4f} → saved to {best_ckpt_path}")

        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch+1:04d}.pth")
            trainer.save_checkpoint(ckpt_path, extra={"epoch": epoch})

    print(f"\n[train] Finished.  Best mIoU = {best_miou:.4f}")
    print(f"[train] Best checkpoint: {best_ckpt_path}")


if __name__ == "__main__":
    main()
