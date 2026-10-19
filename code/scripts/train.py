#!/usr/bin/env python3
"""
train.py — RobustSeg training script with TCPR.

Example usage::

    python scripts/train.py \\
        --backbone segformer \\
        --data_root /data/cityscapes \\
        --epochs 160 \\
        --batch_size 4 \\
        --lr 6e-5 \\
        --lambda_cons 0.3 \\
        --checkpoint_dir runs/robustseg_exp1

The script:
  1. Builds a RobustSegWrapper around the chosen backbone.
  2. Trains with TCPR (CE + KL consistency loss).
  3. Validates every --val_interval epochs and saves the best checkpoint.
  4. Saves epoch checkpoints to --checkpoint_dir.
"""

import argparse
import os
import random
import sys

# Make sure repo root is importable regardless of CWD
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_ROOT  = os.path.dirname(_SCRIPT_DIR)
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

import torch
from torch.utils.data import DataLoader

from models.robustseg import RobustSegWrapper
from models.segformer_wrapper import build_segformer
from training.trainer import Trainer, PolyLR
from data.datasets import CityscapesDataset


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RobustSeg Training (TCPR)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument("--arch",              type=str, default="deeplabv3_resnet50",
                        choices=["deeplabv3_resnet50", "deeplabv3_resnet101",
                                 "fcn_resnet50", "fcn_resnet101"],
                        help="Torchvision segmentation architecture (deeplabv3plus backbone).")
    parser.add_argument("--backbone",          type=str, default="segformer",
                        choices=["segformer", "deeplabv3plus"],
                        help="Backbone architecture.")
    parser.add_argument("--num_classes",       type=int, default=19)
    parser.add_argument("--texture_threshold", type=float, default=0.45)
    parser.add_argument("--lambda_cons",       type=float, default=0.3)
    parser.add_argument("--K",                 type=int,   default=4,
                        help="Perturbed samples per training step.")
    parser.add_argument("--pretrained",        action="store_true",
                        help="Load ImageNet pretrained backbone (timm required).")

    # Data
    parser.add_argument("--data_root",   type=str, required=True,
                        help="Path to Cityscapes root directory.")
    parser.add_argument("--crop_h",      type=int, default=512)
    parser.add_argument("--crop_w",      type=int, default=1024)

    # Training
    parser.add_argument("--epochs",      type=int,   default=160)
    parser.add_argument("--batch_size",  type=int,   default=4)
    parser.add_argument("--lr",          type=float, default=6e-5)
    parser.add_argument("--weight_decay",type=float, default=1e-2)
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--val_interval",type=int,   default=10,
                        help="Run validation every N epochs.")

    # Output
    parser.add_argument("--checkpoint_dir", type=str, default="runs/experiment")
    parser.add_argument("--log_dir",        type=str, default=None,
                        help="Defaults to <checkpoint_dir>/logs.")
    parser.add_argument("--resume",         type=str, default=None,
                        help="Path to a checkpoint to resume from.")

    # Device
    parser.add_argument("--device", type=str, default="",
                        help="'cuda', 'cpu', or '' for auto-detect.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for torch/numpy/random and CUDA init.")

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Build model
# ---------------------------------------------------------------------------

def build_model(args: argparse.Namespace) -> RobustSegWrapper:
    if args.backbone == "segformer":
        backbone, head = build_segformer(
            pretrained=args.pretrained,
            num_classes=args.num_classes,
        )
    elif args.backbone == "deeplabv3plus":
        from models.deeplab_wrapper import build_deeplabv3plus
        backbone, head = build_deeplabv3plus(
            pretrained=args.pretrained,
            num_classes=args.num_classes,
            arch=getattr(args, "arch", "deeplabv3_resnet50"),
        )
    else:
        raise ValueError(f"Unknown backbone: {args.backbone}")

    model = RobustSegWrapper(
        backbone=backbone,
        head=head,
        texture_threshold=args.texture_threshold,
        lambda_cons=args.lambda_cons,
        K=args.K,
    )
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    set_seed(args.seed)
    device = (
        torch.device(args.device) if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[train] Using device: {device}  seed: {args.seed}")

    # Directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    log_dir = args.log_dir or os.path.join(args.checkpoint_dir, "logs")

    # Datasets
    crop_size = (args.crop_h, args.crop_w)
    train_ds  = CityscapesDataset(args.data_root, split="train", crop_size=crop_size)
    val_ds    = CityscapesDataset(args.data_root, split="val",   crop_size=crop_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"[train] Train: {len(train_ds)} images | Val: {len(val_ds)} images")

    # Model
    model = build_model(args)
    model.to(device)
    print(f"[train] Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Poly LR scheduler (steps per epoch × total epochs)
    total_iters = len(train_loader) * args.epochs
    scheduler   = PolyLR(optimizer, total_iters=total_iters, power=0.9)

    # Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        lambda_cons=args.lambda_cons,
        log_dir=log_dir,
        num_classes=args.num_classes,
    )

    # Resume
    start_epoch = 0
    if args.resume:
        payload     = trainer.load_checkpoint(args.resume)
        start_epoch = payload.get("epoch", 0) + 1
        print(f"[train] Resuming from epoch {start_epoch}")

    # Training loop
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
            f"cons={train_metrics['loss_cons']:.4f}  "
            f"lr={train_metrics['lr']:.2e}  "
            f"time={train_metrics['time_s']:.1f}s"
        )

        # Validation
        if (epoch + 1) % args.val_interval == 0 or epoch == args.epochs - 1:
            val_metrics = trainer.validate(val_loader)
            miou        = val_metrics["miou"]
            pix_acc     = val_metrics["pixel_acc"]
            print(f"  [val]   mIoU={miou:.4f}  pixel_acc={pix_acc:.4f}")

            if miou > best_miou:
                best_miou = miou
                trainer.best_miou = best_miou
                trainer.save_checkpoint(best_ckpt_path, extra={"epoch": epoch})
                print(f"  *** New best mIoU={best_miou:.4f} → saved to {best_ckpt_path}")

        # Periodic checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch+1:04d}.pth")
            trainer.save_checkpoint(ckpt_path, extra={"epoch": epoch})

    print(f"\n[train] Finished.  Best mIoU = {best_miou:.4f}")
    print(f"[train] Best checkpoint: {best_ckpt_path}")


if __name__ == "__main__":
    main()
