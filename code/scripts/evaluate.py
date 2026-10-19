#!/usr/bin/env python3
"""
evaluate.py — RobustSeg evaluation script.

Example usage::

    # Evaluate on ACDC (no PDA)
    python scripts/evaluate.py \\
        --checkpoint runs/exp1/best.pth \\
        --dataset acdc \\
        --data_root /data/acdc

    # Evaluate on Cityscapes-C with PDA, all severities
    python scripts/evaluate.py \\
        --checkpoint runs/exp1/best.pth \\
        --dataset cityscapes_c \\
        --data_root /data/cityscapes_c \\
        --cs_root /data/cityscapes \\
        --use_pda \\
        --severity 0 \\
        --corruption_type fog

    # Evaluate on Foggy Cityscapes
    python scripts/evaluate.py \\
        --checkpoint runs/exp1/best.pth \\
        --dataset foggy \\
        --data_root /data/foggy_cityscapes \\
        --cs_root /data/cityscapes
"""

import argparse
import json
import os
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_ROOT  = os.path.dirname(_SCRIPT_DIR)
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

import torch
from torch.utils.data import DataLoader

from models.robustseg import RobustSegWrapper
from models.segformer_wrapper import build_segformer
from evaluation.evaluate import evaluate_on_dataset
from data.datasets import (
    ACDCDataset,
    CityscapesCDataset,
    FoggyCityscapesDataset,
    CityscapesDataset,
    FoggyZurichDataset,
    BDD100KDataset,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RobustSeg Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Checkpoint / model
    parser.add_argument("--checkpoint",        type=str, required=True)
    parser.add_argument("--backbone",          type=str, default="segformer",
                        choices=["segformer", "deeplabv3plus"])
    parser.add_argument("--arch",              type=str, default="deeplabv3_resnet50",
                        choices=["deeplabv3_resnet50", "deeplabv3_resnet101",
                                 "fcn_resnet50", "fcn_resnet101"],
                        help="Torchvision segmentation architecture when "
                             "--backbone deeplabv3plus. Varies the head (ASPP vs FCN) "
                             "and depth independently for the backbone-replication study.")
    parser.add_argument("--num_classes",       type=int, default=19)
    parser.add_argument("--texture_threshold", type=float, default=0.45)
    parser.add_argument("--lambda_cons",       type=float, default=0.3)
    parser.add_argument("--pretrained",        action="store_true",
                        help="Must match how the checkpoint being loaded was trained "
                             "(pretrained backbone vs. from-scratch) — these are "
                             "structurally different and have incompatible state_dicts.")

    # Dataset
    parser.add_argument("--dataset",    type=str, default="acdc",
                        choices=["acdc", "cityscapes_c", "foggy", "cityscapes",
                                 "foggy_zurich", "bdd100k"])
    parser.add_argument("--data_root",  type=str, required=True)
    parser.add_argument("--cs_root",    type=str, default=None,
                        help="Original Cityscapes root (needed for cityscapes_c/foggy).")
    parser.add_argument("--condition",  type=str, default="all",
                        help="ACDC condition: fog|night|rain|snow|all. "
                             "For BDD100K: a weather (rainy/snowy/foggy/...) "
                             "or timeofday (night/daytime) attribute, or all.")
    parser.add_argument("--attr_json",  type=str, default=None,
                        help="BDD100K attributes JSON (image->weather/timeofday) "
                             "for per-condition filtering.")
    parser.add_argument("--corruption_type", type=str, default="fog",
                        help="Cityscapes-C corruption type.")
    parser.add_argument("--severity",   type=int, default=1,
                        help="Corruption severity 1-5 (0 = all).")
    parser.add_argument("--split",      type=str, default="val")
    parser.add_argument("--fog_beta",   type=str, default="0.02",
                        choices=["0.005", "0.01", "0.02"])
    parser.add_argument("--crop_h",     type=int, default=512)
    parser.add_argument("--crop_w",     type=int, default=1024)

    # PDA / TENT
    parser.add_argument("--use_pda",       action="store_true")
    parser.add_argument("--use_tent",      action="store_true",
                        help="Use TENT (entropy minimization) instead of PDA. "
                             "Mutually exclusive with --use_pda.")
    parser.add_argument("--use_oracle",    action="store_true",
                        help="Supervised ORACLE CEILING: same adapted parameters, "
                             "optimiser and step budget as TENT, but the adaptation "
                             "loss is cross-entropy against ground truth. Transductive "
                             "upper bound, never a method. Mutually exclusive with "
                             "--use_pda / --use_tent.")
    parser.add_argument("--use_dua",       action="store_true",
                        help="DUA (Mirza et al., CVPR 2022): gradient-free adaptation "
                             "of BatchNorm running statistics via a decaying-momentum "
                             "forward pass. A second adaptation MECHANISM (statistics, "
                             "not gradients). Mutually exclusive with the others.")
    parser.add_argument("--use_pseudo",    action="store_true",
                        help="Self-training TTA: cross-entropy against the model's own "
                             "frozen hard predictions. Fourth adaptation OBJECTIVE, same "
                             "normalization parameters as TENT/PDA. Not CoTTA (no teacher, "
                             "no augmentation averaging).")
    parser.add_argument("--K",             type=int,   default=4)
    parser.add_argument("--n_adapt_steps", type=int,   default=1)
    parser.add_argument("--lr_pda",        type=float, default=1e-3)
    parser.add_argument("--norm_types",    type=str, default="bn",
                        help="Comma-separated norm kinds PDA adapts: "
                             "any subset of bn,ln,gn (e.g. 'bn', 'bn,ln', 'bn,ln,gn').")
    parser.add_argument("--clean_final",   action="store_true",
                        help="Report predictions from a clean forward pass through the "
                             "adapted model instead of averaging K perturbed passes.")
    parser.add_argument("--continual",     action="store_true",
                        help="Let PDA adaptation persist across the whole dataset instead "
                             "of resetting after every sample (matches TENT/CoTTA-style TTA).")
    parser.add_argument("--restart_prob",  type=float, default=0.0,
                        help="CoTTA-style stochastic restoration probability (continual mode "
                             "only) — per-element chance of resetting a parameter back to its "
                             "pretrained value each step, to bound drift over a long stream.")

    # Misc
    parser.add_argument("--batch_size",  type=int, default=1)
    parser.add_argument("--drop_last",   action="store_true",
                        help="Drop a trailing partial batch. Needed for DeepLab+PDA/TENT "
                             "on odd-sized condition splits: batch_size=1 crashes ASPP's "
                             "global-pool BatchNorm in train mode. Keeps batch_size=2 "
                             "identical to the ACDC/Cityscapes-C protocol.")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device",      type=str, default="")
    parser.add_argument("--results_json", type=str, default=None,
                        help="If set, write metrics + run config as JSON to this path.")
    parser.add_argument("--per_image_cm", type=str, default=None,
                        help="If set, save per-image confusion matrices (and the batch "
                             "id each image was adapted in) to this .npz. They sum to "
                             "the run-level matrix, enabling bootstrap CIs on mIoU.")
    parser.add_argument("--save_preds", type=str, default=None,
                        help="If set (with --save_pred_idx), save argmax predictions and "
                             "labels for those dataset indices to this .npz, from the "
                             "same eval pass that produces the reported metrics.")
    parser.add_argument("--save_pred_idx", type=str, default="",
                        help="Comma-separated dataset indices (eval order) for --save_preds.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed torch/numpy/random. PDA draws K random perturbations "
                             "per batch, so an unseeded eval is not reproducible.")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Cityscapes-19 class names (for pretty printing)
# ---------------------------------------------------------------------------

CS19_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train",
    "motorcycle", "bicycle",
]


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_dataset(args: argparse.Namespace):
    crop_size = (args.crop_h, args.crop_w)

    if args.dataset == "acdc":
        return ACDCDataset(
            root=args.data_root,
            condition=args.condition,
            split=args.split,
            crop_size=crop_size,
        )

    elif args.dataset == "cityscapes_c":
        cs_root = args.cs_root or args.data_root
        if args.severity == 0:
            # Return a list of (severity, dataset) tuples for table printing
            return [
                (sev, CityscapesCDataset(
                    root=args.data_root,
                    cs_root=cs_root,
                    corruption_type=args.corruption_type,
                    severity=sev,
                    crop_size=crop_size,
                ))
                for sev in range(1, 6)
            ]
        else:
            return CityscapesCDataset(
                root=args.data_root,
                cs_root=cs_root,
                corruption_type=args.corruption_type,
                severity=args.severity,
                crop_size=crop_size,
            )

    elif args.dataset == "foggy":
        cs_root = args.cs_root or args.data_root
        return FoggyCityscapesDataset(
            root=args.data_root,
            cs_root=cs_root,
            split=args.split,
            beta=args.fog_beta,
            crop_size=crop_size,
        )

    elif args.dataset == "cityscapes":
        return CityscapesDataset(
            root=args.data_root,
            split=args.split,
            crop_size=crop_size,
        )

    elif args.dataset == "foggy_zurich":
        return FoggyZurichDataset(
            root=args.data_root,
            crop_size=crop_size,
        )

    elif args.dataset == "bdd100k":
        return BDD100KDataset(
            root=args.data_root,
            split=args.split,
            crop_size=crop_size,
            condition=(None if args.condition == "all" else args.condition),
            attr_json=args.attr_json,
        )

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def _print_metrics(metrics: dict, indent: str = "") -> None:
    print(f"{indent}mIoU         : {metrics['miou']:.4f}  ({metrics['miou']*100:.2f}%)")
    print(f"{indent}Pixel Acc    : {metrics['pixel_accuracy']:.4f}")
    print(f"{indent}Mean Acc     : {metrics['mean_accuracy']:.4f}")

    per_cls = metrics.get("per_class_iou", [])
    if per_cls:
        print(f"{indent}Per-class IoU:")
        for i, (name, iou) in enumerate(zip(CS19_NAMES, per_cls)):
            print(f"{indent}  {i:2d} {name:<20s}: {iou:.4f}")


def _print_severity_table(results: list, corruption_type: str) -> None:
    print(f"\nCityscapes-C   corruption={corruption_type}")
    print(f"{'Severity':<10s}  {'mIoU':>8s}  {'PixAcc':>8s}")
    print("-" * 32)
    mious = []
    for sev, metrics in results:
        miou    = metrics["miou"]
        pix_acc = metrics["pixel_accuracy"]
        mious.append(miou)
        print(f"{sev:<10d}  {miou:8.4f}  {pix_acc:8.4f}")
    print("-" * 32)
    mean_miou = sum(mious) / len(mious)
    print(f"{'Mean':<10s}  {mean_miou:8.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    device = (
        torch.device(args.device) if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[eval] Device: {device}")

    if args.seed is not None:
        import random as _random
        import numpy as _np
        _random.seed(args.seed)
        _np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        print(f"[eval] Seed: {args.seed}")

    if sum([args.use_pda, args.use_tent, args.use_oracle, args.use_dua, args.use_pseudo]) > 1:
        raise ValueError("adaptation flags are mutually exclusive")

    norm_types = frozenset(t.strip() for t in args.norm_types.split(",") if t.strip())
    valid_norm_types = {"bn", "ln", "gn"}
    if not norm_types <= valid_norm_types:
        raise ValueError(f"--norm_types has unknown entries: {norm_types - valid_norm_types}")
    method = "PDA" if args.use_pda else ("TENT" if args.use_tent else "vanilla")
    print(f"[eval] Method: {method}  norm_types: {sorted(norm_types) if (args.use_pda or args.use_tent) else 'n/a'}")

    # Build model
    if args.backbone == "segformer":
        backbone, head = build_segformer(pretrained=args.pretrained, num_classes=args.num_classes)
    elif args.backbone == "deeplabv3plus":
        from models.deeplab_wrapper import build_deeplabv3plus
        backbone, head = build_deeplabv3plus(pretrained=args.pretrained,
                                             num_classes=args.num_classes,
                                             arch=args.arch)
    else:
        raise ValueError(f"Unknown backbone: {args.backbone}")
    model = RobustSegWrapper(
        backbone=backbone,
        head=head,
        texture_threshold=args.texture_threshold,
        lambda_cons=args.lambda_cons,
        K=args.K,
    )

    # Load checkpoint
    payload = torch.load(args.checkpoint, map_location="cpu")
    state   = payload.get("model_state", payload)  # support raw state_dict too
    model.load_state_dict(state, strict=True)
    print(f"[eval] Loaded checkpoint: {args.checkpoint}")

    model.eval()

    eval_kwargs = dict(
        device=device,
        use_pda=args.use_pda,
        use_tent=args.use_tent,
        use_oracle=args.use_oracle,
        use_dua=args.use_dua,
        use_pseudo=args.use_pseudo,
        num_classes=args.num_classes,
        K=args.K,
        n_adapt_steps=args.n_adapt_steps,
        lr_pda=args.lr_pda,
        texture_threshold=args.texture_threshold,
        norm_types=norm_types,
        clean_final=args.clean_final,
        continual=args.continual,
        restart_prob=args.restart_prob,
        per_image_cm=bool(args.per_image_cm),
        save_pred_idx=({int(i) for i in args.save_pred_idx.split(",") if i.strip()}
                       if args.save_preds else None),
    )

    # Build dataset(s)
    dataset_or_list = build_dataset(args)
    started_at = time.time()

    if isinstance(dataset_or_list, list):
        # Cityscapes-C multi-severity mode
        severity_results = []
        for sev, ds in dataset_or_list:
            loader = DataLoader(
                ds, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True,
                drop_last=args.drop_last,
            )
            n_eval = len(ds) - (len(ds) % args.batch_size if args.drop_last else 0)
            print(f"\n[eval] Severity {sev}  ({len(ds)} images, {n_eval} evaluated)"
                  f"  use_pda={args.use_pda}")
            metrics = evaluate_on_dataset(model, loader, **eval_kwargs)
            severity_results.append((sev, metrics))

        _print_severity_table(severity_results, args.corruption_type)
        results_payload = {
            "config": vars(args) | {"norm_types": sorted(norm_types)},
            "elapsed_s": time.time() - started_at,
            "severity_results": [
                {"severity": sev, **metrics} for sev, metrics in severity_results
            ],
        }

    else:
        ds = dataset_or_list
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
            drop_last=args.drop_last,
        )
        n_eval = len(ds) - (len(ds) % args.batch_size if args.drop_last else 0)
        print(f"\n[eval] Dataset={args.dataset}  ({len(ds)} images, {n_eval} evaluated)"
              f"  use_pda={args.use_pda}")
        metrics = evaluate_on_dataset(model, loader, **eval_kwargs)

        print("\n" + "="*50)
        print("Results")
        print("="*50)
        _print_metrics(metrics)
        results_payload = {
            "config": vars(args) | {"norm_types": sorted(norm_types)},
            "elapsed_s": time.time() - started_at,
            "metrics": metrics,
        }

    if args.per_image_cm:
        import numpy as _np
        _m = results_payload.get("metrics") or {}
        cm = _m.pop("_per_image_cm", None)
        bid = _m.pop("_cm_batch_ids", None)
        for _sev in results_payload.get("severity_results", []):   # never serialise these
            _sev.pop("_per_image_cm", None)
            _sev.pop("_cm_batch_ids", None)
        if cm is None:
            print("[eval] WARNING: --per_image_cm set but no matrices collected "
                  "(multi-severity mode is not supported); nothing written.")
        else:
            os.makedirs(os.path.dirname(os.path.abspath(args.per_image_cm)) or ".", exist_ok=True)
            _np.savez_compressed(args.per_image_cm, cm=cm, batch_id=_np.asarray(bid))
            print(f"[eval] Per-image confusion matrices ({cm.shape}) -> {args.per_image_cm}")

    if args.save_preds:
        import numpy as _np
        _m = results_payload.get("metrics") or {}
        sp = _m.pop("_saved_preds", None)
        for _sev in results_payload.get("severity_results", []):
            _sev.pop("_saved_preds", None)
        if not sp:
            print("[eval] WARNING: --save_preds set but nothing saved "
                  "(multi-severity mode is not supported, or no index matched).")
        else:
            os.makedirs(os.path.dirname(os.path.abspath(args.save_preds)) or ".", exist_ok=True)
            idx = sorted(sp)
            _np.savez_compressed(
                args.save_preds, idx=_np.asarray(idx),
                pred=_np.stack([sp[i][0] for i in idx]),
                label=_np.stack([sp[i][1] for i in idx]),
            )
            print(f"[eval] Saved predictions for indices {idx} -> {args.save_preds}")

    if args.results_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.results_json)) or ".", exist_ok=True)
        with open(args.results_json, "w") as f:
            json.dump(results_payload, f, indent=2, default=str)
        print(f"\n[eval] Results written to {args.results_json}")


if __name__ == "__main__":
    main()
