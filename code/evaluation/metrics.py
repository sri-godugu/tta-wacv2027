"""
SegmentationMetrics: accumulates a confusion matrix and computes standard
semantic segmentation metrics (mIoU, per-class IoU, pixel accuracy,
mean class accuracy).

All operations use torch; the class works on CPU tensors and is safe to
use with argmax outputs from GPU-accelerated models (move to CPU before
calling update()).
"""

from typing import Dict, List, Optional

import torch


class SegmentationMetrics:
    """
    Confusion-matrix-based evaluator for semantic segmentation.

    Args:
        num_classes (int): number of foreground classes.
        ignore_index (int): label value to exclude from all metrics.
            Defaults to 255 (Cityscapes / ACDC convention).
        device: torch device for the confusion matrix.  Defaults to CPU.
    """

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        device: Optional[torch.device] = None,
    ) -> None:
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.device       = device or torch.device("cpu")
        self._conf_mat    = torch.zeros(
            num_classes, num_classes,
            dtype=torch.long, device=self.device,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear accumulated confusion matrix."""
        self._conf_mat.zero_()

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """
        Accumulate predictions and targets into the confusion matrix.

        Args:
            pred:   (B, H, W) or (H, W) integer prediction tensor.
            target: same shape, integer label tensor.

        Both tensors are moved to CPU and flattened internally.
        """
        pred   = pred.cpu().long()
        target = target.cpu().long()

        assert pred.shape == target.shape, (
            f"pred shape {pred.shape} != target shape {target.shape}"
        )

        self._conf_mat += self.confusion_for(
            pred, target, self.num_classes, self.ignore_index
        ).to(self.device)

    @staticmethod
    def confusion_for(
        pred: torch.Tensor,
        target: torch.Tensor,
        num_classes: int,
        ignore_index: int = 255,
    ) -> torch.Tensor:
        """
        Confusion matrix for a single prediction/target pair (or a batch).

        Factored out of update() so callers can accumulate *per-image* matrices
        that sum exactly to the run-level matrix — needed for image-level
        bootstrap resampling of mIoU.
        """
        pred   = pred.cpu().long()
        target = target.cpu().long()

        mask = target != ignore_index

        pred_m   = pred[mask].clamp(0, num_classes - 1)
        target_m = target[mask].clamp(0, num_classes - 1)

        # Vectorised confusion matrix
        idx = target_m * num_classes + pred_m
        conf_flat = torch.bincount(idx, minlength=num_classes ** 2)
        return conf_flat.reshape(num_classes, num_classes)

    def compute_miou(self) -> Dict[str, object]:
        """
        Compute per-class IoU and mean IoU.

        Returns:
            dict with keys 'per_class_iou' (list of floats) and 'miou' (float).
            Classes with no samples in either pred or target are excluded from
            the mean (but still listed as 0.0 in per_class_iou).
        """
        conf = self._conf_mat.float()
        tp   = conf.diag()                            # (C,)
        fp   = conf.sum(0) - tp                       # (C,) — predicted as c but isn't
        fn   = conf.sum(1) - tp                       # (C,) — is c but not predicted

        denom = tp + fp + fn                          # union
        valid = denom > 0

        iou = torch.where(valid, tp / (denom + 1e-8), torch.zeros_like(tp))  # (C,)

        per_class = iou.tolist()
        miou      = iou[valid].mean().item() if valid.any() else 0.0

        return {
            "miou":          miou,
            "per_class_iou": per_class,
        }

    def compute_all(self) -> Dict[str, object]:
        """
        Compute all standard metrics.

        Returns:
            dict with keys:
              'miou'          — scalar float
              'per_class_iou' — list of floats, length num_classes
              'pixel_accuracy'— scalar float
              'mean_accuracy' — scalar float (mean per-class recall)
        """
        conf = self._conf_mat.float()
        tp   = conf.diag()
        fp   = conf.sum(0) - tp
        fn   = conf.sum(1) - tp

        denom = tp + fp + fn
        valid = denom > 0
        iou   = torch.where(valid, tp / (denom + 1e-8), torch.zeros_like(tp))

        per_class = iou.tolist()
        miou      = iou[valid].mean().item() if valid.any() else 0.0

        # Pixel accuracy
        total_correct = tp.sum()
        total_pixels  = conf.sum()
        pixel_acc     = (total_correct / (total_pixels + 1e-8)).item()

        # Mean per-class accuracy (recall per class)
        n_gt   = conf.sum(1)  # actual occurrences of class c
        valid2 = n_gt > 0
        recall = torch.where(valid2, tp / (n_gt + 1e-8), torch.zeros_like(tp))
        mean_acc = recall[valid2].mean().item() if valid2.any() else 0.0

        return {
            "miou":           miou,
            "per_class_iou":  per_class,
            "pixel_accuracy": pixel_acc,
            "mean_accuracy":  mean_acc,
        }
