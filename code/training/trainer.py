"""
Trainer: orchestrates TCPR training, validation, checkpointing.

The model is expected to be a RobustSegWrapper (or any module that
exposes forward_train(x, y) -> dict and can be called as model(x) for
inference).
"""

import os
import time
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from .losses import CombinedLoss


# ---------------------------------------------------------------------------
# mIoU helper (fast, no extra dependency)
# ---------------------------------------------------------------------------

def _fast_miou(
    all_preds: torch.Tensor,
    all_targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> float:
    """Compute mean IoU from flattened pred/target tensors."""
    mask = all_targets != ignore_index
    preds   = all_preds[mask].long()
    targets = all_targets[mask].long()

    intersection = torch.zeros(num_classes, device=preds.device)
    union        = torch.zeros(num_classes, device=preds.device)

    for c in range(num_classes):
        pred_c   = preds   == c
        target_c = targets == c
        intersection[c] = (pred_c & target_c).sum().float()
        union[c]        = (pred_c | target_c).sum().float()

    valid = union > 0
    iou   = torch.where(valid, intersection / (union + 1e-8), torch.zeros_like(union))
    return iou[valid].mean().item()


# ---------------------------------------------------------------------------
# Poly LR scheduler
# ---------------------------------------------------------------------------

class PolyLR(torch.optim.lr_scheduler._LRScheduler):
    """Polynomial decay LR scheduler (used in segmentation literature)."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_iters: int,
        power: float = 0.9,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ) -> None:
        self.total_iters = max(total_iters, 1)
        self.power = power
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        factor = (1 - self.last_epoch / self.total_iters) ** self.power
        return [
            max(base_lr * factor, self.min_lr)
            for base_lr in self.base_lrs
        ]


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Manages the TCPR training loop for RobustSeg.

    Args:
        model: a RobustSegWrapper (or compatible module).
        optimizer: PyTorch optimizer.
        scheduler: LR scheduler (or None).
        device: torch.device.
        lambda_cons: consistency loss weight (forwarded to CombinedLoss).
        log_dir: directory for tensorboard / text logs.
        num_classes: number of segmentation classes.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        device: torch.device,
        lambda_cons: float = 0.3,
        log_dir: str = "logs",
        num_classes: int = 19,
    ) -> None:
        self.model       = model
        self.optimizer   = optimizer
        self.scheduler   = scheduler
        self.device      = device
        self.lambda_cons = lambda_cons
        self.log_dir     = log_dir
        self.num_classes = num_classes

        self.criterion = CombinedLoss(lambda_cons=lambda_cons)

        os.makedirs(log_dir, exist_ok=True)
        self._log_path = os.path.join(log_dir, "train_log.txt")

        self.best_miou     = 0.0
        self.global_step   = 0

    # ------------------------------------------------------------------
    # train_epoch
    # ------------------------------------------------------------------

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Run one complete training epoch.

        Returns:
            dict with keys: 'loss', 'loss_ce', 'loss_cons', 'lr', 'time_s'.
        """
        self.model.train()
        self.model.to(self.device)

        sum_loss      = 0.0
        sum_loss_ce   = 0.0
        sum_loss_cons = 0.0
        n_batches     = 0
        t0            = time.time()

        iterator = (
            tqdm(dataloader, desc="train", leave=False, dynamic_ncols=True)
            if HAS_TQDM
            else dataloader
        )

        for batch in iterator:
            images, labels = batch
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            out = self.model.forward_train(images, labels)

            loss      = out["loss"]
            loss_ce   = out["loss_ce"]
            loss_cons = out["loss_cons"]

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            sum_loss      += loss.item()
            sum_loss_ce   += loss_ce.item()
            sum_loss_cons += loss_cons.item()
            n_batches     += 1
            self.global_step += 1

            if HAS_TQDM:
                iterator.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "ce":   f"{loss_ce.item():.4f}",
                    "cons": f"{loss_cons.item():.4f}",
                })

        elapsed = time.time() - t0
        metrics = {
            "loss":      sum_loss      / max(n_batches, 1),
            "loss_ce":   sum_loss_ce   / max(n_batches, 1),
            "loss_cons": sum_loss_cons / max(n_batches, 1),
            "lr":        self.optimizer.param_groups[0]["lr"],
            "time_s":    elapsed,
        }
        self._log_metrics("train", metrics)
        return metrics

    # ------------------------------------------------------------------
    # validate
    # ------------------------------------------------------------------

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate on a validation DataLoader.

        Returns:
            dict with keys: 'miou', 'pixel_acc'.
        """
        self.model.eval()
        self.model.to(self.device)

        all_preds   = []
        all_targets = []

        with torch.no_grad():
            iterator = (
                tqdm(dataloader, desc="val", leave=False, dynamic_ncols=True)
                if HAS_TQDM
                else dataloader
            )
            for batch in iterator:
                images, labels = batch
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                logits = self.model(images)

                # Upsample to label resolution if needed
                if logits.shape[-2:] != labels.shape[-2:]:
                    logits = torch.nn.functional.interpolate(
                        logits, size=labels.shape[-2:],
                        mode="bilinear", align_corners=False,
                    )

                preds = logits.argmax(dim=1)
                all_preds.append(preds.cpu())
                all_targets.append(labels.cpu())

        all_preds   = torch.cat(all_preds,   dim=0).reshape(-1)
        all_targets = torch.cat(all_targets, dim=0).reshape(-1)

        miou = _fast_miou(all_preds, all_targets, self.num_classes)

        mask = all_targets != 255
        pixel_acc = (all_preds[mask] == all_targets[mask]).float().mean().item()

        metrics = {"miou": miou, "pixel_acc": pixel_acc}
        self._log_metrics("val", metrics)
        return metrics

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str, extra: Optional[dict] = None) -> None:
        """Save model + optimizer + scheduler state to *path*."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "model_state":     self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "global_step":     self.global_step,
            "best_miou":       self.best_miou,
        }
        if self.scheduler is not None:
            payload["scheduler_state"] = self.scheduler.state_dict()
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        print(f"[Trainer] Saved checkpoint → {path}")

    def load_checkpoint(self, path: str) -> dict:
        """Load checkpoint; returns the full payload dict."""
        payload = torch.load(path, map_location=self.device)
        self.model.load_state_dict(payload["model_state"])
        self.optimizer.load_state_dict(payload["optimizer_state"])
        if self.scheduler is not None and "scheduler_state" in payload:
            self.scheduler.load_state_dict(payload["scheduler_state"])
        self.global_step = payload.get("global_step", 0)
        self.best_miou   = payload.get("best_miou", 0.0)
        print(f"[Trainer] Loaded checkpoint ← {path}  (step {self.global_step})")
        return payload

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _log_metrics(self, phase: str, metrics: dict) -> None:
        parts = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                 for k, v in metrics.items()]
        line = f"[{phase}] step={self.global_step}  " + "  ".join(parts)
        print(line)
        with open(self._log_path, "a") as f:
            f.write(line + "\n")
