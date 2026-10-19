"""
Loss functions for RobustSeg training.

ConsistencyLoss  — pixel-wise KL(P_orig || P_perturbed)
PDALoss          — mean pairwise Jensen-Shannon divergence across K predictions
CombinedLoss     — CE + lambda * ConsistencyLoss (returns dict)
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# ConsistencyLoss
# ---------------------------------------------------------------------------

class ConsistencyLoss(nn.Module):
    """
    Pixel-wise KL divergence between two prediction tensors.

    KL(P_orig || P_perturbed) = sum_c P_orig_c * log(P_orig_c / P_perturbed_c)

    Both inputs are expected to be *raw logits*; softmax is applied internally.

    Args:
        eps (float): small constant for numerical stability.
        reduction (str): 'mean' (default) averages over B, H, W.
    """

    def __init__(self, eps: float = 1e-8, reduction: str = "mean") -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(
        self,
        pred_orig: torch.Tensor,
        pred_perturbed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_orig:      (B, num_classes, H, W) raw logits — the "reference".
            pred_perturbed: (B, num_classes, H, W) raw logits — the "perturbed".

        Returns:
            Scalar loss tensor.
        """
        p = F.softmax(pred_orig, dim=1).clamp(min=self.eps)
        q = F.softmax(pred_perturbed, dim=1).clamp(min=self.eps)

        # KL element-wise
        kl_map = (p * (p.log() - q.log())).sum(dim=1)  # (B, H, W)

        if self.reduction == "mean":
            return kl_map.mean()
        elif self.reduction == "sum":
            return kl_map.sum()
        else:
            return kl_map


# ---------------------------------------------------------------------------
# PDALoss
# ---------------------------------------------------------------------------

class PDALoss(nn.Module):
    """
    Mean pairwise Jensen-Shannon divergence across a list of K predictions.

    JSD(p || q) = 0.5 * KL(p || m) + 0.5 * KL(q || m),   m = 0.5*(p + q)

    This is symmetric and bounded in [0, log 2].

    Args:
        eps (float): small constant for numerical stability.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, predictions_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            predictions_list: list of K tensors, each (B, num_classes, H, W).
                              Raw logits accepted; softmax applied internally.

        Returns:
            Scalar mean pairwise JSD.
        """
        K = len(predictions_list)
        if K < 2:
            return torch.tensor(0.0, device=predictions_list[0].device,
                                dtype=predictions_list[0].dtype)

        probs = [
            F.softmax(p, dim=1).clamp(min=self.eps)
            for p in predictions_list
        ]

        total = torch.tensor(
            0.0, device=predictions_list[0].device, dtype=predictions_list[0].dtype
        )
        n_pairs = 0

        for i in range(K):
            for j in range(i + 1, K):
                p = probs[i]
                q = probs[j]
                m = (0.5 * (p + q)).clamp(min=self.eps)

                kl_pm = (p * (p.log() - m.log())).sum(dim=1)  # (B, H, W)
                kl_qm = (q * (q.log() - m.log())).sum(dim=1)  # (B, H, W)
                jsd   = 0.5 * (kl_pm + kl_qm)

                total = total + jsd.mean()
                n_pairs += 1

        return total / n_pairs


# ---------------------------------------------------------------------------
# CombinedLoss
# ---------------------------------------------------------------------------

class CombinedLoss(nn.Module):
    """
    CE(pred, target) + lambda_cons * ConsistencyLoss(pred, pred_perturbed).

    Returns a dict with keys: 'loss', 'loss_ce', 'loss_cons'.

    Args:
        lambda_cons (float): weight for the consistency term.
        ignore_index (int): label value to ignore in CE (default 255).
        eps (float): passed to ConsistencyLoss.
    """

    def __init__(
        self,
        lambda_cons: float = 0.3,
        ignore_index: int = 255,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.lambda_cons = lambda_cons
        self.ignore_index = ignore_index
        self.cons_loss = ConsistencyLoss(eps=eps)

    def forward(
        self,
        pred_orig: torch.Tensor,
        target: torch.Tensor,
        pred_perturbed: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred_orig:      (B, num_classes, H, W) raw logits.
            target:         (B, H, W) integer labels.
            pred_perturbed: (B, num_classes, H, W) raw logits (optional).
                            When None, consistency loss is zero.

        Returns:
            Dict with 'loss', 'loss_ce', 'loss_cons'.
        """
        loss_ce = F.cross_entropy(pred_orig, target, ignore_index=self.ignore_index)

        if pred_perturbed is not None:
            loss_cons = self.cons_loss(pred_orig, pred_perturbed)
        else:
            loss_cons = torch.tensor(0.0, device=pred_orig.device, dtype=pred_orig.dtype)

        loss = loss_ce + self.lambda_cons * loss_cons

        return {
            "loss": loss,
            "loss_ce": loss_ce.detach(),
            "loss_cons": loss_cons.detach(),
        }
