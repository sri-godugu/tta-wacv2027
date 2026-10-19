"""
OracleAdaptor — supervised upper bound for normalization-parameter adaptation.

Identical machinery to TentAdaptor (same parameters adapted, same optimiser,
same number of steps, same final clean forward pass) with ONE difference: the
adaptation loss is cross-entropy against the ground-truth labels instead of
prediction entropy.

This is a *ceiling*, never a method: it is transductive (it sees the labels of
the very batch it is scored on). It answers "how much of this condition's error
could normalization-parameter adaptation recover if the adaptation signal were
perfect?", which decomposes the measured shortfall of unsupervised TTA into

    recoverability gap = predicted benefit - ceiling      (error this family of
                                                           adaptation cannot touch)
    signal gap         = ceiling - actual benefit          (recoverable error the
                                                           unsupervised signal missed)

The ceiling is specific to the adapted parameters and the step budget it is run
with; it is not a supremum over all possible adaptation.
"""

from typing import FrozenSet

import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptation.tent import TentAdaptor


class OracleAdaptor(TentAdaptor):
    """
    Args:
        model, lr, n_steps, norm_types: identical semantics to TentAdaptor —
            pass exactly what the unsupervised run used, or the comparison is
            not a ceiling for that run.
        ignore_index: label value excluded from the supervised loss.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        n_steps: int = 1,
        norm_types: FrozenSet[str] = frozenset({"bn"}),
        ignore_index: int = 255,
    ) -> None:
        super().__init__(model=model, lr=lr, n_steps=n_steps, norm_types=norm_types)
        self.ignore_index = ignore_index

    # ------------------------------------------------------------------
    def adapt(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        """
        Adapt on batch *x* by minimising cross-entropy against labels *y*, then
        return the prediction of the adapted model from a clean forward pass.

        Logits are upsampled to the label resolution before the loss, matching
        how the evaluation itself scores predictions.
        """
        if y is None:
            raise ValueError("OracleAdaptor.adapt requires ground-truth labels")

        original_grad_state = self._freeze_non_norm()
        norm_params = self._collect_params()

        if not norm_params:
            self._restore_grad_state(original_grad_state)
            return self._plain_forward(x)

        optimizer = torch.optim.Adam(norm_params, lr=self.lr)
        self.model.train()

        self.last_losses = []
        for _ in range(self.n_steps):
            optimizer.zero_grad()
            logits = self.model(x)
            if logits.shape[-2:] != y.shape[-2:]:
                logits = F.interpolate(
                    logits, size=y.shape[-2:], mode="bilinear", align_corners=False,
                )
            loss = F.cross_entropy(logits, y.long(), ignore_index=self.ignore_index)
            self.last_losses.append(loss.detach().item())
            loss.backward()
            optimizer.step()

        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)

        self._restore_grad_state(original_grad_state)
        return logits
