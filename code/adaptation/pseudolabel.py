"""
PseudoLabelAdaptor -- self-training test-time adaptation.

This is the fourth adaptation *objective* in the study, and it exists to make
the mechanism claim of the paper a controlled comparison. Holding the adapted
parameter family fixed (normalization affine, exactly as TENT/PDA/oracle), we
vary only the signal that drives the update:

    disagreement (PDA)  |  entropy (TENT)  |  statistics only (DUA)
    supervised CE (oracle ceiling)  |  self-training CE (this class)

Self-training minimises cross-entropy against the model's own hard predictions,
which is the objective underlying pseudo-label and teacher-student TTA methods.
It is NOT a reimplementation of CoTTA: there is no teacher network, no
augmentation averaging and no stochastic restoration, and like every other
method here it updates normalization parameters only. It therefore tests the
pseudo-labelling *signal*, not the pseudo-labelling *method family*, and the
paper states that distinction explicitly.

Pseudo-labels are taken from a clean forward pass through the *pre-update*
model and then held fixed for the adaptation steps, which is the standard
formulation and avoids the degenerate case of a target that moves with the
prediction it supervises.
"""

from typing import FrozenSet

import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptation.tent import TentAdaptor


class PseudoLabelAdaptor(TentAdaptor):
    """
    Args:
        model, lr, n_steps, norm_types: identical semantics to TentAdaptor --
            pass what the other methods used, or the comparison is not
            controlled.
        conf_threshold: if > 0, pixels whose max softmax probability falls
            below this are excluded from the loss (the usual confidence
            filter). Default 0.0 = use every pixel, the simplest formulation.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        n_steps: int = 1,
        norm_types: FrozenSet[str] = frozenset({"bn"}),
        conf_threshold: float = 0.0,
    ) -> None:
        super().__init__(model=model, lr=lr, n_steps=n_steps, norm_types=norm_types)
        self.conf_threshold = conf_threshold

    # ------------------------------------------------------------------
    def adapt(self, x: torch.Tensor) -> torch.Tensor:
        # 1. pseudo-labels from the pre-adaptation model, then frozen
        self.model.eval()
        with torch.no_grad():
            logits0 = self.model(x)
            probs = F.softmax(logits0, dim=1)
            conf, pseudo = probs.max(dim=1)
            if self.conf_threshold > 0:
                pseudo = torch.where(conf >= self.conf_threshold, pseudo,
                                     torch.full_like(pseudo, 255))

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
            if logits.shape[-2:] != pseudo.shape[-2:]:
                logits = F.interpolate(logits, size=pseudo.shape[-2:],
                                       mode="bilinear", align_corners=False)
            loss = F.cross_entropy(logits, pseudo, ignore_index=255)
            self.last_losses.append(loss.detach().item())
            loss.backward()
            optimizer.step()

        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)

        self._restore_grad_state(original_grad_state)
        return logits
