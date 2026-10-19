"""
DUAAdaptor -- DUA (Mirza et al., CVPR 2022), a *gradient-free* test-time
adaptation baseline.

Where TENT and PDA update normalization **affine parameters** by gradient
descent, DUA only re-estimates normalization **running statistics** by forward
passes, using a decaying momentum. It is therefore a second, mechanistically
distinct adaptation family: statistics vs. gradients. Including it turns the
paper's claim from "two methods, one family" into "three methods, two
mechanisms."

Faithful to the reference implementation (github.com/jmiemirza/DUA,
`methods/dua.py`): per adaptation sample,

    mom_new       = mom_pre * decay_factor          # decay_factor = 0.94
    bn.momentum   = mom_new + min_momentum          # min_momentum = 0.005
    mom_pre       = mom_new

with `mom_pre` initialised to 0.1, BN layers put in train mode (so the forward
pass updates running stats), and the forward run on a batch of augmentations of
a single test image. Defaults follow the reference: decay 0.94, min momentum
0.005, 80 adaptation samples per condition.

Two documented deviations, both forced by the setting:
  * The reference builds its augmentation batch with CIFAR-scale crops; at
    512x1024 segmentation resolution a 64-wide augmentation batch does not fit,
    so `n_aug` defaults to 16 (flip + random-resized-crop, label-free).
  * The reference evaluates after *every* adaptation sample and reports the best
    iterate, which uses test labels for model selection. We adapt for
    `n_adapt_samples` and then evaluate once -- no label-based selection.

Adaptation is continual within a condition (that is DUA's design) and reset
between conditions, mirroring the reference's per-task checkpoint reload.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DUAAdaptor:
    """
    Args:
        model: nn.Module containing BatchNorm layers.
        decay_factor: multiplicative momentum decay per adaptation sample.
        min_momentum: floor added to the decayed momentum.
        mom_pre: initial momentum before the first decay.
        n_aug: size of the augmentation batch built from one test image.
        n_adapt_samples: number of adaptation samples per condition (then the
            statistics are frozen and only inference continues).
    """

    def __init__(
        self,
        model: nn.Module,
        decay_factor: float = 0.94,
        min_momentum: float = 0.005,
        mom_pre: float = 0.1,
        n_aug: int = 16,
        n_adapt_samples: int = 80,
    ) -> None:
        self.model = model
        self.decay_factor = decay_factor
        self.min_momentum = min_momentum
        self.mom_pre = mom_pre
        self.mom_init = mom_pre
        self.n_aug = n_aug
        self.n_adapt_samples = n_adapt_samples
        self.steps_done = 0
        self._original_state = self._save_state()

    # ------------------------------------------------------------------
    def adapt(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run one DUA statistics update from batch *x* (its first image), then
        return the adapted model's prediction for *x* from a clean eval-mode
        forward pass. No gradients are taken anywhere.
        """
        if self.steps_done < self.n_adapt_samples:
            mom_new = self.mom_pre * self.decay_factor
            for m in self.model.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.train()
                    m.momentum = mom_new + self.min_momentum
            self.mom_pre = mom_new
            with torch.no_grad():
                self.model(self._augment(x[:1]))
            self.steps_done += 1

        self.model.eval()
        with torch.no_grad():
            return self.model(x)

    def reset(self) -> None:
        """Restore pre-adaptation statistics and the momentum schedule."""
        self._restore_state(self._original_state)
        self.mom_pre = self.mom_init
        self.steps_done = 0

    # ------------------------------------------------------------------
    def _augment(self, img: torch.Tensor) -> torch.Tensor:
        """Label-free augmentation batch from a single image: horizontal flip
        plus random-resized-crop back to the original resolution."""
        _, _, H, W = img.shape
        views = []
        for i in range(self.n_aug):
            v = torch.flip(img, dims=[3]) if i % 2 else img
            scale = 0.7 + 0.3 * torch.rand(1).item()
            h, w = max(8, int(H * scale)), max(8, int(W * scale))
            top = torch.randint(0, H - h + 1, (1,)).item()
            left = torch.randint(0, W - w + 1, (1,)).item()
            crop = v[:, :, top:top + h, left:left + w]
            views.append(F.interpolate(crop, size=(H, W), mode="bilinear",
                                       align_corners=False))
        return torch.cat(views, dim=0)

    def _save_state(self) -> Dict[str, Dict[str, Optional[torch.Tensor]]]:
        state = {}
        for name, m in self.model.named_modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                state[name] = {
                    "running_mean": m.running_mean.clone() if m.running_mean is not None else None,
                    "running_var": m.running_var.clone() if m.running_var is not None else None,
                    "num_batches_tracked": (m.num_batches_tracked.clone()
                                            if m.num_batches_tracked is not None else None),
                    "momentum": m.momentum,
                }
        return state

    def _restore_state(self, state) -> None:
        for name, m in self.model.named_modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm) and name in state:
                s = state[name]
                if s["running_mean"] is not None:
                    m.running_mean.copy_(s["running_mean"])
                if s["running_var"] is not None:
                    m.running_var.copy_(s["running_var"])
                if s["num_batches_tracked"] is not None:
                    m.num_batches_tracked.copy_(s["num_batches_tracked"])
                m.momentum = s["momentum"]
