"""
TentAdaptor — TENT (Wang et al., ICLR 2021) baseline for direct comparison
against PerturbationDisagreementAdaptor.

TENT minimises the Shannon entropy of the model's own softmax prediction
on a single forward pass, updating only normalization-layer affine
parameters (same norm_types scaffolding as PDA, for an apples-to-apples
comparison of which adaptation *signal* — entropy vs. cross-perturbation
disagreement — is more reliable under corruption, holding everything else
about the adaptation mechanism fixed).

Unlike PDA, TENT needs no feature perturbation and no K parallel views —
just one forward pass per adaptation step.
"""

from typing import Dict, FrozenSet, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


_NORM_KIND_SPEC = {
    "bn": ((nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d), True),
    "ln": ((nn.LayerNorm,), False),
    "gn": ((nn.GroupNorm,), False),
}


def _entropy_loss(logits: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean per-pixel Shannon entropy of the softmax distribution."""
    probs = F.softmax(logits, dim=1).clamp(min=eps)
    return -(probs * probs.log()).sum(dim=1).mean()


class TentAdaptor:
    """
    Args:
        model: RobustSegWrapper instance (or any nn.Module with a plain
            forward(x) -> logits).
        lr: learning rate for the norm-affine Adam optimiser.
        n_steps: number of gradient steps per adapt() call.
        norm_types: which normalization kinds to adapt — subset of
            {"bn", "ln", "gn"}. Should match whatever PDA config is being
            compared against for a fair, matched-capacity comparison.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        n_steps: int = 1,
        norm_types: FrozenSet[str] = frozenset({"bn"}),
    ) -> None:
        unknown = set(norm_types) - set(_NORM_KIND_SPEC)
        if unknown:
            raise ValueError(f"Unknown norm_types entries: {unknown}. "
                              f"Valid: {set(_NORM_KIND_SPEC)}")

        self.model      = model
        self.lr         = lr
        self.n_steps    = n_steps
        self.norm_types = frozenset(norm_types)

        self._norm_classes = tuple(
            cls for key in self.norm_types for cls in _NORM_KIND_SPEC[key][0]
        )

        self._original_state: Dict[str, Dict[str, Optional[torch.Tensor]]] = \
            self._save_state()

        # Pre-update adaptation loss at each step of the most recent adapt()
        # call, for objective-vs-accuracy traces. Subclasses fill it too.
        self.last_losses: List[float] = []

    # ------------------------------------------------------------------
    def adapt(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run TENT on batch *x*: minimise prediction entropy for n_steps,
        updating only the selected normalization affine params, then
        return the (single-forward-pass) prediction from the adapted model.
        """
        original_grad_state = self._freeze_non_norm()
        norm_params = self._collect_params()

        if not norm_params:
            self._restore_grad_state(original_grad_state)
            return self._plain_forward(x)

        optimizer = torch.optim.Adam(norm_params, lr=self.lr)
        self.model.train()

        logits = None
        self.last_losses = []
        for _ in range(self.n_steps):
            optimizer.zero_grad()
            logits = self.model(x)
            loss = _entropy_loss(logits)
            self.last_losses.append(loss.detach().item())
            loss.backward()
            optimizer.step()

        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)

        self._restore_grad_state(original_grad_state)
        return logits

    def reset(self) -> None:
        """Restore normalization params to their pre-adaptation state."""
        self._restore_state(self._original_state)

    # ------------------------------------------------------------------
    # Internal helpers (mirrors PerturbationDisagreementAdaptor's pattern)
    # ------------------------------------------------------------------

    def _plain_forward(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return self.model(x)

    def _freeze_non_norm(self) -> Dict[str, bool]:
        state: Dict[str, bool] = {}
        for name, param in self.model.named_parameters():
            state[name] = param.requires_grad
            param.requires_grad_(False)
        for module in self.model.modules():
            if isinstance(module, self._norm_classes):
                if module.weight is not None:
                    module.weight.requires_grad_(True)
                if module.bias is not None:
                    module.bias.requires_grad_(True)
        return state

    def _restore_grad_state(self, state: Dict[str, bool]) -> None:
        for name, param in self.model.named_parameters():
            param.requires_grad_(state.get(name, True))

    def _collect_params(self) -> List[torch.Tensor]:
        params = []
        for module in self.model.modules():
            if isinstance(module, self._norm_classes):
                if module.weight is not None and module.weight.requires_grad:
                    params.append(module.weight)
                if module.bias is not None and module.bias.requires_grad:
                    params.append(module.bias)
        return params

    def _save_state(self) -> Dict[str, Dict[str, Optional[torch.Tensor]]]:
        state: Dict[str, Dict[str, Optional[torch.Tensor]]] = {}
        for name, module in self.model.named_modules():
            if not isinstance(module, self._norm_classes):
                continue
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                state[name] = {
                    "_kind": "bn",
                    "weight": module.weight.data.clone() if module.weight is not None else None,
                    "bias":   module.bias.data.clone()   if module.bias   is not None else None,
                    "running_mean": module.running_mean.clone()               if module.running_mean         is not None else None,
                    "running_var":  module.running_var.clone()                if module.running_var          is not None else None,
                    "num_batches_tracked": module.num_batches_tracked.clone() if module.num_batches_tracked  is not None else None,
                }
            else:
                state[name] = {
                    "_kind": "ln",
                    "weight": module.weight.data.clone() if module.weight is not None else None,
                    "bias":   module.bias.data.clone()   if module.bias   is not None else None,
                }
        return state

    def _restore_state(self, state: Dict[str, Dict[str, Optional[torch.Tensor]]]) -> None:
        for name, module in self.model.named_modules():
            if name not in state:
                continue
            s = state[name]
            if s.get("weight") is not None and module.weight is not None:
                module.weight.data.copy_(s["weight"])
            if s.get("bias") is not None and module.bias is not None:
                module.bias.data.copy_(s["bias"])
            if s.get("_kind") == "bn":
                if s.get("running_mean") is not None and module.running_mean is not None:
                    module.running_mean.copy_(s["running_mean"])
                if s.get("running_var") is not None and module.running_var is not None:
                    module.running_var.copy_(s["running_var"])
                if s.get("num_batches_tracked") is not None and module.num_batches_tracked is not None:
                    module.num_batches_tracked.copy_(s["num_batches_tracked"])
