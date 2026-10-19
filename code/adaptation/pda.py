"""
PerturbationDisagreementAdaptor (PDA)

Test-time adaptation by minimising disagreement across K perturbed
forward passes.  Only normalization-layer affine parameters are updated;
all other parameters stay frozen.

Which normalization kinds are adapted is controlled by `norm_types`,
so the BN-only / BN+LN / BN+LN+GN ablation is a constructor argument
rather than three forked code paths:
  - "bn" -> nn.BatchNorm{1,2,3}d — weight, bias, and running statistics updated
  - "ln" -> nn.LayerNorm         — weight and bias updated (no running statistics)
  - "gn" -> nn.GroupNorm         — weight and bias updated (no running statistics)

On convolutional backbones (e.g. ResNet), BN layers appear throughout the
backbone; on transformer backbones (e.g. SegFormer's MixTransformer), LN
layers appear in the encoder and BN in the decode head.

After adapt() the adaptor can be reset() to its pre-adaptation norm
state, enabling stateless (per-sample) adaptation.
"""

import copy
from typing import Dict, FrozenSet, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.texture_identifier import TextureChannelIdentifier
from models.feature_perturber import FeaturePerturber


# Maps a norm_types entry to (module classes, kind tag, has_running_stats)
_NORM_KIND_SPEC = {
    "bn": ((nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d), "bn", True),
    "ln": ((nn.LayerNorm,), "ln", False),
    "gn": ((nn.GroupNorm,), "gn", False),
}


# ---------------------------------------------------------------------------
# JSD helper (standalone, no dependency on training.losses)
# ---------------------------------------------------------------------------

def _mean_pairwise_jsd(
    probs_list: List[torch.Tensor],
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Mean pairwise Jensen-Shannon divergence over a list of probability tensors.

    Args:
        probs_list: list of K tensors (B, C, H, W) — already softmax'd.
        eps: clamp floor for numerical safety.

    Returns:
        Scalar JSD.
    """
    K = len(probs_list)
    if K < 2:
        return torch.tensor(0.0, device=probs_list[0].device)

    total  = torch.tensor(0.0, device=probs_list[0].device)
    n_pairs = 0

    for i in range(K):
        for j in range(i + 1, K):
            p = probs_list[i].clamp(min=eps)
            q = probs_list[j].clamp(min=eps)
            m = (0.5 * (p + q)).clamp(min=eps)
            kl_pm = (p * (p.log() - m.log())).sum(dim=1)
            kl_qm = (q * (q.log() - m.log())).sum(dim=1)
            total   = total + (0.5 * (kl_pm + kl_qm)).mean()
            n_pairs += 1

    return total / n_pairs


# ---------------------------------------------------------------------------
# PerturbationDisagreementAdaptor
# ---------------------------------------------------------------------------

class PerturbationDisagreementAdaptor:
    """
    Wraps a RobustSegWrapper (or any backbone+head model that exposes
    the RobustSeg interface) and applies PDA at test time.

    Args:
        model: RobustSegWrapper instance.
        lr: learning rate for the norm-affine Adam optimiser.
        n_steps: number of gradient steps per adapt() call.
        K: number of perturbed forward passes per step.
        texture_threshold: passed to TextureChannelIdentifier.
        norm_types: which normalization kinds to adapt — any subset of
            {"bn", "ln", "gn"}. Defaults to {"bn"} to match the reported
            SegFormer-B2 experiments (decode-head BatchNorm only).
        clean_final: if True, the prediction returned by adapt() comes from
            one clean forward pass through the adapted model, rather than
            averaging K perturbed forward passes. K perturbed views are
            still used to compute the adaptation signal either way.
        continual: if True, the Adam optimiser (and its momentum state) is
            created once and reused across adapt() calls instead of being
            recreated fresh per call — BN affine values persist forward too
            unless the caller explicitly calls reset(). This matches how
            TENT/CoTTA-style continual TTA actually accumulates adaptation
            across a stream, rather than the stateless per-sample default.
        restart_prob: stochastic restoration probability (CoTTA-style). Only
            meaningful when continual=True. After each optimiser step, every
            adapted parameter *element* is reset back to its original
            pretrained value with this per-element probability. Bounds how
            far any single parameter can drift over an arbitrarily long
            stream, preventing the catastrophic error accumulation that
            naive continual BN adaptation is prone to. 0.0 disables it.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        n_steps: int = 1,
        K: int = 4,
        texture_threshold: float = 0.45,
        norm_types: FrozenSet[str] = frozenset({"bn"}),
        clean_final: bool = False,
        continual: bool = False,
        restart_prob: float = 0.0,
    ) -> None:
        unknown = set(norm_types) - set(_NORM_KIND_SPEC)
        if unknown:
            raise ValueError(f"Unknown norm_types entries: {unknown}. "
                              f"Valid: {set(_NORM_KIND_SPEC)}")

        self.model             = model
        self.lr                = lr
        self.n_steps           = n_steps
        self.K                 = K
        self.texture_threshold = texture_threshold
        self.norm_types        = frozenset(norm_types)
        self.clean_final       = clean_final
        self.continual         = continual
        self.restart_prob      = restart_prob

        self._norm_classes = tuple(
            cls
            for key in self.norm_types
            for cls in _NORM_KIND_SPEC[key][0]
        )

        self.identifier = TextureChannelIdentifier()
        self.perturber  = FeaturePerturber()

        # Persistent optimiser for continual mode — created lazily on first
        # adapt() call once bn_params is known, then reused thereafter.
        self._persistent_optimizer: Optional[torch.optim.Optimizer] = None

        # Save a snapshot of BN states at construction time so reset() always
        # returns to the *original* (pre-any-adaptation) state.
        self._original_bn_state: Dict[str, Dict[str, torch.Tensor]] = \
            self._save_bn_state()

        # Mean pairwise JSD from the most recent adapt() call's last step —
        # exposed for diagnostics (e.g. correlating disagreement rate with
        # realized mIoU benefit across corruption types/benchmarks).
        self.last_jsd: Optional[float] = None
        # Pre-update JSD at every step of the most recent adapt() call.
        self.last_losses: List[float] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def adapt(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run PDA on batch *x* and return the adapted prediction.

        Steps:
          1. Freeze all parameters except BN affine.
          2. For n_steps gradient steps:
               a. Identify texture channels.
               b. Generate K perturbed forward passes.
               c. Compute mean pairwise JSD, backward, update BN.
          3. Compute final prediction as mean of K perturbed logits.
          4. Restore original BN state (stateless: no leak between samples).

        Returns:
            logits: (B, num_classes, H, W).
        """
        # --- Freeze all, un-freeze BN affine ---
        original_grad_state = self._freeze_non_bn()
        bn_params = self._collect_bn_params()

        if not bn_params:
            # No BN — just do a standard forward
            self._restore_grad_state(original_grad_state)
            return self._plain_forward(x)

        if self.continual:
            if self._persistent_optimizer is None:
                self._persistent_optimizer = torch.optim.Adam(bn_params, lr=self.lr)
            optimizer = self._persistent_optimizer
        else:
            optimizer = torch.optim.Adam(bn_params, lr=self.lr)

        self.model.train()  # activate BN running-stats accumulation

        self.last_losses = []
        for _ in range(self.n_steps):
            optimizer.zero_grad()

            # Identify texture channels
            with torch.no_grad():
                primary = self._get_primary_features(x)
            channel_mask = self.identifier.identify(
                primary, threshold=self.texture_threshold
            )

            # K perturbed forward passes
            perturbed_logits = []
            for _ in range(self.K):
                lgt = self._perturbed_forward(x, channel_mask)
                perturbed_logits.append(lgt)

            probs = [F.softmax(l, dim=1) for l in perturbed_logits]
            loss  = _mean_pairwise_jsd(probs)
            self.last_jsd = loss.detach().item()
            self.last_losses.append(self.last_jsd)
            loss.backward()
            optimizer.step()

            if self.continual and self.restart_prob > 0.0:
                self._stochastic_restore()

        # --- Final prediction ---
        self.model.eval()
        with torch.no_grad():
            if self.clean_final:
                # One clean forward pass through the now-adapted model —
                # avoids compounding perturbation distortion on top of
                # already-corrupted input for the reported prediction.
                logits = self._plain_forward(x)
            else:
                primary      = self._get_primary_features(x)
                channel_mask = self.identifier.identify(primary, threshold=self.texture_threshold)
                final_logits = []
                for _ in range(self.K):
                    final_logits.append(self._perturbed_forward(x, channel_mask))
                logits = torch.stack(final_logits, dim=0).mean(dim=0)

        # --- Restore parameter grad state ---
        self._restore_grad_state(original_grad_state)

        return logits

    def reset(self) -> None:
        """
        Restore BN parameters to their pre-adaptation state.

        Call this between samples when doing per-sample stateless adaptation.
        """
        self._restore_bn_state(self._original_bn_state)

    def update_base_state(self) -> None:
        """
        Commit the current BN state as the new base (useful for continual
        adaptation where you want to accumulate updates across batches).
        """
        self._original_bn_state = self._save_bn_state()

    def _stochastic_restore(self) -> None:
        """
        CoTTA-style stochastic restoration: reset each adapted weight/bias
        *element* back to its original pretrained value with probability
        self.restart_prob, independently per element. Bounds parameter
        drift over a long continual stream without undoing adaptation
        entirely, unlike a full reset().
        """
        with torch.no_grad():
            for name, module in self.model.named_modules():
                if not isinstance(module, self._norm_classes):
                    continue
                orig = self._original_bn_state.get(name)
                if orig is None:
                    continue
                for pname in ("weight", "bias"):
                    param = getattr(module, pname, None)
                    orig_val = orig.get(pname)
                    if param is None or orig_val is None or not param.requires_grad:
                        continue
                    mask = torch.bernoulli(
                        torch.full_like(param.data, self.restart_prob)
                    ).bool()
                    param.data[mask] = orig_val.to(param.device)[mask]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_primary_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run backbone and return primary (first) feature map."""
        with torch.no_grad():
            feats = self.model.backbone(x)
        if isinstance(feats, (list, tuple)):
            return feats[0]
        return feats

    def _perturbed_forward(
        self,
        x: torch.Tensor,
        channel_mask: torch.Tensor,
    ) -> torch.Tensor:
        """One perturbed forward pass through backbone + head."""
        raw_features = self.model.backbone(x)
        primary = raw_features[0] if isinstance(raw_features, (list, tuple)) else raw_features

        perturbed_primary = self.perturber.perturb(primary, channel_mask)

        if isinstance(raw_features, (list, tuple)):
            features = list(raw_features)
            features[0] = perturbed_primary
        else:
            features = perturbed_primary

        logits = self.model.head(features)
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits

    def _plain_forward(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            feats  = self.model.backbone(x)
            logits = self.model.head(feats)
            if logits.shape[-2:] != x.shape[-2:]:
                logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits

    # ------------------------------------------------------------------
    # Parameter / BN state management
    # ------------------------------------------------------------------

    def _freeze_non_bn(self) -> Dict[str, bool]:
        """Freeze all parameters; un-freeze affine params of self.norm_types. Returns original requires_grad."""
        state: Dict[str, bool] = {}
        for name, param in self.model.named_parameters():
            state[name] = param.requires_grad
            param.requires_grad_(False)
        # Un-freeze normalization-layer affine parameters for the selected kinds
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

    def _collect_bn_params(self) -> List[torch.Tensor]:
        """Collect trainable affine parameters from the selected norm kinds."""
        params = []
        for module in self.model.modules():
            if isinstance(module, self._norm_classes):
                if module.weight is not None and module.weight.requires_grad:
                    params.append(module.weight)
                if module.bias is not None and module.bias.requires_grad:
                    params.append(module.bias)
        return params

    def _save_bn_state(self) -> Dict[str, Dict[str, Optional[torch.Tensor]]]:
        """Save affine state for modules matching the selected norm kinds."""
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
                # LayerNorm / GroupNorm — affine only, no running statistics
                state[name] = {
                    "_kind": "ln",
                    "weight": module.weight.data.clone() if module.weight is not None else None,
                    "bias":   module.bias.data.clone()   if module.bias   is not None else None,
                }
        return state

    def _restore_bn_state(
        self,
        state: Dict[str, Dict[str, Optional[torch.Tensor]]],
    ) -> None:
        """Restore affine state for all BN and LN modules."""
        for name, module in self.model.named_modules():
            if name not in state:
                continue
            s = state[name]
            if s.get("weight") is not None and module.weight is not None:
                module.weight.data.copy_(s["weight"])
            if s.get("bias") is not None and module.bias is not None:
                module.bias.data.copy_(s["bias"])
            # BN-only fields
            if s.get("_kind") == "bn":
                if s.get("running_mean") is not None and module.running_mean is not None:
                    module.running_mean.copy_(s["running_mean"])
                if s.get("running_var") is not None and module.running_var is not None:
                    module.running_var.copy_(s["running_var"])
                if s.get("num_batches_tracked") is not None and module.num_batches_tracked is not None:
                    module.num_batches_tracked.copy_(s["num_batches_tracked"])
