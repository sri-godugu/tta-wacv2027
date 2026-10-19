"""
RobustSegWrapper: wraps any backbone + head pair with the RobustSeg framework.

Training  (HFCPR):
  1. Forward pass through backbone → features.
  2. Identify high-frequency-dominant channels via FFT amplitude analysis.
  3. Generate one phase-randomized perturbed copy of features.
  4. Decode both via head → predictions.
  5. Loss = CE(pred_orig, labels) + lambda_cons * KL(pred_orig || pred_perturbed)

Inference (PDA, optional):
  1. Freeze all parameters except normalization-layer affine params.
       - BatchNorm{1,2,3}d: weight, bias, running_mean, running_var
       - LayerNorm: weight, bias
     On convolutional backbones (ResNet), BN is present throughout.
     On transformer backbones (SegFormer MixTransformer), LN is in the
     encoder and BN is in the decode head; PDA adapts both.
  2. Generate K perturbed forward passes.
  3. Minimise mean pairwise Jensen-Shannon divergence across the K predictions
     for n_adapt_steps gradient steps.
  4. Return the final prediction (average over K).

The wrapper is intentionally backbone-agnostic.  It only requires:
  - backbone(x)  -> features   (tensor *or* list/tuple of tensors)
  - head(features) -> logits   (B, num_classes, H, W)
"""

import copy
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .texture_identifier import TextureChannelIdentifier
from .feature_perturber import FeaturePerturber


class RobustSegWrapper(nn.Module):
    """
    Args:
        backbone: any nn.Module that maps (B, 3, H, W) → feature tensor or
            list of feature tensors.
        head: any nn.Module that maps the backbone output → (B, C_cls, H, W).
        texture_threshold: tau threshold for identifying texture channels.
        lambda_cons: weight of the TCPR consistency loss during training.
        K: number of perturbed samples for PDA.
    """

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        texture_threshold: float = 0.45,
        lambda_cons: float = 0.3,
        K: int = 4,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.texture_threshold = texture_threshold
        self.lambda_cons = lambda_cons
        self.K = K

        self.identifier = TextureChannelIdentifier()
        self.perturber = FeaturePerturber()

    # ------------------------------------------------------------------
    # Feature extraction helpers
    # ------------------------------------------------------------------

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run backbone and return the *primary* feature map used for
        texture analysis.  When the backbone returns a list/tuple (e.g.
        multi-scale SegFormer), we use the first (coarsest) map.
        """
        feats = self.backbone(x)
        if isinstance(feats, (list, tuple)):
            return feats[0]
        return feats

    def _full_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run backbone + head.

        Returns:
            features: primary feature map (B, C_feat, H_f, W_f).
            logits:   (B, num_classes, H, W) — same spatial resolution as x.
        """
        features = self.backbone(x)
        logits = self.head(features)
        # Upsample to input resolution if needed
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(
                logits, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
        primary = features[0] if isinstance(features, (list, tuple)) else features
        return primary, logits

    def _forward_with_perturbed_features(
        self,
        x: torch.Tensor,
        channel_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass where texture-dominant backbone channels are phase-
        randomized before being passed to the head.

        Returns:
            logits: (B, num_classes, H, W)
        """
        raw_features = self.backbone(x)
        primary = raw_features[0] if isinstance(raw_features, (list, tuple)) else raw_features

        perturbed_primary = self.perturber.perturb(primary, channel_mask)

        if isinstance(raw_features, (list, tuple)):
            perturbed_features = list(raw_features)
            perturbed_features[0] = perturbed_primary
        else:
            perturbed_features = perturbed_primary

        logits = self.head(perturbed_features)
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(
                logits, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
        return logits

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    def forward_train(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Full TCPR training forward pass.

        Args:
            x: input images (B, 3, H, W).
            y: segmentation labels (B, H, W) with ignore_index=255.

        Returns:
            dict with keys: 'loss', 'loss_ce', 'loss_cons', 'logits'.
        """
        primary, logits_orig = self._full_forward(x)

        # --- CE loss ---
        loss_ce = F.cross_entropy(logits_orig, y, ignore_index=255)

        # --- TCPR: identify texture channels and generate perturbed pred ---
        channel_mask = self.identifier.identify(primary, threshold=self.texture_threshold)

        logits_pert = self._forward_with_perturbed_features(x, channel_mask)

        # --- Consistency (KL) loss ---
        loss_cons = self._compute_consistency_loss(logits_orig, logits_pert)

        loss = loss_ce + self.lambda_cons * loss_cons

        return {
            "loss": loss,
            "loss_ce": loss_ce.detach(),
            "loss_cons": loss_cons.detach(),
            "logits": logits_orig.detach(),
        }

    # ------------------------------------------------------------------
    # Inference / test forward
    # ------------------------------------------------------------------

    def forward_test(
        self,
        x: torch.Tensor,
        adapt: bool = True,
        n_adapt_steps: int = 1,
    ) -> torch.Tensor:
        """
        Inference forward.  When adapt=True, runs PDA to update BatchNorm
        statistics/affine params before producing the final prediction.

        Args:
            x: input images (B, 3, H, W).
            adapt: whether to run PDA adaptation.
            n_adapt_steps: gradient steps for PDA.

        Returns:
            logits: (B, num_classes, H, W).
        """
        if adapt:
            return self._adapt_norm(x, n_steps=n_adapt_steps)

        with torch.no_grad():
            _, logits = self._full_forward(x)
        return logits

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Dict]:
        """
        Convenience forward: training mode when y is provided, else test mode
        (no PDA) for compatibility with standard PyTorch training loops.
        """
        if y is not None:
            return self.forward_train(x, y)
        _, logits = self._full_forward(x)
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_consistency_loss(
        pred_orig: torch.Tensor,
        pred_perturbed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pixel-wise KL divergence KL(P_orig || P_perturbed), averaged over
        all pixels and batch elements.

        Args:
            pred_orig, pred_perturbed: (B, num_classes, H, W) raw logits.

        Returns:
            scalar loss tensor.
        """
        eps = 1e-8
        p = F.softmax(pred_orig, dim=1).clamp(min=eps)
        q = F.softmax(pred_perturbed, dim=1).clamp(min=eps)
        # KL(p||q) = sum_c p_c * log(p_c / q_c)
        kl = (p * (p.log() - q.log())).sum(dim=1)  # (B, H, W)
        return kl.mean()

    @staticmethod
    def _pda_loss(predictions_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Mean pairwise Jensen-Shannon divergence across K predictions.

        JSD(p||q) = 0.5 * KL(p||m) + 0.5 * KL(q||m),  m = 0.5*(p+q)

        Args:
            predictions_list: list of K tensors, each (B, num_classes, H, W).

        Returns:
            scalar JSD loss.
        """
        eps = 1e-8
        K = len(predictions_list)
        if K < 2:
            return torch.tensor(0.0, device=predictions_list[0].device)

        probs = [F.softmax(p, dim=1).clamp(min=eps) for p in predictions_list]

        total_jsd = torch.tensor(0.0, device=predictions_list[0].device)
        n_pairs = 0

        for i in range(K):
            for j in range(i + 1, K):
                p = probs[i]
                q = probs[j]
                m = 0.5 * (p + q)
                m = m.clamp(min=eps)
                kl_pm = (p * (p.log() - m.log())).sum(dim=1)
                kl_qm = (q * (q.log() - m.log())).sum(dim=1)
                jsd = 0.5 * (kl_pm + kl_qm)  # (B, H, W)
                total_jsd = total_jsd + jsd.mean()
                n_pairs += 1

        return total_jsd / n_pairs

    # ------------------------------------------------------------------
    # PDA adaptation
    # ------------------------------------------------------------------

    def _adapt_norm(
        self,
        x: torch.Tensor,
        n_steps: int = 1,
    ) -> torch.Tensor:
        """
        Perturbation-Disagreement Adaptation (PDA) at test time.

        Adapts normalization-layer affine parameters (BatchNorm weight/bias
        for convolutional backbones; LayerNorm weight/bias for transformer
        backbones).  Running statistics are also updated for BN layers.

        Procedure:
          1. Save original normalization-layer state (BN + LN).
          2. Freeze all parameters except norm affine params.
          3. For n_steps:
               a. Identify high-frequency-dominant channels.
               b. Generate K perturbed forward passes.
               c. Compute mean pairwise JSD.
               d. Gradient step on norm affine params only.
          4. Compute final prediction (mean over K perturbed logits).
          5. Restore original normalization-layer state.

        Returns:
            logits: (B, num_classes, H, W) — averaged over K predictions.
        """
        # --- Save norm state (BN + LN) ---
        bn_state = self._save_bn_state()

        # --- Freeze everything, un-freeze BN and LN affine params ---
        original_requires_grad: Dict[str, bool] = {}
        for name, param in self.named_parameters():
            original_requires_grad[name] = param.requires_grad
            param.requires_grad_(False)

        bn_params = []
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                                   nn.LayerNorm)):
                if module.weight is not None:
                    module.weight.requires_grad_(True)
                    bn_params.append(module.weight)
                if module.bias is not None:
                    module.bias.requires_grad_(True)
                    bn_params.append(module.bias)

        if len(bn_params) == 0:
            # No BN or LN layers — just run standard inference
            with torch.no_grad():
                _, logits = self._full_forward(x)
            self._restore_bn_state(bn_state)
            for name, param in self.named_parameters():
                param.requires_grad_(original_requires_grad.get(name, True))
            return logits

        optimizer = torch.optim.Adam(bn_params, lr=1e-3)

        self.train()  # activate BN running-stats update

        for _ in range(n_steps):
            optimizer.zero_grad()

            primary_feat = self.extract_features(x)
            channel_mask = self.identifier.identify(
                primary_feat.detach(), threshold=self.texture_threshold
            )

            perturbed_logits = []
            for _ in range(self.K):
                logits_k = self._forward_with_perturbed_features(x, channel_mask)
                perturbed_logits.append(logits_k)

            loss = self._pda_loss(perturbed_logits)
            loss.backward()
            optimizer.step()

        # --- Final prediction ---
        self.eval()
        with torch.no_grad():
            primary_feat = self.extract_features(x)
            channel_mask = self.identifier.identify(
                primary_feat, threshold=self.texture_threshold
            )
            final_logits_list = []
            for _ in range(self.K):
                lgt = self._forward_with_perturbed_features(x, channel_mask)
                final_logits_list.append(lgt)
            # Average logits over K samples for the final prediction
            final_logits = torch.stack(final_logits_list, dim=0).mean(dim=0)

        # --- Restore state ---
        self._restore_bn_state(bn_state)
        for name, param in self.named_parameters():
            param.requires_grad_(original_requires_grad.get(name, True))

        return final_logits

    # ------------------------------------------------------------------
    # BN state save / restore
    # ------------------------------------------------------------------

    def _save_bn_state(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Return a deep copy of all BN and LN module states keyed by module name."""
        state: Dict[str, Dict[str, torch.Tensor]] = {}
        for name, module in self.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                state[name] = {
                    "_kind": "bn",
                    "weight": module.weight.data.clone() if module.weight is not None else None,
                    "bias":   module.bias.data.clone()   if module.bias   is not None else None,
                    "running_mean": module.running_mean.clone()               if module.running_mean        is not None else None,
                    "running_var":  module.running_var.clone()                if module.running_var         is not None else None,
                    "num_batches_tracked": module.num_batches_tracked.clone() if module.num_batches_tracked is not None else None,
                }
            elif isinstance(module, nn.LayerNorm):
                state[name] = {
                    "_kind": "ln",
                    "weight": module.weight.data.clone() if module.weight is not None else None,
                    "bias":   module.bias.data.clone()   if module.bias   is not None else None,
                }
        return state

    def _restore_bn_state(self, state: Dict[str, Dict[str, torch.Tensor]]) -> None:
        """Restore BN and LN module states from a previously saved snapshot."""
        for name, module in self.named_modules():
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
