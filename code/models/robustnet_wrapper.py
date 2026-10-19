"""
robustnet_wrapper.py — DeepLabV3+/ResNet-50 with Instance-Selective
Whitening (ISW), as a self-run RobustNet baseline for direct comparison
against RobustSeg (TCPR+PDA) on the same backbone/data/protocol.

RobustNet is a training-time-only method (no test-time adaptation
component) — it is evaluated the same way as a vanilla model at test time.

Compatible with training/trainer.py's Trainer as-is: forward_train()
returns the same {loss, loss_ce, loss_cons, logits} shape Trainer expects,
with the ISW loss standing in the 'loss_cons' slot (weighted by
lambda_cons, repurposed here as the ISW loss weight) purely so the
existing training loop / logging / checkpointing can be reused unchanged.
"""

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .instance_whitening import instance_whitening_loss


class RobustNetWrapper(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        isw_layer_names: List[str] = ("layer1", "layer2"),
        lambda_isw: float = 0.6,
        isw_margin_frac: float = 0.33,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.lambda_cons = lambda_isw  # kept as 'lambda_cons' for CLI/Trainer compatibility
        self.isw_margin_frac = isw_margin_frac

        self._isw_features: Dict[str, torch.Tensor] = {}
        self._hook_handles = []
        # self.backbone is a _DLBackbone wrapping the raw torchvision
        # IntermediateLayerGetter, which preserves the original ResNet's
        # named submodules (layer1..layer4) even though its own forward
        # only returns the "out" key.
        raw_modules = dict(self.backbone.backbone.named_modules())
        for name in isw_layer_names:
            if name not in raw_modules:
                raise ValueError(f"No submodule named '{name}' found in backbone; "
                                  f"available top-level names: "
                                  f"{[n for n in raw_modules if n.count('.') == 0]}")
            handle = raw_modules[name].register_forward_hook(self._make_hook(name))
            self._hook_handles.append(handle)

    def _make_hook(self, name: str):
        def hook(module, inp, out):
            self._isw_features[name] = out
        return hook

    def _full_forward(self, x: torch.Tensor):
        self._isw_features.clear()
        primary = self.backbone(x)
        logits = self.head(primary)
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._full_forward(x)

    def forward_train(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self._full_forward(x)
        loss_ce = F.cross_entropy(logits, y, ignore_index=255)

        loss_isw = torch.tensor(0.0, device=x.device)
        for feat in self._isw_features.values():
            loss_isw = loss_isw + instance_whitening_loss(feat, margin_frac=self.isw_margin_frac)
        if self._isw_features:
            loss_isw = loss_isw / len(self._isw_features)

        loss = loss_ce + self.lambda_cons * loss_isw

        return {
            "loss": loss,
            "loss_ce": loss_ce.detach(),
            "loss_cons": loss_isw.detach(),  # ISW loss, logged in the 'cons' slot
            "logits": logits.detach(),
        }
