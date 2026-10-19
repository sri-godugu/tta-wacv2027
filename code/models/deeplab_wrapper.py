"""
deeplab_wrapper.py — builds a DeepLabV3+/ResNet-50 model split into
(backbone, head) so it fits the RobustSegWrapper API.

ResNet uses BatchNorm throughout (no LayerNorm/GroupNorm anywhere), unlike
SegFormer's MixTransformer — this backbone exists specifically to test
whether PDA shows a clearer benefit on a BN-only-normalized architecture,
where norm_types="bn" adapts the entire backbone rather than just a small
decode head.
"""

from typing import Tuple

import torch.nn as nn


class _DLBackbone(nn.Module):
    """
    Wraps torchvision's IntermediateLayerGetter-based backbone, which
    returns an OrderedDict keyed by output name (e.g. "out") rather than
    a plain tensor — unwrap it here so RobustSegWrapper sees a single
    tensor, matching its (tensor | list/tuple of tensors) contract.
    """

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)["out"]


class _DLHead(nn.Module):
    def __init__(self, classifier: nn.Module) -> None:
        super().__init__()
        self.classifier = classifier

    def forward(self, features):
        return self.classifier(features)


# Architectures available for the backbone-replication study. All are
# BatchNorm-rich (the normalization family our adaptation protocol acts on),
# and they vary the two axes independently: segmentation head (ASPP vs plain
# FCN) and backbone depth (ResNet-50 vs ResNet-101).
_ARCHS = {
    "deeplabv3_resnet50":  ("deeplabv3_resnet50",  "ResNet50_Weights"),
    "deeplabv3_resnet101": ("deeplabv3_resnet101", "ResNet101_Weights"),
    "fcn_resnet50":        ("fcn_resnet50",        "ResNet50_Weights"),
    "fcn_resnet101":       ("fcn_resnet101",       "ResNet101_Weights"),
}


def build_deeplabv3plus(
    pretrained: bool = False,
    num_classes: int = 19,
    arch: str = "deeplabv3_resnet50",
) -> Tuple[nn.Module, nn.Module]:
    """
    Build a torchvision segmentation (backbone, head) pair.

    Args:
        pretrained:  load an ImageNet-pretrained backbone. The head is always
                     freshly initialized for num_classes (torchvision's
                     COCO-pretrained full model has a mismatched, 21-class
                     head, so only the backbone weights are reused here —
                     same pattern as the SegFormer HF-pretrained-backbone path).
        num_classes: number of segmentation classes.
        arch:        one of _ARCHS. Default preserves the original
                     DeepLabV3+/ResNet-50 behaviour exactly.

    Returns:
        (backbone, head) both as nn.Module.
    """
    import torchvision.models.segmentation as seg_models
    import torchvision.models as tv_models

    if arch not in _ARCHS:
        raise ValueError(f"Unknown arch {arch!r}; valid: {sorted(_ARCHS)}")
    ctor_name, weights_name = _ARCHS[arch]

    if pretrained:
        weights_backbone = getattr(tv_models, weights_name).IMAGENET1K_V2
    else:
        weights_backbone = None

    net = getattr(seg_models, ctor_name)(
        weights=None,
        weights_backbone=weights_backbone,
        num_classes=num_classes,
    )

    backbone = _DLBackbone(net.backbone)
    head     = _DLHead(net.classifier)
    return backbone, head
