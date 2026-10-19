"""
evaluate_on_dataset: standard evaluation loop.

When use_pda=True the model is wrapped with PerturbationDisagreementAdaptor
so each batch goes through test-time adaptation before prediction.

The function is backend-agnostic: it works with RobustSegWrapper or any
nn.Module whose forward(x) returns (B, num_classes, H, W) logits.
"""

import sys
import os
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Make imports work whether called as a module or script
_CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

from evaluation.metrics import SegmentationMetrics


def evaluate_on_dataset(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    use_pda: bool = False,
    use_tent: bool = False,
    use_oracle: bool = False,
    use_dua: bool = False,
    use_pseudo: bool = False,
    num_classes: int = 19,
    ignore_index: int = 255,
    K: int = 4,
    n_adapt_steps: int = 1,
    lr_pda: float = 1e-3,
    texture_threshold: float = 0.45,
    norm_types: frozenset = frozenset({"bn"}),
    clean_final: bool = False,
    continual: bool = False,
    restart_prob: float = 0.0,
    verbose: bool = True,
    per_image_cm: bool = False,
    save_pred_idx: Optional[set] = None,
) -> Dict[str, object]:
    """
    Evaluate *model* on *dataloader* and return segmentation metrics.

    Args:
        model:             nn.Module.  Moved to *device* internally.
        dataloader:        yields (images, labels) batches.
        device:            computation device.
        use_pda:           if True, apply PDA test-time adaptation per batch.
        use_tent:          if True, apply TENT (entropy-minimization) test-time
                            adaptation per batch instead of PDA. Mutually exclusive
                            with use_pda.
        num_classes:       number of segmentation classes.
        ignore_index:      label value to ignore (255 for Cityscapes).
        K:                 PDA — number of perturbed forward passes.
        n_adapt_steps:     PDA — gradient steps per batch.
        lr_pda:            PDA — learning rate.
        texture_threshold: PDA — threshold for texture channel detection.
        norm_types:        PDA — which norm kinds to adapt, subset of {"bn","ln","gn"}.
        clean_final:       PDA — report predictions from a clean forward pass through
                            the adapted model instead of averaging K perturbed passes.
        continual:         PDA — let adaptation (BN values + optimiser state) persist
                            across the whole dataloader instead of resetting per sample.
        verbose:           print progress bar.

    Returns:
        dict with keys: 'miou', 'per_class_iou', 'pixel_accuracy',
                        'mean_accuracy'.
    """
    if sum([use_pda, use_tent, use_oracle, use_dua, use_pseudo]) > 1:
        raise ValueError("adaptation modes are mutually exclusive")

    model = model.to(device)
    model.eval()

    # Optionally wrap with a test-time adaptor (PDA or TENT)
    adaptor = None
    if use_pda:
        from adaptation.pda import PerturbationDisagreementAdaptor
        adaptor = PerturbationDisagreementAdaptor(
            model=model,
            lr=lr_pda,
            n_steps=n_adapt_steps,
            K=K,
            texture_threshold=texture_threshold,
            norm_types=norm_types,
            clean_final=clean_final,
            continual=continual,
            restart_prob=restart_prob,
        )
    elif use_tent:
        from adaptation.tent import TentAdaptor
        adaptor = TentAdaptor(
            model=model,
            lr=lr_pda,
            n_steps=n_adapt_steps,
            norm_types=norm_types,
        )
    elif use_dua:
        # Gradient-free: re-estimates BN running statistics only (Mirza et al.).
        from adaptation.dua import DUAAdaptor
        adaptor = DUAAdaptor(model=model)
    elif use_pseudo:
        # Self-training: CE against the model's own frozen hard predictions.
        from adaptation.pseudolabel import PseudoLabelAdaptor
        adaptor = PseudoLabelAdaptor(
            model=model, lr=lr_pda, n_steps=n_adapt_steps, norm_types=norm_types,
        )
    elif use_oracle:
        # Supervised ceiling: same parameters/optimiser/steps as TENT, but the
        # adaptation loss is CE against the labels. Transductive by construction.
        from adaptation.oracle import OracleAdaptor
        adaptor = OracleAdaptor(
            model=model,
            lr=lr_pda,
            n_steps=n_adapt_steps,
            norm_types=norm_types,
            ignore_index=ignore_index,
        )

    metrics = SegmentationMetrics(
        num_classes=num_classes,
        ignore_index=ignore_index,
    )

    jsd_sum = 0.0
    jsd_count = 0
    conf_sum = 0.0
    conf_count = 0
    # Per-image confusion matrices (+ the batch each image was adapted in).
    # These sum exactly to the run-level matrix, so a bootstrap over them
    # reproduces the reported mIoU; batch ids let the resampling respect the
    # fact that PDA/TENT adapt on a whole batch, not per image.
    cm_per_image: list = []
    cm_batch_ids: list = []
    batch_idx = 0
    # Elementwise sum over batches of the adaptor's per-step loss trace.
    trace_sum: Optional[list] = None
    trace_count = 0
    # Saved argmax predictions (+ labels) for selected dataset indices, in
    # eval order, so qualitative figures show exactly the scored pixels.
    saved_preds: Dict[int, object] = {}
    image_offset = 0

    iterator = (
        tqdm(dataloader, desc="eval", dynamic_ncols=True, leave=False)
        if (verbose and HAS_TQDM)
        else dataloader
    )

    for batch in iterator:
        images, labels = batch
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(use_pda or use_tent or use_oracle or use_pseudo):  # DUA needs none
            if adaptor is not None:
                logits = adaptor.adapt(images, labels) if use_oracle else adaptor.adapt(images)
                if use_pda and getattr(adaptor, "last_jsd", None) is not None:
                    jsd_sum += adaptor.last_jsd
                    jsd_count += 1
                trace = getattr(adaptor, "last_losses", None)
                if trace:
                    if trace_sum is None:
                        trace_sum = [0.0] * len(trace)
                    trace_sum = [a + b for a, b in zip(trace_sum, trace)]
                    trace_count += 1
                if use_dua:
                    pass  # DUA is continual within a condition, by design
                elif not (use_pda and continual):
                    # Reset BN to pre-adaptation state for stateless adaptation
                    adaptor.reset()
            else:
                with torch.no_grad():
                    if hasattr(model, "forward_test"):
                        logits = model.forward_test(images, adapt=False)
                    else:
                        logits = model(images)

        # Upsample to label resolution if needed
        if logits.shape[-2:] != labels.shape[-2:]:
            logits = F.interpolate(
                logits, size=labels.shape[-2:],
                mode="bilinear", align_corners=False,
            )

        # Mean max-softmax confidence of the prediction used for this batch.
        # For a vanilla (no-adaptation) run this is the pre-adaptation model
        # confidence — a label-free proxy for baseline difficulty, logged for
        # the "when does TTA help" analysis.
        with torch.no_grad():
            conf_sum += F.softmax(logits, dim=1).max(dim=1).values.mean().item()
            conf_count += 1

        preds = logits.argmax(dim=1)
        preds_c, labels_c = preds.cpu(), labels.cpu()
        metrics.update(preds_c, labels_c)

        if save_pred_idx:
            for i in range(preds_c.shape[0]):
                if image_offset + i in save_pred_idx:
                    saved_preds[image_offset + i] = (
                        preds_c[i].to(torch.uint8).numpy(),
                        labels_c[i].to(torch.uint8).numpy(),
                    )
        image_offset += preds_c.shape[0]

        if per_image_cm:
            for i in range(preds_c.shape[0]):
                cm_per_image.append(
                    SegmentationMetrics.confusion_for(
                        preds_c[i], labels_c[i], num_classes, ignore_index
                    )
                )
                cm_batch_ids.append(batch_idx)
        batch_idx += 1

    result = metrics.compute_all()
    if per_image_cm:
        result["_per_image_cm"] = torch.stack(cm_per_image).numpy()
        result["_cm_batch_ids"] = cm_batch_ids
    if jsd_count > 0:
        result["mean_jsd"] = jsd_sum / jsd_count
    if trace_count > 0:
        result["mean_loss_trace"] = [t / trace_count for t in trace_sum]
    if save_pred_idx:
        result["_saved_preds"] = saved_preds
    if conf_count > 0:
        result["mean_confidence"] = conf_sum / conf_count
    return result
