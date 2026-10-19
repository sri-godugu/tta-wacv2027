"""
instance_whitening.py — Instance-Selective Whitening (ISW) loss.

Reimplementation of the core mechanism from RobustNet (Choi et al., CVPR
2021, "RobustNet: Improving Domain Generalization in Urban-Scene
Segmentation via Instance Selective Whitening") for use as a baseline
comparison. Written from the paper's description, not copied from the
authors' released code (not available in this environment).

Idea: for a backbone feature map, compute each instance's channel-wise
covariance matrix. Off-diagonal covariance terms that vary a lot *across*
instances in a batch are presumed to encode photometric/style-sensitive
correlations (harmful for domain generalization); terms with low
cross-instance variance are presumed semantic/content correlations that
should be preserved. Only the high-variance terms are penalized
("selective" whitening), rather than decorrelating every channel pair
uniformly.
"""

import torch


def instance_whitening_loss(
    feat: torch.Tensor,
    margin_frac: float = 0.33,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Args:
        feat: (B, C, H, W) backbone feature map.
        margin_frac: fraction of off-diagonal covariance entries (ranked by
            cross-instance variance) to penalize. 0.33 follows the
            proportion used in the original paper's ablation.
        eps: numerical floor.

    Returns:
        scalar loss — mean absolute covariance over the selected
        high-variance off-diagonal entries, averaged over the batch.
    """
    B, C, H, W = feat.shape
    if B < 2:
        # Variance across instances is undefined for a single-sample batch.
        return torch.tensor(0.0, device=feat.device)

    x = feat.reshape(B, C, H * W)
    x = x - x.mean(dim=2, keepdim=True)
    cov = torch.bmm(x, x.transpose(1, 2)) / max(H * W - 1, 1)  # (B, C, C)

    off_diag_mask = ~torch.eye(C, dtype=torch.bool, device=feat.device)
    var_cov = cov.var(dim=0, unbiased=False)  # (C, C)
    var_off = var_cov[off_diag_mask]

    k = max(1, int(margin_frac * var_off.numel()))
    threshold = torch.topk(var_off, k, largest=True).values.min()
    select_mask = (var_cov >= threshold) & off_diag_mask

    if select_mask.sum() == 0:
        return torch.tensor(0.0, device=feat.device)

    selected_cov = cov[:, select_mask]  # (B, num_selected)
    return selected_cov.abs().mean()
