"""
FeaturePerturber: applies Fourier phase randomization to texture-dominant channels.

For each channel flagged by the texture mask:
  1. Compute 2-D FFT.
  2. Extract amplitude and phase.
  3. Replace phase with uniform random values in [-pi, pi].
  4. Reconstruct via inverse FFT.

Channels where mask=False are passed through unchanged.
"""

import torch
import torch.nn as nn
from typing import List


class FeaturePerturber(nn.Module):
    """
    Perturbs feature maps by randomizing the Fourier phase of texture-dominant
    channels while leaving amplitude (and non-texture channels) intact.
    """

    def __init__(self) -> None:
        super().__init__()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def perturb(
        self,
        features: torch.Tensor,
        channel_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply one round of Fourier phase randomization.

        Args:
            features: (B, C, H, W) float tensor.
            channel_mask: (C,) bool tensor.  True marks texture-dominant
                channels that will be perturbed.

        Returns:
            perturbed: (B, C, H, W) float tensor (same dtype/device as input).
        """
        if not channel_mask.any():
            # Nothing to perturb
            return features.clone()

        output = features.clone()
        B, C, H, W = features.shape

        # Indices of channels to perturb
        perturb_idx = torch.where(channel_mask)[0]  # (P,)

        # Extract the subset to process in one batched FFT call
        subset = features[:, perturb_idx, :, :]     # (B, P, H, W)

        perturbed_subset = self._phase_randomize(subset)

        # Write back — keep real part only (imaginary artefacts are negligible
        # but we enforce real to avoid downstream surprises)
        output[:, perturb_idx, :, :] = perturbed_subset

        return output

    def perturb_k_times(
        self,
        features: torch.Tensor,
        channel_mask: torch.Tensor,
        K: int = 4,
    ) -> List[torch.Tensor]:
        """
        Generate K independently perturbed copies of *features*.

        Each call to perturb() uses a fresh random phase draw, so the K
        copies are statistically independent.

        Args:
            features: (B, C, H, W) float tensor.
            channel_mask: (C,) bool tensor.
            K: number of perturbed copies.

        Returns:
            List of K tensors, each (B, C, H, W).
        """
        return [self.perturb(features, channel_mask) for _ in range(K)]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _phase_randomize(x: torch.Tensor) -> torch.Tensor:
        """
        Randomize the Fourier phase of tensor *x* while preserving amplitude.

        Args:
            x: (..., H, W) real float tensor.

        Returns:
            reconstructed: (..., H, W) real float tensor.
        """
        freq = torch.fft.fft2(x)            # (..., H, W) complex

        amplitude = freq.abs()              # (..., H, W)

        # Sample random phases uniform in [-pi, pi]
        random_phase = (
            torch.rand_like(amplitude) * 2.0 * torch.pi - torch.pi
        )  # (..., H, W)

        # Build new complex spectrum with original amplitude, random phase
        new_freq = torch.polar(amplitude, random_phase)  # (..., H, W) complex

        # Inverse FFT; take real part to discard numerical imaginary residuals
        reconstructed = torch.fft.ifft2(new_freq).real  # (..., H, W)

        return reconstructed
