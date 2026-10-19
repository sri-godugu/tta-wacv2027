"""
TextureChannelIdentifier: identifies texture-dominant channels in backbone feature maps
by analysing per-channel Fourier amplitude spectra.

A channel is texture-dominant if its high-frequency energy ratio tau_k exceeds a
threshold.  High-frequency energy is defined as energy outside the central low-
frequency region (inner (H/4) x (W/4) window in the shifted spectrum).
"""

import torch
import torch.nn as nn
from typing import Tuple


class TextureChannelIdentifier(nn.Module):
    """
    Identifies texture-dominant channels in a feature map tensor.

    Args:
        low_freq_fraction (float): fraction of the spectrum (centered) considered
            low-frequency.  Defaults to 0.25, meaning the inner 25% in each
            spatial dimension.
    """

    def __init__(self, low_freq_fraction: float = 0.25) -> None:
        super().__init__()
        self.low_freq_fraction = low_freq_fraction

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_texture_ratios(self, feature_maps: torch.Tensor) -> torch.Tensor:
        """
        Compute per-channel texture energy ratio tau_k.

        tau_k = (total_energy - low_freq_energy) / (total_energy + eps)

        Args:
            feature_maps: (B, C, H, W) float tensor.

        Returns:
            ratios: (C,) tensor with values in [0, 1].
        """
        B, C, H, W = feature_maps.shape
        eps = 1e-8

        # 2-D FFT over spatial dims; result shape (B, C, H, W) complex
        freq = torch.fft.fft2(feature_maps)          # un-shifted
        freq_shifted = torch.fft.fftshift(freq, dim=(-2, -1))  # center DC

        amplitude = freq_shifted.abs()               # (B, C, H, W)

        # Low-frequency mask: central rectangle
        lf_mask = self._low_freq_mask(H, W, feature_maps.device)  # (H, W) bool

        # Energy = sum of squared amplitudes
        energy_total = (amplitude ** 2).sum(dim=(-2, -1))          # (B, C)
        energy_lf    = (amplitude ** 2 * lf_mask.float()).sum(dim=(-2, -1))  # (B, C)
        energy_hf    = energy_total - energy_lf                    # (B, C)

        tau = energy_hf / (energy_total + eps)  # (B, C)

        # Average over batch
        return tau.mean(dim=0)  # (C,)

    def identify(
        self,
        feature_maps: torch.Tensor,
        threshold: float = 0.45,
    ) -> torch.Tensor:
        """
        Return a binary mask of texture-dominant channels.

        Args:
            feature_maps: (B, C, H, W) float tensor.
            threshold: channels with tau_k > threshold are flagged as
                texture-dominant.

        Returns:
            mask: (C,) bool tensor.  True where channel is texture-dominant.
        """
        tau = self.compute_texture_ratios(feature_maps)
        return tau > threshold

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _low_freq_mask(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """
        Build a boolean mask marking the low-frequency central region.

        The mask is True inside the central (H * frac) x (W * frac) rectangle
        of the *shifted* spectrum.

        Returns:
            mask: (H, W) bool tensor on *device*.
        """
        h_lf = max(1, int(H * self.low_freq_fraction))
        w_lf = max(1, int(W * self.low_freq_fraction))

        h_start = H // 2 - h_lf // 2
        h_end   = h_start + h_lf
        w_start = W // 2 - w_lf // 2
        w_end   = w_start + w_lf

        mask = torch.zeros(H, W, dtype=torch.bool, device=device)
        mask[h_start:h_end, w_start:w_end] = True
        return mask
