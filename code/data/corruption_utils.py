"""
corruption_utils.py — synthetic image corruptions for benchmarking.

apply_corruption(image, corruption_type, severity) applies one of eight
corruption types at five severity levels to a PIL image or torch tensor.

Supported corruptions:
  fog               — additive haze via depth-blended white noise
  rain              — random dark streaks
  blur_motion       — directional motion blur via scipy
  blur_defocus      — Gaussian blur simulating out-of-focus lens
  noise_gaussian    — additive Gaussian noise
  jpeg              — JPEG compression artefacts
  brightness        — global brightness shift
  contrast          — contrast scaling around mean

All corruptions are implemented with NumPy / SciPy / PIL only — no
external imagecorruptions package required.

Severity mapping: severity in {1,2,3,4,5} maps to increasing intensity.
"""

import io
import math
from typing import Union

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

try:
    from scipy.ndimage import uniform_filter, gaussian_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

import torch
import torchvision.transforms.functional as TF


# ---------------------------------------------------------------------------
# Severity tables
# ---------------------------------------------------------------------------

# Each entry: (param_for_sev1, param_for_sev2, ..., param_for_sev5)

_SEV_TABLES = {
    # (fog_density, blur_sigma)
    "fog":            [(0.05, 0.5), (0.10, 1.0), (0.20, 1.5), (0.35, 2.0), (0.50, 2.5)],
    # rain streak alpha
    "rain":           [0.15, 0.25, 0.35, 0.45, 0.60],
    # motion blur kernel half-length (pixels)
    "blur_motion":    [5, 10, 15, 20, 25],
    # Gaussian blur sigma
    "blur_defocus":   [1.0, 2.0, 3.0, 4.5, 6.0],
    # Gaussian noise std (0-255 scale)
    "noise_gaussian": [5, 15, 25, 35, 50],
    # JPEG quality
    "jpeg":           [90, 70, 50, 30, 15],
    # brightness additive delta (0-255 scale)
    "brightness":     [30, 60, 90, 120, 150],
    # contrast factor (> 1 = more contrast, < 1 = less)
    "contrast":       [0.4, 0.3, 0.2, 0.15, 0.1],
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_corruption(
    image: Union[Image.Image, torch.Tensor],
    corruption_type: str,
    severity: int,
) -> Union[Image.Image, torch.Tensor]:
    """
    Apply a synthetic corruption to *image*.

    Args:
        image:           PIL Image (H, W, 3) or torch tensor (3, H, W) float in [0,1].
        corruption_type: one of the supported corruption names (see module docstring).
        severity:        integer in {1, 2, 3, 4, 5}.

    Returns:
        Corrupted image in the same format/type as input.
    """
    if severity < 1 or severity > 5:
        raise ValueError(f"severity must be 1-5, got {severity}")
    if corruption_type not in _SEV_TABLES:
        raise ValueError(
            f"Unknown corruption_type '{corruption_type}'. "
            f"Supported: {list(_SEV_TABLES.keys())}"
        )

    is_tensor = isinstance(image, torch.Tensor)
    if is_tensor:
        pil_img = TF.to_pil_image(image.clamp(0.0, 1.0))
    else:
        pil_img = image

    fn = _CORRUPTION_FNS[corruption_type]
    corrupted_pil = fn(pil_img, severity - 1)  # 0-indexed into table

    if is_tensor:
        return TF.to_tensor(corrupted_pil)
    return corrupted_pil


# ---------------------------------------------------------------------------
# Corruption implementations
# ---------------------------------------------------------------------------

def _to_float_array(pil: Image.Image) -> np.ndarray:
    """Return float64 array in [0, 255]."""
    return np.array(pil, dtype=np.float64)


def _from_float_array(arr: np.ndarray) -> Image.Image:
    """Clip to [0,255], convert to uint8 PIL image."""
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


# ---- Fog ------------------------------------------------------------------

def _fog(pil: Image.Image, sev_idx: int) -> Image.Image:
    density, blur_sigma = _SEV_TABLES["fog"][sev_idx]
    img = _to_float_array(pil)
    H, W, _ = img.shape

    # Generate a foggy layer using fractal-like noise
    rng  = np.random.default_rng(seed=0)
    haze = rng.random((H, W)).astype(np.float64)

    if HAS_SCIPY:
        haze = gaussian_filter(haze, sigma=blur_sigma * max(H, W) / 100.0)
    else:
        # Fallback: box filter approximation
        k = max(1, int(blur_sigma * max(H, W) / 100.0))
        for _ in range(3):
            haze = _simple_box_filter(haze, k)

    # Normalise haze to [0, 1]
    haze = (haze - haze.min()) / (haze.max() - haze.min() + 1e-8)
    haze = haze[:, :, np.newaxis]  # (H, W, 1)

    # Blend toward white (255)
    foggy = img * (1.0 - density) + 255.0 * density * (0.5 + 0.5 * haze)
    return _from_float_array(foggy)


def _simple_box_filter(arr: np.ndarray, k: int) -> np.ndarray:
    """Simple box blur on 2D array using cumulative sums."""
    k = max(1, k)
    from numpy import cumsum
    cs = cumsum(cumsum(arr, axis=0), axis=1)
    H, W = arr.shape
    result = np.zeros_like(arr)
    r = k
    for y in range(H):
        y0 = max(0, y - r)
        y1 = min(H - 1, y + r)
        for x in range(W):
            x0 = max(0, x - r)
            x1 = min(W - 1, x + r)
            area = (y1 - y0 + 1) * (x1 - x0 + 1)
            total = cs[y1, x1]
            if y0 > 0:
                total -= cs[y0 - 1, x1]
            if x0 > 0:
                total -= cs[y1, x0 - 1]
            if y0 > 0 and x0 > 0:
                total += cs[y0 - 1, x0 - 1]
            result[y, x] = total / area
    return result


# ---- Rain -----------------------------------------------------------------

def _rain(pil: Image.Image, sev_idx: int) -> Image.Image:
    alpha = _SEV_TABLES["rain"][sev_idx]
    img   = _to_float_array(pil)
    H, W, _ = img.shape

    rng          = np.random.default_rng(seed=1)
    n_streaks    = int(H * W * 0.0005 * (sev_idx + 1))
    streak_len   = int(H * 0.05 * (sev_idx + 1))
    streak_width = max(1, sev_idx // 2 + 1)

    mask = np.zeros((H, W), dtype=np.float64)

    for _ in range(n_streaks):
        y0 = rng.integers(0, H)
        x0 = rng.integers(0, W)
        angle = rng.uniform(-15, 15)  # nearly vertical
        dy = int(streak_len * math.cos(math.radians(angle)))
        dx = int(streak_len * math.sin(math.radians(angle)))

        y1 = min(H - 1, max(0, y0 + dy))
        x1 = min(W - 1, max(0, x0 + dx))

        # Bresenham line
        _draw_line(mask, y0, x0, y1, x1, width=streak_width)

    # Dark streaks
    rain_layer = np.stack([mask * 200] * 3, axis=-1)  # dark-ish blue-white
    result     = img * (1.0 - mask[:, :, np.newaxis] * alpha) + rain_layer * alpha
    return _from_float_array(result)


def _draw_line(
    canvas: np.ndarray,
    y0: int, x0: int,
    y1: int, x1: int,
    width: int = 1,
) -> None:
    """Simple rasterised line on a 2D float canvas."""
    dy = abs(y1 - y0)
    dx = abs(x1 - x0)
    sy = 1 if y0 < y1 else -1
    sx = 1 if x0 < x1 else -1
    err = dx - dy
    H, W = canvas.shape

    for _ in range(max(dx, dy) + 1):
        for wy in range(-width // 2, width // 2 + 1):
            for wx in range(-width // 2, width // 2 + 1):
                ny, nx = y0 + wy, x0 + wx
                if 0 <= ny < H and 0 <= nx < W:
                    canvas[ny, nx] = 1.0
        if y0 == y1 and x0 == x1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0  += sx
        if e2 < dx:
            err += dx
            y0  += sy


# ---- Motion blur ----------------------------------------------------------

def _blur_motion(pil: Image.Image, sev_idx: int) -> Image.Image:
    k = _SEV_TABLES["blur_motion"][sev_idx]
    kernel_size = 2 * k + 1

    img = _to_float_array(pil)

    # Horizontal motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float64)
    kernel[k, :] = 1.0 / kernel_size

    if HAS_SCIPY:
        from scipy.ndimage import convolve
        channels = [convolve(img[:, :, c], kernel) for c in range(3)]
        result   = np.stack(channels, axis=-1)
    else:
        # Fallback: uniform filter along axis 1
        result = np.stack(
            [_uniform_filter_1d(img[:, :, c], size=kernel_size) for c in range(3)],
            axis=-1,
        )

    return _from_float_array(result)


def _uniform_filter_1d(arr: np.ndarray, size: int) -> np.ndarray:
    """Simple 1-D uniform filter along last axis (axis=1)."""
    pad = size // 2
    padded = np.pad(arr, ((0, 0), (pad, pad)), mode="edge")
    result = np.zeros_like(arr)
    for i in range(size):
        result += padded[:, i: i + arr.shape[1]]
    return result / size


# ---- Defocus blur ---------------------------------------------------------

def _blur_defocus(pil: Image.Image, sev_idx: int) -> Image.Image:
    sigma = _SEV_TABLES["blur_defocus"][sev_idx]
    img   = _to_float_array(pil)

    if HAS_SCIPY:
        channels = [gaussian_filter(img[:, :, c], sigma=sigma) for c in range(3)]
        result   = np.stack(channels, axis=-1)
    else:
        # Fallback: repeated box filter approximation
        result = img.copy()
        k = max(1, int(sigma * 2))
        for _ in range(3):
            result = np.stack(
                [_uniform_filter_1d(_uniform_filter_1d(result[:, :, c], k).T, k).T
                 for c in range(3)],
                axis=-1,
            )

    return _from_float_array(result)


# ---- Gaussian noise -------------------------------------------------------

def _noise_gaussian(pil: Image.Image, sev_idx: int) -> Image.Image:
    std  = _SEV_TABLES["noise_gaussian"][sev_idx]
    img  = _to_float_array(pil)
    rng  = np.random.default_rng(seed=2)
    noise = rng.normal(0, std, img.shape)
    return _from_float_array(img + noise)


# ---- JPEG compression -----------------------------------------------------

def _jpeg(pil: Image.Image, sev_idx: int) -> Image.Image:
    quality = _SEV_TABLES["jpeg"][sev_idx]
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


# ---- Brightness -----------------------------------------------------------

def _brightness(pil: Image.Image, sev_idx: int) -> Image.Image:
    delta = _SEV_TABLES["brightness"][sev_idx]
    img   = _to_float_array(pil)
    return _from_float_array(img + delta)


# ---- Contrast -------------------------------------------------------------

def _contrast(pil: Image.Image, sev_idx: int) -> Image.Image:
    factor = _SEV_TABLES["contrast"][sev_idx]
    img    = _to_float_array(pil)
    mean   = img.mean(axis=(0, 1), keepdims=True)
    result = mean + factor * (img - mean)
    return _from_float_array(result)


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_CORRUPTION_FNS = {
    "fog":            _fog,
    "rain":           _rain,
    "blur_motion":    _blur_motion,
    "blur_defocus":   _blur_defocus,
    "noise_gaussian": _noise_gaussian,
    "jpeg":           _jpeg,
    "brightness":     _brightness,
    "contrast":       _contrast,
}
