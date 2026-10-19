#!/usr/bin/env python3
"""
demo.py — RobustSeg visual demo.

Given an image path (or no path at all), the script visualises:
  1. Original / synthetic input image
  2. Entropy map   — pixel-wise prediction entropy of vanilla inference
  3. Disagreement map — std-dev of K perturbed predictions (RobustSeg)
  4. Final segmentation prediction (colour-coded)

If no image path is given a synthetic foggy-looking test image is generated
on the fly (Gaussian noise + blur on a gradient background).

All outputs are saved as a single PNG grid.

Usage::

    # With a real image
    python scripts/demo.py --image /path/to/image.png --output demo_out.png

    # Synthetic test (no dataset needed)
    python scripts/demo.py --output demo_out.png
"""

import argparse
import os
import sys
from typing import Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_ROOT  = os.path.dirname(_SCRIPT_DIR)
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from models.robustseg import RobustSegWrapper
from models.segformer_wrapper import build_segformer
from models.texture_identifier import TextureChannelIdentifier
from models.feature_perturber import FeaturePerturber


# ---------------------------------------------------------------------------
# Cityscapes colour palette (19 classes)
# ---------------------------------------------------------------------------

CS19_PALETTE = np.array([
    [128, 64,  128],  # road
    [244, 35,  232],  # sidewalk
    [70,  70,  70 ],  # building
    [102, 102, 156],  # wall
    [190, 153, 153],  # fence
    [153, 153, 153],  # pole
    [250, 170, 30 ],  # traffic light
    [220, 220, 0  ],  # traffic sign
    [107, 142, 35 ],  # vegetation
    [152, 251, 152],  # terrain
    [70,  130, 180],  # sky
    [220, 20,  60 ],  # person
    [255, 0,   0  ],  # rider
    [0,   0,   142],  # car
    [0,   0,   70 ],  # truck
    [0,   60,  100],  # bus
    [0,   80,  100],  # train
    [0,   0,   230],  # motorcycle
    [119, 11,  32 ],  # bicycle
], dtype=np.uint8)

CS19_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train",
    "motorcycle", "bicycle",
]


# ---------------------------------------------------------------------------
# Synthetic image generator
# ---------------------------------------------------------------------------

def _make_synthetic_image(H: int = 512, W: int = 1024) -> Image.Image:
    """
    Generate a synthetic foggy-looking test image.

    The image is a gradient (sky-like top, ground-like bottom) with
    additive Gaussian noise and a Gaussian blur to simulate haze.
    """
    rng = np.random.default_rng(42)

    # Sky-to-ground gradient
    sky_colour    = np.array([135, 206, 235], dtype=np.float32)
    ground_colour = np.array([80,  90,  60 ], dtype=np.float32)

    t        = np.linspace(0, 1, H)[:, np.newaxis, np.newaxis]  # (H,1,1)
    gradient = sky_colour * (1.0 - t) + ground_colour * t        # (H,1,3)
    img      = np.broadcast_to(gradient, (H, W, 3)).copy().astype(np.float32)

    # Add some "structure" with random Gaussian blobs
    for _ in range(20):
        cy, cx  = rng.integers(50, H - 50), rng.integers(50, W - 50)
        sigma   = rng.integers(20, 80)
        colour  = rng.uniform(50, 200, 3).astype(np.float32)
        yy, xx  = np.ogrid[:H, :W]
        blob    = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
        img    += blob[:, :, np.newaxis] * (colour - img) * 0.4

    # Gaussian noise
    noise = rng.normal(0, 15, img.shape).astype(np.float32)
    img  += noise

    # Fog: blend toward white
    fog_density = 0.35
    haze        = rng.random((H, W)).astype(np.float32)[:, :, np.newaxis]
    img = img * (1.0 - fog_density) + 255.0 * fog_density * (0.5 + 0.5 * haze)

    img = img.clip(0, 255).astype(np.uint8)

    pil_img = Image.fromarray(img, mode="RGB")

    # Blur to simulate haze
    from PIL import ImageFilter
    pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=3))

    return pil_img


# ---------------------------------------------------------------------------
# Image pre-processing
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def _preprocess(pil_img: Image.Image, H: int = 512, W: int = 1024) -> torch.Tensor:
    """Returns (1, 3, H, W) normalised tensor."""
    transform = T.Compose([
        T.Resize((H, W), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform(pil_img).unsqueeze(0)


def _denormalize(tensor: torch.Tensor) -> np.ndarray:
    """(1, 3, H, W) or (3, H, W) → (H, W, 3) uint8."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  dtype=tensor.dtype).view(3, 1, 1)
    img  = tensor * std + mean
    img  = (img.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return img


# ---------------------------------------------------------------------------
# Entropy and disagreement maps
# ---------------------------------------------------------------------------

def _entropy_map(logits: torch.Tensor) -> np.ndarray:
    """
    Compute per-pixel Shannon entropy from logits.

    Args:
        logits: (1, C, H, W)

    Returns:
        entropy: (H, W) float32, values in [0, log C].
    """
    probs   = F.softmax(logits, dim=1)  # (1, C, H, W)
    eps     = 1e-8
    entropy = -(probs * (probs + eps).log()).sum(dim=1)  # (1, H, W)
    return entropy.squeeze(0).cpu().numpy()


def _disagreement_map(logits_list: list) -> np.ndarray:
    """
    Compute per-pixel std-dev of predicted class probabilities across K samples.

    Args:
        logits_list: list of K tensors, each (1, C, H, W).

    Returns:
        disagreement: (H, W) float32.
    """
    probs  = torch.stack(
        [F.softmax(l, dim=1).squeeze(0) for l in logits_list], dim=0
    )  # (K, C, H, W)
    std_map = probs.std(dim=0).mean(dim=0)  # (H, W)
    return std_map.cpu().numpy()


# ---------------------------------------------------------------------------
# Prediction colouriser
# ---------------------------------------------------------------------------

def _colourise_prediction(pred: torch.Tensor) -> np.ndarray:
    """
    Args:
        pred: (1, H, W) or (H, W) long tensor of class indices.

    Returns:
        coloured: (H, W, 3) uint8.
    """
    if pred.dim() == 3:
        pred = pred.squeeze(0)
    pred_np = pred.cpu().numpy().astype(np.int64)
    H, W    = pred_np.shape

    colour_img = np.zeros((H, W, 3), dtype=np.uint8)
    for c in range(len(CS19_PALETTE)):
        colour_img[pred_np == c] = CS19_PALETTE[c]

    return colour_img


# ---------------------------------------------------------------------------
# Main demo function
# ---------------------------------------------------------------------------

def run_demo(
    image_path: Optional[str],
    output_path: str,
    checkpoint: Optional[str],
    num_classes: int,
    K: int,
    device: torch.device,
    crop_h: int = 512,
    crop_w: int = 1024,
) -> None:
    """
    Generate and save the 4-panel visualisation.
    """
    # --- Load / generate image ---
    if image_path and os.path.isfile(image_path):
        pil_img = Image.open(image_path).convert("RGB")
        print(f"[demo] Loaded image: {image_path}  size={pil_img.size}")
    else:
        print("[demo] No image provided — generating synthetic foggy test image.")
        pil_img = _make_synthetic_image(crop_h, crop_w)

    input_tensor = _preprocess(pil_img, H=crop_h, W=crop_w).to(device)  # (1, 3, H, W)

    # --- Build model ---
    backbone, head = build_segformer(num_classes=num_classes)
    model = RobustSegWrapper(
        backbone=backbone,
        head=head,
        texture_threshold=0.45,
        K=K,
    )

    if checkpoint and os.path.isfile(checkpoint):
        payload = torch.load(checkpoint, map_location="cpu")
        state   = payload.get("model_state", payload)
        model.load_state_dict(state, strict=True)
        print(f"[demo] Loaded checkpoint: {checkpoint}")
    else:
        print("[demo] No checkpoint — running with random weights (for visual layout demo).")

    model.to(device)
    model.eval()

    # --- Vanilla inference ---
    with torch.no_grad():
        primary_feat = model.extract_features(input_tensor)
        vanilla_logits = model(input_tensor)
        if vanilla_logits.shape[-2:] != input_tensor.shape[-2:]:
            vanilla_logits = F.interpolate(
                vanilla_logits, size=input_tensor.shape[-2:],
                mode="bilinear", align_corners=False,
            )

    entropy = _entropy_map(vanilla_logits)  # (H, W)

    # --- K perturbed predictions (for disagreement map) ---
    identifier = TextureChannelIdentifier()
    perturber  = FeaturePerturber()

    with torch.no_grad():
        channel_mask = identifier.identify(primary_feat, threshold=0.45)
        perturbed_logits_list = []
        for _ in range(K):
            raw_feats = model.backbone(input_tensor)
            primary   = raw_feats[0] if isinstance(raw_feats, (list, tuple)) else raw_feats
            pert_primary = perturber.perturb(primary, channel_mask)
            if isinstance(raw_feats, (list, tuple)):
                feats      = list(raw_feats)
                feats[0]   = pert_primary
            else:
                feats = pert_primary
            lgt = model.head(feats)
            if lgt.shape[-2:] != input_tensor.shape[-2:]:
                lgt = F.interpolate(lgt, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
            perturbed_logits_list.append(lgt)

    disagreement = _disagreement_map(perturbed_logits_list)  # (H, W)

    # Average perturbed logits for final prediction
    avg_logits   = torch.stack(perturbed_logits_list, dim=0).mean(dim=0)
    pred_classes = avg_logits.argmax(dim=1)  # (1, H, W)

    # --- Build visualisation ---
    img_np       = _denormalize(input_tensor.cpu())           # (H, W, 3) uint8
    pred_colour  = _colourise_prediction(pred_classes)        # (H, W, 3) uint8

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle("RobustSeg Demo", fontsize=16, fontweight="bold")

    # Panel 1 — Input image
    axes[0].imshow(img_np)
    axes[0].set_title("Input Image", fontsize=13)
    axes[0].axis("off")

    # Panel 2 — Entropy map (vanilla)
    im_ent = axes[1].imshow(entropy, cmap="hot", interpolation="bilinear")
    axes[1].set_title("Entropy Map\n(Vanilla Prediction)", fontsize=13)
    axes[1].axis("off")
    plt.colorbar(im_ent, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3 — Disagreement map (RobustSeg)
    im_dis = axes[2].imshow(disagreement, cmap="viridis", interpolation="bilinear")
    axes[2].set_title(f"Disagreement Map\n(K={K} perturbed preds)", fontsize=13)
    axes[2].axis("off")
    plt.colorbar(im_dis, ax=axes[2], fraction=0.046, pad=0.04)

    # Panel 4 — Segmentation prediction
    axes[3].imshow(pred_colour)
    axes[3].set_title("Segmentation\n(RobustSeg avg)", fontsize=13)
    axes[3].axis("off")

    # Colour legend for segmentation
    legend_patches = [
        matplotlib.patches.Patch(
            facecolor=tuple(c / 255.0 for c in CS19_PALETTE[i]),
            label=CS19_NAMES[i],
        )
        for i in range(len(CS19_NAMES))
    ]
    axes[3].legend(
        handles=legend_patches,
        loc="lower left",
        fontsize=5,
        ncol=2,
        framealpha=0.7,
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[demo] Saved output → {output_path}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RobustSeg visual demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image",       type=str, default=None,
                        help="Path to input image.  If omitted, a synthetic "
                             "image is generated.")
    parser.add_argument("--checkpoint",  type=str, default=None,
                        help="Path to model checkpoint (.pth).  If omitted, "
                             "random weights are used.")
    parser.add_argument("--output",      type=str, default="demo_output.png")
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--K",           type=int, default=4,
                        help="Number of perturbed forward passes.")
    parser.add_argument("--crop_h",      type=int, default=512)
    parser.add_argument("--crop_w",      type=int, default=1024)
    parser.add_argument("--device",      type=str, default="",
                        help="'cuda', 'cpu', or '' for auto.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.patches  # noqa: F401 — ensure patches available in scope
    args   = parse_args()
    device = (
        torch.device(args.device) if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    run_demo(
        image_path=args.image,
        output_path=args.output,
        checkpoint=args.checkpoint,
        num_classes=args.num_classes,
        K=args.K,
        device=device,
        crop_h=args.crop_h,
        crop_w=args.crop_w,
    )
