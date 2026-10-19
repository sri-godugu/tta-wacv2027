"""
Dataset classes for RobustSeg evaluation.

CityscapesDataset        — standard Cityscapes (train/val/test splits)
ACDCDataset              — ACDC adverse-condition dataset
CityscapesCDataset       — Cityscapes-C corruption benchmark
FoggyCityscapesDataset   — Foggy Cityscapes (beta-0.005/0.01/0.02)

All datasets:
  • return (image_tensor, label_tensor) where
      image_tensor  : (3, H, W)  float32, ImageNet normalised
      label_tensor  : (H, W)     int64,   255 = ignore
  • resize images to (crop_size[0], crop_size[1]) — default (512, 1024)
  • apply Cityscapes 34→19 label mapping

Directory layouts assumed:
  Cityscapes:
    <root>/leftImg8bit/<split>/<city>/*.png
    <root>/gtFine/<split>/<city>/*_labelIds.png

  ACDC:
    <root>/rgb_anon/<condition>/<split>/<scene>/<scene>_frame_<idx>_rgb_anon.png
    <root>/gt/<condition>/<split>/<scene>/<scene>_frame_<idx>_gt_labelIds.png
    (condition ∈ {fog, night, rain, snow}; split ∈ {train, val, test} —
    train/val/test also have sibling *_ref/ dirs of normal-condition images,
    not used here since we only evaluate ACDC, never train on it)

  Cityscapes-C:
    <root>/<corruption_type>/<severity>/<city>/*.png
    Labels reused from a companion Cityscapes root (set cs_root=... kwarg)

  Foggy Cityscapes:
    <root>/leftImg8bit_foggy/<split>/<city>/*_beta_<beta>.png
    Labels from a companion <cs_root>/gtFine directory
"""

import os
import glob
from typing import Callable, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as T


# ---------------------------------------------------------------------------
# Cityscapes 34-class → 19-class label mapping
# ---------------------------------------------------------------------------

# fmt: off
CITYSCAPES_ID_TO_TRAINID: dict = {
    0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255,
    6: 255, 7: 0,   8: 1,   9: 255, 10: 255, 11: 2,
    12: 3,  13: 4,  14: 255, 15: 255, 16: 255, 17: 5,
    18: 255, 19: 6, 20: 7,  21: 8,  22: 9,  23: 10,
    24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 255,
    30: 255, 31: 16, 32: 17, 33: 18,
}
# fmt: on

_LABEL_MAP: np.ndarray = np.full(256, 255, dtype=np.int64)
for _k, _v in CITYSCAPES_ID_TO_TRAINID.items():
    _LABEL_MAP[_k] = _v


def _map_labels(label_array: np.ndarray) -> np.ndarray:
    """Apply 34→19 Cityscapes label mapping."""
    return _LABEL_MAP[label_array.astype(np.int64)]


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def _build_image_transform(
    crop_size: Tuple[int, int] = (512, 1024),
) -> Callable:
    return T.Compose([
        T.Resize(crop_size, interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _resize_label(
    label_pil: Image.Image,
    size: Tuple[int, int],
) -> np.ndarray:
    """Resize a label image with nearest-neighbour interpolation."""
    label_pil = label_pil.resize((size[1], size[0]), Image.NEAREST)
    return np.array(label_pil, dtype=np.int64)


# ---------------------------------------------------------------------------
# Base dataset helper
# ---------------------------------------------------------------------------

class _BaseSegDataset(Dataset):
    """Shared logic: image + label lists, transforms, __getitem__."""

    def __init__(self, crop_size: Tuple[int, int] = (512, 1024)) -> None:
        self.crop_size       = crop_size
        self.image_paths: List[str] = []
        self.label_paths: List[str] = []
        self._img_transform = _build_image_transform(crop_size)
        # Cityscapes labelIds (34-class) datasets need the 34->19 trainId
        # remap; datasets whose masks are already in trainId form (e.g.
        # BDD100K sem_seg) set this False.
        self._apply_cs_mapping = True

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path   = self.image_paths[idx]
        label_path = self.label_paths[idx]

        image = Image.open(img_path).convert("RGB")
        label = Image.open(label_path)

        image_tensor = self._img_transform(image)

        label_arr = _resize_label(label, self.crop_size)
        if self._apply_cs_mapping:
            label_arr = _map_labels(label_arr)
        label_tensor = torch.from_numpy(label_arr).long()

        return image_tensor, label_tensor

    def _check_lengths(self) -> None:
        assert len(self.image_paths) == len(self.label_paths), (
            f"image/label count mismatch: {len(self.image_paths)} vs "
            f"{len(self.label_paths)}"
        )
        assert len(self.image_paths) > 0, "Dataset is empty — check data_root."


# ---------------------------------------------------------------------------
# CityscapesDataset
# ---------------------------------------------------------------------------

class CityscapesDataset(_BaseSegDataset):
    """
    Cityscapes dataset.

    Directory layout::

        <root>/leftImg8bit/<split>/<city>/*_leftImg8bit.png
        <root>/gtFine/<split>/<city>/*_gtFine_labelIds.png

    Args:
        root:      Cityscapes root directory.
        split:     'train', 'val', or 'test'.
        crop_size: (H, W) resize target.
    """

    def __init__(
        self,
        root: str,
        split: str = "val",
        crop_size: Tuple[int, int] = (512, 1024),
    ) -> None:
        super().__init__(crop_size)

        img_dir   = os.path.join(root, "leftImg8bit",  split)
        label_dir = os.path.join(root, "gtFine",       split)

        img_paths = sorted(glob.glob(
            os.path.join(img_dir, "**", "*_leftImg8bit.png"), recursive=True
        ))

        for ip in img_paths:
            # Derive label path from image path
            city   = os.path.basename(os.path.dirname(ip))
            fname  = os.path.basename(ip).replace("_leftImg8bit.png", "_gtFine_labelIds.png")
            lp     = os.path.join(label_dir, city, fname)
            self.image_paths.append(ip)
            self.label_paths.append(lp)

        self._check_lengths()


# ---------------------------------------------------------------------------
# ACDCDataset
# ---------------------------------------------------------------------------

class ACDCDataset(_BaseSegDataset):
    """
    ACDC adverse-condition dataset.

    Directory layout::

        <root>/rgb_anon/<condition>/<split>/<scene>/<scene>_frame_<idx>_rgb_anon.png
        <root>/gt/<condition>/<split>/<scene>/<scene>_frame_<idx>_gt_labelIds.png

    Args:
        root:      ACDC root directory.
        condition: 'fog', 'night', 'rain', 'snow', or 'all'.
        split:     'train', 'val', or 'test'.
        crop_size: (H, W) resize target.
    """

    CONDITIONS = ["fog", "night", "rain", "snow"]

    def __init__(
        self,
        root: str,
        condition: str = "all",
        split: str = "val",
        crop_size: Tuple[int, int] = (512, 1024),
    ) -> None:
        super().__init__(crop_size)

        conditions = self.CONDITIONS if condition == "all" else [condition]

        for cond in conditions:
            img_base   = os.path.join(root, "rgb_anon", cond, split)
            label_base = os.path.join(root, "gt",       cond, split)

            if not os.path.isdir(img_base):
                continue

            img_paths = sorted(glob.glob(
                os.path.join(img_base, "**", "*.png"), recursive=True
            ))

            for ip in img_paths:
                # Construct label path by mirroring directory structure
                rel     = os.path.relpath(ip, img_base)
                # ACDC gt filenames end with _gt_labelIds.png
                gt_name = os.path.basename(rel).replace(
                    "_rgb_anon.png", "_gt_labelIds.png"
                )
                scene   = os.path.dirname(rel)
                lp = os.path.join(label_base, scene, gt_name)

                if os.path.exists(lp):
                    self.image_paths.append(ip)
                    self.label_paths.append(lp)

        if len(self.image_paths) == 0:
            import warnings
            warnings.warn(
                f"ACDCDataset: no samples found under {root}. "
                "Check directory structure."
            )


# ---------------------------------------------------------------------------
# CityscapesCDataset
# ---------------------------------------------------------------------------

class CityscapesCDataset(_BaseSegDataset):
    """
    Cityscapes-C corruption benchmark.

    Directory layout::

        <root>/<corruption_type>/<severity>/<city>/*.png
        Labels sourced from a companion Cityscapes root:
        <cs_root>/gtFine/val/<city>/*_gtFine_labelIds.png

    Args:
        root:            Cityscapes-C root.
        cs_root:         original Cityscapes root (for labels).
        corruption_type: e.g. 'fog', 'noise_gaussian', or 'all'.
        severity:        1-5, or 0 for all severities.
        crop_size:       (H, W) resize target.
    """

    CORRUPTION_TYPES = [
        "brightness", "contrast", "defocus_blur", "elastic_transform",
        "fog", "frost", "gaussian_blur", "gaussian_noise", "glass_blur",
        "impulse_noise", "jpeg_compression", "motion_blur", "pixelate",
        "saturate", "shot_noise", "snow", "spatter", "speckle_noise", "zoom_blur",
    ]

    def __init__(
        self,
        root: str,
        cs_root: str,
        corruption_type: str = "fog",
        severity: int = 1,
        crop_size: Tuple[int, int] = (512, 1024),
    ) -> None:
        super().__init__(crop_size)

        corruptions = (
            self.CORRUPTION_TYPES if corruption_type == "all" else [corruption_type]
        )
        severities  = list(range(1, 6)) if severity == 0 else [severity]

        label_dir = os.path.join(cs_root, "gtFine", "val")

        for corr in corruptions:
            for sev in severities:
                img_dir = os.path.join(root, corr, str(sev))
                if not os.path.isdir(img_dir):
                    continue
                img_paths = sorted(glob.glob(
                    os.path.join(img_dir, "**", "*.png"), recursive=True
                ))
                for ip in img_paths:
                    city = os.path.basename(os.path.dirname(ip))
                    # Map corrupted filename back to original label name
                    base  = os.path.basename(ip)
                    # Cityscapes-C images keep city_seq_frame prefix
                    parts = base.replace(".png", "").split("_")
                    # Typical: city_000000_000019_leftImg8bit.png
                    label_base = "_".join(parts[:3]) + "_gtFine_labelIds.png"
                    lp = os.path.join(label_dir, city, label_base)
                    if os.path.exists(lp):
                        self.image_paths.append(ip)
                        self.label_paths.append(lp)

        if len(self.image_paths) == 0:
            import warnings
            warnings.warn(
                f"CityscapesCDataset: no samples found. "
                f"root={root}, cs_root={cs_root}, "
                f"corruption={corruption_type}, severity={severity}"
            )


# ---------------------------------------------------------------------------
# FoggyCityscapesDataset
# ---------------------------------------------------------------------------

class FoggyCityscapesDataset(_BaseSegDataset):
    """
    Foggy Cityscapes dataset.

    Directory layout::

        <root>/leftImg8bit_foggy/<split>/<city>/*_leftImg8bit_foggy_beta_<beta>.png
        <cs_root>/gtFine/<split>/<city>/*_gtFine_labelIds.png

    Args:
        root:     Foggy Cityscapes root (contains leftImg8bit_foggy/).
        cs_root:  Original Cityscapes root (for labels).  If same as *root*,
                  pass the same path.
        split:    'train' or 'val'.
        beta:     fog density — '0.005', '0.01', or '0.02'.
        crop_size: (H, W) resize target.
    """

    def __init__(
        self,
        root: str,
        cs_root: str,
        split: str = "val",
        beta: str = "0.02",
        crop_size: Tuple[int, int] = (512, 1024),
    ) -> None:
        super().__init__(crop_size)

        img_dir   = os.path.join(root,    "leftImg8bit_foggy", split)
        label_dir = os.path.join(cs_root, "gtFine",            split)

        img_paths = sorted(glob.glob(
            os.path.join(img_dir, "**", f"*_beta_{beta}.png"), recursive=True
        ))

        for ip in img_paths:
            city = os.path.basename(os.path.dirname(ip))
            base = os.path.basename(ip)
            # Strip foggy suffix to recover original stem
            label_base = base.replace(
                f"_leftImg8bit_foggy_beta_{beta}.png",
                "_gtFine_labelIds.png",
            )
            lp = os.path.join(label_dir, city, label_base)
            if os.path.exists(lp):
                self.image_paths.append(ip)
                self.label_paths.append(lp)

        if len(self.image_paths) == 0:
            import warnings
            warnings.warn(
                f"FoggyCityscapesDataset: no samples found under {img_dir} "
                f"with beta={beta}."
            )


# ---------------------------------------------------------------------------
# FoggyZurichDataset  (real fog, ETH; Cityscapes labelIds)
# ---------------------------------------------------------------------------

class FoggyZurichDataset(_BaseSegDataset):
    """
    Foggy Zurich (real fog) --- Sakaridis et al., ETH. The finely-annotated
    test split (Foggy_Zurich-test, ~40 images) has pixel labels in the
    Cityscapes labelIds scheme, so the standard 34->19 remap applies.

    Directory layout is tolerant: we glob RGB images and match each to its
    label by filename stem. Typical release layout::

        <root>/RGB/**/<name>.png
        <root>/gt_labelIds/**/<name>_gt_labelIds.png   (or <name>.png)

    Args:
        root:      Foggy Zurich root.
        crop_size: (H, W) resize target.
        img_glob:  glob (relative to root) for RGB images.
        label_dir: subdir (relative to root) containing labelIds masks.
    """

    def __init__(
        self,
        root: str,
        crop_size: Tuple[int, int] = (512, 1024),
        img_glob: str = "RGB/**/*.png",
        label_dir: str = "gt_labelTrainIds",
    ) -> None:
        super().__init__(crop_size)
        # Foggy Zurich ships gt_labelTrainIds (already 0..18/255 trainId);
        # the annotated test set is the ~45 frames that have a mask here.
        self._apply_cs_mapping = False

        # Build a stem -> label-path index over all *labelIds*.png under label_dir.
        label_root = os.path.join(root, label_dir)
        label_index = {}
        for lp in glob.glob(os.path.join(label_root, "**", "*.png"), recursive=True):
            stem = os.path.basename(lp)
            for suf in ("_gt_labelIds.png", "_gtFine_labelIds.png", ".png"):
                if stem.endswith(suf):
                    label_index[stem[: -len(suf)]] = lp
                    break

        for ip in sorted(glob.glob(os.path.join(root, img_glob), recursive=True)):
            stem = os.path.basename(ip)
            for suf in ("_rgb_anon.png", "_leftImg8bit.png", ".png"):
                if stem.endswith(suf):
                    key = stem[: -len(suf)]
                    break
            else:
                key = os.path.splitext(stem)[0]
            lp = label_index.get(key)
            if lp is not None:
                self.image_paths.append(ip)
                self.label_paths.append(lp)

        if len(self.image_paths) == 0:
            import warnings
            warnings.warn(
                f"FoggyZurichDataset: no image/label pairs matched under {root} "
                f"(img_glob={img_glob!r}, label_dir={label_dir!r}). "
                "Adjust globs once the release layout is confirmed."
            )


# ---------------------------------------------------------------------------
# BDD100KDataset  (independent real driving; sem_seg masks already trainId)
# ---------------------------------------------------------------------------

class BDD100KDataset(_BaseSegDataset):
    """
    BDD100K semantic-segmentation subset. Masks are already in the 19-class
    Cityscapes *trainId* space (0..18, 255 ignore), so NO 34->19 remap is
    applied. Optionally filters the val split by a weather/time-of-day
    attribute so we can form per-condition real points (e.g. rainy, snowy,
    foggy, night).

    Typical layout::

        <root>/images/10k/val/<name>.jpg
        <root>/labels/sem_seg/masks/val/<name>.png     (trainId masks)

    Args:
        root:        BDD100K root.
        split:       'val' (default; the labeled eval split we use).
        crop_size:   (H, W) resize target.
        condition:   None/'all' for the whole split, or one of
                     {'rainy','snowy','foggy','clear','overcast',...} (weather)
                     or {'night','daytime','dawn/dusk'} (timeofday).
        attr_json:   path to a BDD100K attributes JSON mapping image name ->
                     {'weather':..., 'timeofday':..., 'scene':...}. Required
                     only when `condition` filtering is requested.
    """

    def __init__(
        self,
        root: str,
        split: str = "val",
        crop_size: Tuple[int, int] = (512, 1024),
        condition: Optional[str] = None,
        attr_json: Optional[str] = None,
    ) -> None:
        super().__init__(crop_size)
        self._apply_cs_mapping = False  # masks already trainId

        img_dir  = os.path.join(root, "images", "10k", split)
        mask_dir = os.path.join(root, "labels", split)   # trainId masks

        keep = None  # set of allowed image basenames (no ext) if filtering
        if condition and condition != "all":
            if not attr_json or not os.path.exists(attr_json):
                import warnings
                warnings.warn(
                    f"BDD100KDataset: condition={condition!r} requested but "
                    f"attr_json missing ({attr_json!r}); using whole split."
                )
            else:
                import json as _json
                attrs = _json.load(open(attr_json))
                # aggregated list of {'name':..., 'attributes':{...}}
                keep = set()
                for rec in attrs:
                    name = os.path.splitext(rec.get("name", ""))[0]
                    a = rec.get("attributes", {}) or {}
                    if condition in (a.get("weather"), a.get("timeofday"), a.get("scene")):
                        keep.add(name)

        # masks are named "<stem>_train_id.png"; image is "<stem>.jpg"
        for mp in sorted(glob.glob(os.path.join(mask_dir, "*.png"))):
            base = os.path.basename(mp)
            for suf in ("_train_id.png", "_train_labelIds.png", ".png"):
                if base.endswith(suf):
                    stem = base[: -len(suf)]
                    break
            if keep is not None and stem not in keep:
                continue
            ip = os.path.join(img_dir, stem + ".jpg")
            if not os.path.exists(ip):
                ip_png = os.path.join(img_dir, stem + ".png")
                ip = ip_png if os.path.exists(ip_png) else None
            if ip:
                self.image_paths.append(ip)
                self.label_paths.append(mp)

        if len(self.image_paths) == 0:
            import warnings
            warnings.warn(
                f"BDD100KDataset: no pairs found under {root} "
                f"(split={split!r}, condition={condition!r}). "
                "Confirm images/10k and labels/sem_seg/masks layout."
            )
