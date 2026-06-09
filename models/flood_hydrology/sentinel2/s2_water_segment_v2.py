"""
SOTA Water Body Segmentation — Prithvi-EO-2.0 (IBM/NASA Geospatial Foundation Model)
======================================================================================
Replaces the CNN-based WatNet with Prithvi-EO-2.0, a geospatial foundation model
fine-tuned on SEN1Floods11 (446 global flood scenes, Sentinel-1 + Sentinel-2).

Model
-----
  Prithvi-EO-2.0-300M-TL-Sen1Floods11
  — Architecture: Masked Autoencoder ViT encoder (300M params) + UperNet decoder
  — Pre-training: self-supervised MAE on 1M Sentinel-2 time series globally
  — Fine-tuning: SEN1Floods11 binary flood/water segmentation
  — Input: Sentinel-2 6-band [BLUE, GREEN, RED, NIR, SWIR1, SWIR2], scaled ×0.0001
  — Output: Binary water mask (class 0 = land, class 1 = water)
  — Benchmark: F1 ~0.93 on SEN1Floods11 (vs WatNet CNN F1 ~0.87)
  — HuggingFace: ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11
  — Requires: terratorch ≥ 0.99.8 (Python ≥ 3.10 only)
              pip install terratorch torch torchvision timm einops albumentations

Fallback (Python 3.9 / no terratorch)
--------------------------------------
  Physics-based multi-index ensemble: NDWI + MNDWI + AWEInsh + AWEIsh
  Per-scene Otsu thresholding + majority vote (≥ 2 of 4 indices agree = water)
  NOTE: this is a spectral-index method, NOT deep learning.

Outputs (sentinel2/data/)
    segformer_mask_{date}.tif    — binary water mask (uint8 0/1)
    segformer_prob_{date}.tif    — water probability (float32 0–1)
    segformer_metrics.csv        — F1, IoU, precision, recall vs lake_mask_s2 reference
    segformer_comparison.png     — RGB | MNDWI | Prithvi/ensemble | difference

Usage
    # With terratorch (Python 3.10+):
    python3 sentinel2/s2_water_segment_v2.py --model prithvi

    # Fallback physics ensemble (Python 3.9):
    python3 sentinel2/s2_water_segment_v2.py --model ensemble

    # Auto-detect best available model:
    python3 sentinel2/s2_water_segment_v2.py
"""

import os
import sys
import glob
import argparse
import warnings
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DEM_DIR  = os.path.join(BASE_DIR, "..", "dem", "data")

PRITHVI_REPO   = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11"
PRITHVI_CKPT   = "Prithvi-EO-V2-300M-TL-Sen1Floods11.pt"
PRITHVI_CONFIG = "config.yaml"

# SEN1Floods11 per-band mean/std for Prithvi normalization (BLUE,GREEN,RED,NIR,SWIR1,SWIR2)
# Source: terratorch Sen1Floods11NonGeoDataModule defaults (constant_scale=0.0001)
PRITHVI_MEAN = [0.0582, 0.0874, 0.1070, 0.2988, 0.2069, 0.1434]
PRITHVI_STD  = [0.0298, 0.0344, 0.0424, 0.0773, 0.0756, 0.0686]

BAND_ORDER = ["B02", "B03", "B04", "B08", "B11", "B12"]  # = BLUE,GREEN,RED,NIR,SWIR1,SWIR2

DEFAULT_THRESHOLD = 0.50
PATCH_SIZE = 512
OVERLAP    = 64


# ---------------------------------------------------------------------------
# Band loading
# ---------------------------------------------------------------------------

def load_scene_6band(date, data_dir=DATA_DIR):
    """Return (6,H,W) float32 [0,1] array, transform, crs, (H,W) or Nones."""
    import rasterio
    from rasterio.warp import reproject, Resampling

    bands = []
    ref_transform = ref_crs = ref_shape = None

    for band_name in BAND_ORDER:
        fpath = os.path.join(data_dir, f"s2_{date}_{band_name}.tif")
        if not os.path.exists(fpath):
            print(f"    Missing band {band_name} for {date}")
            return None, None, None, None

        with rasterio.open(fpath) as src:
            arr = src.read(1).astype(np.float32)
            if ref_transform is None:
                ref_transform = src.transform
                ref_crs = src.crs
                ref_shape = (src.height, src.width)
            elif arr.shape != ref_shape:
                out = np.zeros(ref_shape, dtype=np.float32)
                reproject(source=arr, destination=out,
                          src_transform=src.transform, src_crs=src.crs,
                          dst_transform=ref_transform, dst_crs=ref_crs,
                          resampling=Resampling.bilinear)
                arr = out
        bands.append(arr)

    stack = np.stack(bands, axis=0) / 10000.0  # (6, H, W) in [0,1]
    return np.clip(stack, 0.0, 1.0), ref_transform, ref_crs, ref_shape


# ---------------------------------------------------------------------------
# Prithvi-EO-2.0 inference  (requires terratorch + Python ≥ 3.10)
# ---------------------------------------------------------------------------

def load_prithvi_model():
    """
    Download Prithvi checkpoint from HuggingFace and load via terratorch.
    Returns (lightning_model, img_size) or raises ImportError if unavailable.
    """
    try:
        from terratorch.cli_tools import LightningInferenceModel
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "terratorch is required for Prithvi-EO-2.0.  "
            "Install it on Python ≥ 3.10:  pip install terratorch"
        )

    print(f"  Downloading Prithvi checkpoint from {PRITHVI_REPO} …")
    ckpt_path   = hf_hub_download(PRITHVI_REPO, PRITHVI_CKPT)
    config_path = hf_hub_download(PRITHVI_REPO, PRITHVI_CONFIG)

    print("  Loading Prithvi-EO-2.0 via terratorch …")
    lightning_model = LightningInferenceModel.from_config(config_path, ckpt_path)
    lightning_model.model.eval()
    print("  ✓ Prithvi-EO-2.0 loaded")
    return lightning_model, 512  # SEN1Floods11 training tile size


def prithvi_predict(lightning_model, scene_6band, img_size=512):
    """
    Run Prithvi-EO-2.0 inference on a full Sentinel-2 scene.
    scene_6band: (6, H, W) float32 [0,1] (already /10000)
    Returns binary water mask (H, W) uint8.
    """
    import torch
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from einops import rearrange

    device = next(lightning_model.model.parameters()).device

    # Normalize per Prithvi training statistics
    img = scene_6band.copy()  # (6, H, W)
    for c in range(6):
        img[c] = (img[c] - PRITHVI_MEAN[c]) / PRITHVI_STD[c]

    _, H, W = img.shape

    # Pad to multiple of img_size
    pad_h = (img_size - H % img_size) % img_size
    pad_w = (img_size - W % img_size) % img_size
    img_pad = np.pad(img, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")

    # Prithvi input shape: (batch, bands, time, H, W) — single time step
    inp = torch.tensor(img_pad, dtype=torch.float32).unsqueeze(0).unsqueeze(2)  # (1,6,1,H,W)

    # Sliding window tiling
    windows = inp.unfold(3, img_size, img_size).unfold(4, img_size, img_size)
    h1, w1  = windows.shape[3:5]
    windows = rearrange(windows, "b c t h1 w1 h w -> (b h1 w1) c t h w",
                        h=img_size, w=img_size)

    transform = A.Compose([ToTensorV2(transpose_mask=False)])

    pred_tiles = []
    for i in range(windows.shape[0]):
        tile = windows[i]  # (6, 1, img_size, img_size)
        tile_2d = tile[:, 0, :, :]  # (6, img_size, img_size) — drop time dim for albumentations
        x = transform(image=tile_2d.numpy().transpose(1, 2, 0))["image"]  # (img_size,img_size,6)
        x = x.unsqueeze(0).unsqueeze(2).to(device)  # (1,6,1,img_size,img_size) — re-add time

        with torch.no_grad():
            out = lightning_model.model(x, temporal_coords=None, location_coords=None)
            pred = out.output.detach().cpu()  # (1, 2, img_size, img_size)

        pred_class = pred.argmax(dim=1)  # (1, img_size, img_size)
        pred_tiles.append(pred_class)

    pred_tiles = torch.cat(pred_tiles, dim=0)  # (h1*w1, img_size, img_size)

    # Reassemble
    pred_img = rearrange(pred_tiles, "(h1 w1) h w -> (h1 h) (w1 w)",
                         h1=h1, w1=w1, h=img_size, w=img_size)
    pred_img = pred_img[:H, :W]  # strip padding

    return pred_img.numpy().astype(np.uint8)


# ---------------------------------------------------------------------------
# Physics-based ensemble fallback  (Python 3.9 compatible, no deep learning)
# ---------------------------------------------------------------------------

def enhanced_spectral_predict(scene_6band):
    """
    Physics-based multi-index ensemble: NDWI + MNDWI + AWEInsh + AWEIsh.
    Each index gets a per-scene Otsu threshold; majority vote (≥2/4) = water.
    Returns vote fraction [0,1]; threshold at 0.5 → majority agrees.

    NOTE: this is NOT deep learning — it is a spectral-index method.
    Use Prithvi-EO-2.0 (--model prithvi) for true SOTA deep learning results.
    """
    b02 = scene_6band[0]  # Blue
    b03 = scene_6band[1]  # Green
    b08 = scene_6band[3]  # NIR
    b11 = scene_6band[4]  # SWIR1
    b12 = scene_6band[5]  # SWIR2
    eps = 1e-9

    with np.errstate(divide="ignore", invalid="ignore"):
        ndwi  = np.where((b03 + b08) > 0, (b03 - b08) / (b03 + b08 + eps), 0.0)
        mndwi = np.where((b03 + b11) > 0, (b03 - b11) / (b03 + b11 + eps), 0.0)
    awei_nsh = 4.0 * (b03 - b11) - (0.25 * b08 + 2.75 * b11)
    awei_sh  = b02 + 2.5 * b03 - 1.5 * (b08 + b11) - 0.25 * b12

    votes = np.zeros_like(b03, dtype=np.float32)
    for arr in [ndwi, mndwi, awei_nsh, awei_sh]:
        flat = arr.ravel()
        valid = flat[np.isfinite(flat)]
        if len(valid) < 10:
            continue
        try:
            from skimage.filters import threshold_otsu
            t = threshold_otsu(valid)
        except Exception:
            t = 0.0
        votes += (arr > t).astype(np.float32)

    return np.clip(votes / 4.0, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(pred_mask, ref_mask, method="unknown"):
    pred = pred_mask.astype(bool)
    ref  = ref_mask.astype(bool)
    tp = int((pred & ref).sum())
    fp = int((pred & ~ref).sum())
    fn = int((~pred & ref).sum())
    tn = int((~pred & ~ref).sum())
    precision = tp / (tp + fp)  if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn)  if (tp + fn) > 0 else 0.0
    f1        = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0.0
    iou       = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    accuracy  = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
    return dict(method=method, tp=tp, fp=fp, fn=fn, tn=tn,
                precision=round(precision,4), recall=round(recall,4),
                f1=round(f1,4), iou=round(iou,4), accuracy=round(accuracy,4))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(dates=None, threshold=DEFAULT_THRESHOLD, model_choice="auto"):
    import rasterio

    # Resolve scene list
    if dates:
        date_list = dates
    else:
        b03_files = sorted(glob.glob(os.path.join(DATA_DIR, "s2_*_B03.tif")))
        date_list = [os.path.basename(f).replace("s2_","").replace("_B03.tif","")
                     for f in b03_files]
    if not date_list:
        sys.exit("No Sentinel-2 scenes found. Run s2_download.py first.")

    # Load model
    use_prithvi = False
    prithvi_model = None

    if model_choice in ("prithvi", "auto"):
        try:
            prithvi_model, _ = load_prithvi_model()
            use_prithvi = True
            method_label = "Prithvi-EO-2.0 (IBM/NASA, SEN1Floods11)"
        except ImportError as e:
            if model_choice == "prithvi":
                sys.exit(f"ERROR: {e}")
            warnings.warn(str(e) + " — falling back to physics-based ensemble.", RuntimeWarning)

    if not use_prithvi:
        method_label = "Physics-based multi-index ensemble (NDWI+MNDWI+AWEInsh+AWEIsh) [NOT deep learning]"
        print(f"\n  ⚠  Prithvi unavailable. Using spectral fallback.")
        print(f"     To run Prithvi: install Python 3.10+ then pip install terratorch")
        print(f"     Then rerun: python3 s2_water_segment_v2.py --model prithvi\n")

    print(f"SOTA Water Segmentation")
    print(f"  Method : {method_label}")
    print(f"  Scenes : {len(date_list)}")
    print(f"  Threshold: {threshold}")

    # Reference mask for metrics
    ref_mask_path = (os.path.join(DEM_DIR, "lake_mask_s2.tif")
                     if os.path.exists(os.path.join(DEM_DIR, "lake_mask_s2.tif"))
                     else os.path.join(DEM_DIR, "lake_mask.tif"))
    ref_mask_raw = None
    if os.path.exists(ref_mask_path):
        with rasterio.open(ref_mask_path) as src:
            ref_mask_raw = src.read(1).astype(np.uint8)
        print(f"  Reference mask: {os.path.basename(ref_mask_path)}")

    metrics_rows = []

    for date in date_list:
        print(f"\n  Processing {date} …")
        scene_6band, transform, crs, shape = load_scene_6band(date)
        if scene_6band is None:
            continue
        H, W = shape

        # Run inference
        if use_prithvi:
            binary = prithvi_predict(prithvi_model, scene_6band)
            prob   = binary.astype(np.float32)  # Prithvi gives hard labels directly
        else:
            prob   = enhanced_spectral_predict(scene_6band)
            binary = (prob > threshold).astype(np.uint8)

        # Apply cloud mask if available
        cm_path = os.path.join(DATA_DIR, f"cloud_mask_{date}.tif")
        if os.path.exists(cm_path):
            with rasterio.open(cm_path) as src:
                cm = src.read(1).astype(np.uint8)
            if cm.shape != binary.shape:
                from scipy.ndimage import zoom
                scale = (binary.shape[0]/cm.shape[0], binary.shape[1]/cm.shape[1])
                cm = zoom(cm.astype(np.float32), scale, order=0).astype(np.uint8)
            binary = np.where(cm > 0, 0, binary)
            prob   = np.where(cm > 0, 0.0, prob)

        # Save probability map
        profile = dict(driver="GTiff", dtype="float32", count=1,
                       height=H, width=W, crs=crs, transform=transform,
                       compress="lzw", nodata=-1.0)
        with rasterio.open(os.path.join(DATA_DIR, f"segformer_prob_{date}.tif"), "w", **profile) as dst:
            dst.write(prob, 1)

        # Save binary mask
        pu8 = {**profile, "dtype": "uint8", "nodata": 255}
        mask_path = os.path.join(DATA_DIR, f"segformer_mask_{date}.tif")
        with rasterio.open(mask_path, "w", **pu8) as dst:
            dst.write(binary, 1)

        area_ha = float(binary.sum() * abs(transform.a)**2 / 1e4)
        print(f"    Water area: {area_ha:.2f} ha | Mask saved → {os.path.basename(mask_path)}")

        # Metrics vs reference
        if ref_mask_raw is not None:
            from scipy.ndimage import zoom as ndzoom
            ref = ref_mask_raw
            if ref.shape != binary.shape:
                sr = binary.shape[0]/ref.shape[0]; sc = binary.shape[1]/ref.shape[1]
                ref = ndzoom(ref.astype(np.float32), (sr, sc), order=0).astype(np.uint8)
            m = compute_metrics(binary, ref, method_label)
            m["date"] = date
            metrics_rows.append(m)
            print(f"    F1={m['f1']:.3f} | IoU={m['iou']:.3f} | "
                  f"Precision={m['precision']:.3f} | Recall={m['recall']:.3f}")

    # Save metrics
    if metrics_rows:
        mdf = pd.DataFrame(metrics_rows)
        metrics_path = os.path.join(DATA_DIR, "segformer_metrics.csv")
        mdf.to_csv(metrics_path, index=False)
        print(f"\nMetrics → {metrics_path}")
        print(f"Mean F1 : {mdf['f1'].mean():.3f}")
        print(f"Mean IoU: {mdf['iou'].mean():.3f}")

        watnet_path = os.path.join(DATA_DIR, "watnet_metrics.csv")
        if os.path.exists(watnet_path):
            wdf = pd.read_csv(watnet_path)
            wf1 = wdf["watnet_f1"].mean() if "watnet_f1" in wdf.columns else float("nan")
            mf1 = wdf["mndwi_f1"].mean()  if "mndwi_f1"  in wdf.columns else float("nan")
            print(f"\nBenchmark on Johns Lake / Winter Garden FL:")
            print(f"  WatNet CNN (2021):           mean F1 = {wf1:.3f}")
            print(f"  MNDWI spectral index:         mean F1 = {mf1:.3f}")
            print(f"  {method_label[:40]}: mean F1 = {mdf['f1'].mean():.3f}")

    _make_comparison_viz(date_list[:4], method_label)


def _make_comparison_viz(dates, method_label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio

    n = min(len(dates), 4)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 4, figsize=(16, 4*n))
    if n == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(f"Water Segmentation: RGB | MNDWI | {method_label[:60]} | Difference",
                 fontsize=9, fontweight="bold")

    for i, date in enumerate(dates[:n]):
        b04 = os.path.join(DATA_DIR, f"s2_{date}_B04.tif")
        b03 = os.path.join(DATA_DIR, f"s2_{date}_B03.tif")
        b02 = os.path.join(DATA_DIR, f"s2_{date}_B02.tif")
        ax = axes[i, 0]
        if all(os.path.exists(p) for p in [b04, b03, b02]):
            with rasterio.open(b04) as s: r = s.read(1).astype(np.float32)
            with rasterio.open(b03) as s: g = s.read(1).astype(np.float32)
            with rasterio.open(b02) as s: b = s.read(1).astype(np.float32)
            rgb = np.stack([r,g,b], axis=-1) / 10000.0
            p2, p98 = np.percentile(rgb, 2), np.percentile(rgb, 98)
            rgb = np.clip((rgb-p2)/(p98-p2+1e-9), 0, 1)
            ax.imshow(rgb)
        ax.set_title(f"{date}\nS2 True-Color", fontsize=8); ax.axis("off")

        ax = axes[i, 1]
        mndwi_p = os.path.join(DATA_DIR, f"water_mask_{date}.tif")
        if os.path.exists(mndwi_p):
            with rasterio.open(mndwi_p) as s: ax.imshow(s.read(1), cmap="Blues", vmin=0, vmax=1)
        ax.set_title("MNDWI\n(spectral baseline)", fontsize=8); ax.axis("off")

        ax = axes[i, 2]
        sf_p = os.path.join(DATA_DIR, f"segformer_mask_{date}.tif")
        if os.path.exists(sf_p):
            with rasterio.open(sf_p) as s: sf = s.read(1)
            ax.imshow(sf, cmap="Blues", vmin=0, vmax=1)
        ax.set_title("This method", fontsize=8); ax.axis("off")

        ax = axes[i, 3]
        if os.path.exists(mndwi_p) and os.path.exists(sf_p):
            with rasterio.open(mndwi_p) as s: mm = s.read(1).astype(np.int16)
            with rasterio.open(sf_p)    as s: sm = s.read(1).astype(np.int16)
            diff = sm - mm
            ax.imshow(diff, cmap="RdBu", vmin=-1, vmax=1)
            agree = int((diff==0).sum())*100//diff.size
            ax.set_title(f"Difference\n(blue=this only, red=MNDWI only)\nAgreement {agree}%",
                         fontsize=7)
        ax.axis("off")

    plt.tight_layout()
    out = os.path.join(BASE_DIR, "segformer_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison visualization → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SOTA water body segmentation for Sentinel-2")
    parser.add_argument("--model", choices=["prithvi","ensemble","auto"], default="auto",
                        help="prithvi=Prithvi-EO-2.0 (needs terratorch+Python3.10), "
                             "ensemble=physics fallback, auto=try prithvi then fall back")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--dates", nargs="+", default=None)
    args = parser.parse_args()
    main(args.dates, args.threshold, args.model)
