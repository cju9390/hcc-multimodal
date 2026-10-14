import argparse
import csv
import glob
import os

import numpy as np
import tiffslide
from PIL import Image

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
WSI_DIR = os.path.join(BASE_DIR, "wsi")
MANIFEST_PATH = os.path.join(BASE_DIR, "pilot_manifest.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "tiles")
TILE_MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "tile_manifest.csv")

TARGET_MAG = 20.0
PATCH_SIZE = 256
TISSUE_FRACTION_MIN = 0.5

# Macenko et al. 2009 reference H&E stain matrix / max concentrations,
# used as the normalization target so every tile ends up in the same stain space.
HE_REFERENCE = np.array([[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]])
MAX_C_REFERENCE = np.array([1.9705, 1.0308])


def native_magnification(slide):
    mag = slide.properties.get("aperio.AppMag")
    return float(mag) if mag else 40.0


def analyze_thumbnail(slide, Io=240):
    """Derive the tissue-saturation threshold and a slide-level stain concentration
    scale from one thumbnail, reused for every tile of that slide.

    Note: this deliberately does NOT re-estimate H&E stain *vectors* per slide via
    eigendecomposition (the textbook Macenko approach). On real TCGA patches, stroma-
    heavy regions have too little independent H vs. E color variation, so the two
    estimated stain vectors collapse onto nearly the same direction (condition number
    in the hundreds) and reconstruction blows up into black/white blob artifacts.
    Deconvolving every tile with the fixed literature reference matrix instead is far
    more stable and still normalizes staining intensity/brightness across slides via
    the per-slide concentration scale computed here."""
    thumb = slide.get_thumbnail((1024, 1024))
    s_channel = np.array(thumb.convert("HSV"))[:, :, 1]
    sat_threshold = otsu_threshold(s_channel)

    img = np.array(thumb).reshape(-1, 3).astype(float)
    od = -np.log10((img + 1) / Io)
    concentrations = np.linalg.lstsq(HE_REFERENCE, od.T, rcond=None)[0]
    max_c = np.percentile(concentrations, 99, axis=1)
    max_c[max_c == 0] = 1e-6

    return sat_threshold, max_c


def otsu_threshold(values):
    hist, bin_edges = np.histogram(values, bins=256, range=(0, 256))
    hist = hist.astype(float)
    total = hist.sum()
    sum_all = np.dot(np.arange(256), hist)
    sum_bg, weight_bg, best_thresh, best_var = 0.0, 0.0, 0, 0.0
    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_all - sum_bg) / weight_fg
        var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if var_between > best_var:
            best_var, best_thresh = var_between, t
    return best_thresh


def tissue_fraction(patch_rgb, sat_threshold):
    hsv = np.array(patch_rgb.convert("HSV"))
    return float((hsv[:, :, 1] > sat_threshold).mean())


def macenko_normalize(patch_rgb, max_c, Io=240):
    img = np.array(patch_rgb).reshape(-1, 3).astype(float)
    od = -np.log10((img + 1) / Io)

    concentrations = np.linalg.lstsq(HE_REFERENCE, od.T, rcond=None)[0]
    concentrations = concentrations / max_c[:, None] * MAX_C_REFERENCE[:, None]

    od_norm = HE_REFERENCE.dot(concentrations)
    img_norm = Io * np.exp(-od_norm * np.log(10))
    img_norm = np.clip(img_norm, 0, 255).T.reshape(patch_rgb.size[1], patch_rgb.size[0], 3)
    return Image.fromarray(img_norm.astype(np.uint8))


def tile_slide(slide_path, submitter_id, max_tiles, writer):
    slide = tiffslide.TiffSlide(slide_path)
    downsample = native_magnification(slide) / TARGET_MAG
    level = slide.get_best_level_for_downsample(downsample)
    level_downsample = slide.level_downsamples[level]
    read_size = round(PATCH_SIZE * downsample / level_downsample)

    sat_threshold, max_c = analyze_thumbnail(slide)
    width, height = slide.dimensions
    step = round(PATCH_SIZE * downsample)

    out_dir = os.path.join(OUT_DIR, submitter_id)
    os.makedirs(out_dir, exist_ok=True)

    n_saved = 0
    for y in range(0, height - step, step):
        if n_saved >= max_tiles:
            break
        for x in range(0, width - step, step):
            if n_saved >= max_tiles:
                break
            patch = slide.read_region((x, y), level, (read_size, read_size)).convert("RGB")
            if read_size != PATCH_SIZE:
                patch = patch.resize((PATCH_SIZE, PATCH_SIZE), Image.LANCZOS)

            frac = tissue_fraction(patch, sat_threshold)
            if frac < TISSUE_FRACTION_MIN:
                continue

            patch = macenko_normalize(patch, max_c)
            tile_path = os.path.join(out_dir, f"{submitter_id}_x{x}_y{y}.jpg")
            patch.save(tile_path, quality=90)
            writer.writerow(
                {
                    "submitter_id": submitter_id,
                    "tile_path": os.path.relpath(tile_path, os.path.dirname(TILE_MANIFEST_PATH)),
                    "x": x,
                    "y": y,
                    "tissue_fraction": round(frac, 3),
                }
            )
            n_saved += 1

    print(f"  {submitter_id}: saved {n_saved} tiles (level={level}, native_mag={native_magnification(slide)})")
    return n_saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tiles", type=int, default=100, help="max tiles to keep per slide")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N slides (smoke test)")
    args = parser.parse_args()

    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]

    os.makedirs(os.path.dirname(TILE_MANIFEST_PATH), exist_ok=True)
    with open(TILE_MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["submitter_id", "tile_path", "x", "y", "tissue_fraction"])
        writer.writeheader()
        for i, row in enumerate(rows, 1):
            matches = glob.glob(os.path.join(WSI_DIR, row["slide_file_name"]))
            if not matches:
                print(f"[{i}/{len(rows)}] {row['submitter_id']}: WSI file missing, skip")
                continue
            print(f"[{i}/{len(rows)}] {row['submitter_id']}")
            tile_slide(matches[0], row["submitter_id"], args.max_tiles, writer)


if __name__ == "__main__":
    main()
