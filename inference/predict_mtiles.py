"""
Batch-predict FOV-tiles (M-tiles) using a trained
Keikhosravi U-Net generator checkpoint.

Reads brightfield TIFF tiles, groups them by parent M-tile using filename
metadata, predicts reconstructed cPOL tiles, and blends overlapping patch
predictions into full reconstructed M-tiles using Hann-window weighting.

Supports checkpoints with either:
    checkpoint["generator"]   # GAN checkpoint
    checkpoint["model"]       # SSIM/MSE checkpoint
    raw state_dict            # fallback

Example: 

Input:
    Directory containing BF tiles exported from the original CZI M-tiles.

    Expected filename format:
        wsi4_mXXXXX_bxX_byY_bwW_bhH_pxXXXX_pyYYYY_BF.tif

    Each patch contains a 512 × 512 brightfield image.
    Tiles overlap according to the original tiling stride (256 px).

    Tiles are grouped by their parent M-tile using filename metadata.
    Each tiles is passed through the trained U-Net generator
    to predict the corresponding reconstructed cPOL image.
    
Outputs:
    wsi4_mXXXXX_bxX_byY_bwW_bhH_pred_recon.tif
"""

import sys
from pathlib import Path
from collections import defaultdict
import argparse
import re

import numpy as np
import tifffile as tiff
import torch
from tqdm import tqdm

ROOT = Path("/.../repository root/")
sys.path.append(str(ROOT))
from models.architectures.keikhosravi_unet import KeikhosraviUnet 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def tif_chw(path: Path) -> torch.Tensor:
    arr = tiff.imread(path).astype(np.float32)
    arr /= 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)

def save_uint8_tif(path: Path, arr: np.ndarray):
    tiff.imwrite(path, np.clip(arr, 0, 255).astype(np.uint8))

def make_weight(size=512):
    w = np.hanning(size)
    W = np.outer(w, w).astype(np.float32)
    W = np.maximum(W, 1e-6)
    W /= W.max()
    return W

def parse_args():
    parser = argparse.ArgumentParser()

    # allows for running from both ends of the WSI.
    parser.add_argument(
        "--from-end",
        action="store_true",
        help="Process from the end of the sorted list.",
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/home/jsundi/proj/users/jsundi/data/WSI_4_export/BF_patches"),
        help="BF patch directory.",
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("/home/jsundi/proj/users/jsundi/development/ship/results/results_mtiles_05/uncon_unet"),
        help="Directory where reconstructed M-tiles are saved.",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Model checkpoint.",
    )

    parser.add_argument(
        "--tile-size",
        type=int,
        default=512,
        help="Nominal patch tile size.",
    )

    parser.add_argument(
        "--canvas-width",
        type=int,
        default=2056,
        help="Width of reconstructed output.",
    )

    parser.add_argument(
        "--canvas-height",
        type=int,
        default=2464,
        help="Height of reconstructed output.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of same-shaped patches to predict at once.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    input_dir = args.input_dir
    outdir = args.outdir
    checkpoint_path = args.checkpoint
    tile_size = args.tile_size
    max_x = args.canvas_width
    max_y = args.canvas_height
    batch_size = args.batch_size
    C = 3

    outdir.mkdir(parents=True, exist_ok=True)

    model = KeikhosraviUnet(3, 3).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "generator" in checkpoint:
        model.load_state_dict(checkpoint["generator"])
        print("Loaded checkpoint key: generator")
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
        print("Loaded checkpoint key: model")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded checkpoint as raw state_dict")

    model.eval()

    input_paths = sorted(input_dir.glob("wsi4_m*_bx*_by*_bw*_bh*_px*_py*_BF.tif"))

    if args.from_end:
        input_paths = list(reversed(input_paths))
        print("Processing from end")
    else:
        print("Processing from beginning")

    print(f"Found {len(input_paths)} input tiles")

    tile_pattern = re.compile(
        r"wsi4_m(?P<m_idx>\d+)_"
        r"bx(?P<bx>-?\d+)_"
        r"by(?P<by>-?\d+)_"
        r"bw(?P<bw>\d+)_"
        r"bh(?P<bh>\d+)_"
        r"px(?P<px>\d+)_"
        r"py(?P<py>\d+)_BF\.tif$"
    )

    grouped_tiles = defaultdict(list)

    for f in input_paths:
        m = tile_pattern.search(f.name)
        if m is None:
            continue

        g = m.groupdict()

        m_idx = int(g["m_idx"])
        px = int(g["px"])
        py = int(g["py"])
        bx = int(g["bx"])
        by = int(g["by"])
        bw = int(g["bw"])
        bh = int(g["bh"])

        grouped_tiles[m_idx].append((f, px, py, bx, by, bw, bh))

    print(f"Found {len(grouped_tiles)} grouped M-tiles")

    weight_window = make_weight(tile_size).astype(np.float32)

    m_items = sorted(grouped_tiles.items(), reverse=args.from_end)
    written = 0

    with torch.no_grad():
        for m_idx, items in tqdm(m_items):
            _, _, _, bx, by, bw, bh = items[0]

            recon_name = (
                f"wsi4_m{m_idx:05d}_"
                f"bx{bx}_by{by}_bw{bw}_bh{bh}_"
                f"pred_recon.tif"
            )

            out_path = outdir / recon_name

            if out_path.exists():
                continue

            canvas = np.zeros((max_y, max_x, C), dtype=np.float32)
            count = np.zeros((max_y, max_x), dtype=np.float32)

            batch_tensors = []
            batch_meta = []
            batch_shape = None

            def flush_batch():
                nonlocal batch_tensors, batch_meta, batch_shape, canvas, count

                if len(batch_tensors) == 0:
                    return

                x_batch = torch.stack(batch_tensors, dim=0).to(device, non_blocking=True)

                y_batch = model(x_batch)
                y_batch = y_batch.detach().cpu().numpy()  # N,C,H,W

                for y, meta in zip(y_batch, batch_meta):
                    tile_name, px, py, h, w = meta

                    y = np.transpose(y, (1, 2, 0)).astype(np.float32)  # H,W,C

                    if y.max() <= 1.0:
                        y *= 255.0

                    x_end = min(px + w, max_x)
                    y_end = min(py + h, max_y)

                    crop_h = y_end - py
                    crop_w = x_end - px

                    tile_crop = y[:crop_h, :crop_w]
                    weight_crop = weight_window[:crop_h, :crop_w]
                    weight_crop_3d = weight_crop[..., None]

                    canvas[py:y_end, px:x_end] += tile_crop * weight_crop_3d
                    count[py:y_end, px:x_end] += weight_crop

                batch_tensors = []
                batch_meta = []
                batch_shape = None

            for tile_path, px, py, _, _, _, _ in items:
                x = tif_chw(tile_path)  # C,H,W
                _, h, w = x.shape
                current_shape = tuple(x.shape)

                if batch_shape is not None and current_shape != batch_shape:
                    flush_batch()

                batch_tensors.append(x)
                batch_meta.append((tile_path.name, px, py, h, w))
                batch_shape = current_shape

                if len(batch_tensors) == batch_size:
                    flush_batch()

            flush_batch()

            recon = canvas / np.maximum(count[..., None], 1e-6)
            save_uint8_tif(out_path, recon)
            written += 1

    print(f"Written {written} reconstructed M-tiles.")


if __name__ == "__main__":
    main()