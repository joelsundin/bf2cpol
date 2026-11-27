"""
Stitch tiled TIFF images into a pyramidal OME-TIFF.

Default filename format:
    wsi4_m{m_idx}_bx{bx}_by{by}_bw{bw}_bh{bh}_cPOL.tif

Example:
    python stitch_tiles_to_ome_tiff.py \
        /path/to/tiles \
        /path/to/output.ome.tif \

With options:
    python stitch_tiles_to_ome_tiff.py \
        /path/to/tiles \
        /path/to/output.ome.tif \
        --tile_size 512 \
        --jpeg_quality 90 \
        --pixelsize_um 0.1726
"""
import argparse
import re
import tempfile
from pathlib import Path

import cv2
import numpy as np
import tifffile as tiff
from tqdm import tqdm


DEFAULT_TILE_PATTERN = (
    r"wsi4_m(?P<m_idx>\d+)_"
    r"bx(?P<bx>-?\d+)_"
    r"by(?P<by>-?\d+)_"
    r"bw(?P<bw>\d+)_"
    r"bh(?P<bh>\d+)_"
    r"cPOL\.tif$"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stitch tiled TIFF images into a pyramidal OME-TIFF."
    )

    parser.add_argument(
        "tile_dir",
        type=Path,
        help="Folder containing input tile TIFF files.",
    )

    parser.add_argument(
        "out_path",
        type=Path,
        help="Output pyramidal OME-TIFF path.",
    )

    parser.add_argument(
        "--tile_pattern",
        type=str,
        default=DEFAULT_TILE_PATTERN,
        help=(
            "Regex pattern for tile filenames. Must contain named groups: "
            "m_idx, bx, by, bw, bh."
        ),
    )

    parser.add_argument(
        "--glob",
        type=str,
        default="*.tif",
        help="Glob pattern for input tile files. Default: *.tif",
    )

    parser.add_argument(
        "--tile_size",
        type=int,
        default=512,
        help="OME-TIFF internal tile size. Default: 512.",
    )

    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=90,
        help="JPEG compression quality. Default: 90.",
    )

    parser.add_argument(
        "--pixelsize_um",
        type=float,
        default=0.1726,
        help="Pixel size in micrometers. Default: 0.1726.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )

    return parser.parse_args()


def collect_tiles(tile_dir: Path, tile_pattern: str, glob_pattern: str):
    pattern = re.compile(tile_pattern)
    tiles = []

    required_groups = {"m_idx", "bx", "by", "bw", "bh"}
    found_groups = set(pattern.groupindex.keys())
    missing_groups = required_groups - found_groups

    if missing_groups:
        raise ValueError(
            f"tile_pattern is missing required named groups: {sorted(missing_groups)}"
        )

    for f in tile_dir.glob(glob_pattern):
        if not f.is_file():
            continue

        match = pattern.search(f.name)
        if match is None:
            continue

        g = match.groupdict()

        tiles.append(
            {
                "file": f,
                "m_idx": int(g["m_idx"]),
                "bx": int(g["bx"]),
                "by": int(g["by"]),
                "bw": int(g["bw"]),
                "bh": int(g["bh"]),
            }
        )

    if not tiles:
        raise RuntimeError(f"No matching tiles found in {tile_dir}")

    return tiles


def build_base_memmap(tiles, tmp_path: Path):
    min_x = min(t["bx"] for t in tiles)
    min_y = min(t["by"] for t in tiles)
    max_x = max(t["bx"] + t["bw"] for t in tiles)
    max_y = max(t["by"] + t["bh"] for t in tiles)

    wsi_w = max_x - min_x
    wsi_h = max_y - min_y

    print(f"Found {len(tiles)} tiles")
    print("Global bounds:")
    print(f"  min_x={min_x}, min_y={min_y}")
    print(f"  max_x={max_x}, max_y={max_y}")
    print(f"  output size = ({wsi_h}, {wsi_w})")

    sample = tiff.imread(tiles[0]["file"])

    if sample.ndim == 2:
        raise ValueError("Expected RGB tiles, got 2D grayscale image.")

    if sample.ndim != 3 or sample.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB tiles, got shape {sample.shape}")

    shape = (wsi_h, wsi_w, sample.shape[2])
    dtype = sample.dtype

    print(f"Base level dtype = {dtype}, shape = {shape}")

    wsi = tiff.memmap(
        tmp_path,
        shape=shape,
        dtype=dtype,
        bigtiff=True,
    )

    wsi[...] = 0

    for tile in tqdm(
        sorted(tiles, key=lambda d: d["m_idx"]),
        desc="Stitching base level",
    ):
        img = tiff.imread(tile["file"])

        x0 = tile["bx"] - min_x
        y0 = tile["by"] - min_y
        x1 = x0 + tile["bw"]
        y1 = y0 + tile["bh"]

        if img.shape[0] != tile["bh"] or img.shape[1] != tile["bw"]:
            raise ValueError(
                f"Tile shape mismatch for {tile['file'].name}: "
                f"image {img.shape[:2]} vs metadata {(tile['bh'], tile['bw'])}"
            )

        if img.ndim != 3 or img.shape[2] != shape[2]:
            raise ValueError(
                f"Tile channel mismatch for {tile['file'].name}: "
                f"expected {shape[2]} channels, got shape {img.shape}"
            )

        wsi[y0:y1, x0:x1] = img

    wsi.flush()
    return wsi


def get_pyramid_shapes(h: int, w: int, tile_size: int):
    pyramid_shapes = []

    level_h = h
    level_w = w

    while level_h > tile_size or level_w > tile_size:
        level_h = max(1, (level_h + 1) // 2)
        level_w = max(1, (level_w + 1) // 2)
        pyramid_shapes.append((level_h, level_w))

    return pyramid_shapes


def write_pyramid(
    base_img,
    out_path: Path,
    tile_size: int,
    jpeg_quality: int,
    pixelsize_um: float,
):
    h, w, c = base_img.shape
    axes = "YXC"

    pyramid_shapes = get_pyramid_shapes(h, w, tile_size)
    num_subifds = len(pyramid_shapes)

    print(f"Writing {num_subifds + 1} pyramid levels total")
    print("Shapes:", [(h, w)] + pyramid_shapes)

    options = dict(
        tile=(tile_size, tile_size),
        photometric="rgb",
        compression="jpeg",
        compressionargs={"level": jpeg_quality},
        resolutionunit="CENTIMETER",
    )

    metadata = {
        "axes": axes,
        "PhysicalSizeX": pixelsize_um,
        "PhysicalSizeXUnit": "µm",
        "PhysicalSizeY": pixelsize_um,
        "PhysicalSizeYUnit": "µm",
        "Description": "WSI pyramid",
        "MapAnnotation": {
            "Namespace": "openmicroscopy.org/PyramidResolution",
            **{
                str(i): f"{level_w} {level_h}"
                for i, (level_h, level_w) in enumerate(pyramid_shapes, start=1)
            },
        },
    }

    with tiff.TiffWriter(out_path, bigtiff=True, ome=True) as tif:
        tif.write(
            base_img,
            subifds=num_subifds,
            metadata=metadata,
            resolution=(1e4 / pixelsize_um, 1e4 / pixelsize_um),
            **options,
        )

        level = np.asarray(base_img)

        for i, (level_h, level_w) in enumerate(pyramid_shapes, start=1):
            level = cv2.resize(
                level,
                (level_w, level_h),
                interpolation=cv2.INTER_AREA,
            )

            mag = w / level_w

            print(f"Writing level {i}: shape={level.shape}")

            tif.write(
                level,
                subfiletype=1,
                resolution=(
                    1e4 / (pixelsize_um * mag),
                    1e4 / (pixelsize_um * mag),
                ),
                **options,
            )


def main():
    args = parse_args()

    tile_dir = args.tile_dir
    out_path = args.out_path

    if not tile_dir.exists():
        raise FileNotFoundError(f"Tile folder not found: {tile_dir}")

    if not tile_dir.is_dir():
        raise NotADirectoryError(f"Tile path is not a folder: {tile_dir}")

    if out_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output file already exists: {out_path}\n"
            "Use --overwrite to replace it."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    tiles = collect_tiles(
        tile_dir=tile_dir,
        tile_pattern=args.tile_pattern,
        glob_pattern=args.glob,
    )

    with tempfile.TemporaryDirectory(prefix="wsi_stitch_") as tmpdir:
        tmp_base = Path(tmpdir) / "base_level_tmp.tif"

        base_img = build_base_memmap(tiles, tmp_base)

        write_pyramid(
            base_img=base_img,
            out_path=out_path,
            tile_size=args.tile_size,
            jpeg_quality=args.jpeg_quality,
            pixelsize_um=args.pixelsize_um,
        )

    print(f"Saved pyramidal OME-TIFF to: {out_path}")


if __name__ == "__main__":
    main()
