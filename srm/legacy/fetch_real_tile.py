"""
fetch_real_tile.py — Pull a real Sentinel-2 L2A scene and tile it for SRM.

Uses Microsoft Planetary Computer's public STAC catalog (no account/API key
needed). Downloads B02 (Blue), B03 (Green), B04 (Red), B08 (NIR) at 10m for a
small window, stacks them in the order dataset.py expects, and writes one or
more 256x256 UINT16 GeoTIFF tiles into data/raw_tiles/.

Usage:
    python fetch_real_tile.py --lat 12.9716 --lon 77.5946 --out data/raw_tiles
    python fetch_real_tile.py --lat 40.7128 --lon -74.0060 --max-cloud 10

Requires: pystac-client, planetary-computer, rasterio, numpy
    pip install pystac-client planetary-computer
"""

import argparse
import os

import numpy as np
import rasterio
import rasterio.warp
from rasterio.windows import Window
from rasterio.transform import Affine
import pystac_client
import planetary_computer


BANDS = ["B02", "B03", "B04", "B08"]  # Blue, Green, Red, NIR — order dataset.py expects
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"


def parse_args():
    p = argparse.ArgumentParser(description="Fetch a real Sentinel-2 L2A tile for SRM")
    p.add_argument("--lat", type=float, required=True, help="Latitude of area of interest")
    p.add_argument("--lon", type=float, required=True, help="Longitude of area of interest")
    p.add_argument("--out", default="data/raw_tiles", help="Output directory")
    p.add_argument("--max-cloud", type=float, default=15.0, help="Max cloud cover %%")
    p.add_argument("--window-px", type=int, default=1024,
                    help="Size of the square window (in 10m pixels) to pull "
                         "from the scene before cutting it into 256x256 tiles.")
    p.add_argument("--tile-size", type=int, default=256)
    p.add_argument("--months-back", type=int, default=6,
                    help="How far back to search for a recent low-cloud scene")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)

    from datetime import datetime, timedelta
    end = datetime.utcnow()
    start = end - timedelta(days=30 * args.months_back)
    date_range = f"{start.date()}/{end.date()}"

    print(f"[fetch] Searching Sentinel-2 L2A near ({args.lat}, {args.lon}), "
          f"cloud<{args.max_cloud}%, {date_range}...")
    search = catalog.search(
        collections=[COLLECTION],
        intersects={"type": "Point", "coordinates": [args.lon, args.lat]},
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": args.max_cloud}},
        sortby=[{"field": "eo:cloud_cover", "direction": "asc"}],
        max_items=1,
    )
    items = list(search.get_items())
    if not items:
        raise RuntimeError(
            f"No scenes found under {args.max_cloud}% cloud in the last "
            f"{args.months_back} months at this location. Try --max-cloud 40 "
            f"or a different --lat/--lon, or increase --months-back."
        )
    item = items[0]
    print(f"[fetch] Using scene {item.id}  (cloud cover: "
          f"{item.properties.get('eo:cloud_cover', '?'):.1f}%, "
          f"date: {item.properties.get('datetime', '?')})")

    # Read a window centered on (lat, lon) from each band, at native 10m GSD.
    band_arrays = []
    ref_transform = None
    ref_crs = None
    half = args.window_px // 2

    for band in BANDS:
        href = item.assets[band].href
        with rasterio.open(href) as src:
            xs, ys = rasterio.warp.transform("EPSG:4326", src.crs, [args.lon], [args.lat])
            row, col = src.index(xs[0], ys[0])
            row0 = max(0, row - half)
            col0 = max(0, col - half)
            window = Window(col0, row0, args.window_px, args.window_px)
            arr = src.read(1, window=window, out_dtype="uint16",
                            boundless=True, fill_value=0)
            if ref_transform is None:
                ref_transform = src.window_transform(window)
                ref_crs = src.crs
            band_arrays.append(arr)
            print(f"[fetch]   {band}: read {arr.shape} window OK")

    stack = np.stack(band_arrays, axis=0)  # (4, H, W), order [B2,B3,B4,B8]
    print(f"[fetch] Stacked array shape: {stack.shape}, dtype: {stack.dtype}")

    # Cut into non-overlapping tile_size x tile_size tiles and write each as
    # its own GeoTIFF, matching what Sentinel2SISRDataset expects to find.
    ts = args.tile_size
    _, h, w = stack.shape
    n_written = 0
    for r in range(0, h - ts + 1, ts):
        for c in range(0, w - ts + 1, ts):
            tile = stack[:, r:r + ts, c:c + ts]
            if (tile == 0).all():
                continue  # skip empty/boundless-fill tiles
            tile_transform = Affine(
                ref_transform.a, ref_transform.b,
                ref_transform.c + c * ref_transform.a,
                ref_transform.d, ref_transform.e,
                ref_transform.f + r * ref_transform.e,
            )
            out_path = os.path.join(args.out, f"real_tile_{item.id}_{r}_{c}.tif")
            with rasterio.open(
                out_path, "w", driver="GTiff", height=ts, width=ts, count=4,
                dtype="uint16", crs=ref_crs, transform=tile_transform,
            ) as dst:
                dst.write(tile)
            n_written += 1

    print(f"[fetch] Wrote {n_written} tile(s) of {ts}x{ts} to '{args.out}'.")
    if n_written == 0:
        print("[fetch] WARNING: all tiles were empty (out of scene bounds or "
              "no-data). Try a larger --window-px or a different location.")


if __name__ == "__main__":
    main()
