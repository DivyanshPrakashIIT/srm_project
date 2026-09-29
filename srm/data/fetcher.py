from pathlib import Path
from typing import Dict, List, Tuple, Union
import numpy as np
import pystac
import pystac_client
import rasterio
import torch

try:
    import planetary_computer as pc

    HAS_PLANETARY_COMPUTER = True
except ImportError:
    pc = None
    HAS_PLANETARY_COMPUTER = False


BAND_MAP: Dict[str, str] = {
    "B2": "B02",
    "B3": "B03",
    "B4": "B04",
    "B8": "B08",
    "B02": "B02",
    "B03": "B03",
    "B04": "B04",
    "B08": "B08",
}

STAC_CATALOG_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


class SentinelSTACFetcher:
    """
    Satellite Data Acquisition Fetcher querying Sentinel-2 L2A scenes via PySTAC Client
    and Microsoft Planetary Computer STAC catalog.
    """

    def __init__(self, catalog_url: str = STAC_CATALOG_URL):
        self.catalog_url = catalog_url

    def _sign_item(self, item: pystac.Item) -> pystac.Item:
        """Signs PySTAC item for Planetary Computer access if library is available."""
        if HAS_PLANETARY_COMPUTER and pc is not None:
            return pc.sign(item)
        return item

    def search_scenes(
        self,
        bbox: Union[List[float], Tuple[float, float, float, float]],
        datetime_range: str,
        max_cloud_cover: float = 15.0,
        limit: int = 10,
    ) -> List[pystac.Item]:
        """
        Queries STAC API for Sentinel-2 L2A items matching bounding box, datetime, and cloud cover.

        Args:
            bbox: Bounding box [min_lon, min_lat, max_lon, max_lat].
            datetime_range: ISO-8601 string or range e.g. "2023-01-01/2023-06-30".
            max_cloud_cover: Maximum allowed cloud cover percentage (default: 15.0).
            limit: Maximum number of items to return.

        Returns:
            List of matching PySTAC Item objects.
        """
        client = pystac_client.Client.open(self.catalog_url)
        search = client.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=datetime_range,
            query={"eo:cloud_cover": {"lt": max_cloud_cover}},
            max_items=limit,
        )
        return list(search.items())

    def download_tile_bands(
        self,
        item: pystac.Item,
        target_bands: List[str] = None,
        output_dir: Union[str, Path] = "data/raw",
    ) -> Path:
        """
        Downloads requested bands [B2, B3, B4, B8], normalizes Digital Numbers (DN / 10000.0),
        and stacks them into a 4-band GeoTIFF raster file.

        Args:
            item: PySTAC Item representing a Sentinel-2 L2A tile.
            target_bands: Ordered list of band names (default ['B2', 'B3', 'B4', 'B8']).
            output_dir: Path to directory where multi-band raster will be saved.

        Returns:
            Path to saved 4-band GeoTIFF file.
        """
        if target_bands is None:
            target_bands = ["B2", "B3", "B4", "B8"]

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{item.id}_stacked.tif"

        # Sign item for Planetary Computer access if available
        signed_item = self._sign_item(item)

        band_data_list = []
        profile = None

        for band in target_bands:
            asset_key = BAND_MAP.get(band, band)
            if asset_key not in signed_item.assets:
                raise KeyError(
                    f"Asset band '{band}' (mapped to '{asset_key}') not found in item assets. "
                    f"Available assets: {list(signed_item.assets.keys())}"
                )

            asset_href = signed_item.assets[asset_key].href

            with rasterio.open(asset_href) as src:
                if profile is None:
                    profile = src.profile.copy()

                band_array = src.read(1).astype(np.float32)
                # Reflectance scale: rho = DN / 10000.0, clamped to [0.0, 1.0]
                scaled_band = np.clip(band_array / 10000.0, 0.0, 1.0)
                band_data_list.append(scaled_band)

        stacked_data = np.stack(band_data_list, axis=0)  # Shape: (4, H, W)

        # Update metadata profile for 4-band float32 GeoTIFF
        profile.update(
            count=len(target_bands),
            dtype="float32",
            driver="GTiff",
        )

        with rasterio.open(out_path, "w", **profile) as dst:
            for b_idx in range(len(target_bands)):
                dst.write(stacked_data[b_idx], b_idx + 1)

        return out_path

    def extract_patches(
        self,
        raster_path: Union[str, Path],
        patch_size: int = 128,
        stride: int = 128,
    ) -> torch.Tensor:
        """
        Extracts patch tensors of shape (N, 4, patch_size, patch_size) from a 4-band raster file.

        Args:
            raster_path: Path to stacked 4-band raster file.
            patch_size: Height/Width of square patches (default 128).
            stride: Stride for sliding window extraction (default 128).

        Returns:
            PyTorch float32 tensor of shape (N, 4, patch_size, patch_size).
        """
        raster_path = Path(raster_path)
        with rasterio.open(raster_path) as src:
            data = src.read().astype(np.float32)  # Shape: (C, H, W)

        c, h, w = data.shape
        patches = []

        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch = data[:, y : y + patch_size, x : x + patch_size]
                patches.append(patch)

        if not patches:
            return torch.empty((0, c, patch_size, patch_size), dtype=torch.float32)

        patches_arr = np.stack(patches, axis=0)
        return torch.from_numpy(patches_arr)