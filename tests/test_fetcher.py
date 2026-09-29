from unittest.mock import MagicMock, patch
import numpy as np
import pystac
import pytest
import rasterio
from rasterio.transform import from_origin
import torch

from srm.data.fetcher import SentinelSTACFetcher


@pytest.fixture
def fetcher():
    return SentinelSTACFetcher()


def test_search_scenes_query_parameters(fetcher):
    """Verifies bounding box, datetime, and cloud cover search parameters sent to STAC API."""
    mock_client = MagicMock()
    mock_search = MagicMock()
    mock_item = MagicMock(spec=pystac.Item)
    mock_search.items.return_value = [mock_item]
    mock_client.search.return_value = mock_search

    bbox = [77.0, 28.5, 77.3, 28.8]
    datetime_range = "2023-01-01/2023-06-30"
    max_cloud_cover = 12.5

    with patch("pystac_client.Client.open", return_value=mock_client) as mock_open:
        results = fetcher.search_scenes(
            bbox=bbox,
            datetime_range=datetime_range,
            max_cloud_cover=max_cloud_cover,
            limit=5,
        )

        mock_open.assert_called_once_with(fetcher.catalog_url)
        mock_client.search.assert_called_once_with(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=datetime_range,
            query={"eo:cloud_cover": {"lt": max_cloud_cover}},
            max_items=5,
        )
        assert len(results) == 1
        assert results[0] == mock_item


def test_download_tile_bands_mapping_and_scaling(fetcher, tmp_path):
    """Verifies band ordering [B2, B3, B4, B8], asset mapping, and reflectance scaling (DN / 10000.0)."""
    mock_item = MagicMock(spec=pystac.Item)
    mock_item.id = "S2A_TEST_TILE"

    # Define mock assets for target bands
    assets = {}
    band_keys = ["B02", "B03", "B04", "B08"]
    for key in band_keys:
        mock_asset = MagicMock()
        mock_asset.href = f"https://example.com/rasters/{key}.tif"
        assets[key] = mock_asset

    mock_item.assets = assets

    # Create dummy raw DN band array (5000 DN = 0.5 reflectance)
    raw_dn_array = np.full((100, 100), fill_value=5000, dtype=np.uint16)

    class DummyRasterReader:
        def __init__(self, *args, **kwargs):
            self.profile = {
                "driver": "GTiff",
                "width": 100,
                "height": 100,
                "count": 1,
                "dtype": "uint16",
                "crs": "EPSG:4326",
                "transform": from_origin(0, 0, 10, 10),
            }

        def read(self, indexes=1):
            return raw_dn_array

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    real_open = rasterio.open

    def open_side_effect(fp, mode="r", **kwargs):
        # Route output writing and verification reads back to real rasterio
        if mode == "w" or str(tmp_path) in str(fp):
            return real_open(fp, mode, **kwargs)
        # Mock virtual input URL assets
        return DummyRasterReader()

    with patch.object(fetcher, "_sign_item", side_effect=lambda item: item), patch(
        "rasterio.open", side_effect=open_side_effect
    ):
        out_path = fetcher.download_tile_bands(
            item=mock_item,
            target_bands=["B2", "B3", "B4", "B8"],
            output_dir=tmp_path,
        )

        assert out_path.exists()
        assert out_path.name == "S2A_TEST_TILE_stacked.tif"

        # Read generated output raster and check values
        with rasterio.open(out_path) as src:
            assert src.count == 4
            assert src.dtypes[0] == "float32"
            stacked = src.read()
            # 5000 DN / 10000.0 = 0.5 reflectance
            assert np.allclose(stacked, 0.5, atol=1e-5)


def test_extract_patches_tensor_shape(fetcher, tmp_path):
    """Verifies that patch extraction yields tensors of shape (N, 4, 128, 128)."""
    raster_path = tmp_path / "synthetic_4band.tif"

    # Generate synthetic (4, 256, 256) raster file
    data = np.random.rand(4, 256, 256).astype(np.float32)
    profile = {
        "driver": "GTiff",
        "height": 256,
        "width": 256,
        "count": 4,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": from_origin(0, 0, 10, 10),
    }

    with rasterio.open(raster_path, "w", **profile) as dst:
        dst.write(data)

    patches = fetcher.extract_patches(raster_path, patch_size=128, stride=128)

    # 256x256 image with patch size 128 and stride 128 yields 4 patches
    assert isinstance(patches, torch.Tensor)
    assert patches.shape == (4, 4, 128, 128)
    assert patches.dtype == torch.float32