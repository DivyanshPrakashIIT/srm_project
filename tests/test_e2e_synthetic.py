"""make e2e: synthetic tile -> reference model -> 5-band COG -> read back and check contract C5."""
import pytest
import torch

from srm import contracts as C

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402

pytestmark = pytest.mark.e2e


def test_synthetic_tile_to_cog(tmp_path, ref_model, synthetic_lr):
    lr = synthetic_lr[:1]  # (1,4,64,64)
    C.validate_reflectance(lr, "lr")
    with torch.no_grad():
        out = ref_model(lr)
    C.validate_model_output(out, lr)
    prod = C.assemble_product_array(out["sr"], out["logvar"])

    src_px = 10.0
    transform = from_origin(77.0, 13.0, src_px / C.SCALE, src_px / C.SCALE)  # 2.5 m pixels
    path = tmp_path / "product.tif"
    profile = dict(driver="COG", dtype="float32", count=5, width=256, height=256,
                   crs="EPSG:32643", transform=transform, compress="DEFLATE", predictor=3)
    try:
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(prod)
            dst.descriptions = C.PRODUCT_BAND_DESCRIPTIONS
    except Exception as exc:  # GDAL < 3.1 has no COG driver
        pytest.skip(f"COG driver unavailable: {exc}")

    with rasterio.open(path) as src:
        assert src.count == 5 and src.dtypes == ("float32",) * 5
        assert (src.height, src.width) == (C.INFER_SR_SIZE, C.INFER_SR_SIZE)
        assert src.res == (2.5, 2.5)
        assert src.descriptions == C.PRODUCT_BAND_DESCRIPTIONS
        data = src.read()
    assert data[:4].min() >= 0.0 and data[:4].max() <= 1.0 and data[4].min() >= 0.0
