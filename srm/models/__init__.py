from typing import Any, Dict
import torch.nn as nn

from srm.models.m0_bicubic import BicubicModel
from srm.models.m1_rrdb import RRDBNet

__all__ = ["BicubicModel", "RRDBNet", "build_model"]


def build_model(config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to build model instances from configuration dict.

    Args:
        config: Dictionary containing model specification.

    Returns:
        Instantiated nn.Module model complying with CONTRACTS.md.
    """
    model_cfg = config.get("model", config)
    name = model_cfg.get("name", "").lower()

    if name in ["m0_bicubic", "bicubic", "m0"]:
        return BicubicModel(
            scale_factor=model_cfg.get("scale_factor", 4),
            in_nc=model_cfg.get("in_nc", 4),
            out_nc=model_cfg.get("out_nc", 4),
        )
    elif name in ["m1_rrdb", "rrdb", "rrdbnet", "m1"]:
        return RRDBNet(
            in_nc=model_cfg.get("in_nc", 4),
            out_nc=model_cfg.get("out_nc", 4),
            num_filters=model_cfg.get("num_filters", 64),
            num_blocks=model_cfg.get("num_blocks", 6),
            gc=model_cfg.get("gc", 32),
            scale_factor=model_cfg.get("scale_factor", 4),
        )
    else:
        raise ValueError(f"Unknown model name: '{name}'")
