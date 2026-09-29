from typing import Any, Dict, Union
import torch.nn as nn

from srm.models.heads import UncertaintyHead
from srm.models.m0_bicubic import BicubicModel
from srm.models.m1_rrdb import RRDBNet
from srm.models.m2_swin import SRMSwinIR, SwinIRNet

__all__ = [
    "BicubicModel",
    "RRDBNet",
    "SwinIRNet",
    "SRMSwinIR",
    "UncertaintyHead",
    "build_model",
]


def build_model(config: Union[Dict[str, Any], str], **kwargs: Any) -> nn.Module:
    """
    Factory function to build model instances from a configuration dict or model name string.

    Args:
        config: Dictionary containing model specification or string model name.
        **kwargs: Optional overrides or kwargs when passing model name directly.

    Returns:
        Instantiated nn.Module model complying with CONTRACTS.md.
    """
    if isinstance(config, str):
        model_cfg = {"name": config, **kwargs}
    else:
        model_cfg = config.get("model", config)
        if kwargs:
            model_cfg = {**model_cfg, **kwargs}

    name = str(model_cfg.get("name", "")).lower().strip()

    in_nc = model_cfg.get("in_nc", model_cfg.get("in_chans", 4))
    out_nc = model_cfg.get("out_nc", model_cfg.get("out_chans", 4))
    scale_factor = model_cfg.get("scale_factor", model_cfg.get("upscale", 4))

    if name in ["m0_bicubic", "bicubic", "m0"]:
        return BicubicModel(
            scale_factor=scale_factor,
            in_nc=in_nc,
            out_nc=out_nc,
        )
    elif name in ["m1_rrdb", "rrdb", "rrdbnet", "m1"]:
        return RRDBNet(
            in_nc=in_nc,
            out_nc=out_nc,
            num_filters=model_cfg.get("num_filters", 64),
            num_blocks=model_cfg.get("num_blocks", 6),
            gc=model_cfg.get("gc", 32),
            scale_factor=scale_factor,
        )
    elif name in ["m2_swin", "swin", "swinir", "srm_swinir", "m2"]:
        return SwinIRNet(
            in_chans=in_nc,
            out_chans=out_nc,
            upscale=scale_factor,
            embed_dim=model_cfg.get("embed_dim", 64),
            depths=model_cfg.get("depths", [4, 4, 4]),
            num_heads=model_cfg.get("num_heads", [4, 4, 4]),
            window_size=model_cfg.get("window_size", 8),
            mlp_ratio=model_cfg.get("mlp_ratio", 2.0),
        )
    else:
        raise ValueError(f"Unknown model name: '{name}'")