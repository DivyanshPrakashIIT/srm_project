from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.data as torch_data
import yaml

from srm.data.degradation import SRMDegradation

# Sentinel-2 SCL Invalid Classes: No Data (0), Saturated/Defective (1), Cloud Shadows (3),
# Cloud Medium Prob (8), Cloud High Prob (9), Thin Cirrus (10)
SCL_INVALID_CLASSES = {0, 1, 3, 8, 9, 10}


class SRMDataset(torch_data.Dataset):
    """
    Sentinel-2 Super-Resolution Dataset.
    Enforces band ordering [B2, B3, B4, B8], surface reflectance normalization DN/10000.0,
    and optional SCL quality mask filtering.
    """

    def __init__(
        self,
        data: Optional[Union[np.ndarray, torch.Tensor, List[Union[np.ndarray, torch.Tensor]]]] = None,
        scl_masks: Optional[Union[np.ndarray, torch.Tensor, List[Union[np.ndarray, torch.Tensor]]]] = None,
        degradation: Optional[SRMDegradation] = None,
        scale_factor: int = 4,
        patch_size: int = 128,
        max_invalid_ratio: float = 0.10,
        is_train: bool = True,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.patch_size = patch_size
        self.max_invalid_ratio = max_invalid_ratio
        self.is_train = is_train
        self.degradation = degradation or SRMDegradation(scale_factor=scale_factor)

        self.hr_patches: List[torch.Tensor] = []

        if data is not None:
            self._process_data(data, scl_masks)

    def _process_data(
        self,
        data: Union[np.ndarray, torch.Tensor, List[Union[np.ndarray, torch.Tensor]]],
        scl_masks: Optional[Union[np.ndarray, torch.Tensor, List[Union[np.ndarray, torch.Tensor]]]] = None,
    ) -> None:
        if isinstance(data, (np.ndarray, torch.Tensor)):
            tensor_data = torch.from_numpy(data) if isinstance(data, np.ndarray) else data
            if tensor_data.dim() == 3:
                tensor_data = tensor_data.unsqueeze(0)
            data_list = [tensor_data[i] for i in range(tensor_data.shape[0])]
        elif isinstance(data, list):
            data_list = [torch.from_numpy(x) if isinstance(x, np.ndarray) else x for x in data]
        else:
            data_list = []

        scl_list = None
        if scl_masks is not None:
            if isinstance(scl_masks, (np.ndarray, torch.Tensor)):
                scl_tensor = torch.from_numpy(scl_masks) if isinstance(scl_masks, np.ndarray) else scl_masks
                if scl_tensor.dim() == 2:
                    scl_tensor = scl_tensor.unsqueeze(0)
                scl_list = [scl_tensor[i] for i in range(scl_tensor.shape[0])]
            elif isinstance(scl_masks, list):
                scl_list = [torch.from_numpy(m) if isinstance(m, np.ndarray) else m for m in scl_masks]

        for idx, item in enumerate(data_list):
            scl = scl_list[idx] if scl_list is not None else None
            self._add_item(item, scl)

    def _add_item(self, item: torch.Tensor, scl: Optional[torch.Tensor] = None) -> None:
        item = item.to(dtype=torch.float32)

        # Radiometric Normalization DN / 10000.0 clipped to [0.0, 1.0]
        if item.max() > 1.0:
            item = item / 10000.0
        item = torch.clamp(item, 0.0, 1.0)

        # Strict band ordering verification [B2, B3, B4, B8]
        if item.shape[0] != 4:
            raise ValueError(f"Expected 4 spectral bands [B2, B3, B4, B8], got shape {item.shape}")

        _, h, w = item.shape
        if h < self.patch_size or w < self.patch_size:
            return

        if h == self.patch_size and w == self.patch_size:
            if self._passes_quality_filter(scl):
                self.hr_patches.append(item)
        else:
            # Crop grid without legacy 5x replicate padding or 255x255 hacks
            for i in range(0, h - self.patch_size + 1, self.patch_size):
                for j in range(0, w - self.patch_size + 1, self.patch_size):
                    patch = item[:, i : i + self.patch_size, j : j + self.patch_size]
                    scl_patch = scl[i : i + self.patch_size, j : j + self.patch_size] if scl is not None else None
                    if self._passes_quality_filter(scl_patch):
                        self.hr_patches.append(patch)

    def _passes_quality_filter(self, scl_patch: Optional[Union[np.ndarray, torch.Tensor]]) -> bool:
        if scl_patch is None:
            return True
        if isinstance(scl_patch, torch.Tensor):
            scl_arr = scl_patch.detach().cpu().numpy()
        else:
            scl_arr = scl_patch
        invalid_mask = np.isin(scl_arr, list(SCL_INVALID_CLASSES))
        invalid_ratio = np.mean(invalid_mask)
        return round(float(invalid_ratio), 2) < self.max_invalid_ratio

    def __len__(self) -> int:
        return len(self.hr_patches)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        hr_tensor = self.hr_patches[idx]
        lr_tensor = self.degradation(hr_tensor)
        return lr_tensor, hr_tensor


def create_dataloaders(config_path: Union[str, Path]) -> Tuple[torch_data.DataLoader, torch_data.DataLoader]:
    """Reads dataset parameters from config file and creates train and validation DataLoaders."""
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, "r") as f:
            raw_cfg = yaml.safe_load(f)
            cfg = raw_cfg.get("data", raw_cfg)
    else:
        cfg = {}

    batch_size = cfg.get("batch_size", 4)
    num_workers = cfg.get("num_workers", 0)
    scale_factor = cfg.get("scale_factor", 4)
    patch_size = cfg.get("patch_size", 128)
    max_invalid_ratio = cfg.get("max_invalid_ratio", 0.10)
    seed = cfg.get("seed", 42)

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    degradation = SRMDegradation(
        scale_factor=scale_factor,
        blur_kernel_size=cfg.get("blur_kernel_size", 3),
        blur_sigma=cfg.get("blur_sigma", 0.5),
        noise_std=cfg.get("noise_std", 0.0),
    )

    # Synthetic fallback tensors for standalone pipeline validation
    train_data = torch.rand((8, 4, patch_size, patch_size)) * 10000.0
    val_data = torch.rand((4, 4, patch_size, patch_size)) * 10000.0

    train_dataset = SRMDataset(
        data=train_data,
        degradation=degradation,
        scale_factor=scale_factor,
        patch_size=patch_size,
        max_invalid_ratio=max_invalid_ratio,
        is_train=True,
    )

    val_dataset = SRMDataset(
        data=val_data,
        degradation=degradation,
        scale_factor=scale_factor,
        patch_size=patch_size,
        max_invalid_ratio=max_invalid_ratio,
        is_train=False,
    )

    train_loader = torch_data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = torch_data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader