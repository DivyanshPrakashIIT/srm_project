"""
SwinIR Backbone for Sentinel-2 Super-Resolution Mapping.
Implements Swin Transformer Blocks (STB) and Residual Swin Transformer Blocks (RSTB)
with dual-head reconstruction output (SR image and uncertainty logvar map).
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from srm.models.heads import UncertaintyHead


class Mlp(nn.Module):
    """Multi-Layer Perceptron (FFN) used in Swin Transformer Block."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Partition feature map into non-overlapping local windows.
    Args:
        x: (B, H, W, C)
        window_size: Size of local window
    Returns:
        windows: (num_windows * B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """
    Merge non-overlapping local windows back to feature map.
    Args:
        windows: (num_windows * B, window_size, window_size, C)
        window_size: Window size
        H: Feature map height
        W: Feature map width
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """
    Window-based multi-head self attention (W-MSA) module with relative position bias.
    """

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: float = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = (window_size, window_size) if isinstance(window_size, int) else window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads)
        )

        # Compute pair-wise relative position index inside window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block with optional shifted window attention (SW-MSA)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 8,
        shift_size: int = 0,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if self.shift_size > 0:
            assert 0 <= self.shift_size < self.window_size, "shift_size must be in [0, window_size)"

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)

    def forward(self, x: torch.Tensor, x_size: Tuple[int, int]) -> torch.Tensor:
        H, W = x_size
        B, L, C = x.shape
        assert L == H * W, f"Input feature length {L} does not match resolution {H}x{W}"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Pad feature map if height/width is not divisible by window_size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))

        Hp, Wp = x.shape[1], x.shape[2]

        # Shifted window masking
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
                attn_mask == 0, float(0.0)
            )
        else:
            shifted_x = x
            attn_mask = None

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA / SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # Reverse window partition
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN with residual connection
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB)."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                )
                for i in range(depth)
            ]
        )
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x: torch.Tensor, x_size: Tuple[int, int]) -> torch.Tensor:
        H, W = x_size
        B, C, _, _ = x.shape
        res = x
        x_flat = x.flatten(2).transpose(1, 2)
        for block in self.blocks:
            x_flat = block(x_flat, x_size)
        x = x_flat.transpose(1, 2).view(B, C, H, W)
        x = self.conv(x) + res
        return x


class Upsample4x(nn.Module):
    """4x Sub-pixel Convolution (PixelShuffle) Upsampler."""

    def __init__(self, in_channels: int = 64, out_channels: int = 64):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, in_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class SwinIRNet(nn.Module):
    """
    SwinIR Backbone adapted for 4-channel Sentinel-2 Super-Resolution Mapping.
    Outputs reconstructed high-resolution image ('sr') and heteroscedastic
    uncertainty log-variance ('logvar').
    """

    def __init__(
        self,
        in_chans: int = 4,
        out_chans: int = 4,
        upscale: int = 4,
        embed_dim: int = 64,
        depths: List[int] = [4, 4, 4],
        num_heads: List[int] = [4, 4, 4],
        window_size: int = 8,
        mlp_ratio: float = 2.0,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.upscale = upscale
        self.embed_dim = embed_dim

        # 1. Shallow feature extraction
        self.conv_first = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1)

        # 2. Deep feature extraction
        self.rstb_blocks = nn.ModuleList()
        for d, n in zip(depths, num_heads):
            self.rstb_blocks.append(
                RSTB(
                    dim=embed_dim,
                    depth=d,
                    num_heads=n,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                )
            )
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1)

        # 3. Upsampling module
        if upscale == 4:
            self.upsampler = Upsample4x(in_channels=embed_dim, out_channels=embed_dim)
        else:
            raise ValueError(f"Unsupported upscale factor: {upscale}. Expected 4.")

        # 4. Dual Heads
        self.sr_head = nn.Conv2d(embed_dim, out_chans, kernel_size=3, stride=1, padding=1)
        self.unc_head = UncertaintyHead(in_channels=embed_dim, out_channels=out_chans)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input LR tensor of shape (B, 4, H, W)
        Returns:
            dict containing:
                - 'sr': Reconstructed SR image of shape (B, 4, H*upscale, W*upscale)
                - 'logvar': Predicted log-variance of shape (B, 4, H*upscale, W*upscale)
        """
        _, _, H, W = x.shape
        x_first = self.conv_first(x)
        feat = x_first
        for rstb in self.rstb_blocks:
            feat = rstb(feat, (H, W))
        feat = self.conv_after_body(feat) + x_first

        feat_up = self.upsampler(feat)

        sr_tensor = self.sr_head(feat_up)
        logvar_tensor = self.unc_head(feat_up)

        return {"sr": sr_tensor, "logvar": logvar_tensor}


# Alias for SwinIRNet
SRMSwinIR = SwinIRNet