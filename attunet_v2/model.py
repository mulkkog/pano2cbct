import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """(Conv2d -> BN -> ReLU) x 2"""
    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),

            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UpConv(nn.Module):
    """Upsample -> Conv2d -> BN -> ReLU"""
    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)


class AttentionBlock(nn.Module):
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttUNet2Dto3D(nn.Module):
    """
    input:
      - (B, H, W) or (B, 1, H, W)
    output:
      - (B, out_depth, H, W)
    """
    def __init__(self, out_depth: int = 120, base_ch: int = 64):
        super().__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = ConvBlock(1, base_ch)
        self.conv2 = ConvBlock(base_ch, base_ch * 2)
        self.conv3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.conv4 = ConvBlock(base_ch * 4, base_ch * 8)
        self.conv5 = ConvBlock(base_ch * 8, base_ch * 16)

        self.up5 = UpConv(base_ch * 16, base_ch * 8)
        self.att5 = AttentionBlock(F_g=base_ch * 8, F_l=base_ch * 8, F_int=base_ch * 4)
        self.upconv5 = ConvBlock(base_ch * 16, base_ch * 8)

        self.up4 = UpConv(base_ch * 8, base_ch * 4)
        self.att4 = AttentionBlock(F_g=base_ch * 4, F_l=base_ch * 4, F_int=base_ch * 2)
        self.upconv4 = ConvBlock(base_ch * 8, base_ch * 4)

        self.up3 = UpConv(base_ch * 4, base_ch * 2)
        self.att3 = AttentionBlock(F_g=base_ch * 2, F_l=base_ch * 2, F_int=base_ch)
        self.upconv3 = ConvBlock(base_ch * 4, base_ch * 2)

        self.up2 = UpConv(base_ch * 2, base_ch)
        self.att2 = AttentionBlock(F_g=base_ch, F_l=base_ch, F_int=max(base_ch // 2, 1))
        self.upconv2 = ConvBlock(base_ch * 2, base_ch)

        self.out_conv = nn.Conv2d(base_ch, out_depth, kernel_size=1, stride=1, padding=0)

    @staticmethod
    def _resize_like(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if src.shape[2:] != ref.shape[2:]:
            src = F.interpolate(src, size=ref.shape[2:], mode="bilinear", align_corners=False)
        return src

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() == 4:
            if x.size(1) != 1:
                raise ValueError(
                    f"Expected input shape (B,H,W) or (B,1,H,W), but got {tuple(x.shape)}"
                )
        else:
            raise ValueError(
                f"Expected input with 3 or 4 dims, but got {x.dim()} dims: {tuple(x.shape)}"
            )

        x1 = self.conv1(x)
        x2 = self.maxpool(x1)
        x2 = self.conv2(x2)

        x3 = self.maxpool(x2)
        x3 = self.conv3(x3)

        x4 = self.maxpool(x3)
        x4 = self.conv4(x4)

        x5 = self.maxpool(x4)
        x5 = self.conv5(x5)

        d5 = self.up5(x5)
        d5 = self._resize_like(d5, x4)
        x4_att = self.att5(g=d5, x=x4)
        d5 = torch.cat((x4_att, d5), dim=1)
        d5 = self.upconv5(d5)

        d4 = self.up4(d5)
        d4 = self._resize_like(d4, x3)
        x3_att = self.att4(g=d4, x=x3)
        d4 = torch.cat((x3_att, d4), dim=1)
        d4 = self.upconv4(d4)

        d3 = self.up3(d4)
        d3 = self._resize_like(d3, x2)
        x2_att = self.att3(g=d3, x=x2)
        d3 = torch.cat((x2_att, d3), dim=1)
        d3 = self.upconv3(d3)

        d2 = self.up2(d3)
        d2 = self._resize_like(d2, x1)
        x1_att = self.att2(g=d2, x=x1)
        d2 = torch.cat((x1_att, d2), dim=1)
        d2 = self.upconv2(d2)

        out = self.out_conv(d2)
        return out


# -----------------------------
# 3D refine network
# -----------------------------
class ConvBlock3D(nn.Module):
    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(ch_out, affine=True),
            nn.SiLU(inplace=True),
            nn.Conv3d(ch_out, ch_out, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(ch_out, affine=True),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock3D(nn.Module):
    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(ch_in, ch_out, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(ch_out, affine=True),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class RefineUNet3D(nn.Module):
    """
    input : (B,1,Z,H,W) coarse volume
    output: (B,1,Z,H,W) residual volume
    """
    def __init__(self, in_ch: int = 1, base_ch: int = 16):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc1 = ConvBlock3D(in_ch, base_ch)
        self.enc2 = ConvBlock3D(base_ch, base_ch * 2)
        self.enc3 = ConvBlock3D(base_ch * 2, base_ch * 4)
        self.enc4 = ConvBlock3D(base_ch * 4, base_ch * 8)

        self.up3 = UpBlock3D(base_ch * 8, base_ch * 4)
        self.dec3 = ConvBlock3D(base_ch * 8, base_ch * 4)

        self.up2 = UpBlock3D(base_ch * 4, base_ch * 2)
        self.dec2 = ConvBlock3D(base_ch * 4, base_ch * 2)

        self.up1 = UpBlock3D(base_ch * 2, base_ch)
        self.dec1 = ConvBlock3D(base_ch * 2, base_ch)

        self.out_conv = nn.Conv3d(base_ch, 1, kernel_size=1)

    @staticmethod
    def _resize_like(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if src.shape[2:] != ref.shape[2:]:
            src = F.interpolate(src, size=ref.shape[2:], mode="trilinear", align_corners=False)
        return src

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))

        d3 = self.up3(x4)
        d3 = self._resize_like(d3, x3)
        d3 = torch.cat([x3, d3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self._resize_like(d2, x2)
        d2 = torch.cat([x2, d2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self._resize_like(d1, x1)
        d1 = torch.cat([x1, d1], dim=1)
        d1 = self.dec1(d1)

        residual = self.out_conv(d1)
        return residual


class CoarseRefineNet(nn.Module):
    """
    pano -> coarse 2D-to-3D -> residual refine 3D -> refined volume
    """
    def __init__(
        self,
        out_depth: int = 120,
        coarse_base_ch: int = 128,
        refine_base_ch: int = 16,
    ):
        super().__init__()
        self.coarse = AttUNet2Dto3D(
            out_depth=out_depth,
            base_ch=coarse_base_ch,
        )
        self.refine = RefineUNet3D(
            in_ch=1,
            base_ch=refine_base_ch,
        )

    def forward(self, pano: torch.Tensor) -> Dict[str, torch.Tensor]:
        coarse_4d = self.coarse(pano)              # (B,Z,H,W)
        coarse_5d = coarse_4d.unsqueeze(1)         # (B,1,Z,H,W)
        residual = self.refine(coarse_5d)          # (B,1,Z,H,W)
        refined = coarse_5d + residual             # residual learning
        return {
            "coarse_4d": coarse_4d,
            "coarse_5d": coarse_5d,
            "residual": residual,
            "refined": refined,
        }