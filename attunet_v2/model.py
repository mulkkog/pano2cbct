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
    def __init__(self, out_depth: int = 120, base_ch: int = 32):
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
 
