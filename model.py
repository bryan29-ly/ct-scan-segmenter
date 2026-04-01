"""
PlainConv U-Net for abdominal CT segmentation.

7-stage encoder (32→512→512→512), 6-stage decoder with skip connections.
Deep supervision heads at the 3 finest resolutions.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, 3, stride=stride, padding=1, bias=True)
        self.norm = nn.InstanceNorm2d(out_ch, eps=1e-5, affine=True)
        self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))


class EncoderStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int):
        super().__init__()
        self.conv1 = ConvBlock(in_ch, out_ch, stride=stride)
        self.conv2 = ConvBlock(out_ch, out_ch, stride=1)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class DecoderStage(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_ch, in_ch, kernel_size=2, stride=2, bias=False)
        self.conv1 = ConvBlock(in_ch + skip_ch, out_ch, stride=1)
        self.conv2 = ConvBlock(out_ch, out_ch, stride=1)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


class PlainConvUNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 54, deepsupervision: bool = True):
        super().__init__()
        self.deepsupervision = deepsupervision
        features = [32, 64, 128, 256, 512, 512, 512]
        strides = [1, 2, 2, 2, 2, 2, 2]

        # Encoder
        self.encoder = nn.ModuleList()
        ch = in_channels
        for i, (f, s) in enumerate(zip(features, strides)):
            self.encoder.append(EncoderStage(ch, f, stride=s))
            ch = f

        # Decoder
        self.decoder = nn.ModuleList()
        for i in range(len(features) - 1, 0, -1):
            self.decoder.append(DecoderStage(
                features[i], features[i - 1], features[i - 1]))

        # Deep supervision heads (needed for weight compatibility)
        self.heads = nn.ModuleList([
            nn.Conv2d(features[0], num_classes, 1, bias=True),
            nn.Conv2d(features[1], num_classes, 1, bias=True),
            nn.Conv2d(features[2], num_classes, 1, bias=True),
        ])

    def forward(self, x):
        skips = []
        for i, stage in enumerate(self.encoder):
            x = stage(x)
            if i < len(self.encoder) - 1:
                skips.append(x)

        for i, stage in enumerate(self.decoder):
            x = stage(x, skips[-(i + 1)])

        return self.heads[0](x)
