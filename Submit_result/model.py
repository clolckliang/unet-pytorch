import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class LightConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LightConvBlock, self).__init__()
        mid_channels = max(8, out_channels // 2)  # 确保中间通道数不少于8
        self.conv = nn.Sequential(
            # 使用1x1卷积降低通道数
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # 使用深度可分离卷积
            DepthwiseSeparableConv(mid_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class LightSEBlock(nn.Module):
    def __init__(self, channels):
        super(LightSEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduced_channels = max(4, channels // 8)  # 确保不少于4个通道
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class self_net(nn.Module):
    def __init__(self, num_classes=21):
        super(self_net, self).__init__()

        # Encoder - slightly increase channel numbers
        self.enc1 = LightConvBlock(3, 32)
        self.enc2 = LightConvBlock(32, 64)
        self.enc3 = LightConvBlock(64, 128)
        self.enc4 = LightConvBlock(128, 256)

        # Bridge
        self.bridge = LightConvBlock(256, 512)

        # Decoder - with skip connections
        self.dec4 = LightConvBlock(512 + 256, 256)
        self.dec3 = LightConvBlock(256 + 128, 128)
        self.dec2 = LightConvBlock(128 + 64, 64)
        self.dec1 = LightConvBlock(64 + 32, 32)

        # Final output layer
        self.final = nn.Conv2d(32, num_classes, 1)

        # Dropout for regularization
        self.dropout = nn.Dropout2d(0.1)

        # Max pooling
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # Bridge
        bridge = self.bridge(self.pool(enc4))

        # Decoder with skip connections
        dec4 = self.dec4(
            torch.cat([F.interpolate(bridge, size=enc4.shape[2:], mode='bilinear', align_corners=True), enc4], 1))
        dec3 = self.dec3(
            torch.cat([F.interpolate(dec4, size=enc3.shape[2:], mode='bilinear', align_corners=True), enc3], 1))
        dec2 = self.dec2(
            torch.cat([F.interpolate(dec3, size=enc2.shape[2:], mode='bilinear', align_corners=True), enc2], 1))
        dec1 = self.dec1(
            torch.cat([F.interpolate(dec2, size=enc1.shape[2:], mode='bilinear', align_corners=True), enc1], 1))

        # Final output
        final = self.final(dec1)
        return F.interpolate(final, size=x.shape[2:], mode='bilinear', align_corners=True)