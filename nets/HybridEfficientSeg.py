import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class EfficientSEBlock(nn.Module):
    """Lightweight Squeeze-and-Excitation block"""

    def __init__(self, channels):
        super().__init__()
        reduced_channels = max(8, channels // 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class STDCBlock(nn.Module):
    """Modified STDC block with SE attention"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        mid_channels = out_channels // 2

        self.conv1 = DepthwiseSeparableConv(in_channels, mid_channels, stride=stride)
        self.conv2 = DepthwiseSeparableConv(mid_channels, mid_channels)
        self.se = EfficientSEBlock(out_channels)

        self.skip = None
        if stride > 1 or in_channels != out_channels:
            self.skip = DepthwiseSeparableConv(in_channels, out_channels, stride=stride)

    def forward(self, x):
        identity = x

        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out = torch.cat([out1, out2], dim=1)

        if self.skip is not None:
            identity = self.skip(x)

        out = out + identity
        out = self.se(out)
        return out


class DecoderBlock(nn.Module):
    """Efficient decoder block with skip connection and SE attention"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels),
            DepthwiseSeparableConv(out_channels, out_channels)
        )
        self.se = EfficientSEBlock(out_channels)

    def forward(self, x, skip=None):
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.se(x)
        return x


class HybridEfficientSeg(nn.Module):
    """Hybrid architecture combining STDC and U-Net features for steel defect segmentation"""

    def __init__(self, num_classes=3):
        super().__init__()

        # Encoder with STDC blocks
        self.enc1 = STDCBlock(3, 32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc2 = STDCBlock(32, 64, stride=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.enc3 = STDCBlock(64, 128, stride=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.enc4 = STDCBlock(128, 256, stride=1)
        self.pool4 = nn.MaxPool2d(2, 2)

        # Bridge
        self.bridge = STDCBlock(256, 512)
        self.dropout = nn.Dropout2d(0.2)

        # Decoder with skip connections
        self.dec4 = DecoderBlock(512 + 256, 256)
        self.dec3 = DecoderBlock(256 + 128, 128)
        self.dec2 = DecoderBlock(128 + 64, 64)
        self.dec1 = DecoderBlock(64 + 32, 32)

        # Output projection
        self.final_conv = nn.Sequential(
            DepthwiseSeparableConv(32, 32),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))

        # Bridge
        bridge = self.dropout(self.bridge(self.pool4(enc4)))

        # Decoder
        dec4 = self.dec4(F.interpolate(bridge, size=enc4.shape[2:],
                                       mode='bilinear', align_corners=True), enc4)
        dec3 = self.dec3(F.interpolate(dec4, size=enc3.shape[2:],
                                       mode='bilinear', align_corners=True), enc3)
        dec2 = self.dec2(F.interpolate(dec3, size=enc2.shape[2:],
                                       mode='bilinear', align_corners=True), enc2)
        dec1 = self.dec1(F.interpolate(dec2, size=enc1.shape[2:],
                                       mode='bilinear', align_corners=True), enc1)

        # Final output
        out = self.final_conv(dec1)
        return F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=True)


# 使用示例
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = HybridEfficientSeg(num_classes=3)
    dummy_input = torch.randn(1, 3, 256, 256)
    output = model(dummy_input)
    print(f'Output shape: {output.shape}')
    print(f'Number of parameters: {count_parameters(model):,}')