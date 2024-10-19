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

class LightConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_se=True):
        super(LightConvBlock, self).__init__()
        mid_channels = max(8, out_channels // 2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(mid_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.se = LightSEBlock(out_channels) if use_se else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.se(x)
        return x

class FeatureFusion(nn.Module):
    def __init__(self, in_channels):
        super(FeatureFusion, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.se = LightSEBlock(in_channels)
        
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.conv1x1(x)
        x = self.se(x)
        return x

class ImprovedUltraLightweightUnet(nn.Module):
    def __init__(self, num_classes=21):
        super(ImprovedUltraLightweightUnet, self).__init__()

        # Encoder
        self.enc1 = LightConvBlock(3, 32, use_se=True)
        self.enc2 = LightConvBlock(32, 64, use_se=True)
        self.enc3 = LightConvBlock(64, 128, use_se=True)
        self.enc4 = LightConvBlock(128, 256, use_se=True)

        # Bridge
        self.bridge = LightConvBlock(256, 512, use_se=True)

        # Feature Fusion modules
        self.fusion4 = FeatureFusion(256)
        self.fusion3 = FeatureFusion(128)
        self.fusion2 = FeatureFusion(64)
        self.fusion1 = FeatureFusion(32)

        # Decoder
        self.dec4 = LightConvBlock(256, 256, use_se=True)
        self.dec3 = LightConvBlock(128, 128, use_se=True)
        self.dec2 = LightConvBlock(64, 64, use_se=True)
        self.dec1 = LightConvBlock(32, 32, use_se=True)

        # Auxiliary outputs for deep supervision
        self.aux_out4 = nn.Conv2d(256, num_classes, 1)
        self.aux_out3 = nn.Conv2d(128, num_classes, 1)
        self.aux_out2 = nn.Conv2d(64, num_classes, 1)

        # Final output
        self.final = nn.Conv2d(32, num_classes, 1)

        self.dropout = nn.Dropout2d(0.1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # Bridge
        bridge = self.bridge(self.pool(enc4))

        # Decoder with improved skip connections
        up4 = F.interpolate(bridge, size=enc4.shape[2:], mode='bilinear', align_corners=True)
        merge4 = self.fusion4(up4, enc4)
        dec4 = self.dec4(merge4)

        up3 = F.interpolate(dec4, size=enc3.shape[2:], mode='bilinear', align_corners=True)
        merge3 = self.fusion3(up3, enc3)
        dec3 = self.dec3(merge3)

        up2 = F.interpolate(dec3, size=enc2.shape[2:], mode='bilinear', align_corners=True)
        merge2 = self.fusion2(up2, enc2)
        dec2 = self.dec2(merge2)

        up1 = F.interpolate(dec2, size=enc1.shape[2:], mode='bilinear', align_corners=True)
        merge1 = self.fusion1(up1, enc1)
        dec1 = self.dec1(merge1)

        # Final output
        final = self.final(dec1)
        final = F.interpolate(final, size=x.shape[2:], mode='bilinear', align_corners=True)

        if self.training:
            # Auxiliary outputs for deep supervision
            aux4 = F.interpolate(self.aux_out4(dec4), size=x.shape[2:], mode='bilinear', align_corners=True)
            aux3 = F.interpolate(self.aux_out3(dec3), size=x.shape[2:], mode='bilinear', align_corners=True)
            aux2 = F.interpolate(self.aux_out2(dec2), size=x.shape[2:], mode='bilinear', align_corners=True)
            return final, aux4, aux3, aux2
        
        return final
