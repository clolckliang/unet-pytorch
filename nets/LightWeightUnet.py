import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels, momentum=0.1),  # 调整momentum以适应小batch_size
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels, momentum=0.1)
        self.se = SEBlock(channels)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.se(x)
        x += residual
        x = self.relu(x)
        return x


class LightweightVGG(nn.Module):
    def __init__(self, in_channels=3, pretrained=False):
        super(LightweightVGG, self).__init__()
        # 减少通道数以适应小数据集
        self.stage1 = nn.Sequential(
            ConvBlock(in_channels, 24),
            ResidualBlock(24),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.stage2 = nn.Sequential(
            ConvBlock(24, 48),
            ResidualBlock(48),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.stage3 = nn.Sequential(
            ConvBlock(48, 96),
            ResidualBlock(96),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.stage4 = nn.Sequential(
            ConvBlock(96, 192),
            ResidualBlock(192),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.stage5 = nn.Sequential(
            ConvBlock(192, 384),
            ResidualBlock(384),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 添加Dropout防止过拟合
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        feat1 = self.stage1(x)
        feat1 = self.dropout(feat1)

        feat2 = self.stage2(feat1)
        feat2 = self.dropout(feat2)

        feat3 = self.stage3(feat2)
        feat3 = self.dropout(feat3)

        feat4 = self.stage4(feat3)
        feat4 = self.dropout(feat4)

        feat5 = self.stage5(feat4)
        feat5 = self.dropout(feat5)

        return [feat1, feat2, feat3, feat4, feat5]


class LightweightUnetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(LightweightUnetUp, self).__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = nn.Sequential(
            ConvBlock(in_size, out_size),
            ResidualBlock(out_size)
        )
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv(outputs)
        outputs = self.dropout(outputs)
        return outputs


class LightweightUnet(nn.Module):
    def __init__(self, num_classes=21, pretrained=False, backbone='lightweight_vgg', in_channels=3):
        super(LightweightUnet, self).__init__()

        if backbone == 'lightweight_vgg':
            self.backbone = LightweightVGG(in_channels=in_channels, pretrained=pretrained)
        else:
            raise ValueError('Unsupported backbone - `{}`, Only lightweight_vgg is supported.'.format(backbone))

        # 调整解码器通道数
        self.up_concat4 = LightweightUnetUp(576, 192)  # 384 + 192 -> 192
        self.up_concat3 = LightweightUnetUp(288, 96)  # 192 + 96 -> 96
        self.up_concat2 = LightweightUnetUp(144, 48)  # 96 + 48 -> 48
        self.up_concat1 = LightweightUnetUp(72, 24)  # 48 + 24 -> 24

        # 添加额外的特征融合
        self.final_conv = nn.Sequential(
            ConvBlock(24, 24),
            nn.Dropout2d(0.1),
            ResidualBlock(24),
            nn.Conv2d(24, num_classes, 1)
        )

        self.backbone_name = backbone
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        [feat1, feat2, feat3, feat4, feat5] = self.backbone(inputs)

        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        final = self.final_conv(up1)
        return final

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True



# Calculate model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    model = LightweightUnet(num_classes=21)
    num_params = count_parameters(model)
    print(f"Number of parameters: {num_params:,}")