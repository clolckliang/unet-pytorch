import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class TraditionalUnet(nn.Module):
    def __init__(self, in_channels=3, num_classes=21):
        super(TraditionalUnet, self).__init__()

        # 减少基础通道数和层数
        # 原始: 64-128-256-512-1024
        # 现在: 32-64-128-256

        # 编码器部分
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)

        # 解码器部分
        self.up1 = Up(256 + 128, 128)
        self.up2 = Up(128 + 64, 64)
        self.up3 = Up(64 + 32, 32)

        # 最终输出层
        self.outc = nn.Conv2d(32, num_classes, kernel_size=1)

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

    def forward(self, x):
        # 编码路径
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # 解码路径
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        # 输出层
        output = self.outc(x)
        return output

    def freeze_encoder(self):
        """冻结编码器部分的参数"""
        for param in self.inc.parameters():
            param.requires_grad = False
        for param in self.down1.parameters():
            param.requires_grad = False
        for param in self.down2.parameters():
            param.requires_grad = False
        for param in self.down3.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """解冻编码器部分的参数"""
        for param in self.inc.parameters():
            param.requires_grad = True
        for param in self.down1.parameters():
            param.requires_grad = True
        for param in self.down2.parameters():
            param.requires_grad = True
        for param in self.down3.parameters():
            param.requires_grad = True


# 计算模型参数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 比较两个模型的参数量
    light_model = TraditionalUnet(in_channels=3, num_classes=21)
    light_params = count_parameters(light_model)
    print(f"Lightweight U-Net parameters: {light_params:,}")