import torch
import torch.nn as nn
import torch.nn.functional as F


class LightSEBlock(nn.Module):
    def __init__(self, channels):
        super(LightSEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduced_channels = max(8, channels // 4)  # 保持SE模块的通道压缩比
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



class RepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
        super(RepVGGBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        # 3x3 conv branch
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                               padding, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 1x1 conv branch
        self.conv2 = nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Identity branch
        if in_channels == out_channels and stride == 1:
            self.identity = True
            self.id_bn = nn.BatchNorm2d(out_channels)
        else:
            self.identity = False

        self.relu = nn.ReLU(inplace=True)
        self.deploy = False

    def forward(self, x):
        if self.deploy:
            return self.relu(self.reparam_conv(x))

        id_out = self.id_bn(x) if self.identity else 0
        return self.relu(self.bn1(self.conv1(x)) + self.bn2(self.conv2(x)) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1, self.bn1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2, self.bn2)
        kernelid, biasid = self._fuse_bn_tensor(None, self.id_bn) if self.identity else (0, 0)

        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def switch_to_deploy(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        self.reparam_conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size,
                                      self.stride, self.padding, groups=self.groups)
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias
        self.deploy = True

    def _fuse_bn_tensor(self, conv, bn):
        if conv is None:
            kernel = torch.zeros(self.out_channels, self.out_channels, 3, 3)
            for i in range(self.out_channels):
                kernel[i, i, 1, 1] = 1
            return kernel, bn.bias
        kernel = conv.weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])


class FusedMBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_ratio=4):
        super(FusedMBConv, self).__init__()

        exp_channels = int(in_channels * expansion_ratio)
        self.conv = nn.Sequential(
            # Fused Conv = MBConv的DWConv和PWConv的组合
            nn.Conv2d(in_channels, exp_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(exp_channels),
            nn.ReLU6(inplace=True),
            # Project
            nn.Conv2d(exp_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # Shortcut connection
        self.use_shortcut = in_channels == out_channels

    def forward(self, x):
        out = self.conv(x)
        if self.use_shortcut:
            out = out + x
        return out


class LightweightConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_repvgg=True):
        super(LightweightConvBlock, self).__init__()
        mid_channels = max(16, out_channels // 2)

        if use_repvgg:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, 1),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                RepVGGBlock(mid_channels, out_channels)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, 1),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                FusedMBConv(mid_channels, out_channels)
            )

    def forward(self, x):
        return self.conv(x)


class ImprovedSegNet(nn.Module):
    def __init__(self, num_classes=21, use_repvgg=True):
        super(ImprovedSegNet, self).__init__()

        # Encoder
        self.enc1 = LightweightConvBlock(3, 44, use_repvgg)
        self.enc2 = LightweightConvBlock(44, 88, use_repvgg)
        self.enc3 = LightweightConvBlock(88, 176, use_repvgg)
        self.enc4 = LightweightConvBlock(176, 352, use_repvgg)

        # Bridge
        self.bridge = LightweightConvBlock(352, 704, use_repvgg)

        # Decoder
        self.dec4 = LightweightConvBlock(704 + 352, 352, use_repvgg)
        self.dec3 = LightweightConvBlock(352 + 176, 176, use_repvgg)
        self.dec2 = LightweightConvBlock(176 + 88, 88, use_repvgg)
        self.dec1 = LightweightConvBlock(88 + 44, 44, use_repvgg)

        # SE modules
        self.se1 = LightSEBlock(44)
        self.se2 = LightSEBlock(88)
        self.se3 = LightSEBlock(176)
        self.se4 = LightSEBlock(352)

        self.final = nn.Conv2d(44, num_classes, 1)
        self.dropout = nn.Dropout2d(0.15)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # Encoder
        enc1 = self.se1(self.enc1(x))
        enc2 = self.se2(self.enc2(self.pool(enc1)))
        enc3 = self.se3(self.enc3(self.pool(enc2)))
        enc4 = self.se4(self.enc4(self.pool(enc3)))

        # Bridge
        bridge = self.dropout(self.bridge(self.pool(enc4)))

        # Decoder
        dec4 = self.dec4(torch.cat([F.interpolate(bridge, size=enc4.shape[2:],
                                                  mode='bilinear', align_corners=True), enc4], 1))
        dec3 = self.dec3(torch.cat([F.interpolate(dec4, size=enc3.shape[2:],
                                                  mode='bilinear', align_corners=True), enc3], 1))
        dec2 = self.dec2(torch.cat([F.interpolate(dec3, size=enc2.shape[2:],
                                                  mode='bilinear', align_corners=True), enc2], 1))
        dec1 = self.dec1(torch.cat([F.interpolate(dec2, size=enc1.shape[2:],
                                                  mode='bilinear', align_corners=True), enc1], 1))

        # Output
        final = self.final(dec1)
        return F.interpolate(final, size=x.shape[2:], mode='bilinear', align_corners=True)

    def switch_to_deploy(self):
        """将RepVGG块转换为推理模式"""
        for module in self.modules():
            if isinstance(module, RepVGGBlock):
                module.switch_to_deploy()