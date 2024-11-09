import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
import torchvision


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


class SKConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, M=2, r=16, L=32):
        super(SKConv, self).__init__()
        d = max(in_channels // r, L)
        self.M = M
        self.out_channels = out_channels
        self.conv = nn.ModuleList()
        for i in range(M):
            self.conv.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride, padding=1 + i,
                          dilation=1 + i, groups=32, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Conv2d(out_channels, d, 1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Conv2d(d, out_channels * M, 1, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        batch_size = input.size(0)
        output = []
        for conv in self.conv:
            output.append(conv(input))
        U = reduce(lambda x, y: x + y, output)
        s = self.global_pool(U)
        z = self.fc1(s)
        a_b = self.fc2(z)
        a_b = a_b.reshape(batch_size, self.M, self.out_channels, -1)
        a_b = self.softmax(a_b)
        a_b = list(a_b.chunk(self.M, dim=1))
        a_b = list(map(lambda x: x.reshape(batch_size, self.out_channels, 1, 1), a_b))
        V = list(map(lambda x, y: x * y, output, a_b))
        V = reduce(lambda x, y: x + y, V)
        return V


class LightSKBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LightSKBlock, self).__init__()
        mid_channels = max(16, out_channels // 2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.sk_conv = SKConv(mid_channels, out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.sk_conv(x)
        x = self.bn(x)
        return self.relu(x)


class AttentionGatedDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            LightweightConvBlock(in_channels, out_channels),
            LightweightConvBlock(out_channels, out_channels)
        )
        self.attention = LightweightSpatialAttention()

    def forward(self, x, skip=None):
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.attention(x)
        return


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积"""
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
    """轻量级卷积块"""
    def __init__(self, in_channels, out_channels):
        super(LightConvBlock, self).__init__()
        mid_channels = max(16, out_channels // 2)  # 保持中间通道数计算方式
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(mid_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class ChannelAttention(nn.Module):
    """通道注意力"""
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.shared_mlp(self.avg_pool(x).view(b, c))
        max_out = self.shared_mlp(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        return self.sigmoid(out).view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    """空间注意力"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))


class CBAM(nn.Module):
    """CBAM注意力模块"""
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class CRFSAttention(nn.Module):
    """CRFS注意力模块"""
    def __init__(self, in_channels, reduction=16):
        super(CRFSAttention, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction

        self.conv_theta = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False)
        self.conv_phi = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False)
        self.conv_g = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False)
        self.conv_attn = nn.Conv2d(in_channels // reduction, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, _, h, w = x.size()

        # 计算位置关系矩阵
        theta = self.conv_theta(x)
        phi = self.conv_phi(x)
        g = self.conv_g(x)
        attn = self.conv_attn(torch.sigmoid(theta + phi))

        # 应用CRFs注意力
        attn = attn.view(batch_size, 1, h, w)
        x_attn = x * attn

        return self.sigmoid(x_attn)

class EdgeEnhancementBlock(nn.Module):
    """边界增强卷积块"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class AtrousConvBlock(nn.Module):
    """空洞卷积块, 使用深度可分离卷积和空洞卷积"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride,
                                   padding=padding, dilation=dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


class LightweightConvBlock(nn.Module):
    """轻量级卷积块,使用深度可分离卷积"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels,
                                   bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


class OptimizedMultiScaleBlock(nn.Module):
    """优化多尺度块,使用轻量级卷积块和空洞卷积块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        branch_channels = out_channels // 4

        self.reduce = nn.Conv2d(in_channels, branch_channels * 2, 1)
        self.branch1 = LightweightConvBlock(branch_channels * 2, branch_channels * 2)
        self.branch2 = nn.Sequential(
            LightweightConvBlock(branch_channels * 2, branch_channels * 2),
            LightweightConvBlock(branch_channels * 2, branch_channels * 2, kernel_size=3, padding=2, stride=1)
        )

    def forward(self, x):
        x = self.reduce(x)
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        size = branch1.shape[2:]
        branch2 = F.interpolate(branch2, size=size, mode='bilinear', align_corners=True)
        return torch.cat([branch1, branch2], dim=1)


class LightweightSpatialAttention(nn.Module):
    """轻量级空间注意力模块,使用全局平均池化和全局最大池化"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(x_cat))
        return x * out


class LightweightSEBlock(nn.Module):
    """轻量级SE块,使用全局平均池化和全连接层"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduced_channels = max(8, channels // reduction)
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


class DecoderBlock(nn.Module):
    """Efficient decoder block with skip connection and SE attention"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels),
            DepthwiseSeparableConv(out_channels, out_channels)
        )
        self.se = LightweightSEBlock(out_channels)

    def forward(self, x, skip=None):
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.se(x)
        return x


class STDCBlock(nn.Module):
    """Modified STDC block with SE attention"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        mid_channels = out_channels // 2

        self.conv1 = DepthwiseSeparableConv(in_channels, mid_channels, stride=stride)
        self.conv2 = DepthwiseSeparableConv(mid_channels, mid_channels)
        self.se = LightweightSEBlock(out_channels)

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


class EfficientAttention(nn.Module):
    """更轻量级的通道注意力模块"""

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        reduced_channels = max(8, channels // reduction)

        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.shared_mlp(self.avg_pool(x).view(b, c))
        max_out = self.shared_mlp(self.max_pool(x).view(b, c))
        y = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * y.expand_as(x)


class EfficientDecoderBlock(nn.Module):
    """解码器块"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            LightweightConvBlock(in_channels, out_channels),
            LightweightConvBlock(out_channels, out_channels)
        )
        self.channel_attention = EfficientAttention(out_channels)
        self.spatial_attention = LightweightSpatialAttention()

    def forward(self, x, skip=None):
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class OptimizedBalancedSegWithFPN_Supervision(nn.Module):
    def __init__(self, num_classes=3):
        super(OptimizedBalancedSegWithFPN_Supervision,self).__init__()
        # 编码器块
        self.enc1 = OptimizedMultiScaleBlock(3, 32)
        self.enc2 = OptimizedMultiScaleBlock(32, 64)
        self.enc3 = OptimizedMultiScaleBlock(64, 128)
        self.enc4 = OptimizedMultiScaleBlock(128, 256)

        # 桥接层
        self.bridge = nn.Sequential(
            OptimizedMultiScaleBlock(256, 512),
            nn.Dropout2d(0.1)
        )

        # 横向连接
        self.lateral4 = nn.Conv2d(256, 256, kernel_size=1)
        self.lateral3 = nn.Conv2d(128, 128, kernel_size=1)
        self.lateral2 = nn.Conv2d(64, 64, kernel_size=1)
        self.lateral1 = nn.Conv2d(32, 32, kernel_size=1)

        # 解码器块
        self.dec4 = EfficientDecoderBlock(512, 256)
        self.dec3 = EfficientDecoderBlock(256 + 128, 128)
        self.dec2 = EfficientDecoderBlock(128 + 64, 64)
        self.dec1 = EfficientDecoderBlock(64 + 32, 32)

        # 边界增强块
        self.edge_enhancement = EdgeEnhancementBlock(32, 32)

        # 最终卷积层
        self.final_conv = nn.Sequential(
            LightweightConvBlock(32, 32),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

        # 深度监督的辅助输出层
        self.aux_output3 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.aux_output2 = nn.Conv2d(64, num_classes, kernel_size=1)
        self.aux_output1 = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # 编码器路径
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))

        # 桥接层
        bridge = self.bridge(F.max_pool2d(enc4, 2))

        # 横向连接
        lat4 = self.lateral4(enc4)
        lat3 = self.lateral3(enc3)
        lat2 = self.lateral2(enc2)
        lat1 = self.lateral1(enc1)

        # 解码器路径
        dec4 = self.dec4(F.interpolate(bridge, size=enc4.shape[2:], mode='bilinear', align_corners=True))
        dec4 = F.interpolate(lat4, size=dec4.shape[2:], mode='bilinear', align_corners=True) + dec4

        dec3 = self.dec3(F.interpolate(dec4, size=enc3.shape[2:], mode='bilinear', align_corners=True), enc3)
        dec3 = F.interpolate(lat3, size=dec3.shape[2:], mode='bilinear', align_corners=True) + dec3

        dec2 = self.dec2(F.interpolate(dec3, size=enc2.shape[2:], mode='bilinear', align_corners=True), enc2)
        dec2 = F.interpolate(lat2, size=dec2.shape[2:], mode='bilinear', align_corners=True) + dec2

        dec1 = self.dec1(F.interpolate(dec2, size=enc1.shape[2:], mode='bilinear', align_corners=True), enc1)
        dec1 = F.interpolate(lat1, size=dec1.shape[2:], mode='bilinear', align_corners=True) + dec1

        # 边界增强
        enhanced_edges = self.edge_enhancement(dec1)

        # 最终输出
        final = self.final_conv(dec1 + enhanced_edges)
        final = F.interpolate(final, size=x.shape[2:], mode='bilinear', align_corners=True)

        # 深度监督的辅助输出
        aux_out3 = self.aux_output3(dec3)
        aux_out3 = F.interpolate(aux_out3, size=x.shape[2:], mode='bilinear', align_corners=True)

        aux_out2 = self.aux_output2(dec2)
        aux_out2 = F.interpolate(aux_out2, size=x.shape[2:], mode='bilinear', align_corners=True)

        aux_out1 = self.aux_output1(dec1)
        aux_out1 = F.interpolate(aux_out1, size=x.shape[2:], mode='bilinear', align_corners=True)

        # 返回主输出和辅助输出
        return final, aux_out1, aux_out2, aux_out3






class OptimizedBalancedSegWithFPN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        # 编码器块
        self.enc1 = OptimizedMultiScaleBlock(3, 32)  # 输出通道数：32
        self.enc2 = OptimizedMultiScaleBlock(32, 64)  # 输出通道数：64
        self.enc3 = OptimizedMultiScaleBlock(64, 128)  # 输出通道数：128
        self.enc4 = OptimizedMultiScaleBlock(128, 256)  # 输出通道数：256

        # 桥接层
        self.bridge = nn.Sequential(
            OptimizedMultiScaleBlock(256, 512),
            nn.Dropout2d(0.1)
        )

        # 横向连接 (特征金字塔式横向连接)
        # 修改输入通道数以匹配编码器的输出通道数
        self.lateral4 = nn.Conv2d(256, 256, kernel_size=1)  # 输入通道应为 enc4 的输出通道 256
        self.lateral3 = nn.Conv2d(128, 128, kernel_size=1)  # 输入通道应为 enc3 的输出通道 128
        self.lateral2 = nn.Conv2d(64, 64, kernel_size=1)  # 输入通道应为 enc2 的输出通道 64
        self.lateral1 = nn.Conv2d(32, 32, kernel_size=1)  # 输入通道应为 enc1 的输出通道 32

        # 解码器块，使用密集连接
        self.dec4 = EfficientDecoderBlock(512, 256)
        self.dec3 = EfficientDecoderBlock(256 + 128, 128)
        self.dec2 = EfficientDecoderBlock(128 + 64, 64)
        self.dec1 = EfficientDecoderBlock(64 + 32, 32)

        # 边界增强块
        self.edge_enhancement = EdgeEnhancementBlock(32, 32)

        # 最终卷积层
        self.final_conv = nn.Sequential(
            LightweightConvBlock(32, 32),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # 编码器路径
        enc1 = self.enc1(x)  # 输出特征维度：32
        enc2 = self.enc2(F.max_pool2d(enc1, 2))  # 输出特征维度：64
        enc3 = self.enc3(F.max_pool2d(enc2, 2))  # 输出特征维度：128
        enc4 = self.enc4(F.max_pool2d(enc3, 2))  # 输出特征维度：256

        # 桥接层
        bridge = self.bridge(F.max_pool2d(enc4, 2))  # 输出特征维度：512

        # 特征金字塔融合 (FPN)
        lat4 = self.lateral4(enc4)  # 横向连接特征 256
        lat3 = self.lateral3(enc3)  # 横向连接特征 128
        lat2 = self.lateral2(enc2)  # 横向连接特征 64
        lat1 = self.lateral1(enc1)  # 横向连接特征 32

        # 解码器路径，融合特征金字塔
        dec4 = self.dec4(F.interpolate(bridge, size=enc4.shape[2:], mode='bilinear', align_corners=True))
        dec4 = F.interpolate(lat4, size=dec4.shape[2:], mode='bilinear', align_corners=True) + dec4

        dec3 = self.dec3(F.interpolate(dec4, size=enc3.shape[2:], mode='bilinear', align_corners=True), enc3)
        dec3 = F.interpolate(lat3, size=dec3.shape[2:], mode='bilinear', align_corners=True) + dec3

        dec2 = self.dec2(F.interpolate(dec3, size=enc2.shape[2:], mode='bilinear', align_corners=True), enc2)
        dec2 = F.interpolate(lat2, size=dec2.shape[2:], mode='bilinear', align_corners=True) + dec2

        dec1 = self.dec1(F.interpolate(dec2, size=enc1.shape[2:], mode='bilinear', align_corners=True), enc1)
        dec1 = F.interpolate(lat1, size=dec1.shape[2:], mode='bilinear', align_corners=True) + dec1

        # 边界增强
        enhanced_edges = self.edge_enhancement(dec1)

        # 最终输出
        final = self.final_conv(dec1 + enhanced_edges)
        return F.interpolate(final, size=x.shape[2:], mode='bilinear', align_corners=True)


class OptimizedBalancedSeg(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.edge_enhancement = EdgeEnhancementBlock(32, 32)  # 参数根据需要调整
        self.enc1 = OptimizedMultiScaleBlock(3, 32)
        self.enc2 = OptimizedMultiScaleBlock(32, 64)
        self.enc3 = OptimizedMultiScaleBlock(64, 128)
        # self.enc3 = AtrousConvBlock(64, 128, dilation=2)  # 使用空洞卷积
        self.enc4 = OptimizedMultiScaleBlock(128, 256)

        self.bridge = nn.Sequential(
            OptimizedMultiScaleBlock(256, 512),
            nn.Dropout2d(0.1)
        )

        self.dec4 = EfficientDecoderBlock(512 + 256, 256)
        self.dec3 = EfficientDecoderBlock(256 + 128, 128)
        self.dec2 = EfficientDecoderBlock(128 + 64, 64)
        self.dec1 = EfficientDecoderBlock(64 + 32, 32)

        self.final_conv = nn.Sequential(
            LightweightConvBlock(32, 32),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

        self.enc1_se = LightweightSEBlock(32)
        self.enc2_se = LightweightSEBlock(64)
        self.enc3_se = LightweightSEBlock(128)
        self.enc4_se = LightweightSEBlock(256)

        # CBAM 注意力模块集成
        self.enc1_cbam = CBAM(32)
        self.enc2_cbam = CBAM(64)
        self.enc3_cbam = CBAM(128)
        self.enc4_cbam = CBAM(256)

        self.dec4_spa = LightweightSpatialAttention()
        self.dec3_spa = LightweightSpatialAttention()
        self.dec2_spa = LightweightSpatialAttention()
        self.dec1_spa = LightweightSpatialAttention()

        self.aux_head = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1_se(self.enc1(x))
        enc2 = self.enc2_se(self.enc2(F.max_pool2d(enc1, 2)))
        enc3 = self.enc3_se(self.enc3(F.max_pool2d(enc2, 2)))
        enc4 = self.enc4_se(self.enc4(F.max_pool2d(enc3, 2)))

        bridge = self.bridge(F.max_pool2d(enc4, 2))

        dec4 = self.dec4(F.interpolate(bridge, size=enc4.shape[2:]), enc4)
        dec4 = self.dec4_spa(dec4)
        dec3 = self.dec3(F.interpolate(dec4, size=enc3.shape[2:]), enc3)
        dec3 = self.dec3_spa(dec3)
        dec2 = self.dec2(F.interpolate(dec3, size=enc2.shape[2:]), enc2)
        dec2 = self.dec2_spa(dec2)
        dec1 = self.dec1(F.interpolate(dec2, size=enc1.shape[2:]), enc1)
        dec1 = self.dec1_spa(dec1)
        # enhanced_edges = self.edge_enhancement(dec1)
        # final = self.final_conv(dec1 + enhanced_edges)  # Combine features with edge-enhanced features
        final = self.final_conv(dec1)
        aux = F.interpolate(self.aux_head(enc3), size=x.shape[2:])
        return F.interpolate(final, size=x.shape[2:]), aux


class UltraLightweightUnet(nn.Module):
    def __init__(self, num_classes=21):
        super(UltraLightweightUnet, self).__init__()

        # Encoder - 微调通道数以达到目标参数量
        self.enc1 = LightConvBlock(3, 44)
        self.enc2 = LightConvBlock(44, 88)
        self.enc3 = LightConvBlock(88, 176)
        self.enc4 = LightConvBlock(176, 352)

        # Bridge
        self.bridge = LightConvBlock(352, 704)

        # Decoder - with skip connections
        self.dec4 = LightConvBlock(704 + 352, 352)
        self.dec3 = LightConvBlock(352 + 176, 176)
        self.dec2 = LightConvBlock(176 + 88, 88)
        self.dec1 = LightConvBlock(88 + 44, 44)

        # Final output layer
        self.final = nn.Conv2d(44, num_classes, 1)

        # Dropout for regularization
        self.dropout = nn.Dropout2d(0.15)  # 保持dropout率不变

        # Max pooling
        self.pool = nn.MaxPool2d(2, 2)

        # SE模块
        self.se1 = LightweightSEBlock(44)
        self.se2 = LightweightSEBlock(88)
        self.se3 = LightweightSEBlock(176)
        self.se4 = LightweightSEBlock(352)

    def forward(self, x):
        # Encoder
        enc1 = self.se1(self.enc1(x))
        enc2 = self.se2(self.enc2(self.pool(enc1)))
        enc3 = self.se3(self.enc3(self.pool(enc2)))
        enc4 = self.se4(self.enc4(self.pool(enc3)))

        # Bridge
        bridge = self.dropout(self.bridge(self.pool(enc4)))

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


class TraditionalUnet(nn.Module):
    """传统Unet模型"""
    def __init__(self, in_channels=3, num_classes=21):
        super(TraditionalUnet, self).__init__()

        # 减少基础通道数和层数

        # 编码器部分
        self.inc = DoubleConv(in_channels, 22)
        self.down1 = Down(22, 44)
        self.down2 = Down(44, 88)
        self.down3 = Down(88, 176)

        # 解码器部分
        self.up1 = Up(176 + 88, 88)
        self.up2 = Up(88 + 44, 44)
        self.up3 = Up(44 + 22, 22)

        # 最终输出层
        self.outc = nn.Conv2d(22, num_classes, kernel_size=1)

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


class OptimizedBalancedSegWithCRFS(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        # 编码器部分
        self.enc1 = OptimizedMultiScaleBlock(3, 32)
        self.enc2 = OptimizedMultiScaleBlock(32, 64)
        self.enc3 = OptimizedMultiScaleBlock(64, 128)
        self.enc4 = OptimizedMultiScaleBlock(128, 256)

        # 桥接层
        self.bridge = nn.Sequential(
            OptimizedMultiScaleBlock(256, 512),
            nn.Dropout2d(0.1)
        )

        # 解码器部分
        self.dec4 = EfficientDecoderBlock(512 + 256, 256)
        self.dec3 = EfficientDecoderBlock(256 + 128, 128)
        self.dec2 = EfficientDecoderBlock(128 + 64, 64)
        self.dec1 = EfficientDecoderBlock(64 + 32, 32)

        # 引入CRFs注意力模块
        self.crfs_att4 = CRFSAttention(256)
        self.crfs_att3 = CRFSAttention(128)
        self.crfs_att2 = CRFSAttention(64)
        self.crfs_att1 = CRFSAttention(32)

        # 边界增强模块
        self.edge_enhancement = EdgeEnhancementBlock(32, 32)

        # 最终输出层
        self.final_conv = nn.Sequential(
            LightweightConvBlock(32, 32),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # 编码器路径
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))

        # 桥接层
        bridge = self.bridge(F.max_pool2d(enc4, 2))

        # 解码器路径,引入CRFs注意力
        dec4 = self.dec4(F.interpolate(bridge, size=enc4.shape[2:], mode='bilinear', align_corners=True), enc4)
        dec4 = self.crfs_att4(dec4)
        dec3 = self.dec3(F.interpolate(dec4, size=enc3.shape[2:], mode='bilinear', align_corners=True), enc3)
        dec3 = self.crfs_att3(dec3)
        dec2 = self.dec2(F.interpolate(dec3, size=enc2.shape[2:], mode='bilinear', align_corners=True), enc2)
        dec2 = self.crfs_att2(dec2)
        dec1 = self.dec1(F.interpolate(dec2, size=enc1.shape[2:], mode='bilinear', align_corners=True), enc1)
        dec1 = self.crfs_att1(dec1)

        # 边界增强
        enhanced_edges = self.edge_enhancement(dec1)

        # 最终输出
        final = self.final_conv(dec1 + enhanced_edges)
        return F.interpolate(final, size=x.shape[2:], mode='bilinear', align_corners=True)


if __name__ == '__main__':
    model = OptimizedBalancedSegWithFPN_Supervision(num_classes=3)
    dummy_input = torch.randn(1, 3, 256, 256)
    output = model(dummy_input)
    if isinstance(output, tuple):
        print(f'Output shapes: {[o.shape for o in output]}')
    else:
        print(f'Output shape: {output.shape}')
    print(f'Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
