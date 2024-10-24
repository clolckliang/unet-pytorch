import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce


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

# SKConv implementation from the second file
class SKConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, M=2, r=16, L=32):
        super(SKConv, self).__init__()
        d = max(in_channels//r, L)
        self.M = M
        self.out_channels = out_channels
        self.conv = nn.ModuleList()
        for i in range(M):
            self.conv.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride, padding=1+i,
                         dilation=1+i, groups=32, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Conv2d(out_channels, d, 1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Conv2d(d, out_channels*M, 1, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        batch_size = input.size(0)
        output = []
        for conv in self.conv:
            output.append(conv(input))
        U = reduce(lambda x,y: x+y, output)
        s = self.global_pool(U)
        z = self.fc1(s)
        a_b = self.fc2(z)
        a_b = a_b.reshape(batch_size, self.M, self.out_channels, -1)
        a_b = self.softmax(a_b)
        a_b = list(a_b.chunk(self.M, dim=1))
        a_b = list(map(lambda x: x.reshape(batch_size, self.out_channels, 1, 1), a_b))
        V = list(map(lambda x,y: x*y, output, a_b))
        V = reduce(lambda x,y: x+y, V)
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

class ModifiedSelfNet(nn.Module):
    def __init__(self, num_classes=21):
        super(ModifiedSelfNet, self).__init__()

        # Encoder - 使用SK blocks替换原来的LightConvBlock
        self.enc1 = LightSKBlock(3, 44)
        self.enc2 = LightSKBlock(44, 88)
        self.enc3 = LightSKBlock(88, 176)
        self.enc4 = LightSKBlock(176, 352)

        # Bridge
        self.bridge = LightSKBlock(352, 704)

        # Decoder - with skip connections
        self.dec4 = LightSKBlock(704 + 352, 352)
        self.dec3 = LightSKBlock(352 + 176, 176)
        self.dec2 = LightSKBlock(176 + 88, 88)
        self.dec1 = LightSKBlock(88 + 44, 44)

        # Final output layer
        self.final = nn.Conv2d(44, num_classes, 1)

        # Dropout for regularization
        self.dropout = nn.Dropout2d(0.15)

        # Max pooling
        self.pool = nn.MaxPool2d(2, 2)

        # SE modules可以保留，与SK模块互补
        self.se1 = LightSEBlock(44)
        self.se2 = LightSEBlock(88)
        self.se3 = LightSEBlock(176)
        self.se4 = LightSEBlock(352)

    def forward(self, x):
        # Encoder
        enc1 = self.se1(self.enc1(x))
        enc2 = self.se2(self.enc2(self.pool(enc1)))
        enc3 = self.se3(self.enc3(self.pool(enc2)))
        enc4 = self.se4(self.enc4(self.pool(enc3)))

        # Bridge
        bridge = self.dropout(self.bridge(self.pool(enc4)))

        # Decoder with skip connections
        dec4 = self.dec4(torch.cat([F.interpolate(bridge, size=enc4.shape[2:],
                                                 mode='bilinear', align_corners=True), enc4], 1))
        dec3 = self.dec3(torch.cat([F.interpolate(dec4, size=enc3.shape[2:],
                                                 mode='bilinear', align_corners=True), enc3], 1))
        dec2 = self.dec2(torch.cat([F.interpolate(dec3, size=enc2.shape[2:],
                                                 mode='bilinear', align_corners=True), enc2], 1))
        dec1 = self.dec1(torch.cat([F.interpolate(dec2, size=enc1.shape[2:],
                                                 mode='bilinear', align_corners=True), enc1], 1))

        # Final output
        final = self.final(dec1)
        return F.interpolate(final, size=x.shape[2:], mode='bilinear', align_corners=True)