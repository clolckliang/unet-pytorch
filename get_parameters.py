import torch
import numpy as np
from nets.unet import Unet  # 确保你根据实际情况修改这个路径
from nets.LightWeightUnet import LightweightUnet
from nets.UltraLightweightUnet import UltraLightweightUnet
# Model parameters based on the config file
num_classes = 4  # 从配置文件中获取
pretrained = False
#backbone = "vgg"  # 从配置文件中获取
backbone = "lightweight_vgg"
# Instantiate the model
#model = Unet(num_classes=num_classes, pretrained=pretrained, backbone=backbone)
model = UltraLightweightUnet(num_classes=num_classes)
# Load the model state dict from a .pth file
model_path = "logs/ep100-loss0.147-val_loss0.238.pth"  # 根据你的模型路径修改
model.load_state_dict(torch.load(model_path, weights_only=True))

# Calculate the number of parameters
num_parameters = sum(p.numel() for p in model.parameters())

# Print the total number of parameters
print(f'Total number of parameters: {num_parameters}')
