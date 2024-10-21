import torch


def get_model_info():
    """Get model parameters for baseline and custom models."""

    # Import model definitions
    from nets.unet import Unet  # 确保你根据实际情况修改这个路径
    from nets.UltraLightweightUnet_large import UltraLightweightUnet_large

    # 基准模型参数
    num_classes = 4
    pretrained = False
    backbone = "vgg"  # 这里的backbone可以根据需要修改

    # 实例化基准模型
    baseline_model = Unet(num_classes=num_classes, pretrained=pretrained, backbone=backbone)
    baseline_model_path = "model_data/result_model/best_unet_baseline.pth"
    baseline_model.load_state_dict(
        torch.load(baseline_model_path, map_location='cpu', weights_only=True))  # 使用weights_only=True

    # 计算基准模型参数数量
    baseline_params = sum(p.numel() for p in baseline_model.parameters())

    # 自定义模型参数
    my_model = UltraLightweightUnet_large(num_classes=num_classes)
    my_model_path = "logs/ep005-loss0.792-val_loss0.753.pth"  # 根据实际路径修改
    my_model.load_state_dict(torch.load(my_model_path, map_location='cpu', weights_only=True))  # 使用weights_only=True

    # 计算自定义模型参数数量
    my_model_params = sum(p.numel() for p in my_model.parameters())

    return baseline_params, my_model_params


# 调用示例
baseline_params, my_model_params = get_model_info()
print(f'Baseline model parameters: {baseline_params}')
print(f'My model parameters: {my_model_params}')
