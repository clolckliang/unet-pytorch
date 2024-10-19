import torch
from nets.unet import Unet  # 根据你的模型文件路径导入

# Model parameters based on the config file
num_classes = 4  # 根据你的配置文件设置
pretrained = False
backbone = "vgg"  # 根据你的配置文件设置

# Instantiate the model
model = Unet(num_classes=num_classes, pretrained=pretrained, backbone=backbone)

# Load the model state dict from a .pth file
model_path = "model_data/result_model/best_epoch_weights.pth"  # 根据实际路径调整
model.load_state_dict(torch.load(model_path, weights_only=True), strict=False)

# Set the model to evaluation mode
model.eval()

# Print model parameters before quantization
print("Model parameters before quantization:")
for name, param in model.named_parameters():
    print(f"{name}: {param.size()}")

# Set up the quantization configuration
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

# Prepare the model for quantization
torch.quantization.prepare(model, inplace=True)

# Dummy input for calibration (adjust size if necessary)
dummy_input = torch.randn(1, 3, 256, 256)  # Example input shape
with torch.no_grad():
    model(dummy_input)  # Forward pass to calibrate

# Convert the model to a quantized version
torch.quantization.convert(model, inplace=True)

# Calculate the number of parameters after quantization
num_parameters = sum(p.numel() for p in model.parameters())

# Print model parameters after quantization
print("Model parameters after quantization:")
for name, param in model.named_parameters():
    print(f"{name}: {param.size()}")

# Check if the model has parameters after quantization
if num_parameters == 0:
    print("The model has no parameters after quantization. Please check the model structure.")
else:
    print(f'Total number of parameters (quantized model): {num_parameters}')

# Save the quantized model
quantized_model_path = "model_data/unet_vgg_quantized.pth"
torch.save(model.state_dict(), quantized_model_path)

print(f'Quantized model saved to {quantized_model_path}')
