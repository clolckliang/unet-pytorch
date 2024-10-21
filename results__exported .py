import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import time
import torch
# baseline_model
from unet import Unet as baseline_model
# myselfs' model
from UltraLightweightUnet_config import Unet as UltraLightweightUnet
from utils.utils_metrics import compute_mIoU, show_results, compute_mIoU_npy

def get_model_info():
    """Get model parameters for baseline and custom models."""

    # Import model definitions
    from nets.unet import Unet  # 确保你根据实际情况修改这个路径
    from nets.UltraLightweightUnet import UltraLightweightUnet

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
    my_model = UltraLightweightUnet(num_classes=num_classes)
    my_model_path = "Submit_result/model.pth"  # 根据实际路径修改
    my_model.load_state_dict(torch.load(my_model_path, map_location='cpu', weights_only=True))  # 使用weights_only=True

    # 计算自定义模型参数数量
    my_model_params = sum(p.numel() for p in my_model.parameters())

    return baseline_params, my_model_params



def calculate_fps(model, image, num_iterations=100):
    """Calculate FPS for a model"""
    start_time = time.time()
    for _ in range(num_iterations):
        _ = model.get_miou_png(image)
    end_time = time.time()
    fps = num_iterations / (end_time - start_time)
    return fps


def save_metrics_to_txt(metrics_dict, save_path):
    """Save metrics dictionary to txt file in the desired format"""
    results = {
        "UNet": {
            "Class1_IoU": metrics_dict.get('UNet_Class1_IoU', 0.0),
            "Class2_IoU": metrics_dict.get('UNet_Class2_IoU', 0.0),
            "Class3_IoU": metrics_dict.get('UNet_Class3_IoU', 0.0),
            "mIoU": metrics_dict.get('UNet_mIoU', 0.0),
            "FPS": metrics_dict.get('UNet_FPS', 0.0),
            "Parameters": metrics_dict.get('UNet_Parameters', 0)
        },
        "OursModel": {
            "Class1_IoU": metrics_dict.get('MyModel_Class1_IoU', 0.0),
            "Class2_IoU": metrics_dict.get('MyModel_Class2_IoU', 0.0),
            "Class3_IoU": metrics_dict.get('MyModel_Class3_IoU', 0.0),
            "mIoU": metrics_dict.get('MyModel_mIoU', 0.0),
            "FPS": metrics_dict.get('MyModel_FPS', 0.0),
            "Parameters": metrics_dict.get('MyModel_Parameters', 0)
        }
    }
    with open(save_path, 'w') as f:
        f.write(str(results))


if __name__ == "__main__":
    miou_mode = 0
    num_classes = 4
    name_classes = ["Background", "Inclusions", "Patches", "Scratches"]
    VOCdevkit_path = 'VOCdevkit'

    image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), 'r').read().splitlines()
    gt_dir = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass/")
    miou_out_path = "Submit_result"
    pred_dir = os.path.join(miou_out_path, 'detection-results')

    # New directories for the required outputs
    baseline_pred_dir = os.path.join(miou_out_path, 'baseline_predictions')
    gt_npy_dir = os.path.join(miou_out_path, 'test_ground_truths')
    model_pred_dir = os.path.join(miou_out_path, 'test_predictions')

    # Dictionary to store all metrics
    metrics_dict = {}

    if miou_mode == 0 or miou_mode == 1:
        for dir_path in [baseline_pred_dir, gt_npy_dir, model_pred_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

        print("Load model.")
        baseline_unet = baseline_model()
        my_unet = UltraLightweightUnet()
        print("Load model done.")

        # Calculate model parameters using the verified method
        print("Calculating model parameters...")
        try:
            baseline_params, my_model_params = get_model_info()
            print(f"Baseline UNet parameters: {baseline_params}")
            print(f"My model parameters: {my_model_params}")
        except Exception as e:
            print(f"Error calculating parameters: {e}")
            baseline_params = 0
            my_model_params = 0

        # Calculate FPS using the first image
        print("Calculating FPS...")
        first_image_path = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/" + image_ids[0] + ".jpg")
        first_image = Image.open(first_image_path)
        baseline_fps = calculate_fps(baseline_unet, first_image)
        my_model_fps = calculate_fps(my_unet, first_image)
        print(f"Baseline UNet FPS: {baseline_fps:.2f}")
        print(f"My model FPS: {my_model_fps:.2f}")

        print("Get predict result.")
        for index, image_id in enumerate(tqdm(image_ids), start=1):
            file_name = f"prediction_{index:06d}.npy"
            gt_file_name = f"ground_truth_{index:06d}.npy"

            image_path = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/" + image_id + ".jpg")
            image = Image.open(image_path)

            # Get predictions and save
            baseline_pred = baseline_unet.get_miou_png(image)
            baseline_pred_np = np.array(baseline_pred)
            np.save(os.path.join(baseline_pred_dir, file_name), baseline_pred_np)

            gt_path_npy = os.path.join(gt_dir, f"{image_id}.png")
            gt = Image.open(gt_path_npy)
            gt_np = np.array(gt)
            np.save(os.path.join(gt_npy_dir, gt_file_name), gt_np)

            model_pred = my_unet.get_miou_png(image)
            model_pred_np = np.array(model_pred)
            np.save(os.path.join(model_pred_dir, file_name), model_pred_np)

        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou for baseline model.")
        baseline_hist, baseline_IoUs, baseline_PA_Recall, baseline_Precision = compute_mIoU_npy(
            gt_npy_dir, baseline_pred_dir, image_ids, num_classes, name_classes)

        print("Get miou for my model.")
        my_hist, my_IoUs, my_PA_Recall, my_Precision = compute_mIoU_npy(
            gt_npy_dir, model_pred_dir, image_ids, num_classes, name_classes)

        if baseline_hist is None or my_hist is None:
            print("Error: Unable to compute metrics due to processing failures.")
        else:
            # Store metrics in dictionary
            metrics_dict = {
                'UNet_Class1_IoU': float(baseline_IoUs[1]),  # Skip background class
                'UNet_Class2_IoU': float(baseline_IoUs[2]),
                'UNet_Class3_IoU': float(baseline_IoUs[3]),
                'UNet_mIoU': float(np.nanmean(baseline_IoUs[1:])),  # Calculate mIoU excluding background
                'UNet_FPS': float(baseline_fps),
                'UNet_Parameters': int(baseline_params),
                'MyModel_Class1_IoU': float(my_IoUs[1]),  # Skip background class
                'MyModel_Class2_IoU': float(my_IoUs[2]),
                'MyModel_Class3_IoU': float(my_IoUs[3]),
                'MyModel_mIoU': float(np.nanmean(my_IoUs[1:])),  # Calculate mIoU excluding background
                'MyModel_FPS': float(my_model_fps),
                'MyModel_Parameters': int(my_model_params)
            }

            # Save metrics to txt file
            metrics_txt_path = os.path.join(miou_out_path, '关键指标数据文档.txt')
            print("关键指标数据文档.txt输出完成")
            save_metrics_to_txt(metrics_dict, metrics_txt_path)

            # Show results for both models
            print("Baseline UNet Results:")
            show_results(miou_out_path, baseline_hist, baseline_IoUs, baseline_PA_Recall, baseline_Precision,
                         name_classes)
            print("\nMy Model Results:")
            show_results(miou_out_path, my_hist, my_IoUs, my_PA_Recall, my_Precision, name_classes)