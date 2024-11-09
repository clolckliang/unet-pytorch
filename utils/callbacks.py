import os

import matplotlib
import torch
import torch.nn.functional as F
import wandb

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import scipy.signal

import cv2
import shutil
import numpy as np

from PIL import Image
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from .utils import cvtColor, preprocess_input, resize_image
from .utils_metrics import compute_mIoU

from typing import Optional, Union, Dict, Callable

import torch
import os
import warnings
from pathlib import Path


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
            self,
            patience: int = 10,
            verbose: bool = False,
            delta: float = 0,
            save_path: str = 'best_model.pth',
            mode: str = 'min',
            trace_func: Callable = print
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            save_path (str): Path to save the best model.
            mode (str): 'min' for minimize metric, 'max' for maximize metric.
            trace_func (callable): Function for logging/printing information.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.save_path = Path(save_path)  # 使用 Path 对象处理路径
        self.mode = mode
        self.trace_func = trace_func

        # Validate mode
        if mode not in ['min', 'max']:
            raise ValueError(f"mode '{mode}' is not supported. Use 'min' or 'max'")

        # Create directory for save_path if it doesn't exist
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.best_state: Optional[Dict[str, torch.Tensor]] = None

        # Set the direction for optimization
        self.mode_worse = np.Inf if mode == 'min' else -np.Inf
        self.mode_better = -np.Inf if mode == 'min' else np.Inf

    def __call__(
            self,
            val_metric: float,
            model: torch.nn.Module,
            use_wandb: bool = False,
            epoch: Optional[int] = None
    ) -> bool:
        """
        Args:
            val_metric (float): Validation metric to monitor
            model (torch.nn.Module): Model to save
            use_wandb (bool): Whether to use weights & biases logging
            epoch (int, optional): Current epoch number for logging

        Returns:
            bool: True if training should stop early, False otherwise
        """
        score = -val_metric if self.mode == 'min' else val_metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_metric, model, use_wandb)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if epoch is not None:
                    self.trace_func(f'Current epoch: {epoch}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_metric, model, use_wandb)
            self.counter = 0

        return self.early_stop

    def save_checkpoint(
            self,
            val_metric: float,
            model: torch.nn.Module,
            use_wandb: bool = False
    ) -> None:
        """Saves model when validation metric improves."""
        if self.verbose:
            improvement = 'decreased' if self.mode == 'min' else 'increased'
            self.trace_func(f'Validation metric {improvement} ({val_metric:.6f}). Saving model...')

        # Save model state
        self.best_state = {
            k: v.cpu() for k, v in model.state_dict().items()
        }

        try:
            torch.save({
                'model_state_dict': self.best_state,
                'best_score': self.best_score,
                'counter': self.counter,
                'val_metric': val_metric
            }, self.save_path)
        except Exception as e:
            warnings.warn(f"Error saving checkpoint: {str(e)}")

        if use_wandb:
            try:
                import wandb
                wandb.save(str(self.save_path))
                wandb.log({
                    'best_val_metric': val_metric,
                    'early_stopping_counter': self.counter
                })
            except ImportError:
                warnings.warn("wandb not installed. Skipping wandb logging.")
            except Exception as e:
                warnings.warn(f"Error logging to wandb: {str(e)}")

    def load_best_model(self, model: torch.nn.Module) -> None:
        """
        Loads the best model state.

        Args:
            model (torch.nn.Module): Model to load the best state into
        """
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
        elif self.save_path.exists():
            try:
                checkpoint = torch.load(self.save_path)
                model.load_state_dict(checkpoint['model_state_dict'])
            except Exception as e:
                raise RuntimeError(f"Error loading checkpoint: {str(e)}")
        else:
            raise RuntimeError("No best model state found to load")

    def get_best_score(self) -> float:
        """Returns the best score achieved."""
        return self.best_score if self.best_score is not None else self.mode_worse



class LossHistory():
    def __init__(self, log_dir, model, input_shape, val_loss_flag=True):
        self.log_dir        = log_dir
        self.val_loss_flag  = val_loss_flag

        self.losses         = []
        if self.val_loss_flag:
            self.val_loss   = []
        
        os.makedirs(self.log_dir)
        self.writer     = SummaryWriter(self.log_dir)
        try:
            dummy_input     = torch.randn(2, 3, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except:
            pass

    def append_loss(self, epoch, loss, val_loss = None):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        if self.val_loss_flag:
            self.val_loss.append(val_loss)
        
        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        if self.val_loss_flag:
            with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
                f.write(str(val_loss))
                f.write("\n")
            
        self.writer.add_scalar('loss', loss, epoch)
        if self.val_loss_flag:
            self.writer.add_scalar('val_loss', val_loss, epoch)
            
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        if self.val_loss_flag:
            plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
            
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            if self.val_loss_flag:
                plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")

class EvalCallback():
    def __init__(self, net, input_shape, num_classes, image_ids, dataset_path, log_dir, cuda, \
            miou_out_path=".temp_miou_out", eval_flag=True, period=1):
        super(EvalCallback, self).__init__()
        
        self.net                = net
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.image_ids          = image_ids
        self.dataset_path       = dataset_path
        self.log_dir            = log_dir
        self.cuda               = cuda
        self.miou_out_path      = miou_out_path
        self.eval_flag          = eval_flag
        self.period             = period
        
        self.image_ids          = [image_id.split()[0] for image_id in image_ids]
        self.mious      = [0]
        self.epoches    = [0]
        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write(str(0))
                f.write("\n")

    def get_miou_png(self, image):
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data, nw, nh  = resize_image(image, (self.input_shape[1],self.input_shape[0]))
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
                
            #---------------------------------------------------#
            #   图片传入网络进行预测
            #---------------------------------------------------#
            pr = self.net(images)[0]
            #---------------------------------------------------#
            #   取出每一个像素点的种类
            #---------------------------------------------------#
            pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy()
            #--------------------------------------#
            #   将灰条部分截取掉
            #--------------------------------------#
            pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
            #---------------------------------------------------#
            #   进行图片的resize
            #---------------------------------------------------#
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
            #---------------------------------------------------#
            #   取出每一个像素点的种类
            #---------------------------------------------------#
            pr = pr.argmax(axis=-1)
    
        image = Image.fromarray(np.uint8(pr))
        return image

    def get_miou_png_Supervision(self, image):
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            # ---------------------------------------------------#
            #   图片传入网络进行预测
            # ---------------------------------------------------#
            outputs = self.net(images)
            if isinstance(outputs, (tuple, list)):
                pr = outputs[0]  # 仅取主输出
            else:
                pr = outputs

            # ---------------------------------------------------#
            #   确保 pr 是一个 [1, num_classes, H, W] 的张量
            # ---------------------------------------------------#
            if pr.dim() == 4:
                pr = pr.squeeze(0)  # 变为 [num_classes, H, W]
            elif pr.dim() != 3:
                raise ValueError(f"Expected pr to be 3D tensor, but got {pr.dim()}D tensor.")

            # ---------------------------------------------------#
            #   取出每一个像素点的种类
            # ---------------------------------------------------#
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()  # 变为 [H, W, num_classes]

            # --------------------------------------#
            #   将灰条部分截取掉
            # --------------------------------------#
            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh),
                 int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
            # ---------------------------------------------------#
            #   进行图片的resize
            # ---------------------------------------------------#
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
            # ---------------------------------------------------#
            #   取出每一个像素点的种类
            # ---------------------------------------------------#
            pr = pr.argmax(axis=-1)

        image = Image.fromarray(np.uint8(pr))
        return image

    def on_epoch_end(self, epoch, model_eval):
        if epoch % self.period == 0 and self.eval_flag:
            self.net = model_eval
            gt_dir = os.path.join(self.dataset_path, "DataB/SegmentationClass/")
            pred_dir = os.path.join(self.miou_out_path, 'detection-results')
            if not os.path.exists(self.miou_out_path):
                os.makedirs(self.miou_out_path)
            if not os.path.exists(pred_dir):
                os.makedirs(pred_dir)
            print("Get miou.")
            for image_id in tqdm(self.image_ids):
                image_path = os.path.join(self.dataset_path, "DataB/JPEGImages/" + image_id + ".jpg")
                image = Image.open(image_path)
                try:
                    image = self.get_miou_png(image)
                except Exception as e:
                    print(f'seme error rise{e},try to switch get_miou_png_Supervision')
                    image = self.get_miou_png_Supervision(image)

                pred_dir = '/root/autodl-tmp/unet-pytorch/.temp_miou_out/detection-results/'
                if not os.path.exists(pred_dir):
                    os.makedirs(pred_dir)

                image.save(os.path.join(pred_dir, image_id + ".png"))

            print("Calculate miou.")
            _, IoUs, _, _ = compute_mIoU(gt_dir, pred_dir, self.image_ids, self.num_classes, None)

            # 初始化所需的列表（只在第一次调用时进行）
            if not hasattr(self, 'class_ious'):
                self.class_ious = {}
                for i in range(1, self.num_classes):
                    self.class_ious[i] = []
            if not hasattr(self, 'epoches'):
                self.epoches = []
            if not hasattr(self, 'mious'):
                self.mious = []

            # 当前epoch
            curr_epoch = epoch + 1

            # 如果是第一个epoch，需要特殊处理
            if len(self.epoches) == 0:
                self.epoches = [curr_epoch]
                self.mious = []
                for i in range(1, self.num_classes):
                    self.class_ious[i] = []
            elif curr_epoch not in self.epoches:
                self.epoches.append(curr_epoch)

            # 存储IoU值并记录到wandb
            temp_ious = []
            wandb_data = {}  # 用于存储要记录到wandb的数据

            for class_id in range(1, self.num_classes):
                iou = IoUs[class_id] * 100
                temp_ious.append(iou)

                # 确保class_ious[class_id]的长度与epoches相同
                while len(self.class_ious[class_id]) < len(self.epoches) - 1:
                    self.class_ious[class_id].append(0)
                self.class_ious[class_id].append(iou)

                # 添加到wandb数据
                wandb_data[f'metrics/class_{class_id}_iou'] = iou

            # 计算并存储平均mIoU
            temp_miou = np.mean(temp_ious)
            if len(self.mious) < len(self.epoches):
                self.mious.append(temp_miou)

            # 添加平均mIoU到wandb数据
            wandb_data['metrics/mean_iou'] = temp_miou

            # 保存到文件
            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write(f"Epoch {curr_epoch}: mean_mIoU={temp_miou:.2f}\n")
                for class_id in range(1, self.num_classes):
                    f.write(f"class{class_id}={IoUs[class_id] * 100:.2f}, ")
                f.write("\n")

            # 确保所有数据长度一致
            for class_id in range(1, self.num_classes):
                while len(self.class_ious[class_id]) < len(self.epoches):
                    self.class_ious[class_id].append(0)

            # 绘图
            try:
                colors = plt.cm.rainbow(np.linspace(0, 1, self.num_classes - 1))
                plt.figure(figsize=(12, 8))

                # 绘制每个类别的IoU曲线
                for class_id in range(1, self.num_classes):
                    plt.plot(self.epoches, self.class_ious[class_id],
                             color=colors[class_id - 1],
                             linewidth=2,
                             label=f'Class {class_id} IoU')

                # 绘制平均IoU曲线
                plt.plot(self.epoches, self.mious,
                         color='black',
                         linewidth=2,
                         linestyle='--',
                         label='Mean IoU')

                plt.grid(True)
                plt.xlabel('Epoch')
                plt.ylabel('IoU (%)')
                plt.title(f'IoU Curves for {self.num_classes - 1} Classes')

                # 调整图例
                if self.num_classes > 10:
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
                else:
                    plt.legend(loc="upper right")

                plt.tight_layout()

                # 保存图片
                plot_path = os.path.join(self.log_dir, "epoch_miou.png")
                plt.savefig(plot_path, bbox_inches='tight', dpi=300)

                # 如果启用了wandb，记录图表
                if hasattr(self, 'use_wandb') and self.use_wandb:
                    # 创建wandb的图表对象
                    iou_plot = wandb.Image(plot_path, caption=f"IoU Curves - Epoch {curr_epoch}")
                    wandb_data['visualizations/iou_curves'] = iou_plot

                    # 创建wandb的折线图
                    wandb_data['charts/iou_curves'] = {
                        'epoch': curr_epoch,
                        'mean_iou': temp_miou,
                        **{f'class_{i}_iou': self.class_ious[i][-1] for i in range(1, self.num_classes)}
                    }

                    # 记录所有数据到wandb
                    wandb.log(wandb_data)

                plt.cla()
                plt.close("all")

            except Exception as e:
                print(f"Warning: Error while plotting: {str(e)}")
                print(f"Current data lengths - Epochs: {len(self.epoches)}, mIoUs: {len(self.mious)}")
                for class_id in range(1, self.num_classes):
                    print(f"Class {class_id} IoUs: {len(self.class_ious[class_id])}")

            print("Get miou done.")
            miou_out_path = self.miou_out_path  # 假设这是你用的路径
            if os.path.exists(miou_out_path):
                shutil.rmtree(miou_out_path)
            else:
                print(f"目录 {miou_out_path} 不存在，跳过删除操作")