import datetime
import os
from functools import partial
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.ImprovedUltraLightweightUnet import ImprovedUltraLightweightUnet
from nets.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import UltraLightweightUnetDataset, unet_dataset_collate
from utils.utils import download_weights, seed_everything, show_config, worker_init_fn
from utils.utils_fit import fit_one_epoch


def transfer_weights(old_model_path, new_model, device):
    if os.path.exists(old_model_path):
        print(f'Loading weights from {old_model_path} for transfer learning...')
        old_state_dict = torch.load(old_model_path, map_location=device)
        new_state_dict = new_model.state_dict()

        transfer_dict = {}
        # 编码器层权重迁移
        for i in range(1, 5):
            old_prefix = f'enc{i}.conv.'
            for k, v in old_state_dict.items():
                if k.startswith(old_prefix) and k in new_state_dict:
                    transfer_dict[k] = v

        # Bridge层权重迁移
        for k, v in old_state_dict.items():
            if k.startswith('bridge.conv.') and k in new_state_dict:
                transfer_dict[k] = v

        # 解码器层权重迁移
        for i in range(1, 5):
            old_prefix = f'dec{i}.conv.'
            for k, v in old_state_dict.items():
                if k.startswith(old_prefix) and k in new_state_dict:
                    transfer_dict[k] = v

        # 最终输出层权重迁移
        if 'final.weight' in old_state_dict:
            transfer_dict['final.weight'] = old_state_dict['final.weight']
        if 'final.bias' in old_state_dict:
            transfer_dict['final.bias'] = old_state_dict['final.bias']

        # 更新新模型的权重
        new_state_dict.update(transfer_dict)
        new_model.load_state_dict(new_state_dict, strict=False)

        print("Weight transfer completed!")
        print(f"Number of transferred layers: {len(transfer_dict)}")
        print(f"Number of new layers requiring initialization: {len(new_state_dict) - len(transfer_dict)}")
    return new_model


if __name__ == "__main__":
    # 基础设置
    Cuda = True
    seed = 11
    distributed = False
    sync_bn = False
    fp16 = True
    num_classes = 4

    # 模型相关设置
    input_shape = [256, 256]
    backbone = None
    pretrained = False
    model_path = "model_data/result_model/best_epoch_weights_1020_1:50.pth"

    # 训练策略设置
    Init_Epoch = 0
    Freeze_Epoch = 50  # 第一阶段训练轮数
    UnFreeze_Epoch = 400  # 总训练轮数
    Freeze_batch_size = 16  # 第一阶段batch size
    Unfreeze_batch_size = 16  # 第二阶段batch size
    Freeze_Train = True  # 是否采用分阶段训练

    # 优化器设置
    Init_lr = 1e-3  # 初始学习率
    Min_lr = Init_lr * 0.01
    optimizer_type = "adamw"  # 使用AdamW优化器
    momentum = 0.9
    weight_decay = 1e-2  # 增大权重衰减以防止过拟合
    lr_decay_type = 'cos'

    # 损失函数设置
    dice_loss = True
    focal_loss = True
    cls_weights = np.array([1, 15, 1.5, 2], np.float32)

    # 其他设置
    save_period = 5
    save_dir = 'logs'
    eval_flag = True
    eval_period = 5
    num_workers = 4

    # 数据集设置
    VOCdevkit_path = 'VOCdevkit'

    # ------------------------------------------------------#
    #   设置用到的显卡
    # ------------------------------------------------------#
    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0
        rank = 0


    model = ImprovedUltraLightweightUnet(num_classes=num_classes).train()

    # 权重迁移
    if model_path:
        model = transfer_weights(model_path, model, device)

    if Freeze_Train:
        # 第一阶段：冻结原有层，只训练新增层
        for name, param in model.named_parameters():
            if 'se' not in name and 'fusion' not in name and 'aux' not in name:
                param.requires_grad = False

    # 优化器配置
    if optimizer_type == "adamw":
        optimizer = optim.AdamW([
            {'params': [p for n, p in model.named_parameters() if
                        'se' not in n and 'fusion' not in n and 'aux' not in n],
             'lr': Init_lr * 0.1},
            {'params': [p for n, p in model.named_parameters() if 'se' in n or 'fusion' in n or 'aux' in n],
             'lr': Init_lr}
        ], weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=Init_lr, momentum=momentum, nesterov=True,
                              weight_decay=weight_decay)

    # 余弦退火学习率调度器
    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr, Min_lr, UnFreeze_Epoch)

    # 后续训练流程与原代码相似，但需要注意处理deep supervision的输出
    # ... (训练循环代码保持不变)
    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history = None

    # ------------------------------------------------------------------#
    #   torch 1.2不支持amp，建议使用torch 1.7.1及以上正确使用fp16
    #   因此torch1.2这里显示"could not be resolve"
    # ------------------------------------------------------------------#
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler

        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()
    # ----------------------------#
    #   多卡同步Bn
    # ----------------------------#
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            # ----------------------------#
            #   多卡平行运行
            # ----------------------------#
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank],
                                                                    find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    # ---------------------------#
    #   读取数据集对应的txt
    # ---------------------------#
    with open(os.path.join(VOCdevkit_path, "VOC2012/ImageSets/Segmentation/train.txt"), "r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "VOC2012/ImageSets/Segmentation/val.txt"), "r") as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    if local_rank == 0:
        show_config(
            num_classes=num_classes, backbone=backbone, model_path=model_path, input_shape=input_shape, \
            Init_Epoch=Init_Epoch, Freeze_Epoch=Freeze_Epoch, UnFreeze_Epoch=UnFreeze_Epoch,
            Freeze_batch_size=Freeze_batch_size, Unfreeze_batch_size=Unfreeze_batch_size, Freeze_Train=Freeze_Train, \
            Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum,
            lr_decay_type=lr_decay_type, \
            save_period=save_period, save_dir=save_dir, num_workers=num_workers, num_train=num_train, num_val=num_val
        )
    # ---------------------------------------#
    #   判断每一个世代的长度
    # ---------------------------------------#
    # -------------------------------------------------------------------#
    #   如果不冻结训练的话，直接设置batch_size为Unfreeze_batch_size
    # -------------------------------------------------------------------#
    batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size
    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

    train_dataset = UltraLightweightUnetDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
    val_dataset = UltraLightweightUnetDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, )
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, )
        batch_size = batch_size // ngpus_per_node
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True

    gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                     pin_memory=True,
                     drop_last=True, collate_fn=unet_dataset_collate, sampler=train_sampler,
                     worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
    gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                         pin_memory=True,
                         drop_last=True, collate_fn=unet_dataset_collate, sampler=val_sampler,
                         worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

    # ----------------------#
    #   记录eval的map曲线
    # ----------------------#
    if local_rank == 0:
        eval_callback = EvalCallback(model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda, \
                                     eval_flag=eval_flag, period=eval_period)
    else:
        eval_callback = None

    # ---------------------------------------#
    #   开始模型训练
    # ---------------------------------------#
    for epoch in range(Init_Epoch, UnFreeze_Epoch):
        # ---------------------------------------#
        #   如果模型有冻结学习部分
        #   则解冻，并设置参数
        # ---------------------------------------#
        if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
            batch_size = Unfreeze_batch_size

            # -------------------------------------------------------------------#
            #   判断当前batch_size，自适应调整学习率
            # -------------------------------------------------------------------#
            nbs = 16
            lr_limit_max = 1e-4 if optimizer_type == 'adam' else 1e-1
            lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
            Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
            Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
            # ---------------------------------------#
            #   获得学习率下降的公式
            # ---------------------------------------#
            lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

            model.unfreeze_backbone()

            epoch_step = num_train // batch_size
            epoch_step_val = num_val // batch_size

            if epoch_step == 0 or epoch_step_val == 0:
                raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

            if distributed:
                batch_size = batch_size // ngpus_per_node

            gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=True,
                             drop_last=True, collate_fn=unet_dataset_collate, sampler=train_sampler,
                             worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
            gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                 pin_memory=True,
                                 drop_last=True, collate_fn=unet_dataset_collate, sampler=val_sampler,
                                 worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

            UnFreeze_flag = True

        if distributed:
            train_sampler.set_epoch(epoch)

        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

        fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch,
                      epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, dice_loss, focal_loss,
                      cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank)

        if distributed:
            dist.barrier()

    if local_rank == 0:
        loss_history.writer.close()