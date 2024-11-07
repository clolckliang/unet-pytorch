import os
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from PIL import Image
import albumentations as A



class SteelDefectDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, dataset_path, train=True):
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.input_shape = input_shape
        self.dataset_path = dataset_path
        self.train = train

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # 从注释行中提取文件名
        file_name = self.annotation_lines[index].strip()

        # 从文件中读取灰度图像
        img = Image.open(os.path.join(os.path.join(self.dataset_path, "Images"), f"{file_name}.png")).convert('L')

        # 从文件中读取标签
        label = Image.open(os.path.join(os.path.join(self.dataset_path, "Labels"), f"{file_name}.png")).convert('L')

        # 数据增强
        img, label = self.get_random_data(img, label, self.input_shape, random=self.train, file_name=file_name)

        # 将图像转换为numpy数组并添加通道维度
        img_array = np.array(img)[np.newaxis, :, :]
        label_array = np.array(label)[np.newaxis, :, :]

        # 将图像和标签转换为PyTorch张量
        img_tensor = torch.from_numpy(img_array).type(torch.FloatTensor)
        label_tensor = torch.from_numpy(label_array).type(torch.LongTensor)

        return img_tensor, label_tensor

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, input_shape, jitter=.3, random=True, file_name=None):
        w, h = input_shape


        # 定义数据增强变换
        transform = A.Compose([
            A.RandomScale(scale_limit=0.5, interpolation=cv2.INTER_NEAREST, p=0.5),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Resize(width=w, height=h, interpolation=cv2.INTER_NEAREST),
        ])

        # 对图像和标签应用数据增强
        augmented = transform(image=image, mask=label)
        new_image, new_label = augmented['image'], augmented['mask']

        # 新增的拼接式数据增强
        if self.train and self.rand() < 0.5 and file_name is not None:
            # 随机从4张200x200的图像和标签中选择2张进行拼接
            patch_images = [new_image]
            patch_labels = [new_label]
            for _ in range(3):
                patch_img = Image.open(os.path.join(os.path.join(self.dataset_path, "Images"),
                                                    f"{file_name}_{len(patch_images)}.png")).convert('L')
                patch_label = Image.open(os.path.join(os.path.join(self.dataset_path, "Labels"),
                                                      f"{file_name}_{len(patch_labels)}.png")).convert('L')
                patch_img = patch_img.resize((200, 200), Image.BICUBIC)
                patch_label = patch_label.resize((200, 200), Image.NEAREST)
                patch_images.append(patch_img)
                patch_labels.append(patch_label)

            # 拼接4张200x200的图像和标签到一张200x200的图像和标签
            new_image = Image.new('L', (400, 400), 0)
            new_label = Image.new('L', (400, 400), 0)
            new_image.paste(patch_images[0], (0, 0))
            new_image.paste(patch_images[1], (200, 0))
            new_image.paste(patch_images[2], (0, 200))
            new_image.paste(patch_images[3], (200, 200))
            new_label.paste(patch_labels[0], (0, 0))
            new_label.paste(patch_labels[1], (200, 0))
            new_label.paste(patch_labels[2], (0, 200))
            new_label.paste(patch_labels[3], (200, 200))
            new_image = new_image.resize((w, h), Image.BICUBIC)
            new_label = new_label.resize((w, h), Image.NEAREST)

        return new_image, new_label


def steel_defect_collate(batch):
    images, labels,pngs = [], [],[]
    pngs = []
    for img, label,png in batch:
        images.append(img)
        labels.append(label)
        pngs.append(png)
    images = torch.stack(images, 0)
    labels = torch.stack(labels, 0)
    pngs = torch.stack(pngs,0)
    return images, labels,pngs
