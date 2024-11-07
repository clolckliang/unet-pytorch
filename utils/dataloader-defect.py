import os
import numpy as np
from torch.utils.data.dataset import Dataset
from PIL import Image

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
        annotation_line = self.annotation_lines[index]
        name = annotation_line.split()[0]

        # 从文件中读取灰度图像
        img = Image.open(os.path.join(os.path.join(self.dataset_path, "Images"), name + ".png")).convert('L')
        
        # 从文件中读取标签
        label = Image.open(os.path.join(os.path.join(self.dataset_path, "Labels"), name + ".png")).convert('L')
        
        # 数据增强
        img, label = self.get_random_data(img, label, self.input_shape, random=self.train)

        # 将图像转换为numpy数组并添加通道维度
        img_array = np.array(img)[np.newaxis, :, :]
        label_array = np.array(label)[np.newaxis, :, :]

        # 将图像和标签转换为PyTorch张量
        img_tensor = torch.from_numpy(img_array).type(torch.FloatTensor)
        label_tensor = torch.from_numpy(label_array).type(torch.LongTensor)

        return img_tensor, label_tensor

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, input_shape, jitter=.3, random=True):
        w, h = input_shape
        iw, ih = image.size

        if not random:
            # 不进行随机数据增强,只进行缩放
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            image = image.resize((nw, nh), Image.BICUBIC)
            label = label.resize((nw, nh), Image.NEAREST)
            new_image = Image.new('L', [w, h], 0)
            new_label = Image.new('L', [w, h], 0)
            new_image.paste(image, ((w-nw)//2, (h-nh)//2))
            new_label.paste(label, ((w-nw)//2, (h-nh)//2))
            return new_image, new_label

        # 随机缩放
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(0.5, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)
        label = label.resize((nw,nh), Image.NEAREST)

        # 随机翻转
        if self.rand() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        # 添加灰色边框
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('L', (w,h), 0)
        new_label = Image.new('L', (w,h), 0)
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        
        # 新增的拼接式数据增强
        if self.train and self.rand() < 0.5:
            # 随机从4张200x200的图像和标签中选择2张进行拼接
            patch_images = [new_image]
            patch_labels = [new_label]
            for _ in range(3):
                patch_img = Image.open(os.path.join(os.path.join(self.dataset_path, "Images"), f"{name}_{len(patch_images)}.png")).convert('L')
                patch_label = Image.open(os.path.join(os.path.join(self.dataset_path, "Labels"), f"{name}_{len(patch_labels)}.png")).convert('L')
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
    images, labels = [], []
    for img, label in batch:
        images.append(img)
        labels.append(label)
    images = torch.stack(images, 0)
    labels = torch.stack(labels, 0)
    return images, labels
```

主要的改动有:

1. 在`__getitem__`方法中,从标签文件夹中读取标签图像,并将其转换为PIL图像。
2. 在`get_random_data`方法中,对标签图像也进行数据增强操作,包括缩放、翻转和拼接。
3. 在拼接后,同时更新标签图像。
4. 在`steel_defect_collate`方法中,同时返回图像和标签张量。

这样就确保了图像和标签能够对应,可以用于后续的模型训练和评估。

如果还有任何其他问题,欢迎继续询问。