# -*- coding:utf-8 -*-
# 猫狗分类：数据集加载
import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T


class DogCat(data.Dataset):

    def __init__(self, root, transforms=None, train=True, test=False):
        """
        主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据
        """
        self.test = test
        imgs = [os.path.join(root, img) for img in os.listdir(root)]

        # test1: data/test1/8973.jpg
        # train: data/train/cat.10004.jpg

        if self.test:
            # 如果是进行测试，截取得到数据的数字标码，如上面的8973，根据key=8973的值进行排序，返回对所有的图片路径进行排序后返回
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('/')[-1]))
        else:
            # 如果是进行训练，截取得到数据的数字标识，如上面的10004，根据key=10004的值进行排序，返回对所有的图片路径进行排序后返回
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2]))

        imgs_num = len(imgs)  # 然后就可以得到数据的大小

        if self.test:
            # 如果是进行测试
            self.imgs = imgs
        elif train:
            # 如果是进行训练,使用前70%的数据
            self.imgs = imgs[:int(0.7 * imgs_num)]
        else:
            # 如果是进行验证，使用后30%的数据
            self.imgs = imgs[int(0.7 * imgs_num):]

        if transforms is None:
            # 数据转换操作，测试验证和训练的数据转换有所区别

            # 对数据进行归一化
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

            # 当是测试集和验证集时进行的操作
            if self.test or not train:
                self.transforms = T.Compose([
                    T.Resize(224),  # 重新设定大小
                    T.CenterCrop(227),  # 从图片中心截取
                    T.ToTensor(),  # 转成Tensor格式，大小范围为[0,1]
                    normalize  # 归一化处理,大小范围为[-1,1]
                ])
            else:
                self.transforms = T.Compose([
                    T.Resize(256),
                    T.RandomResizedCrop(227),  # 从图片的任何部位随机截取224*224大小的图
                    T.RandomHorizontalFlip(),  # 随机水平翻转给定的PIL.Image,翻转概率为0.5
                    T.ToTensor(),
                    normalize
                ])

    def __getitem__(self, index):
        """
        一次返回一张图片的数据
        """
        img_path = self.imgs[index]
        if self.test:
            # 如果是测试，得到图片路径中的数字标识作为label
            label = int(self.imgs[index].split('.')[-2].split('/')[-1])
        else:
            # 如果是训练，判断图片路径中是猫狗来设定label，猫为0，狗为1
            label = 1 if 'dog' in img_path.split('/')[-1] else 0
        data = Image.open(img_path)  # 打开该路径获得数据
        data = self.transforms(data)  # 然后对图片数据进行transform
        return data, label  # 最后得到统一的图片信息和label信息

    def __len__(self):  # 图片数据的大小
        return len(self.imgs)
