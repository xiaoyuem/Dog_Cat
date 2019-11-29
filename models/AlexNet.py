# -*- coding:utf-8 -*-
# 猫狗分类：AlexNet模型
from torch import nn
from .BasicModule import BasicModule

class AlexNet(BasicModule):
    def __init__(self, num_class=2):
        nn.Module.__init__(self)

        self.features = nn.Sequential(
            # conv1
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),  # 对原变量进行覆盖
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),

            # conv2
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),

            # conv3
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # conv4
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # conv5
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.classifier = nn.Sequential(
            # fc6
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            # fc7
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            # fc8
            nn.Linear(4096, num_class)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x
