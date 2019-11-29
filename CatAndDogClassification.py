# -*- coding:utf-8 -*-
# 猫狗分类
# 导入库
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# 设置超参数
BATCH_SIZE = 50
EPOCHS = 30
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据预处理
transform = transforms.Compose([
    transforms.RandomResizedCrop(150),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# 读取数据
root = 'kaggle_dogandcat_small'
dataset_train = datasets.ImageFolder(root + '/train', transform)
dataset_test = datasets.ImageFolder(root + '/test', transform)

# 导入数据
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True)


# 定义网络
class ConvNet(nn.Module):
    def __init__(self):
        # 四层卷积层+池化层
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.max_pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.max_pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.max_pool3 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(128, 128, 3)
        self.max_pool4 = nn.MaxPool2d(2)
        # 全链接层
        self.fc1 = nn.Linear(6272, 512)
        self.fc2 = nn.Linear(512, 1)

    # 前向神经网络
    def forward(self, x):
        in_size = x.size(0)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.max_pool3(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.max_pool4(x)
        # 展开 全连接层Linear，需要将高维数据平铺变为低位数据
        x = x.view(in_size, -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)

        return x


# 实例化模型并且移动到GPU
model = ConvNet().to(DEVICE)
# 选择简单暴力的Adam优化器，学习率调低
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 定义训练过程
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device).float().reshape(50, 1)
        optimizer.zero_grad()           # 模型的参数梯度设成0,pytorch中的backward()函数的计算，当网络参量进行反馈时，
        output = model(data)            # 梯度是被积累的而不是被替换掉；但是在每一个batch时毫无疑问并不需要将两个batch的梯度混合起来累积
        # print(output)
        loss = F.binary_cross_entropy(output, target)
        loss.backward()
        optimizer.step()        #模型更新，优化器优化
        if (batch_idx + 1) % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), loss.item()))


# 定义测试过程
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():       # 导数非空
        for data, target in test_loader:
            data, target = data.to(device), target.to(device).float().reshape(50, 1)
            output = model(data)
            # print(output)
            test_loss += F.binary_cross_entropy(output, target, reduction='sum').item()  # 将一批的损失相加
            pred = torch.tensor([[1] if num[0] >= 0.5 else [0] for num in output]).to(device) # 预测结果
            correct += pred.eq(target.long()).sum().item()           # 正确率
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

# # 训练
for epoch in range(1, EPOCHS + 1):
    train(model, DEVICE, train_loader, optimizer, epoch)
    test(model, DEVICE, test_loader)
