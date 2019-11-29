# -*- coding:utf-8 -*-
# 猫狗分类：训练模型
# import fire
from models.AlexNet import AlexNet
from data.dataset import DogCat
from torch.utils import data
import torch as t
from torch.autograd import Variable as V
from torchnet import meter
from config import DefaultConfig
import numpy as np


opt = DefaultConfig()

# 训练
def train(**kwargs):
    # 根据命令行参数更新配置
    opt.parse(kwargs)

    # step1:模型
    model = AlexNet()
    if opt.load_model_path:
        model.load_state_dict(t.load(opt.load_model_path))

    # step2:数据
    # 训练数据
    train_data = DogCat(opt.train_data_root, train=True)
    # 验证数据
    val_data = DogCat(opt.train_data_root, train=False)
    # 由训练数据构成批量的数据，神经网络有效的输入格式，即增加了一个batch_size维度
    train_dataloader = data.DataLoader(train_data, batch_size=opt.batch_size, shuffle=True)
    # 批量的验证数据，神经网络有效的输入格式
    val_dataloader = data.DataLoader(val_data, batch_size=opt.batch_size, shuffle=False)

    # step3:目标函数和优化器,交叉熵损失函数（内部其实就是softmax机制）
    criterion = t.nn.CrossEntropyLoss()
    lr = opt.lr
    # 随机梯度下降优化模型
    optimizer = t.optim.SGD(model.parameters(), lr=lr, weight_decay=opt.weight_decay)

    # step4:统计指标:平滑处理之后的损失，还有混淆矩阵
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)  # 用来统计分类问题中的分类情况，是一个比准确率更详细的统计指标
    previous_loss = 1e10

    # 训练
    for epoch in range(opt.max_epoch):
        loss_meter.reset()
        confusion_matrix.reset()

        # 按批次训练模型，每次训练所有图像后利用验证集进行验证测试
        for i, (datas, labels) in enumerate(train_dataloader):
            # 训练模型参数
            input = V(datas)
            target = V(labels)
            optimizer.zero_grad()  # 梯度清零
            score = model(input)
            loss = criterion(score, target)
            loss.backward()
            # 所有optimizer都实现了step()方法，调用这个方法可以更新参数
            # 每次用backward()这类方法计算出了梯度后，就可以调用一次这个方法来更新参数。
            optimizer.step()

            if (i + 1) % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, (i + 1) * len(datas), len(train_dataloader.dataset),
                           100. * (i + 1) / len(train_dataloader), loss.item()))

            # 更新统计指标
            loss_meter.add(loss.data)
            # detach 一下更安全保险
            confusion_matrix.add(score.detach(), target.detach())

        # 保存模型
        t.save(model.state_dict(), opt.save_model_path + '/AlexNet.pth')

        # 计算验证集上的指标及可视化
        val_accuracy = val(model, val_dataloader)
        print(val_accuracy)
        # 如果损失不再下降，则降低学习率
        if loss_meter.value()[0] > previous_loss:
            lr = lr * opt.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        previous_loss = loss_meter.value()[0]

# 验证
def val(model, dataloader):
    # 计算模型在验证集上的准确率等信息
    # 把模型设为验证模式，验证模式和训练模式dropout层工作不一样
    model.eval()
    a = 0.0
    num = 0
    for i, (datas, labels) in enumerate(dataloader):
        with t.no_grad():
            val_input = V(datas)
            val_label = V(labels.long())
        score = model(val_input)
        soft_score = t.nn.functional.softmax(score)
        # 统计标签和预测相同性，计算正确率
        a = a + sum((np.argmax(np.array(soft_score.data), axis=1) == np.array(val_label)))
        num = num + len(labels)
        # 把模式恢复为训练模式
    model.train()
    accuracy = a / num

    return accuracy

def help():
    """
    打印帮助的信息： python file.py help
    """

    print("""
    usage : python file.py <function> [--args=value]
    <function> := train | test | help
    example: 
            python {0} train --env='env0701' --lr=0.01
            python {0} help
    avaiable args:""".format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)


# if __name__ == '__main__':
#     fire.Fire()
# 主函数程序
new_config = {'lr':0.1,
              'train_data_root':'/home/maxiaoyue/PycharmProjects/CatandDogClassification/kaggle_dogandcat_small/train',
              'batch_size':100, 'save_model_path': '/home/maxiaoyue/PycharmProjects/CatandDogClassification/checkpoints'}
opt.parse(new_config)
train()
# help()