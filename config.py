# -*- coding:utf-8 -*-
# 猫狗分类：参数设置
import os


class DefaultConfig(object):
    # 使用的模型
    model = 'AlexNet'
    # 训练集路径
    train_data_root = os.path.join(os.getcwd(), 'kaggle/train')
    # 测试集路径
    test_data_root = os.path.join(os.getcwd(), 'kaggle/test')
    use_gpu = True  # user GPU or not
    # 加载预训练模型的路径，为None代表不加载
    load_model_path = None
    # 保存训练模型的路径目录
    save_model_path = os.path.join(os.getcwd(), 'checkpoints')
    # 批处理大小
    batch_size = 2
    # 进程个数
    num_workers = 0
    # 学习率
    lr = 0.1
    lr_decay = 0.95  # when val_loss increase,lr=lr*0.95
    weight_decay = 1e-4  # 损失函数
    max_epoch = 10

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    # 根据字典kwargs更新config参数
    def parse(self, kwargs):
        # 更新配置参数
        for k, v, in kwargs.items():
            if not hasattr(self, k):
                print('has not attribut %s' % k)
            setattr(self, k, v)

    # hasattr(self,k):判断sekf里是否有k属性
    # setattr(self,k,v):给self里的k属性赋值为v
    # getattr(self,k):获得self的k属性

opt = DefaultConfig()
