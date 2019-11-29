# -*- coding:utf-8 -*-
# 准备猫狗数据分成训练、验证、测试数据集
import os
import shutil


def make_dir(folder):
    """ 创建目录，若目录存在则需要先清空
    :param folder: 目录
    :return: bool
    """
    try:
        if os.path.isdir(folder):
            shutil.rmtree(folder)
        os.mkdir(folder)
        return True
    except BaseException:
        return False

# 原始目录所在的路径
# 数据集未压缩
original_dataset_train_dir = '/home/maxiaoyue/PycharmProjects/CatandDogClassification/kaggle/train'
original_dataset_test_dir = '/home/maxiaoyue/PycharmProjects/CatandDogClassification/kaggle/test1'
# 存储较小数据集的目录
base_dir = '/home/maxiaoyue/PycharmProjects/CatandDogClassification/kaggle_dogandcat_small'
make_dir(base_dir)
# 训练、验证、测试数据集的目录
train_dir = os.path.join(base_dir, 'train')
make_dir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
make_dir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
make_dir(test_dir)
# 猫训练图片所在目录
train_cats_dir = os.path.join(train_dir, 'cats')
make_dir(train_cats_dir)
# 狗训练图片所在目录
train_dogs_dir = os.path.join(train_dir, 'dogs')
make_dir(train_dogs_dir)
# 猫验证图片所在目录
validation_cats_dir = os.path.join(validation_dir, 'cats')
make_dir(validation_cats_dir)
# 狗验证数据集所在目录
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
make_dir(validation_dogs_dir)
# 猫测试数据集所在目录
test_cats_dir = os.path.join(test_dir, 'cats')
make_dir(test_cats_dir)
# 狗测试数据集所在目录
test_dogs_dir = os.path.join(test_dir, 'dogs')
make_dir(test_dogs_dir)


# 复制最开始的1000张猫图片到 train_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(2000)]
for fname in fnames:
    src = os.path.join(original_dataset_train_dir, fname)
    dst = os.path.join(train_dir, fname)
    shutil.copyfile(src, dst)
 # 复制接下来500张猫图片到 validation_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_train_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)
# 复制接下来500张图片到 test_cats_dir
fnames = ['{}.jpg'.format(i) for i in range(1, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_test_dir, fname)
    dst = os.path.join(test_dir, fname)
    shutil.copyfile(src, dst)
 # 复制最开始的1000张狗图片到 train_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(2000)]
for fname in fnames:
    src = os.path.join(original_dataset_train_dir, fname)
    dst = os.path.join(train_dir, fname)
    shutil.copyfile(src, dst)
 # 复制接下来500张狗图片到 validation_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_train_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)
 # 复制接下来500张狗图片到 test_dogs_dir
# fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
# for fname in fnames:
#     src = os.path.join(original_dataset_test_dir, fname)
#     dst = os.path.join(test_dogs_dir, fname)
#     shutil.copyfile(src, dst)

print('total training cat images:', len(os.listdir(train_cats_dir)))
print('total training dog images:', len(os.listdir(train_dogs_dir)))
print('total validation cat images:', len(os.listdir(validation_cats_dir)))
print('total validation dog images:', len(os.listdir(validation_dogs_dir)))
print('total test cat images:', len(os.listdir(test_cats_dir)))
print('total test dog images:', len(os.listdir(test_dogs_dir)))
