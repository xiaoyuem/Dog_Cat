# -*- coding:utf-8 -*-
# 猫狗分类：测试模型
import models
import torch as t
from data.dataset import DogCat
from torch.utils.data import DataLoader
from config import DefaultConfig
from tqdm import tqdm
from torch.autograd import Variable as V

opt = DefaultConfig()

# 测试已训练模型
@t.no_grad()  # pytorch>=0.5
def test(**kwargs):
    opt.parse(kwargs)

    # configure model  模型
    model = getattr(models, opt.model)().eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)

    # data 加载数据
    test_data = DogCat(opt.test_data_root, test=True)
    test_dataloader = DataLoader(test_data,
                                 batch_size=opt.batch_size,
                                 shuffle=False,
                                 num_workers=opt.num_workers)

    results = []
    for ii, (data, path) in tqdm(enumerate(test_dataloader)):
        test_input = V(data)
        test_score = model(test_input)
        probability = t.nn.functional.softmax(test_score, dim=1)[:, 1].detach().tolist()  # 这里改过，github代码有误
        # label = score.max(dim = 1)[1].detach().tolist()

        batch_results = [(path_.item(), probability_) for path_, probability_ in zip(path, probability)]
        results += batch_results

    write_csv(results, opt.result_file)

    return results

# 写CSV文本
def write_csv(results,file_name):
    import csv
    with open(file_name,'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id','label'])
        writer.writerows(results)

# help函数
def help():
    """
    打印帮助的信息： python file.py help
    """

    print("""
    usage : python file.py <function> [--args=value]
    <function> := train | test | help
    example: 
            python {0} test --dataset='path/to/dataset/root/'
            python {0} help
    avaiable args:""".format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)


# 主函数程序
new_config = {'load_model_path':'/home/maxiaoyue/PycharmProjects/CatandDogClassification/checkpoints/AlexNet.pth',
              'batch_size':100, 'test_data_root': '/home/maxiaoyue/PycharmProjects/CatandDogClassification/kaggle_dogandcat_small/test'}
opt.parse(new_config)
test()
# help()