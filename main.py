# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train,init_network
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
parser.add_argument('--AtModel', default='BaseLine', type=str, help='BaseLine or PGD or Free or FGSM')
parser.add_argument('--epochs', default=20, type=int, help='epochs')
parser.add_argument('--epsilon', default=0.1, type=float, help='given some radius')
parser.add_argument('--alpha', default=0.1, type=float, help='adversarial step size')
parser.add_argument('--K', default=3, type=int, help='PGD steps ')
parser.add_argument('--M', default=3, type=int, help='Free minibatch replays')
args = parser.parse_args()


if __name__ == '__main__':
    time1=time.time()
    dataset = 'THUCNews'  # 数据集

    # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    embedding = 'embedding_SougouNews.npz'
    if args.embedding == 'random':
        embedding = 'random'

    x = import_module('TextCNN')
    config = x.Config(dataset, embedding)

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    config.n_vocab = len(vocab)
    config.num_epochs=args.epochs
    model = x.Model(config).to(config.device)
    if args.AtModel=="BaseLine":
        AtModel=None
    else:
        At= import_module('AtModels.'+ args.AtModel)
        AtModel=At.AtModel(model,args)

    init_network(model)
    train(config, model, train_iter, dev_iter, test_iter,AtModel)
    time2=time.time()
    print("耗时{}min".format((time2-time1)/60))