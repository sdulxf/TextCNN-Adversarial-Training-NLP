# TextCNN with Adversarial Training
## 概述
为测试对抗训练在NLP方面的应用，在TextCNN上进行对抗训练，测试了FGSM、 Free、PGD三种对抗训练的效果。

TextCNN的代码来源于github项目：[Chinese-Text-Classification-Pytorch](https://github.com/649453932/Chinese-Text-Classification-Pytorc)。

对抗训练的代码借鉴：[知乎专栏文章](https://zhuanlan.zhihu.com/p/91269728 )和[Fast is better than free: Revisiting adversarial
training论文源码](https://github.com/locuslab/fast_adversarial)

## 数据集说明

 从[THUCNews](http://thuctc.thunlp.org/)中抽取了20万条新闻标题，文本长度在20到30之间。一共10个类别，每类2万条。

类别：财经、房产、股票、教育、科技、社会、时政、体育、游戏、娱乐。

数据集划分：

| 数据集 | 数据量 |
| ------ | ------ |
| 训练集 | 18万   |
| 验证集 | 1万    |
| 测试集 | 1万    |

## 环境

python 3.7  

pytorch 1.9  

tqdm  

sklearn  

## 资源

GPU: 12G

## 目录结构

    main.py         # 运行主程序
    train_eval.py   # 训练和评测代码
    utils.py    `   # 数据集处理和词向量生成
    TextCNN.py      #TextCNN模型
    AtModels/       # 各对抗训练代码。  
        FGSM.py     # FGSM 对抗训练模型
        PGD.py      # PGD 对抗训练模型
        Free.py     # Free 对抗训练模型
    THUCNEWS/       # THUCNEWS数据集和运行结果
        data/       # 训练测试数据集
            train.txt   # 训练数据集
            dev.txt     # 开发数据集
            test.txt    # 测试数据集
            class.txt   # 类别定义
            embedding_SougouNews.npz    # 从Sougou数据生成的词向量数据
            vocab.pkl   # 词汇表数据
        save_dict/      # 输出模型文件


## 使用说明
```
# 训练并测试：
# TextCNN + 无对抗训练
python main.py

# TextCNN + PGD
python main.py --AtModel=PGD --epsilon=0.0313 --alpha=0.0627 --K=5

# TextCNN + Free
python main.py --AtModel=Free --epsilon=0.1 --M=8

# TextCNN + FGSM
python main.py --AtModel=FGSM --epsilon=0.0313 --alpha=0.0627

```

## 试验结果

| 方法         | 参数配置                                       | Acc    | Precison | Recall | F1     | 耗时(min) | Epoch |
| ------------ | ---------------------------------------------- | ------ | -------- | ------ | ------ | --------- | ----- |
| TextCNN      | early_stop                                     | 90.24% | 90.27%   | 90.24% | 90.22% | 4.3       | 3     |
| TextCNN+PGD  | early_stop,epsilon=8/255,     alpha=16/255,K=5 | 90.91% | 90.96%   | 90.91% | 90.91% | 28        | 3     |
| TextCNN+Free | early_stop,epsilon=0.1,     M=7                | 88.44% | 88.54%   | 88.44% | 88.46% | 30.95     | 2     |
| TextCNN+FGSM | early_stop,epsilon=8/255,     alpha=16/255     | 91.16% | 91.16%   | 91.16% | 91.15% | 12.3      | 4     |

## 参考

[1] Fast is better than free: Revisiting adversarial training.    
    https://arxiv.org/abs/2001.03994    

[2] https://github.com/locuslab/fast_adversarial    

[3] https://github.com/649453932/Chinese-Text-Classification-Pytorch    

[4] https://zhuanlan.zhihu.com/p/91269728