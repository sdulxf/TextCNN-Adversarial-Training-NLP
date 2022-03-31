# !/usr/bin/env python
# -*-coding:utf-8 -*-

import torch
import torch.nn.functional as F

class AtModel():
    def __init__(self, model,args):
        self.model = model
        self.emb_name = 'embedding' #模型embedding的参数名
        self.emb_backup = {}
        self.epsilon=args.epsilon
        self.alpha = args.alpha

    #初始化扰动
    def initial(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.emb_backup[name] = param.data.clone()
                self.r = torch.zeros_like(param.data)
                self.r = self.r.uniform_(-self.epsilon, self.epsilon)
                param.data.add_(self.r)

    # 干扰
    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    self.r = self.r + self.alpha * torch.sign(param.grad)
                    if torch.norm(self.r) > self.epsilon:
                        self.r = self.epsilon * self.r / torch.norm(self.r)
                    param.data.add_(self.r)

    def project(self, param_name, param_data):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > self.epsilon:
            r = self.epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    # 恢复embedding参数
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def train(self, trains, labels, optimizer):
        '''
        FGSM 对抗训练
        :param train:训练数据
        :param labels: 标签
        :return: outputs: 分类结果
                  loss:   损失值
        '''
        # 正常训练
        self.initial()  # random初始化
        outputs = self.model(trains)
        loss = F.cross_entropy(outputs, labels)
        loss.backward() # 反向传播，得到正常的grad
        self.restore()  # 恢复embedding
        self.attack()  # 根据x+r的梯度更新r并得到新的x+r
        outputs = self.model(trains)
        loss_adv = F.cross_entropy(outputs, labels)
        loss_adv.backward()  # 更新梯度
        self.restore()  # 恢复embedding参数
        optimizer.step()  # 根据梯度更新参数
        self.model.zero_grad()
        return outputs, loss