# !/usr/bin/env python
# -*-coding:utf-8 -*-
import torch
import torch.nn.functional as F


class AtModel():
    def __init__(self, model,args):
        self.model = model
        self.emb_backup = {}
        self.r = 0
        self.epsilon=args.epsilon
        self.M=args.M

    def attack(self, emb_name='embedding', is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                norm = torch.norm(param.grad)
                if is_first_attack:
                    self.r = torch.zeros_like(param.data)
                    self.emb_backup[name] = param.data.clone()
                    assert name in self.emb_backup
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    self.r = self.r + r_at
                    param.data.add_(self.r)
                    param.data = self.project(name, param.data)


    def restore(self, emb_name='embedding'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]

    def project(self, param_name, param_data):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > self.epsilon:
            r = self.epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def clear(self):
        self.emb_backup = {}

    def train(self, trains, labels, optimizer):
        '''
        Free对抗训练过程
        :param train:训练数据为Tensor
        :param labels: 标签为Tensor
        :return: outputs: 分类结果
                  loss:   损失值
        '''
        outputs = self.model(trains)
        self.model.zero_grad()
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        # free 对抗训练
        for m in range(self.M):
            self.attack(is_first_attack=(m == 0))  # 在embedding上添加对抗扰动
            outputs = self.model(trains)
            loss_adv = F.cross_entropy(outputs, labels)
            loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            self.restore()  # 恢复embedding参数
            optimizer.step()  # 梯度下降，更新参数
            self.model.zero_grad()
        self.clear()
        return outputs, loss
