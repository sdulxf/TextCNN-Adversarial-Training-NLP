# !/usr/bin/env python
# -*-coding:utf-8 -*-

import torch
import torch.nn.functional as F

class AtModel():
    def __init__(self, model,args):
        self.model = model
        self.emb_name = 'embedding'#模型embedding的参数名
        self.emb_backup = {}
        self.grad_backup = {}
        self.epsilon=args.epsilon
        self.alpha = args.alpha
        self.K=args.K

    def attack(self,is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > self.epsilon:
            r = self.epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]

    def train(self, trains, labels, optimizer):
        '''
        PGD对抗训练过程
        :param train:训练数据
        :param labels: 标签
        :return: outputs: 分类结果
                  loss:   损失值
        '''
        # 正常训练
        outputs = self.model(trains)
        self.model.zero_grad()
        loss = F.cross_entropy(outputs, labels)
        loss.backward()# 反向传播，得到正常的grad
        self.backup_grad()# 备份正常训练的grad
        # PGD 对抗训练
        for j in range(self.K):
            self.attack(is_first_attack=(j == 0))# 在embedding上添加对抗扰动, first attack时备份param.data
            # 最后一次attack完，需要还原gradients,用作下一次attack使用
            if j != self.K - 1:
                self.model.zero_grad()
            else:
                self.restore_grad()
            outputs = self.model(trains)
            loss_adv = F.cross_entropy(outputs, labels)
            loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
        self.restore() # 恢复embedding参数
        optimizer.step()# 梯度下降，更新参数
        return outputs, loss
