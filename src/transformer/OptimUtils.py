"""
@author yy
@date 2020.5.6
"""

import numpy as np


class ScheduleOptim:
    """
    管理optimizer的一个wrap，方便管理learning rate的更新，参考论文中的描述
    """
    def __init__(self, optimizer, init_lr, d_model, n_warmup_steps):
        """
        :param optimizer: pytorch的优化器
        :param init_lr: 初始learning rate
        :param d_model: model中间参数的维度
        :param n_warmup_steps:
        """
        self._optimizer = optimizer
        self.init_lr = init_lr
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.step_num = 0

    def step_and_update_lr(self):
        self._update_lr()
        self._optimizer.step()

    def zero_grad(self):
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        d_model = self.d_model
        step_num, n_warmup_steps = self.step_num, self.n_warmup_steps
        return (d_model ** -0.5) * min(step_num ** -0.5, step_num * n_warmup_steps ** -1.5)

    def _update_lr(self):
        self.step_num += 1
        lr = self.init_lr * self._get_lr_scale()
        for para_group in self._optimizer.param_groups:
            para_group['lr'] = lr


