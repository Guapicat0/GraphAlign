import torch.nn as nn


class HarmonicMeanLoss(nn.Module):
    def __init__(self):
        super(HarmonicMeanLoss, self).__init__()

    def forward(self, loss1, loss2):
        # 防止除以零错误
        epsilon = 1e-7  # 很小的正数
        numerator = 2 * loss1 * loss2
        denominator = loss1 + loss2 + epsilon  # 添加epsilon防止分母为0
        return numerator / denominator

class ADDLoss(nn.Module):
    def __init__(self):
        super(ADDLoss, self).__init__()

    def forward(self, loss1, loss2):
        # 防止除以零错误
        epsilon = 1e-7  # 很小的正数
        denominator = loss1 + loss2  # 添加epsilon防止分母为0
        return denominator