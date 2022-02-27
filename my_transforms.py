# -*- coding:utf-8 -*-
import torch
from torchvision.transforms import functional as F


# adjust_gamma 参考了官方文档
# https://pytorch.org/vision/main/generated/torchvision.transforms.functional.adjust_gamma.html
class AdjustGamma(torch.nn.Module):
    def __init__(self, gamma, gain=1.0):
        super(AdjustGamma, self).__init__()
        self.gamma = gamma
        self.gain = gain

    def forward(self, img):
        return F.adjust_gamma(img, gamma=self.gamma, gain=self.gain)


@DeprecationWarning
class AffineTransform(torch.nn.Module):
    def __init__(self, p):
        super(AffineTransform, self).__init__()
        self.p = p

    def forward(self, img):
        # return F.affine(img, scale=self.p)
        pass
