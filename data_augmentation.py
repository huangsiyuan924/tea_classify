# -*- coding:utf-8 -*-
import os
from torchvision import transforms, datasets
from my_transforms import AdjustGamma


def data_enhance_rotate(data_dir):
    # 从-15度到15度，总共30次
    data_list = []
    for i in range(-15, 16):
        if i == 0: continue  # 0度不要
        x = 'train_data'
        trans = transforms.Compose([
            transforms.RandomRotation(degrees=(i, i)),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 均值，标准差
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 均值，标准差
        ])
        image_datasets = datasets.ImageFolder(os.path.join(data_dir, x), trans)
        data_list.append(image_datasets)

    tmp = data_list[0]
    for i in range(1, len(data_list)):
        tmp += data_list[i]
    return tmp


def data_enhance_gamma(data_dir):
    # 从0.7到1.3， 一共三十次
    data_list = []
    i = 0.7
    num = 0
    while i < 1.3:
        x = 'train_data'
        """
        gamma ( float ) -- 非负实数，与γ在等式中。gamma 大于 1 会使阴影更暗，而 gamma 小于 1 会使暗区更亮。
        gain ( float ) – 常数乘数。
        """
        num += 1
        trans = transforms.Compose([
            AdjustGamma(gamma=i),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 均值，标准差
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 均值，标准差
        ])
        i += 0.02
        image_datasets = datasets.ImageFolder(os.path.join(data_dir, x), trans)
        data_list.append(image_datasets)

    tmp = data_list[0]
    for i in range(1, len(data_list)):
        tmp += data_list[i]
    return tmp
