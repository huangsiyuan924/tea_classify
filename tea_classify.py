# -*- coding:utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

from data_augmentation import data_enhance_rotate, data_enhance_gamma

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = './datasets'

data_transforms = {
    'train_data': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 均值，标准差
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 均值，标准差
    ]),
    'test_data': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

BATCH_SIZE = 256  # 256实测占用5.1G显存
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in
                  ['train_data', 'test_data']}
# 旋转图片，从-15度到15度，每次递增1，跳过0度，一共产生9000张
image_datasets['train_data'] += data_enhance_rotate(data_dir)
# 伽马纠正，从0.7到1.3，每次递增0.02，总共30次，一共产生9000张
image_datasets['train_data'] += data_enhance_gamma(data_dir)
train_loader = torch.utils.data.DataLoader(image_datasets.get("train_data"), batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(image_datasets.get("test_data"), batch_size=BATCH_SIZE, shuffle=True)
# train_loader一共18300数据，test_loader一共600数据

"""
    原文中的网络模型，不过随机池化没有实现
"""
class CNN_SP(nn.Module):
    def __init__(self):
        super().__init__()
        # 下面的卷积层Conv2d的第一个参数指输入通道数，第二个参数指输出通道数，第三个参数指卷积核的大小
        self.conv1 = nn.Conv2d(3, 40, 3, stride=3, padding=1)
        self.conv2 = nn.Conv2d(40, 80, 5, stride=3, padding=0)
        self.conv3 = nn.Conv2d(80, 120, 3, stride=3, padding=1)
        self.conv4 = nn.Conv2d(120, 120, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(120, 120, 3, stride=1, padding=1)
        self.dropout_layer = torch.nn.Dropout(0.1)
        # 下面的全连接层Linear的第一个参数指输入通道数，第二个参数指输出通道数
        self.fc1 = nn.Linear(120 * 10 * 10, 100)
        self.fc2 = nn.Linear(100, 3)

    def forward(self, x):
        in_size = x.size(0)
        out = self.conv1(x)
        out = F.relu(out)
        out = F.max_pool2d(out, 3, 1, 1)
        # out = self.pool1(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = F.max_pool2d(out, 3, 1, 1)
        # out = self.pool2(out)
        out = self.conv3(out)
        out = F.relu(out)
        out = F.max_pool2d(out, 3, 1, 1)
        # out = self.pool3(out)
        out = self.conv4(out)
        out = F.relu(out)
        out = F.max_pool2d(out, 3, 1, 1)
        # out = self.pool4(out)
        out = self.conv5(out)
        out = F.relu(out)
        out = F.max_pool2d(out, 3, 1, 1)
        # out = self.pool5(out)
        out = out.view(in_size, -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.dropout_layer(out)
        out = self.fc2(out)
        out = F.log_softmax(out, dim=1)  # 计算log(softmax(x))
        return out


# model = StochasticPooling().to(DEVICE)
model = CNN_SP().to(DEVICE)
# summary(model, (40, 86, 86))
summary(model, (3, 256, 256))

# tensorboard, 记录loss和acc
writer = SummaryWriter("./logs")
writer.add_images()
start_lr = 0.01
optimizer = optim.SGD(model.parameters(), lr=start_lr, momentum=0.9)
# optimizer = optim.Adam(model.parameters(), lr=start_lr)

'''
    自适应学习率，复现原文中的每10个epoch将学习率减少10倍，这里暂时选择3方便看效果
'''


def adjust_learning_rate(optimizer, epoch, start_lr):
    lr = start_lr * (0.1 ** (epoch // 3))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    adjust_learning_rate(optimizer, epoch, start_lr)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        # loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \tLr:{:.2E}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(),
                optimizer.state_dict()['param_groups'][0]['lr']))
            writer.add_scalar('train_loss', loss.item(), (epoch - 1) * len(train_loader) + batch_idx)


def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    writer.add_scalar('test_acc', 100. * correct / len(test_loader.dataset), epoch)
    writer.add_scalar('test_loss', test_loss, epoch)


EPOCHS = 10
for epoch in range(1, EPOCHS + 1):
    train(model, DEVICE, train_loader, optimizer, epoch)
    test(model, DEVICE, test_loader, epoch)

writer.close()
