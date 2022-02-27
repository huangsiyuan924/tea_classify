原论文地址：https://www.researchgate.net/publication/323419312_Twelve-layer_deep_convolutional_neural_network_with_stochastic_pooling_for_tea_category_classification_on_GPU_platform

鄙人不是人工智能专业的，但是对这个领域非常好奇，最近找到了唐老师的一篇论文，用所学的知识进行了简单的复现。

但是由于理论知识和对pytorch使用不够，其中的随机池化（stochastic pooling），目前我无法实现，只能使用torch自带的模块进行模型的搭建。


### 开发环境
由于本科是做大数据的，经常需要集群，一套大数据服务下来内存动不动占用20G+，所以内存直接给到了32G，但是测试跑深度模型内存占不了太多，重要的是显卡，本人显卡3060 Laptop，功耗130W，6G显存，算力49左右，基本可以跑模型。
* 操作系统：win11
* cpu：i7-11800H
* 显卡：3060 Laptop（6G）
* 内存：32G
* cuda版本：11.6
* python版本：3.6.9
* pytorch版本：1.9



```python
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
# 使用GPU训练，3060显卡，30个epoch一共需要不到半小时
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### 制作数据集
没有原文茶叶数据集，所以我就在kaggle上搜索了相似的花分类数据集

kaggle地址在：https://www.kaggle.com/alxmamaev/flowers-recognition?select=flowers

其中有五类花共4317个图片，只采用了3类，一共选取900张图片

数据集目录树：

    datasets
    
    +---test_data
    
    |   +---daisy
    
    |   +---dandelion
    
    |   \---rose
    
    \---train_data
    
        +---daisy
        
        +---dandelion
        
        \---rose

其中训练集共有300张，三类分别各有100张

测试集共有600张，三类各有200张

经过数据增强后，训练集共有18300张

数据增强函数在[data_augmentation.py](./data_augmentation.py)中


```python
data_dir = './datasets'
BATCH_SIZE = 256  # 256实测占用4-5G显存

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
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in
                  ['train_data', 'test_data']}
# 旋转图片，从-15度到15度，每次递增1，跳过0度，一共产生9000张
image_datasets['train_data'] += data_enhance_rotate(data_dir)
# 伽马纠正，从0.7到1.3，每次递增0.02，总共30次，一共产生9000张
image_datasets['train_data'] += data_enhance_gamma(data_dir)
train_loader = torch.utils.data.DataLoader(image_datasets.get("train_data"), batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(image_datasets.get("test_data"), batch_size=BATCH_SIZE, shuffle=True)
# train_loader一共18300数据，test_loader一共600数据
```

### 查看数据


```python
writer = SummaryWriter("./logs")
# 查看第一批的数据集，总共256张，（256, 3, 256, 256）：
for imgs, labels in train_loader:
    writer.add_images("imgs", imgs)
    break
    
writer.close()
```

<img src="./images/images_show.png" alt="images_show" style="zoom: 80%;" />

### 模型搭建
按照原文进行模型搭建，随机池化没有实现，就用torch自带的最大池化代替了

模型一共1,627,563个参数


```python

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
        # 原文中dropout的比率为0.1，防止过拟合
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
```

    e:\python_venv\torch_venv\lib\site-packages\torch\nn\functional.py:718: UserWarning: Named tensors and all their associated APIs are an experimental feature and subject to change. Please do not use them for anything important until they are released as stable. (Triggered internally at  ..\c10/core/TensorImpl.h:1156.)
      return torch.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)


    ==========================================================================================
    Layer (type:depth-idx)                   Output Shape              Param #
    ==========================================================================================
    ├─Conv2d: 1-1                            [-1, 40, 86, 86]          1,120
    ├─Conv2d: 1-2                            [-1, 80, 28, 28]          80,080
    ├─Conv2d: 1-3                            [-1, 120, 10, 10]         86,520
    ├─Conv2d: 1-4                            [-1, 120, 10, 10]         129,720
    ├─Conv2d: 1-5                            [-1, 120, 10, 10]         129,720
    ├─Linear: 1-6                            [-1, 100]                 1,200,100
    ├─Dropout: 1-7                           [-1, 100]                 --
    ├─Linear: 1-8                            [-1, 3]                   303
    ==========================================================================================
    Total params: 1,627,563
    Trainable params: 1,627,563
    Non-trainable params: 0
    Total mult-adds (M): 106.47
    ==========================================================================================
    Input size (MB): 0.75
    Forward/backward pass size (MB): 3.01
    Params size (MB): 6.21
    Estimated Total Size (MB): 9.97
    ==========================================================================================





    ==========================================================================================
    Layer (type:depth-idx)                   Output Shape              Param #
    ==========================================================================================
    ├─Conv2d: 1-1                            [-1, 40, 86, 86]          1,120
    ├─Conv2d: 1-2                            [-1, 80, 28, 28]          80,080
    ├─Conv2d: 1-3                            [-1, 120, 10, 10]         86,520
    ├─Conv2d: 1-4                            [-1, 120, 10, 10]         129,720
    ├─Conv2d: 1-5                            [-1, 120, 10, 10]         129,720
    ├─Linear: 1-6                            [-1, 100]                 1,200,100
    ├─Dropout: 1-7                           [-1, 100]                 --
    ├─Linear: 1-8                            [-1, 3]                   303
    ==========================================================================================
    Total params: 1,627,563
    Trainable params: 1,627,563
    Non-trainable params: 0
    Total mult-adds (M): 106.47
    ==========================================================================================
    Input size (MB): 0.75
    Forward/backward pass size (MB): 3.01
    Params size (MB): 6.21
    Estimated Total Size (MB): 9.97
    ==========================================================================================



### 自适应学习率，优化器，tensorboard的配置
* 学习率：开始0.01，每10个epch就将lr减少10倍，测试了每3个epoch减少一次lr比固定lr准确率高了5%，但是10个epoch减少一次lr不知为何准确率上不去
* 优化器：相比较SGDM与Adam，Adam拟合速度比较快，但最终结果差不多
* tensorboard：将数据写入logs文件夹中


```python
# tensorboard, 记录loss和acc
writer = SummaryWriter("./logs")
start_lr = 0.01
optimizer = optim.SGD(model.parameters(), lr=start_lr, momentum=0.9)
# optimizer = optim.Adam(model.parameters(), lr=start_lr)

'''
    自适应学习率，复现原文中的每10个epoch将学习率减少10倍
'''
def adjust_learning_rate(optimizer, epoch, start_lr):
    lr = start_lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
```

### 训练与测试函数


```python
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
```

### 开始训练


```python
EPOCHS = 30
for epoch in range(1, EPOCHS + 1):
    train(model, DEVICE, train_loader, optimizer, epoch)
    test(model, DEVICE, test_loader, epoch)
writer.close()
```

    Train Epoch: 1 [2304/18300 (12%)]	Loss: 1.097219 	Lr:1.00E-02
    Train Epoch: 1 [4864/18300 (26%)]	Loss: 1.090676 	Lr:1.00E-02
    Train Epoch: 1 [7424/18300 (40%)]	Loss: 1.069286 	Lr:1.00E-02
    Train Epoch: 1 [9984/18300 (54%)]	Loss: 0.906102 	Lr:1.00E-02
    Train Epoch: 1 [12544/18300 (68%)]	Loss: 0.960054 	Lr:1.00E-02
    Train Epoch: 1 [15104/18300 (82%)]	Loss: 0.902408 	Lr:1.00E-02
    Train Epoch: 1 [17664/18300 (96%)]	Loss: 0.809350 	Lr:1.00E-02
    
    Test set: Average loss: 0.8285, Accuracy: 392/600 (65%)
    
    Train Epoch: 2 [2304/18300 (12%)]	Loss: 0.624967 	Lr:1.00E-02
    Train Epoch: 2 [4864/18300 (26%)]	Loss: 0.600188 	Lr:1.00E-02
    Train Epoch: 2 [7424/18300 (40%)]	Loss: 0.478863 	Lr:1.00E-02
    Train Epoch: 2 [9984/18300 (54%)]	Loss: 0.552383 	Lr:1.00E-02
    Train Epoch: 2 [12544/18300 (68%)]	Loss: 0.554944 	Lr:1.00E-02
    Train Epoch: 2 [15104/18300 (82%)]	Loss: 0.489458 	Lr:1.00E-02
    Train Epoch: 2 [17664/18300 (96%)]	Loss: 0.399592 	Lr:1.00E-02
    
    Test set: Average loss: 0.7292, Accuracy: 444/600 (74%)
    
    Train Epoch: 3 [2304/18300 (12%)]	Loss: 0.338853 	Lr:1.00E-02
    Train Epoch: 3 [4864/18300 (26%)]	Loss: 0.352237 	Lr:1.00E-02
    Train Epoch: 3 [7424/18300 (40%)]	Loss: 0.327742 	Lr:1.00E-02
    Train Epoch: 3 [9984/18300 (54%)]	Loss: 0.212875 	Lr:1.00E-02
    Train Epoch: 3 [12544/18300 (68%)]	Loss: 0.188415 	Lr:1.00E-02
    Train Epoch: 3 [15104/18300 (82%)]	Loss: 0.324034 	Lr:1.00E-02
    Train Epoch: 3 [17664/18300 (96%)]	Loss: 0.620643 	Lr:1.00E-02
    
    Test set: Average loss: 0.8427, Accuracy: 429/600 (72%)
    
    Train Epoch: 4 [2304/18300 (12%)]	Loss: 0.462232 	Lr:1.00E-02
    Train Epoch: 4 [4864/18300 (26%)]	Loss: 0.366939 	Lr:1.00E-02
    Train Epoch: 4 [7424/18300 (40%)]	Loss: 0.399144 	Lr:1.00E-02
    Train Epoch: 4 [9984/18300 (54%)]	Loss: 0.245538 	Lr:1.00E-02
    Train Epoch: 4 [12544/18300 (68%)]	Loss: 0.191642 	Lr:1.00E-02
    Train Epoch: 4 [15104/18300 (82%)]	Loss: 0.162637 	Lr:1.00E-02
    Train Epoch: 4 [17664/18300 (96%)]	Loss: 0.163442 	Lr:1.00E-02
    
    Test set: Average loss: 1.2516, Accuracy: 448/600 (75%)
    
    Train Epoch: 5 [2304/18300 (12%)]	Loss: 0.196951 	Lr:1.00E-02
    Train Epoch: 5 [4864/18300 (26%)]	Loss: 0.182253 	Lr:1.00E-02
    Train Epoch: 5 [7424/18300 (40%)]	Loss: 0.101793 	Lr:1.00E-02
    Train Epoch: 5 [9984/18300 (54%)]	Loss: 0.073624 	Lr:1.00E-02
    Train Epoch: 5 [12544/18300 (68%)]	Loss: 0.063638 	Lr:1.00E-02
    Train Epoch: 5 [15104/18300 (82%)]	Loss: 0.083632 	Lr:1.00E-02
    Train Epoch: 5 [17664/18300 (96%)]	Loss: 1.211800 	Lr:1.00E-02
    
    Test set: Average loss: 1.1084, Accuracy: 393/600 (66%)
    
    Train Epoch: 6 [2304/18300 (12%)]	Loss: 0.514575 	Lr:1.00E-02
    Train Epoch: 6 [4864/18300 (26%)]	Loss: 0.442459 	Lr:1.00E-02
    Train Epoch: 6 [7424/18300 (40%)]	Loss: 0.301550 	Lr:1.00E-02
    Train Epoch: 6 [9984/18300 (54%)]	Loss: 0.277747 	Lr:1.00E-02
    Train Epoch: 6 [12544/18300 (68%)]	Loss: 0.218475 	Lr:1.00E-02
    Train Epoch: 6 [15104/18300 (82%)]	Loss: 0.176061 	Lr:1.00E-02
    Train Epoch: 6 [17664/18300 (96%)]	Loss: 0.129977 	Lr:1.00E-02
    
    Test set: Average loss: 1.4732, Accuracy: 461/600 (77%)
    
    Train Epoch: 7 [2304/18300 (12%)]	Loss: 0.432843 	Lr:1.00E-02
    Train Epoch: 7 [4864/18300 (26%)]	Loss: 0.300809 	Lr:1.00E-02
    Train Epoch: 7 [7424/18300 (40%)]	Loss: 0.259566 	Lr:1.00E-02
    Train Epoch: 7 [9984/18300 (54%)]	Loss: 0.117622 	Lr:1.00E-02
    Train Epoch: 7 [12544/18300 (68%)]	Loss: 0.109836 	Lr:1.00E-02
    Train Epoch: 7 [15104/18300 (82%)]	Loss: 0.087582 	Lr:1.00E-02
    Train Epoch: 7 [17664/18300 (96%)]	Loss: 0.028560 	Lr:1.00E-02
    
    Test set: Average loss: 1.4280, Accuracy: 439/600 (73%)
    
    Train Epoch: 8 [2304/18300 (12%)]	Loss: 0.038407 	Lr:1.00E-02
    Train Epoch: 8 [4864/18300 (26%)]	Loss: 0.025478 	Lr:1.00E-02
    Train Epoch: 8 [7424/18300 (40%)]	Loss: 0.023039 	Lr:1.00E-02
    Train Epoch: 8 [9984/18300 (54%)]	Loss: 0.009913 	Lr:1.00E-02
    Train Epoch: 8 [12544/18300 (68%)]	Loss: 0.005106 	Lr:1.00E-02
    Train Epoch: 8 [15104/18300 (82%)]	Loss: 0.005644 	Lr:1.00E-02
    Train Epoch: 8 [17664/18300 (96%)]	Loss: 0.004291 	Lr:1.00E-02
    
    Test set: Average loss: 2.0407, Accuracy: 447/600 (74%)
    
    Train Epoch: 9 [2304/18300 (12%)]	Loss: 0.003617 	Lr:1.00E-02
    Train Epoch: 9 [4864/18300 (26%)]	Loss: 0.001738 	Lr:1.00E-02
    Train Epoch: 9 [7424/18300 (40%)]	Loss: 0.019854 	Lr:1.00E-02
    Train Epoch: 9 [9984/18300 (54%)]	Loss: 0.007459 	Lr:1.00E-02
    Train Epoch: 9 [12544/18300 (68%)]	Loss: 0.029173 	Lr:1.00E-02
    Train Epoch: 9 [15104/18300 (82%)]	Loss: 0.022766 	Lr:1.00E-02
    Train Epoch: 9 [17664/18300 (96%)]	Loss: 0.008697 	Lr:1.00E-02
    
    Test set: Average loss: 1.9628, Accuracy: 461/600 (77%)
    
    Train Epoch: 10 [2304/18300 (12%)]	Loss: 0.002282 	Lr:1.00E-03
    Train Epoch: 10 [4864/18300 (26%)]	Loss: 0.003669 	Lr:1.00E-03
    Train Epoch: 10 [7424/18300 (40%)]	Loss: 0.000566 	Lr:1.00E-03
    Train Epoch: 10 [9984/18300 (54%)]	Loss: 0.002330 	Lr:1.00E-03
    Train Epoch: 10 [12544/18300 (68%)]	Loss: 0.003311 	Lr:1.00E-03
    Train Epoch: 10 [15104/18300 (82%)]	Loss: 0.001121 	Lr:1.00E-03
    Train Epoch: 10 [17664/18300 (96%)]	Loss: 0.001894 	Lr:1.00E-03
    
    Test set: Average loss: 2.0243, Accuracy: 458/600 (76%)
    
    Train Epoch: 11 [2304/18300 (12%)]	Loss: 0.000987 	Lr:1.00E-03
    Train Epoch: 11 [4864/18300 (26%)]	Loss: 0.001631 	Lr:1.00E-03
    Train Epoch: 11 [7424/18300 (40%)]	Loss: 0.002116 	Lr:1.00E-03
    Train Epoch: 11 [9984/18300 (54%)]	Loss: 0.002778 	Lr:1.00E-03
    Train Epoch: 11 [12544/18300 (68%)]	Loss: 0.000749 	Lr:1.00E-03
    Train Epoch: 11 [15104/18300 (82%)]	Loss: 0.001173 	Lr:1.00E-03
    Train Epoch: 11 [17664/18300 (96%)]	Loss: 0.001320 	Lr:1.00E-03
    
    Test set: Average loss: 2.0913, Accuracy: 457/600 (76%)
    
    Train Epoch: 12 [2304/18300 (12%)]	Loss: 0.000660 	Lr:1.00E-03
    Train Epoch: 12 [4864/18300 (26%)]	Loss: 0.001504 	Lr:1.00E-03
    Train Epoch: 12 [7424/18300 (40%)]	Loss: 0.001177 	Lr:1.00E-03
    Train Epoch: 12 [9984/18300 (54%)]	Loss: 0.001130 	Lr:1.00E-03
    Train Epoch: 12 [12544/18300 (68%)]	Loss: 0.003466 	Lr:1.00E-03
    Train Epoch: 12 [15104/18300 (82%)]	Loss: 0.000927 	Lr:1.00E-03
    Train Epoch: 12 [17664/18300 (96%)]	Loss: 0.003104 	Lr:1.00E-03
    
    Test set: Average loss: 2.1379, Accuracy: 456/600 (76%)
    
    Train Epoch: 13 [2304/18300 (12%)]	Loss: 0.001257 	Lr:1.00E-03
    Train Epoch: 13 [4864/18300 (26%)]	Loss: 0.001908 	Lr:1.00E-03
    Train Epoch: 13 [7424/18300 (40%)]	Loss: 0.001151 	Lr:1.00E-03
    Train Epoch: 13 [9984/18300 (54%)]	Loss: 0.002254 	Lr:1.00E-03
    Train Epoch: 13 [12544/18300 (68%)]	Loss: 0.000466 	Lr:1.00E-03
    Train Epoch: 13 [15104/18300 (82%)]	Loss: 0.001906 	Lr:1.00E-03
    Train Epoch: 13 [17664/18300 (96%)]	Loss: 0.001601 	Lr:1.00E-03
    
    Test set: Average loss: 2.1743, Accuracy: 456/600 (76%)
    
    Train Epoch: 14 [2304/18300 (12%)]	Loss: 0.000527 	Lr:1.00E-03
    Train Epoch: 14 [4864/18300 (26%)]	Loss: 0.001415 	Lr:1.00E-03
    Train Epoch: 14 [7424/18300 (40%)]	Loss: 0.000473 	Lr:1.00E-03
    Train Epoch: 14 [9984/18300 (54%)]	Loss: 0.001148 	Lr:1.00E-03
    Train Epoch: 14 [12544/18300 (68%)]	Loss: 0.000807 	Lr:1.00E-03
    Train Epoch: 14 [15104/18300 (82%)]	Loss: 0.001052 	Lr:1.00E-03
    Train Epoch: 14 [17664/18300 (96%)]	Loss: 0.000796 	Lr:1.00E-03
    
    Test set: Average loss: 2.2068, Accuracy: 454/600 (76%)
    
    Train Epoch: 15 [2304/18300 (12%)]	Loss: 0.000368 	Lr:1.00E-03
    Train Epoch: 15 [4864/18300 (26%)]	Loss: 0.000612 	Lr:1.00E-03
    Train Epoch: 15 [7424/18300 (40%)]	Loss: 0.001022 	Lr:1.00E-03
    Train Epoch: 15 [9984/18300 (54%)]	Loss: 0.000957 	Lr:1.00E-03
    Train Epoch: 15 [12544/18300 (68%)]	Loss: 0.000918 	Lr:1.00E-03
    Train Epoch: 15 [15104/18300 (82%)]	Loss: 0.001273 	Lr:1.00E-03
    Train Epoch: 15 [17664/18300 (96%)]	Loss: 0.000964 	Lr:1.00E-03
    
    Test set: Average loss: 2.2453, Accuracy: 452/600 (75%)
    
    Train Epoch: 16 [2304/18300 (12%)]	Loss: 0.000413 	Lr:1.00E-03
    Train Epoch: 16 [4864/18300 (26%)]	Loss: 0.001036 	Lr:1.00E-03
    Train Epoch: 16 [7424/18300 (40%)]	Loss: 0.002273 	Lr:1.00E-03
    Train Epoch: 16 [9984/18300 (54%)]	Loss: 0.000672 	Lr:1.00E-03
    Train Epoch: 16 [12544/18300 (68%)]	Loss: 0.001812 	Lr:1.00E-03
    Train Epoch: 16 [15104/18300 (82%)]	Loss: 0.002614 	Lr:1.00E-03
    Train Epoch: 16 [17664/18300 (96%)]	Loss: 0.000698 	Lr:1.00E-03
    
    Test set: Average loss: 2.2781, Accuracy: 453/600 (76%)
    
    Train Epoch: 17 [2304/18300 (12%)]	Loss: 0.000867 	Lr:1.00E-03
    Train Epoch: 17 [4864/18300 (26%)]	Loss: 0.000971 	Lr:1.00E-03
    Train Epoch: 17 [7424/18300 (40%)]	Loss: 0.001042 	Lr:1.00E-03
    Train Epoch: 17 [9984/18300 (54%)]	Loss: 0.001154 	Lr:1.00E-03
    Train Epoch: 17 [12544/18300 (68%)]	Loss: 0.000325 	Lr:1.00E-03
    Train Epoch: 17 [15104/18300 (82%)]	Loss: 0.000488 	Lr:1.00E-03
    Train Epoch: 17 [17664/18300 (96%)]	Loss: 0.000628 	Lr:1.00E-03
    
    Test set: Average loss: 2.2891, Accuracy: 450/600 (75%)
    
    Train Epoch: 18 [2304/18300 (12%)]	Loss: 0.001390 	Lr:1.00E-03
    Train Epoch: 18 [4864/18300 (26%)]	Loss: 0.000301 	Lr:1.00E-03
    Train Epoch: 18 [7424/18300 (40%)]	Loss: 0.000283 	Lr:1.00E-03
    Train Epoch: 18 [9984/18300 (54%)]	Loss: 0.004392 	Lr:1.00E-03
    Train Epoch: 18 [12544/18300 (68%)]	Loss: 0.000198 	Lr:1.00E-03
    Train Epoch: 18 [15104/18300 (82%)]	Loss: 0.000510 	Lr:1.00E-03
    Train Epoch: 18 [17664/18300 (96%)]	Loss: 0.000564 	Lr:1.00E-03
    
    Test set: Average loss: 2.3235, Accuracy: 451/600 (75%)
    
    Train Epoch: 19 [2304/18300 (12%)]	Loss: 0.000306 	Lr:1.00E-03
    Train Epoch: 19 [4864/18300 (26%)]	Loss: 0.000792 	Lr:1.00E-03
    Train Epoch: 19 [7424/18300 (40%)]	Loss: 0.001067 	Lr:1.00E-03
    Train Epoch: 19 [9984/18300 (54%)]	Loss: 0.001001 	Lr:1.00E-03
    Train Epoch: 19 [12544/18300 (68%)]	Loss: 0.001125 	Lr:1.00E-03
    Train Epoch: 19 [15104/18300 (82%)]	Loss: 0.002891 	Lr:1.00E-03
    Train Epoch: 19 [17664/18300 (96%)]	Loss: 0.001220 	Lr:1.00E-03
    
    Test set: Average loss: 2.3502, Accuracy: 452/600 (75%)
    
    Train Epoch: 20 [2304/18300 (12%)]	Loss: 0.001024 	Lr:1.00E-04
    Train Epoch: 20 [4864/18300 (26%)]	Loss: 0.001537 	Lr:1.00E-04
    Train Epoch: 20 [7424/18300 (40%)]	Loss: 0.000545 	Lr:1.00E-04
    Train Epoch: 20 [9984/18300 (54%)]	Loss: 0.000716 	Lr:1.00E-04
    Train Epoch: 20 [12544/18300 (68%)]	Loss: 0.001008 	Lr:1.00E-04
    Train Epoch: 20 [15104/18300 (82%)]	Loss: 0.001034 	Lr:1.00E-04
    Train Epoch: 20 [17664/18300 (96%)]	Loss: 0.000150 	Lr:1.00E-04
    
    Test set: Average loss: 2.3517, Accuracy: 452/600 (75%)
    
    Train Epoch: 21 [2304/18300 (12%)]	Loss: 0.000746 	Lr:1.00E-04
    Train Epoch: 21 [4864/18300 (26%)]	Loss: 0.000651 	Lr:1.00E-04
    Train Epoch: 21 [7424/18300 (40%)]	Loss: 0.003437 	Lr:1.00E-04
    Train Epoch: 21 [9984/18300 (54%)]	Loss: 0.001380 	Lr:1.00E-04
    Train Epoch: 21 [12544/18300 (68%)]	Loss: 0.001035 	Lr:1.00E-04
    Train Epoch: 21 [15104/18300 (82%)]	Loss: 0.000353 	Lr:1.00E-04
    Train Epoch: 21 [17664/18300 (96%)]	Loss: 0.000644 	Lr:1.00E-04
    
    Test set: Average loss: 2.3540, Accuracy: 450/600 (75%)
    
    Train Epoch: 22 [2304/18300 (12%)]	Loss: 0.000657 	Lr:1.00E-04
    Train Epoch: 22 [4864/18300 (26%)]	Loss: 0.000473 	Lr:1.00E-04
    Train Epoch: 22 [7424/18300 (40%)]	Loss: 0.000939 	Lr:1.00E-04
    Train Epoch: 22 [9984/18300 (54%)]	Loss: 0.001266 	Lr:1.00E-04
    Train Epoch: 22 [12544/18300 (68%)]	Loss: 0.000466 	Lr:1.00E-04
    Train Epoch: 22 [15104/18300 (82%)]	Loss: 0.000363 	Lr:1.00E-04
    Train Epoch: 22 [17664/18300 (96%)]	Loss: 0.000514 	Lr:1.00E-04
    
    Test set: Average loss: 2.3559, Accuracy: 451/600 (75%)
    
    Train Epoch: 23 [2304/18300 (12%)]	Loss: 0.000833 	Lr:1.00E-04
    Train Epoch: 23 [4864/18300 (26%)]	Loss: 0.000235 	Lr:1.00E-04
    Train Epoch: 23 [7424/18300 (40%)]	Loss: 0.000631 	Lr:1.00E-04
    Train Epoch: 23 [9984/18300 (54%)]	Loss: 0.000782 	Lr:1.00E-04
    Train Epoch: 23 [12544/18300 (68%)]	Loss: 0.000465 	Lr:1.00E-04
    Train Epoch: 23 [15104/18300 (82%)]	Loss: 0.001370 	Lr:1.00E-04
    Train Epoch: 23 [17664/18300 (96%)]	Loss: 0.000462 	Lr:1.00E-04
    
    Test set: Average loss: 2.3566, Accuracy: 450/600 (75%)
    
    Train Epoch: 24 [2304/18300 (12%)]	Loss: 0.000325 	Lr:1.00E-04
    Train Epoch: 24 [4864/18300 (26%)]	Loss: 0.000587 	Lr:1.00E-04
    Train Epoch: 24 [7424/18300 (40%)]	Loss: 0.000962 	Lr:1.00E-04
    Train Epoch: 24 [9984/18300 (54%)]	Loss: 0.000584 	Lr:1.00E-04
    Train Epoch: 24 [12544/18300 (68%)]	Loss: 0.000466 	Lr:1.00E-04
    Train Epoch: 24 [15104/18300 (82%)]	Loss: 0.000846 	Lr:1.00E-04
    Train Epoch: 24 [17664/18300 (96%)]	Loss: 0.000391 	Lr:1.00E-04
    
    Test set: Average loss: 2.3591, Accuracy: 450/600 (75%)
    
    Train Epoch: 25 [2304/18300 (12%)]	Loss: 0.001401 	Lr:1.00E-04
    Train Epoch: 25 [4864/18300 (26%)]	Loss: 0.000639 	Lr:1.00E-04
    Train Epoch: 25 [7424/18300 (40%)]	Loss: 0.000330 	Lr:1.00E-04
    Train Epoch: 25 [9984/18300 (54%)]	Loss: 0.001660 	Lr:1.00E-04
    Train Epoch: 25 [12544/18300 (68%)]	Loss: 0.000475 	Lr:1.00E-04
    Train Epoch: 25 [15104/18300 (82%)]	Loss: 0.000507 	Lr:1.00E-04
    Train Epoch: 25 [17664/18300 (96%)]	Loss: 0.000453 	Lr:1.00E-04
    
    Test set: Average loss: 2.3600, Accuracy: 449/600 (75%)
    
    Train Epoch: 26 [2304/18300 (12%)]	Loss: 0.000462 	Lr:1.00E-04
    Train Epoch: 26 [4864/18300 (26%)]	Loss: 0.001772 	Lr:1.00E-04
    Train Epoch: 26 [7424/18300 (40%)]	Loss: 0.001300 	Lr:1.00E-04
    Train Epoch: 26 [9984/18300 (54%)]	Loss: 0.000458 	Lr:1.00E-04
    Train Epoch: 26 [12544/18300 (68%)]	Loss: 0.000521 	Lr:1.00E-04
    Train Epoch: 26 [15104/18300 (82%)]	Loss: 0.001024 	Lr:1.00E-04
    Train Epoch: 26 [17664/18300 (96%)]	Loss: 0.001051 	Lr:1.00E-04
    
    Test set: Average loss: 2.3633, Accuracy: 449/600 (75%)
    
    Train Epoch: 27 [2304/18300 (12%)]	Loss: 0.000975 	Lr:1.00E-04
    Train Epoch: 27 [4864/18300 (26%)]	Loss: 0.000169 	Lr:1.00E-04
    Train Epoch: 27 [7424/18300 (40%)]	Loss: 0.000472 	Lr:1.00E-04
    Train Epoch: 27 [9984/18300 (54%)]	Loss: 0.000648 	Lr:1.00E-04
    Train Epoch: 27 [12544/18300 (68%)]	Loss: 0.000974 	Lr:1.00E-04
    Train Epoch: 27 [15104/18300 (82%)]	Loss: 0.000657 	Lr:1.00E-04
    Train Epoch: 27 [17664/18300 (96%)]	Loss: 0.000292 	Lr:1.00E-04
    
    Test set: Average loss: 2.3654, Accuracy: 451/600 (75%)
    
    Train Epoch: 28 [2304/18300 (12%)]	Loss: 0.000852 	Lr:1.00E-04
    Train Epoch: 28 [4864/18300 (26%)]	Loss: 0.002082 	Lr:1.00E-04
    Train Epoch: 28 [7424/18300 (40%)]	Loss: 0.000808 	Lr:1.00E-04
    Train Epoch: 28 [9984/18300 (54%)]	Loss: 0.000805 	Lr:1.00E-04
    Train Epoch: 28 [12544/18300 (68%)]	Loss: 0.000585 	Lr:1.00E-04
    Train Epoch: 28 [15104/18300 (82%)]	Loss: 0.000565 	Lr:1.00E-04
    Train Epoch: 28 [17664/18300 (96%)]	Loss: 0.000159 	Lr:1.00E-04
    
    Test set: Average loss: 2.3669, Accuracy: 451/600 (75%)
    
    Train Epoch: 29 [2304/18300 (12%)]	Loss: 0.000581 	Lr:1.00E-04
    Train Epoch: 29 [4864/18300 (26%)]	Loss: 0.001014 	Lr:1.00E-04
    Train Epoch: 29 [7424/18300 (40%)]	Loss: 0.000724 	Lr:1.00E-04
    Train Epoch: 29 [9984/18300 (54%)]	Loss: 0.000292 	Lr:1.00E-04
    Train Epoch: 29 [12544/18300 (68%)]	Loss: 0.000853 	Lr:1.00E-04
    Train Epoch: 29 [15104/18300 (82%)]	Loss: 0.000546 	Lr:1.00E-04
    Train Epoch: 29 [17664/18300 (96%)]	Loss: 0.000566 	Lr:1.00E-04
    
    Test set: Average loss: 2.3699, Accuracy: 449/600 (75%)
    
    Train Epoch: 30 [2304/18300 (12%)]	Loss: 0.000877 	Lr:1.00E-05
    Train Epoch: 30 [4864/18300 (26%)]	Loss: 0.000531 	Lr:1.00E-05
    Train Epoch: 30 [7424/18300 (40%)]	Loss: 0.003624 	Lr:1.00E-05
    Train Epoch: 30 [9984/18300 (54%)]	Loss: 0.000412 	Lr:1.00E-05
    Train Epoch: 30 [12544/18300 (68%)]	Loss: 0.001050 	Lr:1.00E-05
    Train Epoch: 30 [15104/18300 (82%)]	Loss: 0.000485 	Lr:1.00E-05
    Train Epoch: 30 [17664/18300 (96%)]	Loss: 0.001094 	Lr:1.00E-05
    
    Test set: Average loss: 2.3700, Accuracy: 449/600 (75%)


​    

### 训练过程可视化
从上到下，从左往右三图依次是训练集的loss变化、测试集的正确率变化、测试集的loss变化
<img src="./images/res_show.png" />

### 总结
* 测试集的正确率基本在第10个epoch(76%)就开始下降了, 而此时lr为0.001
* 训练集的loss在不断下降，但测试集的loss值在不断上升，推测觉得是数据集太少或者数据集处理不当导致的，即过拟合了
* 最高一次准确率在81%，是自适应学习率每3个epoch减少10倍，但是要有个下限


```python

```
