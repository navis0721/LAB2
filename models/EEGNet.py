import torch.nn as nn
import torch


# TODO implement EEGNet model
class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()
        self.dropout = 0.25
        
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 51), padding=(0, 25), bias=False),
            nn.BatchNorm2d(16)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2, 1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(self.dropout)
        )
        
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15), padding=(0, 7), bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
            nn.Dropout(self.dropout)
        )
        
        self.out = nn.Linear(in_features=736, out_features=2)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        x = x.view(x.size(0), -1)
        x = self.out(x)
        
        return x


# (Optional) implement DeepConvNet model
class DeepConvNet(nn.Module):
    def __init__(self, nCh: int, nTime: int, nCls: int):
        super(DeepConvNet, self).__init__()
        self.nCh = nCh
        self.nTime = nTime
        self.dropout = 0.5
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 5), bias=False),
            nn.Conv2d(25, 25, (nCh, 1), bias=False),
            nn.BatchNorm2d(25),
            nn.LeakyReLU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(self.dropout)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(25, 50, (1, 5), bias=False),
            nn.BatchNorm2d(50),
            nn.LeakyReLU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(self.dropout)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(50, 100, (1, 5), bias=False),
            nn.BatchNorm2d(100),
            nn.LeakyReLU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(self.dropout)
        )
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(100, 200, (1, 5), bias=False),
            nn.BatchNorm2d(200),
            nn.LeakyReLU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(self.dropout)
        )
        self.conv_layer = nn.Sequential(self.layer1, self.layer2, self.layer3, self.layer4)
        
        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._forward_flatten().shape[1], nCls),
        )
        
    def _forward_flatten(self):
        x = torch.rand(1, 1, self.nCh, self.nTime)
        x = self.conv_layer(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        return x


    def forward(self, x):
        # print("size of x:", x.size())
        x = self.conv_layer(x)
        x = self.out(x)
        return x