import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class DWSConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DWSConv, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, 
                                                groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),                                 
            nn.Conv2d(in_channels, out_channels, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.main(x)

class DWSCWConv(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(DWSCWConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, padding=1, groups=in_channels, bias=False),
            GCWConv(1, kernel_size, kernel_size[0]//2)
        )
    
    def forward(self, x):
        return self.conv(x)

class GCWConv(nn.Module):
    def __init__(self, group, kernel_size, padding):
        super(GCWConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1, group, kernel_size,
                                stride=(group, 1, 1), 
                                padding=(padding, 0, 0), bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.view(x.size(0), -1, x.size(3), x.size(4))
        return x


class GM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GM, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, padding=1, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, in_channels, 1, 1, groups=2, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, 1, padding=1, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, 1, 1, groups=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.main(x) + x

class GCWM(nn.Module):
    def __init__(self, in_channels, out_channels, dc):
        super(GCWM, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, padding=1, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, in_channels, 1, 1, groups=2, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, 1, padding=1, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, 1, 1, groups=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            GCWConv(2, (dc, 1, 1), dc//2 - 1)
        )
    def forward(self, x):
        return self.main(x) + x

class CCL(nn.Module):
    def __init__(self, m, n, feature_size):
        super(CCL, self).__init__()
        self.main = nn.Sequential(
            GCWConv(1, (m - n + 1, feature_size, feature_size), 0)
        )
        
    def forward(self, x):
        return self.main(x)

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size
    def forward(self, x):
        return x.view(self.size)


class ChannelNet(nn.Module):
    def __init__(self, v=3, num_class=100):
        super(ChannelNet, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            DWSConv(32, 64, stride=1),
            DWSConv(64, 128, stride=2),
            DWSConv(128, 128, stride=1),
            DWSConv(128, 256, stride=1),
            DWSConv(256, 256, stride=1),
            DWSConv(256, 512, stride=2),
            GCWM(512, 512, 8),
            GCWM(512, 512, 8),
            GM(512, 512),
            DWSConv(512, 1024, stride=1),
        )
        if v==1:
            self.classifier = nn.Sequential(
                DWSConv(1024, 1024, stride=1),
                nn.AvgPool2d(8),
                View([-1, 1024]),
                nn.Linear(1024, num_class),
            )
        elif v==2:
            self.classifier = nn.Sequential(
                DWSCWConv(1024, (65, 1, 1)),
                nn.AvgPool2d(8),
                View([-1, 1024]),
                nn.Linear(1024, num_class),
            )
        elif v==3:
            self.classifier = nn.Sequential(
                DWSCWConv(1024, (65, 1, 1)),
                CCL(1024, num_class, 8),
            )

    def forward(self, x):
        x = self.main(x)
        x = self.classifier(x).squeeze()
        return x