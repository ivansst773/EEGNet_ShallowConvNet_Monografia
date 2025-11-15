import torch
import torch.nn as nn

class EEGNet(nn.Module):
    def __init__(self, n_channels, n_times, n_classes, dropout=0.25):
        super().__init__()
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(16)
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, (n_channels, 1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout)
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, (1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout)
        )
        dummy = torch.zeros(1, 1, n_channels, n_times)
        out = self._forward_features(dummy)
        in_feats = out.view(1, -1).size(1)
        self.classify = nn.Linear(in_feats, n_classes)

    def _forward_features(self, x):
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        return self.classify(x)

class ShallowConvNet(nn.Module):
    def __init__(self, n_channels, n_times, n_classes, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 40, (1, 25), bias=False)
        self.conv2 = nn.Conv2d(40, 40, (n_channels, 1), bias=False)
        self.batchnorm = nn.BatchNorm2d(40)
        self.pool = nn.AvgPool2d((1, 75), stride=(1, 15))
        self.dropout = nn.Dropout(dropout)
        dummy = torch.zeros(1, 1, n_channels, n_times)
        out = self._forward_features(dummy)
        in_feats = out.view(1, -1).size(1)
        self.fc = nn.Linear(in_feats, n_classes)

    def _forward_features(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.batchnorm(x)
        x = torch.square(x)
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))
        x = self.dropout(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
