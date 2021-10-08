
import torch
import torch.nn as nn



class HyperPNN(nn.Module):
    def __init__(self, n_layers = 24):
        super(HyperPNN, self).__init__()
        self.n_layers = n_layers
        self.conv1 = nn.Conv2d(102, 128, 1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(128,128,1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(129, 128, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.feature = []
        for i in range(self.n_layers):
            self.feature += [
                nn.Conv2d(128, 128, 3, 1, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace = True)
                ]
        self.feature = nn.Sequential(*self.feature)
        self.conv7 = nn.Conv2d(128, 102, 1)
        self.upsample = nn.Upsample(scale_factor = 4, mode = "nearest")


    def forward(self, x, y):
        out = self.upsample(x)
        out = self.relu(self.bn1(self.conv1(out)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = torch.cat([y, out], dim = 1)
        out = self.relu(self.bn3(self.conv3(out)))
        out = self.feature(out)
        out = self.conv7(out)
        return out
