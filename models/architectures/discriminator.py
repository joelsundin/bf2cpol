import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscBlock(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            DiscBlock(64, 64, 128),
            DiscBlock(128, 128, 256),
            DiscBlock(256, 256, 512),
            DiscBlock(512, 512, 1024),
            DiscBlock(1024, 1024, 2048),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        # out = SIGMOID[FC[LReLU[FC[z]]]] ([1,2048] -> [1,2048] -> [1,1])
        self.fc = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(2048, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)