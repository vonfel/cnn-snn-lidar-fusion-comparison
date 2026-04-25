import torch.nn as nn


class CNNEncoder(nn.Module):
    """
    3-layer CNN encoder for depth-only (in_channels=1)
    or early fusion depth+time_surface (in_channels=3).

    Architecture:
        Layer 1: Conv(in, 16, 3) + BN + ReLU         -> (B, 16, 480, 640)
        Layer 2: Conv(16, 32, 3, stride=2) + BN + ReLU -> (B, 32, 240, 320)
        Layer 3: Conv(32, 64, 3, stride=2) + BN + ReLU -> (B, 64, 120, 160)

    Output: (B, 64, 120, 160) feature map.
    """

    def __init__(self, in_channels=1):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )  # -> (B, 32, 240, 320)
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )  # -> (B, 64, 120, 160)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
