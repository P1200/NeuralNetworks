import torch.nn as nn
import torch.nn.functional as F


class ConvolutionNN(nn.Module):
    def __init__(self, num_classes):
        super(ConvolutionNN, self).__init__()

        # Warstwa konwolucyjna 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Warstwa konwolucyjna 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Warstwa konwolucyjna 3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Fully Connected Layer
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # (64x64) → (32x32)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # (32x32) → (16x16)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # (16x16) → (8x8)

        x = x.view(x.size(0), -1)  # Flatten (8x8x128)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
