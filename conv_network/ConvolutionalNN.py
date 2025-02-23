import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalNN(nn.Module):
    def __init__(self, num_classes=80):
        super(ConvolutionalNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # (64x64 → 32x32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # (32x32 → 16x16)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # (16x16 → 8x8)

        # Fully Connected
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (64x64 → 32x32)
        x = self.pool(F.relu(self.conv2(x)))  # (32x32 → 16x16)
        x = self.pool(F.relu(self.conv3(x)))  # (16x16 → 8x8)

        x = x.view(x.size(0), -1)  # (8x8x128)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        x = self.fc3(x)
        return x
