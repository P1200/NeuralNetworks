import torch.nn as nn
import torch.nn.functional as F


class MultiLayerNN(nn.Module):
    def __init__(self, input_size=64 * 64, num_classes=80, hidden_sizes=[512, 256]):
        super(MultiLayerNN, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])

        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])

        self.fc3 = nn.Linear(hidden_sizes[1], num_classes)

        self.dropout = nn.Dropout(0.2)  # Dropout dla regularyzacji

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Spłaszczenie wejścia (28x28 -> 784)

        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)

        x = self.fc3(x)  # Brak softmax - CrossEntropyLoss to załatwi

        return x
