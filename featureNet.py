import torch.nn as nn
from torchsummary import summary


class featureNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.BatchNorm2d(1),
                                 nn.Conv2d(1, 11, kernel_size=3, stride=1, padding=1),
                                 nn.Conv2d(11, 11, kernel_size=3, stride=1, padding=1),
                                 nn.ReLU(),
                                 nn.MaxPool2d(kernel_size=2, stride=1),
                                 nn.Conv2d(11, 22, kernel_size=3, stride=1, padding=1),
                                 nn.Conv2d(22, 22, kernel_size=3, stride=1, padding=1),
                                 nn.ReLU(),
                                 nn.MaxPool2d(kernel_size=2, stride=1),
                                 nn.Conv2d(22, 44, kernel_size=3, stride=1, padding=1),
                                 nn.Conv2d(44, 44, kernel_size=3, stride=1, padding=1),
                                 nn.ReLU(),
                                 nn.MaxPool2d(kernel_size=2, stride=1),
                                 nn.Flatten(),
                                 nn.Linear(44*9*197, 4096),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(4096, 4096),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(4096, 8),
                                 )

    def forward(self, x):
        return self.net(x)


net = featureNet()
print(summary(net, (1, 12, 200)))