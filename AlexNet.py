import torch
import torch.nn as nn
class AlexNet(nn.Module):
  def __init__(self):
    super(AlexNet,self).__init__()
    self.features = nn.Sequential(
        nn.Conv2d(3,96,kernel_size = 11, stride = 4, padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(96, 192, kernel_size=5, stride = 1, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(192,384, kernel_size=3, stride = 1, padding = 1),
        nn.ReLU(),
        nn.Conv2d(384,256,kernel_size=3,stride = 2, padding = 1),
        nn.ReLU(), 
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride = 2),
    )
    self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
    self.classifier = nn.Sequential(
        nn.Dropout(),
        nn.Linear(256 * 6 * 6, 4096),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Linear(4096, 1000),
      )
  def forward(self, x):
      x = self.features(x)
      x = self.avgpool(x)
      x = torch.flatten(x, 1)
      x = self.classifier(x)
      return x


x = torch.rand(64,3,227,227)
model = AlexNet()
print(model(x).shape)