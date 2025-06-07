import torch.nn as nn
from torchvision import models

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SEBasicBlock(nn.Module):
    def __init__(self, block):
        super(SEBasicBlock, self).__init__()
        self.block = block
        self.se = SEBlock(block.conv2.out_channels)

    def forward(self, x):
        out = self.block(x)
        out = self.se(out)
        return out

def build_senet_resnet18(num_classes=15, pretrained=True):
    base_model = models.resnet18(pretrained=pretrained)
    base_model.layer4[0] = SEBasicBlock(base_model.layer4[0])
    in_features = base_model.fc.in_features
    base_model.fc = nn.Linear(in_features, num_classes)
    return base_model
