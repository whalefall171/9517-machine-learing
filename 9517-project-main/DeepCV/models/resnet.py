import torch.nn as nn
from torchvision import models

def build_resnet18(num_classes=15, pretrained=True):
    model = models.resnet18(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
