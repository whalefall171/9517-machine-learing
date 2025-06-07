import torch.nn as nn
from torchvision import models

def build_efficientnet_b0(num_classes=15, pretrained=True):
    model = models.efficientnet_b0(pretrained=pretrained)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model
