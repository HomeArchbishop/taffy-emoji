import torch.nn as nn
from torchvision import models

IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def build_model(pretrained: bool = True) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    m = models.efficientnet_b0(weights=weights)
    m.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(m.classifier[1].in_features, 1),
    )
    return m
