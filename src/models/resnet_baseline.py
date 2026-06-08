import torch
import torch.nn as nn
import torchvision.models as models


class ResNet50Baseline(nn.Module):
    def __init__(self, num_classes: int = 8, pretrained: bool = True):
        super().__init__()
        self.backbone = models.resnet50(pretrained=pretrained)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)
