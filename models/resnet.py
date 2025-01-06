from torch import nn
import torch
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class ResNetBackbone(nn.Module):
    def __init__(self, d_model):
        super(ResNetBackbone, self).__init__()
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Remove the last FC and AvgPool layers
        self.channel_projector = nn.Conv2d(2048, d_model, kernel_size=1)  # Project to d_model channels

    def forward(self, x):
        x = self.backbone(x)  # Output shape: [batch_size, 2048, H, W]
        x = self.channel_projector(x)  # Output shape: [batch_size, d_model, H, W]
        return x
