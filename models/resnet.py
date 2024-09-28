from torch import nn
import torch
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class ResNetBackbone(nn.Module):
    def __init__(self):
        super(ResNetBackbone, self).__init__()
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        for param in resnet.parameters():
            param.requires_grad = False
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Remove the last FC and AvgPool layers
        self.fc = nn.Linear(2048, 512)  # ResNet-50 outputs 2048-dim feature maps

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)  # Flatten the feature maps
        x = self.fc(x)
        return x