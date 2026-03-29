import torch
import torch.nn as nn
from torchvision import models

class MobileNetClassifier(nn.Module):
    def __init__(self, pretrained=True):
        super(MobileNetClassifier, self).__init__()
        # Use MobileNetV2 for speed and efficiency
        self.model = models.mobilenet_v2(pretrained=pretrained)
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_ftrs, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

def get_mobilenet_model(pretrained=True):
    return MobileNetClassifier(pretrained=pretrained)
