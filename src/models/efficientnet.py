import torch
import torch.nn as nn
from torchvision import models

class EfficientNetClassifier(nn.Module):
    def __init__(self, pretrained=True):
        super(EfficientNetClassifier, self).__init__()
        # EfficientNet-B0
        self.model = models.efficientnet_b0(pretrained=pretrained)
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_ftrs, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

def get_efficientnet_model(pretrained=True):
    return EfficientNetClassifier(pretrained=pretrained)
