import torch
import torch.nn as nn
from torchvision import models

class BinaryHealthClassifier(nn.Module):
    def __init__(self, pretrained=True):
        super(BinaryHealthClassifier, self).__init__()
        self.model = models.resnet50(pretrained=pretrained)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_ftrs, 1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        return self.model(x)

def get_model(pretrained=True):
    return BinaryHealthClassifier(pretrained=pretrained)
