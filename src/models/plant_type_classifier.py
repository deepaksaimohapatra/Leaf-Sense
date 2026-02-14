import torch
import torch.nn as nn
from torchvision import models

class PlantTypeClassifier(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super(PlantTypeClassifier, self).__init__()
        # Using ResNet18 for plant identification as it's fast and sufficient for 3 classes
        self.model = models.resnet18(pretrained=pretrained)
        
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.model(x)

def get_plant_model(num_classes=3, pretrained=True):
    return PlantTypeClassifier(num_classes=num_classes, pretrained=pretrained)
