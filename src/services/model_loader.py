import torch
import os
from src.models.binary_health_classifier import get_model as get_resnet50
from src.models.custom_cnn import get_custom_cnn_model
from src.models.mobilenet import get_mobilenet_model
from src.models.efficientnet import get_efficientnet_model
from src.utils.config import MODEL_SAVE_PATH

class ModelLoader:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}

    def get_model(self, model_name="resnet50"):
        model_name = model_name.lower()
        if model_name in self.models:
            return self.models[model_name]

        print(f"Loading model architecture: {model_name}...")
        
        if model_name == "resnet50":
            model = get_resnet50(pretrained=False)
            if os.path.exists(MODEL_SAVE_PATH):
                model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=self.device), strict=False)
                print(f"Loaded trained weights for ResNet50 from {MODEL_SAVE_PATH}")
            else:
                print(f"Warning: Model file not found at {MODEL_SAVE_PATH}. Using untrained model fallback.")
                model = get_resnet50(pretrained=True)
                
        elif model_name == "cnn":
            # Initialize custom CNN (No pretrained weights available for custom arch by default)
            model = get_custom_cnn_model()
            print("Loaded Custom CNN (Untrained for specific classes)")
            
        elif model_name == "mobilenet":
            # Initialize MobileNetV2 with ImageNet pretrained features, classification head is new
            model = get_mobilenet_model(pretrained=True)
            print("Loaded MobileNetV2 (Feature extractor pretrained)")
            
        elif model_name == "efficientnet":
            # Initialize EfficientNet-B0 with ImageNet pretrained features, classification head is new
            model = get_efficientnet_model(pretrained=True)
            print("Loaded EfficientNet-B0 (Feature extractor pretrained)")
            
        else:
            raise ValueError(f"Unknown model architecture: {model_name}")

        model.to(self.device)
        model.eval()
        self.models[model_name] = model
        return model
        
    def preload_all_models(self):
        """Preload all models into memory to ensure fast inference on API requests."""
        architectures = ["resnet50", "cnn", "mobilenet", "efficientnet"]
        for arch in architectures:
            try:
                self.get_model(arch)
            except Exception as e:
                print(f"Failed to load {arch}: {e}")

# Singleton instance
model_loader_instance = ModelLoader()
