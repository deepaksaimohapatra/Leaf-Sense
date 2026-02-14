import torch
from src.models.binary_health_classifier import get_model
from src.preprocessing.image_preprocessing import preprocess_image
from src.utils.config import MODEL_SAVE_PATH, IMG_SIZE, CLASSES
import os

class HealthClassifier:
    def __init__(self, model_path=MODEL_SAVE_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_model(pretrained=False)
        
        if os.path.exists(model_path):
            missing_keys, unexpected_keys = self.model.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)
            print(f"Loaded model from {model_path}")
            if missing_keys: print(f"Missing keys: {missing_keys}")
            if unexpected_keys: print(f"Unexpected keys: {unexpected_keys}")
        else:
            print(f"Warning: Model file not found at {model_path}. Using untrained model.")
            
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image_path):
        """
        Predicts whether the leaf is Healthy or Unhealthy.
        """
        tensor = preprocess_image(image_path, IMG_SIZE).to(self.device)
        
        with torch.no_grad():
            output = self.model(tensor)
            confidence = output.item()
            
            # Label mapping: labels were 0 and 1 in training
            # Typically 0:Healthy, 1:Unhealthy if sorted alphabetically by ImageFolder
            # Let's assume class 0 is Healthy, 1 is Unhealthy
            if confidence > 0.5:
                label = "Unhealthy"
                score = confidence
            else:
                label = "Healthy"
                score = 1 - confidence
                
        return {
            "prediction": label,
            "confidence": round(score, 4)
        }

# Singleton instance
_classifier = None

def get_health_classifier():
    global _classifier
    if _classifier is None:
        _classifier = HealthClassifier()
    return _classifier
