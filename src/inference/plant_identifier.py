import torch
import os
from src.models.plant_type_classifier import get_plant_model
from src.preprocessing.image_preprocessing import preprocess_image
from src.utils.config import IMG_SIZE, MODEL_SAVE_PATH

class PlantIdentifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = ["Apple", "Potato", "Tomato"]
        
        # Initialize and load the trained model
        self.model = get_plant_model(num_classes=len(self.classes), pretrained=False)
        model_path = os.path.join(os.path.dirname(MODEL_SAVE_PATH), "plant_type_model.pth")
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded trained plant type model from {model_path}")
        else:
            print(f"WARNING: Trained plant type model not found at {model_path}. Using pre-trained weights.")
            
        self.model.to(self.device).eval()
        
    def predict(self, image_path):
        """
        Identifies the plant type using the trained classifier.
        """
        tensor = preprocess_image(image_path, IMG_SIZE).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
        prob, idx = torch.max(probs, 1)
        detected_plant = self.classes[idx.item()]
        confidence = prob.item()
            
        return {
            "plant": detected_plant,
            "confidence": round(confidence, 4),
            "message": f"Our analysis indicates this is likely an **{detected_plant}** leaf."
        }

# Singleton
_identifier = None

def identify_plant(image_path):
    global _identifier
    if _identifier is None:
        _identifier = PlantIdentifier()
    return _identifier.predict(image_path)
