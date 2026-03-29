import torch
from src.services.model_loader import model_loader_instance
from src.preprocessing.image_preprocessing import preprocess_image
from src.utils.config import IMG_SIZE
import io
from PIL import Image

class PredictorService:
    @staticmethod
    def predict_health(image, model_name="resnet50"):
        """
        Runs inference on an image using the specified model.
        Supports PIL Image or raw bytes.
        """
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
            
        device = model_loader_instance.device
        model = model_loader_instance.get_model(model_name)
        
        tensor = preprocess_image(image, IMG_SIZE).to(device)
        
        with torch.no_grad():
            output = model(tensor)
            confidence = output.item()
            
            if confidence > 0.5:
                label = "Diseased" # Changed from Unhealthy to Diseased per requirement
                score = confidence
            else:
                label = "Healthy"
                score = 1.0 - confidence
                
        return {
            "prediction": label,
            "confidence": round(score, 4),
            "model_used": model_name
        }
