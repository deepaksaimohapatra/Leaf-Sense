import torch
from torchvision import transforms
from PIL import Image

def get_preprocessing_pipeline(img_size=(224, 224), augment=False):
    """
    Returns the preprocessing pipeline. 
    If augment=True, includes data augmentation for training.
    """
    if augment:
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def preprocess_image(image_path, img_size=(224, 224), augment=False):
    """
    Loads an image, applies preprocessing, and adds batch dimension.
    """
    if isinstance(image_path, str):
        image = Image.open(image_path).convert("RGB")
    else:
        # Assume it's a PIL image/file-like object
        image = image_path.convert("RGB")
        
    pipeline = get_preprocessing_pipeline(img_size, augment=augment)
    tensor = pipeline(image).unsqueeze(0) # Add batch dimension
    return tensor
