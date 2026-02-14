import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "PlantDiseaseDataset")

# Dataset splits
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VALID_DIR = os.path.join(DATA_DIR, "valid")
TEST_DIR = os.path.join(DATA_DIR, "test", "random_images")

# Model parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 10

# Model storage
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models_saved", "binary_model.pt")

# Classes
CLASSES = ["Healthy", "Unhealthy"]
# Placeholder for plant identification later
PLANTS = ["Apple"] 
