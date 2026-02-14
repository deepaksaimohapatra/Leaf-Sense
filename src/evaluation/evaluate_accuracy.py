import os
import sys
import torch
from PIL import Image

# Add base directory to path to reach src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.inference.health_classifier import get_health_classifier
from src.utils.config import BASE_DIR

from src.data.dataset_loader import get_dataloaders

def evaluate_with_dataloader(classifier, dataloader, name="Validation Set", limit=None):
    device = classifier.device
    model = classifier.model
    model.eval()
    
    total = 0
    correct = 0
    tp = 0; tn = 0; fp = 0; fn = 0
    
    print(f"\nEvaluating {name} using Batch Processing...")
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            if limit and total >= limit:
                break
                
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Confusion matrix
            tp += ((predicted == 1) & (labels == 1)).sum().item()
            tn += ((predicted == 0) & (labels == 0)).sum().item()
            fp += ((predicted == 1) & (labels == 0)).sum().item()
            fn += ((predicted == 0) & (labels == 1)).sum().item()
            
            if (i+1) % 10 == 0:
                print(f"Processed {total} images...")

    accuracy = correct / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("-" * 50)
    print(f"RESULTS FOR {name}")
    print("-" * 50)
    print(f"Total: {total} | Accuracy: {accuracy*100:.2f}%")
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    return {"accuracy": accuracy, "f1": f1}

def evaluate_directory(classifier, dir_path, name="Test Set"):
    if not os.path.exists(dir_path):
        print(f"Error: Directory not found at {dir_path}")
        return None

    all_files = []
    # Flat structure only for the random test dir
    for img_name in os.listdir(dir_path):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            gt_label = "Healthy" if "healthy" in img_name.lower() else "Unhealthy"
            all_files.append((os.path.join(dir_path, img_name), img_name, gt_label))

    total = len(all_files)
    if total == 0:
        print(f"No images found in {name}.")
        return None

    print(f"\nEvaluating {name} on {total} images...")
    
    correct = 0
    tp = 0; tn = 0; fp = 0; fn = 0

    for i, (full_path, filename, gt_label) in enumerate(all_files):
        try:
            img = Image.open(full_path).convert("RGB")
            result = classifier.predict(img)
            pred_label = result["prediction"]
            
            is_correct = (pred_label == gt_label)
            if is_correct:
                correct += 1
                if gt_label == "Unhealthy": tp += 1
                else: tn += 1
            else:
                if gt_label == "Healthy": fp += 1
                else: fn += 1
            
            if (i+1) % 10 == 0:
                print(f"Processed {i+1}/{total} images...")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    accuracy = correct / total
    print("-" * 50)
    print(f"RESULTS FOR {name}")
    print("-" * 50)
    print(f"Total: {total} | Accuracy: {accuracy*100:.2f}%")
    return {"accuracy": accuracy}

def evaluate_accuracy(limit=None):
    test_dir = os.path.join(BASE_DIR, "data", "PlantDiseaseDataset", "test")
    classifier = get_health_classifier()
    
    # 1. Evaluate on Random Test Set (Flat files)
    evaluate_directory(classifier, test_dir, "Random Test Set")
    
    # 2. Evaluate on Validation Set (Using efficient DataLoader)
    _, valid_loader, _ = get_dataloaders(label_mode='health')
    evaluate_with_dataloader(classifier, valid_loader, "Validation Set", limit=limit)

if __name__ == "__main__":
    # Test with a smaller limit first if requested, otherwise run full
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    evaluate_accuracy(limit=limit)
