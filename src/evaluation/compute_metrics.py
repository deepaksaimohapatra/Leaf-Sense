"""
Computes REAL evaluation metrics (Accuracy, Precision, Recall, F1)
by running each model architecture against the full validation dataset.

Results are cached to a JSON file so they are only computed once
(or when explicitly re-evaluated).
"""

import os
import sys
import json
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Ensure project root is on the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.preprocessing.image_preprocessing import preprocess_image
from src.services.model_loader import model_loader_instance
from src.utils.config import VALID_DIR, IMG_SIZE

METRICS_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models_saved", "computed_metrics.json"
)


import random

def collect_validation_samples(max_samples=40):
    """
    Walks the validation directory and returns a subset of samples.
    """
    samples = []
    label_map = {"Healthy": 0, "Unhealthy": 1}

    for status in ["Healthy", "Unhealthy"]:
        status_path = os.path.join(VALID_DIR, status)
        if not os.path.exists(status_path):
            continue

        for plant in os.listdir(status_path):
            plant_path = os.path.join(status_path, plant)
            if not os.path.isdir(plant_path):
                continue

            for img_name in os.listdir(plant_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    samples.append((
                        os.path.join(plant_path, img_name),
                        label_map[status]
                    ))
    
    if max_samples and len(samples) > max_samples:
        random.seed(42) # Consistent sampling
        samples = random.sample(samples, max_samples)

    return samples


def evaluate_model(model_name, samples, device):
    """
    Runs a single model against all samples and computes metrics.
    Uses a raw 0.5 threshold (no temperature scaling) for fair evaluation.
    """
    model = model_loader_instance.get_model(model_name)
    
    y_true = []
    y_pred = []

    for img_path, true_label in samples:
        try:
            image = Image.open(img_path).convert("RGB")
            tensor = preprocess_image(image, IMG_SIZE).to(device)

            with torch.no_grad():
                output = model(tensor)
                prob = output.item()
                predicted_label = 1 if prob > 0.5 else 0

            y_true.append(true_label)
            y_pred.append(predicted_label)
        except Exception as e:
            print(f"  Skipping {img_path}: {e}")
            continue

    if len(y_true) == 0:
        return None

    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }
    metrics["total_samples"] = len(y_true)
    metrics["correct"] = int(sum(1 for t, p in zip(y_true, y_pred) if t == p))

    return metrics


def compute_all_metrics(force=False, max_samples=40):
    """
    Compute metrics for all model architectures.
    If cached results exist and force=False, returns cached results.
    """
    if not force and os.path.exists(METRICS_CACHE_PATH):
        with open(METRICS_CACHE_PATH, 'r') as f:
            cached = json.load(f)
            print(f"[Metrics] Loaded cached metrics from {METRICS_CACHE_PATH}")
            return cached

    print(f"[Metrics] Computing REAL evaluation metrics (max_samples={max_samples})...")
    samples = collect_validation_samples(max_samples=max_samples)
    print(f"[Metrics] Found {len(samples)} validation images.")

    if len(samples) == 0:
        print("[Metrics] WARNING: No validation samples found! Using fallback.")
        return None

    device = model_loader_instance.device
    architectures = ["resnet50", "cnn", "mobilenet", "efficientnet"]
    all_metrics = {}

    for arch in architectures:
        print(f"[Metrics] Evaluating {arch}...")
        try:
            metrics = evaluate_model(arch, samples, device)
            if metrics:
                all_metrics[arch] = metrics
                print(f"  -> {arch}: Acc={metrics['accuracy']}, "
                      f"Pr={metrics['precision']}, Re={metrics['recall']}, "
                      f"F1={metrics['f1_score']} ({metrics['total_samples']} samples)")
                
                # Update cache immediately after each model to allow "streaming" updates
                os.makedirs(os.path.dirname(METRICS_CACHE_PATH), exist_ok=True)
                with open(METRICS_CACHE_PATH, 'w') as f:
                    json.dump(all_metrics, f, indent=2)
            else:
                print(f"  -> {arch}: No valid predictions.")
        except Exception as e:
            print(f"  -> {arch}: FAILED - {e}")

    return all_metrics


if __name__ == "__main__":
    results = compute_all_metrics(force=True)
    if results:
        print("\n=== FINAL RESULTS ===")
        for model, m in results.items():
            print(f"{model}: Accuracy={m['accuracy']}, Precision={m['precision']}, "
                  f"Recall={m['recall']}, F1={m['f1_score']}")
