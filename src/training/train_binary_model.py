import torch
import torch.nn as nn
import torch.optim as optim
from src.models.binary_health_classifier import get_model
from src.data.dataset_loader import get_dataloaders
from src.utils.config import MODEL_SAVE_PATH, NUM_EPOCHS, LEARNING_RATE
import os
from torch.utils.data import Subset
import random

def train_model(use_subset=True, subset_size=600):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data (now with data augmentation in training transform)
    train_loader_full, valid_loader_full, class_to_idx = get_dataloaders()
    
    if use_subset:
        print(f"Selecting a balanced subset of {subset_size} images per class for improved training speed.")
        train_ds = train_loader_full.dataset
        valid_ds = valid_loader_full.dataset
        
        # Helper to get balanced indices
        def get_balanced_subset(dataset, limit_per_class):
            indices = list(range(len(dataset)))
            # Group indices by class
            class_indices = {}
            for idx in indices:
                _, label = dataset.samples[idx]
                if label not in class_indices:
                    class_indices[label] = []
                class_indices[label].append(idx)
            
            subset_indices = []
            for label in class_indices:
                random.shuffle(class_indices[label])
                subset_indices.extend(class_indices[label][:limit_per_class])
            return subset_indices
            
        train_indices = get_balanced_subset(train_ds, subset_size)
        valid_indices = get_balanced_subset(valid_ds, subset_size // 3) # smaller valid set
        
        train_loader = torch.utils.data.DataLoader(
            Subset(train_ds, train_indices), batch_size=32, shuffle=True
        )
        valid_loader = torch.utils.data.DataLoader(
            Subset(valid_ds, valid_indices), batch_size=32, shuffle=False
        )
    else:
        train_loader = train_loader_full
        valid_loader = valid_loader_full

    print(f"Classes found: {class_to_idx}")

    # Initialize upgraded model (ResNet50)
    model = get_model(pretrained=True).to(device)

    # Loss and optimizer
    criterion = nn.BCELoss()
    # Using AdamW with weight decay for better regularization
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
    # Cosine Annealing scheduler for better performance
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

    # Training loop
    best_val_acc = 0.0

    for epoch in range(15): 
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if i % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}")

        epoch_acc = correct / total
        scheduler.step()

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                outputs = model(inputs)
                predicted = (outputs > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_epoch_acc = val_correct / val_total

        print(f"--- Epoch {epoch+1} Summary: Train Acc: {epoch_acc:.4f} | Val Acc: {val_epoch_acc:.4f} ---")

        # Save weights immediately on first epoch or if improved
        if val_epoch_acc >= best_val_acc or epoch == 0:
            best_val_acc = val_epoch_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"--> Saved improved health model to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    # Using 500 images per class for balanced training
    train_model(use_subset=True, subset_size=500)
