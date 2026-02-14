import torch
import torch.nn as nn
import torch.optim as optim
from src.models.plant_type_classifier import get_plant_model
from src.data.dataset_loader import get_dataloaders
from src.utils.config import MODEL_SAVE_PATH, NUM_EPOCHS, LEARNING_RATE
import os
from torch.utils.data import Subset
import random

def train_plant_model(subset_size=1000):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data for plant classification
    train_loader_full, valid_loader_full, idx_to_class = get_dataloaders(label_mode='plant')
    print(f"Plant Index Mapping: {idx_to_class}")

    # Use a subset for training if dataset is too large, but ensure it is balanced
    def get_balanced_subset(dataset, limit_per_class):
        indices = list(range(len(dataset)))
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

    train_indices = get_balanced_subset(train_loader_full.dataset, subset_size)
    valid_indices = get_balanced_subset(valid_loader_full.dataset, subset_size // 4)

    train_loader = torch.utils.data.DataLoader(
        Subset(train_loader_full.dataset, train_indices), batch_size=32, shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(
        Subset(valid_loader_full.dataset, valid_indices), batch_size=32, shuffle=False
    )

    # Initialize model
    model = get_plant_model(num_classes=len(idx_to_class)).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
    # Cosine Annealing scheduler for better convergence
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # Training loop
    best_val_acc = 0.0
    save_path = os.path.join(os.path.dirname(MODEL_SAVE_PATH), "plant_type_model.pth")

    for epoch in range(10):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
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
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_epoch_acc = val_correct / val_total
        print(f"--- Epoch {epoch+1} Summary: Train Acc: {epoch_acc:.4f} | Val Acc: {val_epoch_acc:.4f} ---")

        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved improved plant type model to {save_path}")

if __name__ == "__main__":
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    train_plant_model(subset_size=1000)
