# Updated train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch import amp
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from densenet2 import initialize_densenet
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
import numpy as np
import os
import json


def main():
    # Configuration
    train_dir = '../../split_data/Train'
    eval_dir = '../../split_data/Evaluate'
    num_classes = 21
    num_epochs = 25
    batch_size = 64
    learning_rate = 0.001
    num_workers = 4

    # Data preparation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Custom Dataset class to handle corrupted images
    class SafeImageFolder(datasets.ImageFolder):
        def __getitem__(self, index):
            try:
                return super(SafeImageFolder, self).__getitem__(index)
            except (OSError, ValueError) as e:
                print(f"Warning: Skipping corrupted image at index {index}: {e}")
                return None

    train_dataset = SafeImageFolder(root=train_dir, transform=transform)
    eval_dataset = SafeImageFolder(root=eval_dir, transform=transform)

    # Filter out None items resulting from corrupted images
    train_dataset.samples = [s for s in train_dataset.samples if os.path.exists(s[0])]
    eval_dataset.samples = [s for s in eval_dataset.samples if os.path.exists(s[0])]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=safe_collate)

    # Initialize the model
    model = initialize_densenet(num_classes=num_classes)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Enable mixed precision training for faster computation
    scaler = amp.GradScaler('cuda')

    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Initialize dictionary to store experimental results
    experiment_results = {}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, data in enumerate(train_loader):
            if data is None:
                continue
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass with mixed precision
            with amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # Backward pass and optimize
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Print statistics
            running_loss += loss.item()
            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {running_loss / 10:.4f}')
                running_loss = 0.0

        # Evaluate the model after each epoch
        eval_metrics = evaluate_model(model, eval_loader, device, num_classes)
        train_metrics = evaluate_model(model, train_loader, device, num_classes)

        # Store metrics for training and evaluation
        experiment_results[epoch + 1] = {
            'train_metrics': train_metrics,
            'eval_metrics': eval_metrics
        }

    # Save the experimental results to a JSON file
    with open('experiment_results.json', 'w') as f:
        json.dump(experiment_results, f, indent=4)

    # Save the final trained model
    torch.save(model.state_dict(), 'densenet_model.pth')

# Define the custom collate function to filter out None items
def safe_collate(batch):
    # Only include valid (non-None) data points
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None  # Return None if all items are invalid
    return torch.utils.data.dataloader.default_collate(batch)

def evaluate_model(model, data_loader, device, num_classes):
    model.eval()
    all_labels = []
    all_predictions = []

    # Use torch.no_grad to prevent gradient calculations during evaluation
    with torch.no_grad():
        for data in data_loader:
            if data is None:
                continue
            
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=1)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=1)
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=1)
    conf_matrix = confusion_matrix(all_labels, all_predictions, normalize='true')

    metrics = {
        'accuracy': accuracy,
        'f1_score': f1,
        'recall': recall,
        'precision': precision,
        'confusion_matrix': conf_matrix.tolist()  # Convert to list for JSON serialization
    }

    return metrics

if __name__ == '__main__':
    main()
