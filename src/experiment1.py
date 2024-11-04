import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import densenet121, DenseNet121_Weights
import matplotlib.pyplot as plt
from collections import defaultdict
import os

# Experiment 1
# To produce graph showing Epoch vs Accuracy and Epoch vs Loss


def train_and_evaluate():
    # Configuration
    train_dir = '../../split_data/Train'
    eval_dir = '../../split_data/Evaluate'
    num_classes = 21  # Update according to your dataset
    num_epochs = 10  # You can adjust as needed
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
                return None, None

    train_dataset = SafeImageFolder(root=train_dir, transform=transform)
    eval_dataset = SafeImageFolder(root=eval_dir, transform=transform)

    # Filter out None items resulting from corrupted images
    train_dataset.samples = [s for s in train_dataset.samples if os.path.exists(s[0])]
    eval_dataset.samples = [s for s in eval_dataset.samples if os.path.exists(s[0])]

    def safe_collate(batch):
        batch = [item for item in batch if item[0] is not None and item[1] is not None]
        return torch.utils.data.dataloader.default_collate(batch)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=safe_collate)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=safe_collate)

    # Initialise the model
    model = densenet121(weights=DenseNet121_Weights.DEFAULT)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # For tracking loss and accuracy
    training_loss_history = []
    validation_loss_history = []
    training_accuracy_history = []
    validation_accuracy_history = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        training_loss_history.append(epoch_loss)
        training_accuracy_history.append(epoch_accuracy)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.2f}%')

        # Evaluation Phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in eval_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(eval_loader)
        epoch_accuracy = 100 * correct / total
        validation_loss_history.append(epoch_loss)
        validation_accuracy_history.append(epoch_accuracy)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {epoch_loss:.4f}, Validation Accuracy: {epoch_accuracy:.2f}%')

    # Plot Loss vs Epoch
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), training_loss_history, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), validation_loss_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Epoch vs Loss')
    plt.legend()
    plt.show()

    # Plot Accuracy vs Epoch
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), training_accuracy_history, label='Training Accuracy')
    plt.plot(range(1, num_epochs + 1), validation_accuracy_history, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Epoch vs Accuracy')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    train_and_evaluate()
