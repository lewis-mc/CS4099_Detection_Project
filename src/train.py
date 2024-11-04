import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from densenet2 import initialize_densenet
from collections import defaultdict
from PIL import Image
import os


def main():
    # Configuration
    train_dir = '../../split_data/Train'
    eval_dir = '../../split_data/Evaluate'
    num_classes = 21 
    num_epochs = 1
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
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Initialize the model
    model = initialize_densenet(num_classes=num_classes)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Enable mixed precision training for faster computation
    scaler = torch.cuda.amp.GradScaler('cuda')

    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, data in enumerate(train_loader):
            if data is None:
                continue
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
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

    # Save the trained model
    torch.save(model.state_dict(), 'densenet_model.pth')

if __name__ == '__main__':
    main()

