import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from preprocessing import train_dataset, train_loader
from cnn import SimpleCNN

# Initialize model, loss function, and optimizer
num_classes = len(train_dataset.classes)  # Adjust based on your dataset
model = SimpleCNN(num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")


