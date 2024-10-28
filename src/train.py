import torch
import torch.optim as optim
import torch.nn as nn
from preprocessing import train_dataset, train_loader
from cnn import SimpleCNN

# Check for GPU and set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("CUDA is available: {}".format(torch.cuda.is_available()))
if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    print("Using CPU")

# Initialize model, loss function, and optimizer
num_classes = len(train_dataset.classes)  # Adjust based on your dataset
model = SimpleCNN(num_classes=num_classes).to(device)  # Move model to the selected device
criterion = nn.CrossEntropyLoss().to(device)            # Move criterion to the device
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 1
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        # Move inputs and labels to the selected device
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item()

    # Print epoch loss
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")



