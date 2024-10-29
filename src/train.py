import torch
import torch.optim as optim
import torch.nn as nn
from preprocessing import train_dataset, train_loader
from cnn import SimpleCNN
from densenet import densenet121

# Check for GPU and set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("CUDA is available: {}".format(torch.cuda.is_available()))
if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    print("Using CPU")


# Initialize model, loss function, and optimizer
num_classes = len(train_dataset.classes)  # Ensure num_classes matches your dataset
model = densenet121(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with progress updates
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
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
        
        # Print progress every 100 batches
        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    
    # Print average loss per epoch
    print(f"Epoch {epoch + 1} completed. Average Loss: {running_loss / len(train_loader):.4f}")

# Optionally save model checkpoint after training
torch.save(model.state_dict(), "densenet121_trained.pth")

