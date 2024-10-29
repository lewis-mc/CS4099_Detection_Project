import torch
import torch.nn as nn
from densenet import densenet121  # Import your DenseNet-121 model implementation
from preprocessing import eval_loader, eval_dataset  # Import your eval_loader and eval_dataset
import os

# Set up device for GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("CUDA is available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    print("Using CPU")

# Set the number of classes based on the eval_dataset directory structure
num_classes = len(eval_dataset.classes)
class_names = eval_dataset.classes  # Assuming eval_dataset has `classes` with class names

# Initialize and load the trained model
model = densenet121(num_classes=num_classes).to(device)
model.load_state_dict(torch.load("densenet121_trained.pth", map_location=device, weights_only=True))  # Load the model
model.eval()  # Set model to evaluation mode

# Define the evaluation function
def evaluate_model(model, eval_loader):
    # Initialize counters for each class
    correct_per_class = [0] * num_classes
    total_per_class = [0] * num_classes

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for inputs, labels in eval_loader:
            # Move data to the device
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass to get predictions
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability

            # Update counts for each class
            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    correct_per_class[label.item()] += 1
                total_per_class[label.item()] += 1

    # Calculate and print the accuracy for each class
    for i in range(num_classes):
        if total_per_class[i] > 0:
            accuracy = 100 * correct_per_class[i] / total_per_class[i]
            print(f"Class '{class_names[i]}' Accuracy: {accuracy:.2f}%")
        else:
            print(f"Class '{class_names[i]}' has no samples in evaluation set.")

# Run evaluation
evaluate_model(model, eval_loader)
