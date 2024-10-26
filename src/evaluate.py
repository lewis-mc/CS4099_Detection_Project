import torch
from train import model
from preprocessing import eval_loader


model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in eval_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Evaluate Accuracy: {100 * correct / total:.2f}%")
