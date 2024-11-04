import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import densenet121, DenseNet121_Weights
from collections import defaultdict
from PIL import Image
import os


def evaluate():
    # Configuration
    eval_dir = '../../split_data/Evaluate'
    batch_size = 64
    num_workers = 4  # Use multiple workers to speed up data loading

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

    eval_dataset = SafeImageFolder(root=eval_dir, transform=transform)

    # Filter out None items resulting from corrupted images
    eval_dataset.samples = [s for s in eval_dataset.samples if os.path.exists(s[0])]

    def safe_collate(batch):
        batch = [item for item in batch if item[0] is not None and item[1] is not None]
        return torch.utils.data.dataloader.default_collate(batch)

    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=safe_collate)

    # Initialize the model
    num_classes = len(eval_dataset.classes)
    model = densenet121(weights=DenseNet121_Weights.DEFAULT)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    # Load the trained model
    model.load_state_dict(torch.load('densenet_model.pth', map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    model.eval()

    # Evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    correct = 0
    total = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    with torch.no_grad():
        for data in eval_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Per-class accuracy
            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    class_correct[label.item()] += 1
                class_total[label.item()] += 1

    print(f'Accuracy of the model on the evaluation dataset: {100 * correct / total:.2f}%')
    
    # Print accuracy for each class by using class names from the folder structure
    class_names = eval_dataset.classes
    for class_idx in range(num_classes):
        if class_total[class_idx] > 0:
            accuracy = 100 * class_correct[class_idx] / class_total[class_idx]
            print(f'Accuracy for class {class_names[class_idx]}: {accuracy:.2f}%')

if __name__ == '__main__':
    evaluate()
