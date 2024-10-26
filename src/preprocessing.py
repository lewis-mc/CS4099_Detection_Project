from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match model input requirements
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalization based on pre-trained model requirements
])

# Load datasets
train_dataset = datasets.ImageFolder(root="dataset/train", transform=transform)
eval_dataset = datasets.ImageFolder(root="dataset/eval", transform=transform)
test_dataset = datasets.ImageFolder(root="dataset/test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

