# preprocessing.py
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import torch
import os

class SafeImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        try:
            return super(SafeImageFolder, self).__getitem__(index)
        except (OSError, IOError) as e:
            print(f"Skipping corrupt image at index {index}: {e}")
            return None

def get_data_loaders(data_dir, batch_size=32, num_workers=4):
    data_transforms = {
        "train": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "eval": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Use SafeImageFolder to handle corrupt images
    train_dataset = SafeImageFolder(os.path.join(data_dir, "../../split_data/Train"), transform=data_transforms["train"])
    eval_dataset = SafeImageFolder(os.path.join(data_dir, "../../split_data/Train"), transform=data_transforms["eval"])

    # Collate function to filter out None values (corrupt images)
    def collate_skip_none(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.default_collate(batch)

    # Create and return individual DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=collate_skip_none)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_skip_none)

    # Get the class names
    class_names = train_dataset.classes
    return train_loader, eval_loader, class_names



