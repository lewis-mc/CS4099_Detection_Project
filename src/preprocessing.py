# preprocessing.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pickle

data_dir = '../../split_data'
batch_size = 32
num_workers = 4

def get_data_loaders(data_dir, batch_size, num_workers=4):
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

    # Directories
    train_dir = f"{data_dir}/Train"
    eval_dir = f"{data_dir}/Evaluate"

    # Load datasets
    image_datasets = {
        "train": datasets.ImageFolder(train_dir, transform=data_transforms["train"]),
        "eval": datasets.ImageFolder(eval_dir, transform=data_transforms["eval"])
    }

    # Data loaders
    dataloaders = {
        "train": DataLoader(image_datasets["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        "eval": DataLoader(image_datasets["eval"], batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    }

    # Save class names for later use in evaluation
    class_names = image_datasets["train"].classes
    with open("class_names.pkl", "wb") as f:
        pickle.dump(class_names, f)

    return dataloaders

get_data_loaders(data_dir, batch_size, num_workers)


