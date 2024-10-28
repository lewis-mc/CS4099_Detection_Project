import os
import shutil
import random
from math import floor, ceil

# Define paths and split ratios
source_dir = '../../raw_data/BM_cytomorphology_data'
train_dir = '../../split_data/Train'
evaluate_dir = '../../split_data/Evaluate'
test_dir = '../../split_data/Test'

train_ratio = 0.8
evaluate_ratio = 0.1
test_ratio = 0.1

# Ensure the destination directories exist
for split_dir in [train_dir, evaluate_dir, test_dir]:
    os.makedirs(split_dir, exist_ok=True)

# Function to gather all images in a class folder, including subdirectories
def gather_images(class_path):
    image_files = []
    for root, _, files in os.walk(class_path):
        for file in files:
            image_files.append(os.path.join(root, file))
    return image_files

# Function to split files into Train, Evaluate, and Test
def split_data(files):
    random.shuffle(files)
    total = len(files)
    
    # Calculate split sizes
    test_size = ceil(total * test_ratio)
    evaluate_size = ceil(total * evaluate_ratio)
    
    test_files = files[:test_size]
    evaluate_files = files[test_size:test_size + evaluate_size]
    train_files = files[test_size + evaluate_size:]
    
    return train_files, evaluate_files, test_files

# Process each class in the source directory
for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)
    if os.path.isdir(class_path):
        
        # Gather all images in the class directory, including subdirectories
        files = gather_images(class_path)
        
        # Split files into Train, Evaluate, and Test
        train_files, evaluate_files, test_files = split_data(files)
        
        # Define destination paths
        train_class_dir = os.path.join(train_dir, class_name)
        evaluate_class_dir = os.path.join(evaluate_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        
        # Create class folders in each split directory
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(evaluate_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)
        
        # Move files to the respective directories
        for file_path in train_files:
            shutil.copy(file_path, os.path.join(train_class_dir, os.path.basename(file_path)))
        for file_path in evaluate_files:
            shutil.copy(file_path, os.path.join(evaluate_class_dir, os.path.basename(file_path)))
        for file_path in test_files:
            shutil.copy(file_path, os.path.join(test_class_dir, os.path.basename(file_path)))

print("Data split complete!")