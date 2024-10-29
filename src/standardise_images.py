import os
from PIL import Image

directory_path = '../../split_data/Evaluate'


import os
from PIL import Image

def check_file_format(directory):
    incorrect_files = []  # List to store paths of files with incorrect formats
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.jpg'):
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        img.verify()  # Verify the image format
                except Exception as e:
                    incorrect_files.append(file_path)
                    print(f"Incorrect format found: {file_path} - Error: {e}")
    print(f"Total files with incorrect formats: {len(incorrect_files)}")
    return incorrect_files

check_file_format(directory_path)
