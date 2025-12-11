"""
@Project ：CVTools
@File ：data_split.py
@Author ：Haruka
@Date ：2025/12/11 16:43 
"""

import os
import random
from tqdm import tqdm
import shutil
import argparse

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


def split_data(source_dir: str, dest_dir: str, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # Ensure the ratios sum to 1
    assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum to 1"

    # Create destination directories if they don't exist
    for subset in ['train', 'val', 'test']:
        os.makedirs(os.path.join(dest_dir, 'images', subset))
        os.makedirs(os.path.join(dest_dir, 'labels', subset))

    # Get all image files in the source directory
    image_files = [f for f in os.listdir(os.path.join(source_dir)) if f.lower().endswith(IMAGE_EXTENSIONS)]
    random.shuffle(image_files)

    total_images = len(image_files)
    train_end = int(total_images * train_ratio)
    val_end = train_end + int(total_images * val_ratio)

    train_files = image_files[:train_end]
    val_files = image_files[train_end:val_end]
    test_files = image_files[val_end:]

    def move_files(file_list, _subset):
        for file_name in tqdm(file_list, desc=f"Processing {_subset} files"):
            # copy image file
            shutil.copy(os.path.join(source_dir, file_name), os.path.join(dest_dir, 'images', _subset, file_name))
            # copy corresponding label file
            label_file_name = os.path.splitext(file_name)[0] + '.txt'
            label_path = os.path.join(source_dir, label_file_name)

            if os.path.exists(label_path):
                shutil.copy(label_path, os.path.join(dest_dir, 'labels', _subset, label_file_name))
            else:
                print(f"⚠️: Label file not found for {file_name}, Label file isn't txt file ? ")

    move_files(train_files, 'train')
    move_files(val_files, 'val')
    move_files(test_files, 'test')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train, val, and test sets.")
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Path to the source directory containing images and labels')
    parser.add_argument('-o', '--output', type=str, required=True, help='Path to the destination directory')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio of training set')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Ratio of validation set')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Ratio of test set')

    args = parser.parse_args()

    split_data(args.input, args.output, train_ratio=args.train_ratio, val_ratio=args.val_ratio,
               test_ratio=args.test_ratio)
    print("Data split completed successfully.")
    print(f"Train: {len(os.listdir(os.path.join(args.output, 'images', 'train')))} images")
    print(f"Val: {len(os.listdir(os.path.join(args.output, 'images', 'val')))} images")
    print(f"Test: {len(os.listdir(os.path.join(args.output, 'images', 'test')))} images")
