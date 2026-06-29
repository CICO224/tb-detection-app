"""
Data Organization Helper Script
================================

This script helps organize your chest X-ray images into the correct
folder structure for training.

Usage:
    python organize_data.py --source "path/to/your/images" --split 0.75

This will:
1. Look for images in your source folder (or subfolders)
2. Organize them into Normal/TB folders
3. Split them into train/test sets
"""

import os
import shutil
import argparse
import random
from pathlib import Path
from PIL import Image

def is_image_file(filename):
    """Check if file is an image."""
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    return any(filename.lower().endswith(ext) for ext in valid_extensions)

def find_images(source_dir):
    """Find all image files in source directory and subdirectories."""
    images = []
    source_path = Path(source_dir)
    
    if not source_path.exists():
        print(f"ERROR: Source directory '{source_dir}' does not exist!")
        return images
    
    print(f"Searching for images in: {source_dir}")
    
    # Search recursively
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if is_image_file(file):
                images.append(os.path.join(root, file))
    
    print(f"Found {len(images)} image files")
    return images

def organize_by_folder_structure(images, source_dir, train_dir, test_dir, split_ratio=0.8):
    """
    Organize images based on folder structure.
    Assumes structure like:
        source/
            Normal/
                image1.jpg
                image2.jpg
            TB/
                image3.jpg
    """
    source_path = Path(source_dir)
    normal_images = []
    tb_images = []
    
    # Check if source has Normal/TB subfolders
    normal_source = source_path / "Normal"
    tb_source = source_path / "TB"
    
    if normal_source.exists() and tb_source.exists():
        print("Detected Normal/TB folder structure in source directory")
        normal_images = [str(f) for f in normal_source.rglob('*') if is_image_file(f.name)]
        tb_images = [str(f) for f in tb_source.rglob('*') if is_image_file(f.name)]
    else:
        # Try to find any subfolders that might indicate classes
        subdirs = [d for d in source_path.iterdir() if d.is_dir()]
        if len(subdirs) == 2:
            print(f"Found 2 subdirectories: {[d.name for d in subdirs]}")
            print("Assuming first folder is Normal, second is TB")
            print("(If this is wrong, organize your source folder manually)")
            normal_images = [str(f) for f in subdirs[0].rglob('*') if is_image_file(f.name)]
            tb_images = [str(f) for f in subdirs[1].rglob('*') if is_image_file(f.name)]
        else:
            print("WARNING: Could not detect Normal/TB structure.")
            print("Please organize your source folder as:")
            print("  source/Normal/  (normal X-rays)")
            print("  source/TB/      (TB X-rays)")
            return False
    
    print(f"Found {len(normal_images)} Normal images")
    print(f"Found {len(tb_images)} TB images")
    
    if len(normal_images) == 0 or len(tb_images) == 0:
        print("ERROR: Need images in both Normal and TB categories!")
        return False
    
    # Split into train/test
    random.shuffle(normal_images)
    random.shuffle(tb_images)
    
    normal_split = int(len(normal_images) * split_ratio)
    tb_split = int(len(tb_images) * split_ratio)
    
    normal_train = normal_images[:normal_split]
    normal_test = normal_images[normal_split:]
    tb_train = tb_images[:tb_split]
    tb_test = tb_images[tb_split:]
    
    # Create directories
    os.makedirs(train_dir / "Normal", exist_ok=True)
    os.makedirs(train_dir / "TB", exist_ok=True)
    os.makedirs(test_dir / "Normal", exist_ok=True)
    os.makedirs(test_dir / "TB", exist_ok=True)
    
    # Copy files
    print("\nCopying files...")
    
    def copy_files(file_list, dest_dir, label):
        for i, src_file in enumerate(file_list, 1):
            filename = os.path.basename(src_file)
            dest_file = dest_dir / filename
            
            # Handle duplicate filenames
            counter = 1
            while dest_file.exists():
                name, ext = os.path.splitext(filename)
                dest_file = dest_dir / f"{name}_{counter}{ext}"
                counter += 1
            
            shutil.copy2(src_file, dest_file)
            if i % 10 == 0:
                print(f"  {label}: {i}/{len(file_list)}")
    
    copy_files(normal_train, train_dir / "Normal", "Normal (train)")
    copy_files(normal_test, test_dir / "Normal", "Normal (test)")
    copy_files(tb_train, train_dir / "TB", "TB (train)")
    copy_files(tb_test, test_dir / "TB", "TB (test)")
    
    print("\n" + "=" * 60)
    print("Data organization complete!")
    print("=" * 60)
    print(f"Training set:")
    print(f"  Normal: {len(normal_train)} images")
    print(f"  TB: {len(tb_train)} images")
    print(f"Test set:")
    print(f"  Normal: {len(normal_test)} images")
    print(f"  TB: {len(tb_test)} images")
    print("=" * 60)
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description='Organize chest X-ray images for TB detection training'
    )
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Path to folder containing your images (should have Normal/ and TB/ subfolders)'
    )
    parser.add_argument(
        '--split',
        type=float,
        default=0.8,
        help='Train/test split ratio (default: 0.8 = 80%% train, 20%% test)'
    )
    parser.add_argument(
        '--train-dir',
        type=str,
        default='data/train',
        help='Training data directory (default: data/train)'
    )
    parser.add_argument(
        '--test-dir',
        type=str,
        default='data/test',
        help='Test data directory (default: data/test)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("TB Detection - Data Organization Tool")
    print("=" * 60)
    print()
    
    # Convert to Path objects
    source_dir = Path(args.source)
    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)
    
    # Organize data
    success = organize_by_folder_structure(
        [],  # Not used in this function
        source_dir,
        train_dir,
        test_dir,
        args.split
    )
    
    if success:
        print("\n✅ Ready to train! Run: python train_model.py")
    else:
        print("\n❌ Please fix the issues above and try again.")

if __name__ == '__main__':
    main()