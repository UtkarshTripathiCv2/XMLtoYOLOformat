import os
import random
import shutil

# --- Configuration ---
# Set the path to your dataset folder (where all the .jpg and .txt files are)
source_folder = '.'  # Use '.' for the current folder

# Set the split ratio
# 80% for training, 10% for validation, 10% for testing
split_ratio = (0.8, 0.1, 0.1) 

# --- End of Configuration ---


def split_dataset(source):
    # Get all image files from the source folder
    all_files = [f for f in os.listdir(source) if f.endswith('.jpg')]
    random.shuffle(all_files)

    # Calculate split indices
    total_files = len(all_files)
    train_end = int(total_files * split_ratio[0])
    val_end = train_end + int(total_files * split_ratio[1])

    # Split the file list
    train_files = all_files[:train_end]
    val_files = all_files[train_end:val_end]
    test_files = all_files[val_end:]

    # Create destination directories
    for folder in ['train', 'val', 'test']:
        os.makedirs(os.path.join(source, 'images', folder), exist_ok=True)
        os.makedirs(os.path.join(source, 'labels', folder), exist_ok=True)

    # Function to move files
    def move_files(files, split_name):
        for file_name in files:
            base_name = os.path.splitext(file_name)[0]
            image_source_path = os.path.join(source, f"{base_name}.jpg")
            label_source_path = os.path.join(source, f"{base_name}.txt")

            image_dest_path = os.path.join(source, 'images', split_name, f"{base_name}.jpg")
            label_dest_path = os.path.join(source, 'labels', split_name, f"{base_name}.txt")

            shutil.move(image_source_path, image_dest_path)
            shutil.move(label_source_path, label_dest_path)
        print(f"Moved {len(files)} files to {split_name} set.")

    # Move files to their respective directories
    move_files(train_files, 'train')
    move_files(val_files, 'val')
    move_files(test_files, 'test')

if __name__ == '__main__':
    split_dataset(source_folder)
    print("\nDataset successfully split into train, val, and test sets!")
    print("Your data is now organized in the 'images' and 'labels' folders.")