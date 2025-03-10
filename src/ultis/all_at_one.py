import os
import shutil
from tqdm import tqdm

def collect_images(source_dir, destination_dir):
    # Define image extensions to look for
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    
    # Create destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)
    
    # Find all image files
    image_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))
    
    print(f"Found {len(image_files)} image files")
    
    # Copy image files to destination
    for src_path in tqdm(image_files, desc="Copying images"):
        filename = os.path.basename(src_path)
        dst_path = os.path.join(destination_dir, filename)
        
        # Handle duplicates
        if os.path.exists(dst_path):
            base_name, extension = os.path.splitext(filename)
            counter = 1
            while os.path.exists(dst_path):
                new_filename = f"{base_name}_{counter}{extension}"
                dst_path = os.path.join(destination_dir, new_filename)
                counter += 1
        
        # Copy the file
        shutil.copy2(src_path, dst_path)
    
    print(f"Successfully copied images to {destination_dir}")

# Example usage
source_directory = ".DATA\HISTORICAL_RAW_DATA"  # Change this to your source folder
destination_directory = ".DATA\HISTORICAL_RAW_DATA\ALL_IMAGES"  # Change this to your destination folder

collect_images(source_directory, destination_directory)
