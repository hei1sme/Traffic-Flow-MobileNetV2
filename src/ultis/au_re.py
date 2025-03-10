import cv2
import numpy as np
import os
import random
import tkinter as tk
from tkinter import filedialog, simpledialog
import re
import time

def adjust_brightness(image, factor):
    """Adjust the brightness of an image"""
    return np.clip(image * factor, 0, 255).astype(np.uint8)

def translate_image(image, tx, ty):
    """Translate an image by tx, ty pixels"""
    rows, cols, _ = image.shape
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(image, M, (cols, rows))

def augment_images(folder_path, target_count):
    """Augment images in folder_path until reaching target_count"""
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('jpg', 'png', 'jpeg'))]
    
    if len(image_files) >= target_count:
        print(f"Folder already contains at least {target_count} images. No augmentation needed.")
        return
    
    while len(image_files) < target_count:
        img_name = random.choice(image_files)
        img_path = os.path.join(folder_path, img_name)
        image = cv2.imread(img_path)
        
        if image is None:
            continue
        
        augmentation_type = random.choice(["brightness", "translate"])
        
        if augmentation_type == "brightness":
            factor = random.uniform(0.8, 1.2)
            augmented_image = adjust_brightness(image, factor)
        else:
            tx = random.randint(-10, 10)  # Small horizontal shift
            ty = random.randint(-10, 10)  # Small vertical shift
            augmented_image = translate_image(image, tx, ty)
        
        new_filename = f"aug_{len(image_files) + 1}.jpg"
        new_path = os.path.join(folder_path, new_filename)
        cv2.imwrite(new_path, augmented_image)
        image_files.append(new_filename)
        print(f"Generated: {new_filename} ({len(image_files)}/{target_count})")

    print(f"Data augmentation complete! Folder now contains {target_count} images.")
    return True

def rename_images(folder_path=None):
    """Rename all images in a folder with sequential numbering"""
    # If no folder path provided, ask user to select one
    if folder_path is None:
        root = tk.Tk()
        root.withdraw()
        folder_path = filedialog.askdirectory(title="Select Folder Containing Images")
        if not folder_path:
            print("No folder selected. Exiting...")
            return False
    
    # Get custom prefix from user
    prefix = input("Enter the custom prefix for renaming (e.g., low_inbound): ")
    
    # Get all image files in the selected folder
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]
    
    # Count original files
    original_count = len(image_files)
    print(f"Found {original_count} image files to rename")
    
    # Sort files to maintain order
    image_files.sort()
    
    # SIMPLIFIED APPROACH: Two-phase renaming
    print("\n--- PHASE 1: RENAMING TO TEMPORARY NAMES ---")
    
    # First, rename all files to temporary names with unique identifiers
    temp_files = []
    for idx, filename in enumerate(image_files):
        ext = os.path.splitext(filename)[1]
        # Use timestamp + random number to ensure uniqueness
        temp_name = f"temp_{int(time.time())}_{random.randint(1000, 9999)}_{idx}{ext}"
        old_path = os.path.join(folder_path, filename)
        temp_path = os.path.join(folder_path, temp_name)
        
        try:
            os.rename(old_path, temp_path)
            temp_files.append(temp_name)
            print(f"Temporary rename: {filename} -> {temp_name}")
        except Exception as e:
            print(f"Error creating temporary name for {filename}: {e}")
    
    # Add a small delay to ensure file system catches up
    time.sleep(1)
    
    print(f"\n--- PHASE 2: RENAMING TO FINAL NAMES ({len(temp_files)} files) ---")
    
    # Now rename all temporary files to the final sequence
    successful_final = 0
    for idx, temp_name in enumerate(temp_files):
        ext = os.path.splitext(temp_name)[1]
        final_name = f"{prefix}_{idx}{ext}"
        temp_path = os.path.join(folder_path, temp_name)
        final_path = os.path.join(folder_path, final_name)
        
        try:
            os.rename(temp_path, final_path)
            print(f"Final rename: {temp_name} -> {final_name}")
            successful_final += 1
        except Exception as e:
            print(f"Error in final renaming of {temp_name}: {e}")
    
    # Print summary
    print("\n--- RENAME SUMMARY ---")
    print(f"Original files found: {original_count}")
    print(f"Successfully renamed to temporary names: {len(temp_files)}")
    print(f"Successfully renamed to final names: {successful_final}")
    
    # Verify results
    verify_results(folder_path, prefix, image_extensions)
    return True

def verify_results(folder_path, prefix, image_extensions):
    """Verify that all files have been properly renamed"""
    print("\n--- VERIFYING RESULTS ---")
    
    # Get all image files in the folder
    all_files = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]
    
    # Count files with the correct prefix pattern
    pattern = re.compile(f"^{re.escape(prefix)}_([0-9]+)\\.(jpg|jpeg|png|bmp|tiff)$", re.IGNORECASE)
    correct_files = [f for f in all_files if pattern.match(f)]
    temp_files = [f for f in all_files if f.startswith("temp_")]
    other_files = [f for f in all_files if not pattern.match(f) and not f.startswith("temp_")]
    
    print(f"Total image files in folder: {len(all_files)}")
    print(f"Files with correct naming pattern: {len(correct_files)}")
    print(f"Files still with temporary names: {len(temp_files)}")
    print(f"Other files: {len(other_files)}")
    
    if len(temp_files) > 0:
        print("\nWARNING: Some files still have temporary names!")
        print("You may need to run the script again.")
    
    if len(other_files) > 0:
        print("\nWARNING: Some files have names that don't match the expected pattern!")
        print("These files may need manual attention.")
    
    if len(correct_files) == len(all_files):
        print("\nSUCCESS: All files have been properly renamed!")

def main():
    """Main function to run the combined script"""
    root = tk.Tk()
    root.withdraw()
    
    # Ask user what they want to do
    choice = simpledialog.askstring("Operation", 
                                   "What would you like to do?\n\n"
                                   "1: Data Augmentation\n"
                                   "2: Rename Images\n"
                                   "3: Both (Augment then Rename)\n\n"
                                   "Enter 1, 2, or 3:", 
                                   initialvalue="3")
    
    if not choice or choice not in ["1", "2", "3"]:
        print("Invalid choice. Exiting.")
        return
    
    # Get folder path for the selected operation
    folder_selected = None
    if choice in ["1", "3"]:
        folder_selected = filedialog.askdirectory(title="Select Image Folder")
        if not folder_selected:
            print("No folder selected. Exiting.")
            return
    
    # Perform the selected operation(s)
    if choice == "1":  # Data Augmentation only
        target_count = simpledialog.askinteger("Input", "Enter target number of images:", 
                                              initialvalue=200, minvalue=1)
        if target_count:
            augment_images(folder_selected, target_count)
        else:
            print("Invalid target count. Exiting.")
    
    elif choice == "2":  # Rename only
        rename_images()
    
    elif choice == "3":  # Both operations
        target_count = simpledialog.askinteger("Input", "Enter target number of images:", 
                                              initialvalue=200, minvalue=1)
        if target_count:
            # First augment
            success = augment_images(folder_selected, target_count)
            
            # Then ask if user wants to rename now
            if success:
                rename_now = simpledialog.askstring("Rename", 
                                                   "Data augmentation complete!\n\n"
                                                   "Would you like to rename the images now? (y/n)",
                                                   initialvalue="y")
                if rename_now and rename_now.lower() == "y":
                    rename_images(folder_selected)
                else:
                    print("Skipping renaming. You can run the script again later to rename.")
        else:
            print("Invalid target count. Exiting.")

if __name__ == "__main__":
    main()
