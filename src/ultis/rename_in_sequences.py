import os
import tkinter as tk
from tkinter import filedialog
import re
import random
import time

def rename_images():
    # Open folder selection dialog
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Select Folder Containing Images")
    if not folder_path:
        print("No folder selected. Exiting...")
        return
    
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

def verify_results(folder_path, prefix, image_extensions):
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

if __name__ == "__main__":
    rename_images()
