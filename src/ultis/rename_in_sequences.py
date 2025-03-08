import os
import tkinter as tk
from tkinter import filedialog

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
    
    # Sort files to maintain order
    image_files.sort()
    
    # Rename images with incremental numbering
    for idx, filename in enumerate(image_files):
        ext = os.path.splitext(filename)[1]  # Get file extension
        new_name = f"{prefix}_{idx}{ext}"
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_name)
        
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_name}")
    
    print("All images have been renamed successfully!")

if __name__ == "__main__":
    rename_images()
