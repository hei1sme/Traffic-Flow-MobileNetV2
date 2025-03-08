import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog

def slice_image(image_path, outbound_folder, inbound_folder):
    """ Slices an image into two parts using predefined points. """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found or unable to read: {image_path}")
    
    # Predefined points
    points_part1 = [(445, 286), (80, 0), (510, 0), (510, 283)]
    points_part2 = [(476, 286), (87, 2), (2, 1), (2, 285)]

    # Convert to numpy arrays
    poly1 = np.array([points_part1], dtype=np.int32)
    poly2 = np.array([points_part2], dtype=np.int32)

    # Create empty masks
    mask1 = np.zeros(image.shape[:2], dtype=np.uint8)
    mask2 = np.zeros(image.shape[:2], dtype=np.uint8)

    # Fill masks
    cv2.fillPoly(mask1, poly1, 255)
    cv2.fillPoly(mask2, poly2, 255)

    # Extract parts
    part1 = cv2.bitwise_and(image, image, mask=mask1)
    part2 = cv2.bitwise_and(image, image, mask=mask2)

    # Define save paths in the respective folders
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    part1_path = os.path.join(inbound_folder, f"{name}_part1{ext}")
    part2_path = os.path.join(outbound_folder, f"{name}_part2{ext}")

    
    # Save results
    cv2.imwrite(part1_path, part1)
    cv2.imwrite(part2_path, part2)

    print(f"Processed: {filename}")

def process_folder():
    """ Browse and process all images in a selected folder. """
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Select Folder Containing Images")
    if not folder_path:
        print("No folder selected.")
        return
    
    # Create outbound and inbound folders inside SLICED
    save_folder = os.path.join(folder_path, "SLICED")
    outbound_folder = os.path.join(save_folder, "outbound")
    inbound_folder = os.path.join(save_folder, "inbound")
    os.makedirs(outbound_folder, exist_ok=True)
    os.makedirs(inbound_folder, exist_ok=True)
    
    # Process all images in the folder
    for file in os.listdir(folder_path):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(folder_path, file)
            slice_image(image_path, outbound_folder, inbound_folder)
    
    print(f"All images processed. Sliced images saved in: {save_folder}")

# Run the function
if __name__ == "__main__":
    process_folder()