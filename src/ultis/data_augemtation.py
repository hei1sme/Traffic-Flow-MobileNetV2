import cv2
import numpy as np
import os
import random
import tkinter as tk
from tkinter import filedialog

def adjust_brightness(image, factor):
    return np.clip(image * factor, 0, 255).astype(np.uint8)

def translate_image(image, tx, ty):
    rows, cols, _ = image.shape
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(image, M, (cols, rows))

def augment_images(folder_path, target_count=200):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('jpg', 'png', 'jpeg'))]
    
    if len(image_files) >= target_count:
        print("Folder already contains at least 200 images. No augmentation needed.")
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

    print("Data augmentation complete! Folder now contains 200 images.")

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory(title="Select Image Folder")
    if folder_selected:
        augment_images(folder_selected)
    else:
        print("No folder selected. Exiting.")
