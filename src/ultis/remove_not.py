import os
import cv2
import numpy as np
import logging
from datetime import datetime
from tkinter import Tk, filedialog
from tqdm import tqdm

# Set up logging
log_filename = f"duplicate_removal_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

def select_folder():
    """Open a dialog to select a folder"""
    root = Tk()
    root.withdraw()  # Hide the main window
    folder_path = filedialog.askdirectory(title="Select folder containing images")
    root.destroy()
    return folder_path

def select_reference_image():
    """Open a dialog to select the reference image"""
    root = Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(
        title="Select reference image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
    )
    root.destroy()
    return file_path

def compare_images(img1, img2, threshold=0.95):
    """
    Compare two images using structural similarity index
    
    Args:
        img1: First image
        img2: Second image
        threshold: Similarity threshold (0-1), higher means more similar
        
    Returns:
        bool: True if images are similar, False otherwise
    """
    # Resize images to same dimensions for comparison
    height = min(img1.shape[0], img2.shape[0])
    width = min(img1.shape[1], img2.shape[1])
    
    img1_resized = cv2.resize(img1, (width, height))
    img2_resized = cv2.resize(img2, (width, height))
    
    # Convert to grayscale
    img1_gray = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)
    
    # Calculate structural similarity index
    try:
        score = cv2.matchTemplate(img1_gray, img2_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(score)
        return max_val >= threshold
    except Exception as e:
        logging.error(f"Error comparing images: {e}")
        return False

def find_and_remove_duplicates(folder_path, reference_image_path, threshold=0.95):
    """
    Find and remove duplicate images in the folder compared to the reference image
    
    Args:
        folder_path: Path to folder containing images
        reference_image_path: Path to reference image
        threshold: Similarity threshold (0-1)
    """
    # Load reference image
    reference_img = cv2.imread(reference_image_path)
    if reference_img is None:
        logging.error(f"Could not load reference image: {reference_image_path}")
        return
    
    logging.info(f"Reference image: {reference_image_path}")
    logging.info(f"Searching for duplicates in: {folder_path}")
    logging.info(f"Similarity threshold: {threshold}")
    
    # Get list of image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
    image_files = [
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f)) and 
        any(f.lower().endswith(ext) for ext in image_extensions)
    ]
    
    logging.info(f"Found {len(image_files)} image files to check")
    
    # Statistics
    total_files = len(image_files)
    deleted_files = 0
    error_files = 0
    
    # Process each image
    for img_path in tqdm(image_files, desc="Checking images"):
        try:
            # Skip if it's the reference image itself
            if os.path.samefile(img_path, reference_image_path):
                logging.info(f"Skipping reference image: {img_path}")
                continue
            
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                logging.warning(f"Could not load image: {img_path}")
                error_files += 1
                continue
            
            # Compare with reference
            if compare_images(reference_img, img, threshold):
                # Log before deleting
                logging.info(f"Duplicate found and deleted: {img_path}")
                
                # Delete the file
                os.remove(img_path)
                deleted_files += 1
        except Exception as e:
            logging.error(f"Error processing {img_path}: {e}")
            error_files += 1
    
    # Log summary
    logging.info(f"Summary:")
    logging.info(f"  Total files checked: {total_files}")
    logging.info(f"  Duplicates deleted: {deleted_files}")
    logging.info(f"  Files with errors: {error_files}")
    
    print(f"\nDone! Check the log file for details: {log_filename}")

if __name__ == "__main__":
    print("This script will find and delete images that are similar to a reference image.")
    print("WARNING: Deleted images cannot be recovered!")
    
    # Ask for confirmation
    confirm = input("Do you want to continue? (yes/no): ")
    if confirm.lower() not in ['yes', 'y']:
        print("Operation cancelled.")
        exit()
    
    # Select folder and reference image
    print("Please select the folder containing images to check...")
    folder_path = select_folder()
    
    if not folder_path:
        print("No folder selected. Exiting.")
        exit()
    
    print("Please select the reference image...")
    reference_image_path = select_reference_image()
    
    if not reference_image_path:
        print("No reference image selected. Exiting.")
        exit()
    
    # Ask for threshold
    threshold_input = input("Enter similarity threshold (0.0-1.0, default 0.95): ")
    threshold = 0.95  # Default value
    
    if threshold_input:
        try:
            threshold = float(threshold_input)
            if threshold < 0 or threshold > 1:
                print("Invalid threshold. Using default value 0.95.")
                threshold = 0.95
        except ValueError:
            print("Invalid threshold. Using default value 0.95.")
    
    # Run the duplicate finder
    find_and_remove_duplicates(folder_path, reference_image_path, threshold)
