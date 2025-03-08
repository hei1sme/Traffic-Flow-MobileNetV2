import os
import pandas as pd
from tkinter import filedialog, Tk

# Select dataset folder
root = Tk()
root.withdraw()
dataset_folder = filedialog.askdirectory(title="Select Dataset Folder")

if not dataset_folder:
    print("No folder selected. Exiting...")
    exit()

# Define folders
inbound_folder = os.path.join(dataset_folder, "inbound")
outbound_folder = os.path.join(dataset_folder, "outbound")
label_file = os.path.join(dataset_folder, "labels.csv")

# Prepare label list
labels = []

def process_images(folder, category):
    """ Process images and assign correct labels based on naming convention """
    if os.path.exists(folder):
        for file in os.listdir(folder):
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                # Default labels (all zero)
                label = [0, 0, 0, 0, 0, 0]  

                # Assign correct label based on filename
                if file.startswith("low_in") and category == "inbound":
                    label[0] = 1  # low_inbound
                elif file.startswith("mid_in") and category == "inbound":
                    label[1] = 1  # medium_inbound
                elif file.startswith("high_in") and category == "inbound":
                    label[2] = 1  # high_inbound
                elif file.startswith("low_out") and category == "outbound":
                    label[3] = 1  # low_outbound
                elif file.startswith("mid_out") and category == "outbound":
                    label[4] = 1  # medium_outbound
                elif file.startswith("high_out") and category == "outbound":
                    label[5] = 1  # high_outbound
                
                labels.append([file] + label)

# Process inbound and outbound images
process_images(inbound_folder, "inbound")
process_images(outbound_folder, "outbound")

# Convert to DataFrame
df = pd.DataFrame(labels, columns=["image_id", "low_inbound", "medium_inbound", "high_inbound", "low_outbound", "medium_outbound", "high_outbound"])

# Save CSV
df.to_csv(label_file, index=False)
print(f"✅ Labels saved: {label_file}")
