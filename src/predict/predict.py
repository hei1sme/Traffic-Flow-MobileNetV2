import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Define paths
DATASET_DIR = "TESTING_DATASET"  # Change this to your dataset path
INBOUND_MODEL_PATH = "model/inbound_classification_final.keras"
OUTBOUND_MODEL_PATH = "model/outbound_classification_final.keras"
OUTPUT_CSV = "predictions.csv"

# Define the custom loss function
def weighted_categorical_crossentropy(class_weights):
    def loss(y_true, y_pred):
        y_pred /= tf.reduce_sum(y_pred, axis=-1, keepdims=True)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        loss = y_true * tf.math.log(y_pred)
        weight_vector = tf.constant([class_weights[0], class_weights[1], class_weights[2]], dtype=tf.float32)
        loss = loss * weight_vector
        return -tf.reduce_sum(loss, axis=-1)
    return loss

# Load models with the custom loss function
inbound_model = load_model(INBOUND_MODEL_PATH, custom_objects={'loss': weighted_categorical_crossentropy})
outbound_model = load_model(OUTBOUND_MODEL_PATH, custom_objects={'loss': weighted_categorical_crossentropy})

# Define cropping coordinates
INBOUND_CROP = [(445, 286), (80, 0), (510, 0), (510, 283)]
OUTBOUND_CROP = [(476, 286), (87, 2), (2, 1), (2, 285)]

# Function to crop an image using polygon mask
def crop_image(img, points):
    mask = np.zeros_like(img, dtype=np.uint8)
    points = np.array([points], dtype=np.int32)
    cv2.fillPoly(mask, points, (255, 255, 255))
    masked_img = cv2.bitwise_and(img, mask)
    x, y, w, h = cv2.boundingRect(points)
    return masked_img[y:y+h, x:x+w]

# Function to preprocess image for model input
def preprocess_image(img):
    img = cv2.resize(img, (128, 128))  # Resize to model input size
    img = img / 255.0  # Normalize
    return img[np.newaxis, ...]  # Add batch dimension

# Process dataset
predictions = []
for filename in os.listdir(DATASET_DIR):
    if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue
    
    img_path = os.path.join(DATASET_DIR, filename)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Could not load image {filename}")
        continue
    
    # Crop inbound and outbound parts
    inbound_img = crop_image(img, INBOUND_CROP)
    outbound_img = crop_image(img, OUTBOUND_CROP)
    
    # Preprocess images
    inbound_img_processed = preprocess_image(inbound_img)
    outbound_img_processed = preprocess_image(outbound_img)
    
    # Predict using models
    inbound_pred = inbound_model.predict(inbound_img_processed)[0]
    outbound_pred = outbound_model.predict(outbound_img_processed)[0]
    
    # Store results
    predictions.append([filename, *inbound_pred, *outbound_pred])
    
    # Print predictions
    print(f"{filename} Predictions:")
    print(f"  Inbound - Low: {inbound_pred[0]:.2f}, Medium: {inbound_pred[1]:.2f}, High: {inbound_pred[2]:.2f}")
    print(f"  Outbound - Low: {outbound_pred[0]:.2f}, Medium: {outbound_pred[1]:.2f}, High: {outbound_pred[2]:.2f}")
    
    # Visualization
    fig, ax = plt.subplots(3, 2, figsize=(10, 12))
    
    # Show full image
    ax[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[0, 0].set_title(f"Full Image - {filename}")
    ax[0, 0].axis("off")
    
    # Show prediction chart for full image
    categories = ["Low", "Medium", "High"]
    x = np.arange(len(categories))
    width = 0.3
    
    ax[0, 1].bar(x - width/2, inbound_pred, width, label="Inbound")
    ax[0, 1].bar(x + width/2, outbound_pred, width, label="Outbound")
    ax[0, 1].set_xticks(x)
    ax[0, 1].set_xticklabels(categories)
    ax[0, 1].set_ylim(0, 1)
    ax[0, 1].set_title("Full Image Prediction")
    ax[0, 1].legend()
    
    # Show inbound cropped image
    ax[1, 0].imshow(cv2.cvtColor(inbound_img, cv2.COLOR_BGR2RGB))
    ax[1, 0].set_title("Inbound Cropped Image")
    ax[1, 0].axis("off")
    
    # Show inbound prediction chart
    ax[1, 1].bar(x, inbound_pred, width, color='blue')
    ax[1, 1].set_xticks(x)
    ax[1, 1].set_xticklabels(categories)
    ax[1, 1].set_ylim(0, 1)
    ax[1, 1].set_title("Inbound Prediction")
    
    # Show outbound cropped image
    ax[2, 0].imshow(cv2.cvtColor(outbound_img, cv2.COLOR_BGR2RGB))
    ax[2, 0].set_title("Outbound Cropped Image")
    ax[2, 0].axis("off")
    
    # Show outbound prediction chart
    ax[2, 1].bar(x, outbound_pred, width, color='orange')
    ax[2, 1].set_xticks(x)
    ax[2, 1].set_xticklabels(categories)
    ax[2, 1].set_ylim(0, 1)
    ax[2, 1].set_title("Outbound Prediction")
    
    plt.tight_layout()
    plt.show()

# Save predictions to CSV
columns = ["Filename", "low_inbound", "medium_inbound", "high_inbound", "low_outbound", "medium_outbound", "high_outbound"]
df = pd.DataFrame(predictions, columns=columns)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Predictions saved to {OUTPUT_CSV}")
