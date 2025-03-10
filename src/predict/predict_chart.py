import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
import pytz
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

# Function to convert timestamp in filename to formatted date-time in GMT+7
def extract_timestamp(filename):
    try:
        timestamp_ms = int(filename.split('_')[-1].split('.')[0])
        timestamp_utc = datetime.datetime.utcfromtimestamp(timestamp_ms / 1000)
        gmt_plus_7 = pytz.timezone("Asia/Ho_Chi_Minh")
        timestamp_gmt7 = timestamp_utc.replace(tzinfo=pytz.utc).astimezone(gmt_plus_7)
        return timestamp_gmt7.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        print(f"Error parsing timestamp from {filename}: {e}")
        return filename

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

# Function to display prediction charts
def display_prediction_charts(img, inbound_img, outbound_img, inbound_pred, outbound_pred, timestamp):
    # Convert BGR to RGB for matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inbound_img_rgb = cv2.cvtColor(inbound_img, cv2.COLOR_BGR2RGB)
    outbound_img_rgb = cv2.cvtColor(outbound_img, cv2.COLOR_BGR2RGB)
    
    # Create labels and values for charts
    categories = ['Low', 'Medium', 'High']
    inbound_values = inbound_pred
    outbound_values = outbound_pred
    
    # Create a single figure with 3 sections
    fig = plt.figure(figsize=(18, 12))
    
    # Section 1: Full image + double column chart with inbound and outbound
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=1)
    ax1.imshow(img_rgb)
    ax1.set_title(f"Full Image - {timestamp}", fontsize=14)
    ax1.axis('off')
    
    ax2 = plt.subplot2grid((3, 3), (0, 2), colspan=1, rowspan=1)
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, inbound_values, width, label='Inbound', color=['green', 'yellow', 'red'])
    bars2 = ax2.bar(x + width/2, outbound_values, width, label='Outbound', color=['lightgreen', 'lightyellow', 'lightcoral'])
    
    ax2.set_ylim(0, 1)
    ax2.set_title("Traffic Predictions", fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.legend()
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    
    # Section 2: Cropped inbound image + inbound chart
    ax3 = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=1)
    ax3.imshow(inbound_img_rgb)
    ax3.set_title("Inbound Area", fontsize=14)
    ax3.axis('off')
    
    ax4 = plt.subplot2grid((3, 3), (1, 2), colspan=1, rowspan=1)
    bars = ax4.bar(categories, inbound_values, color=['green', 'yellow', 'red'])
    ax4.set_ylim(0, 1)
    ax4.set_title("Inbound Traffic Prediction", fontsize=14)
    
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    
    # Section 3: Cropped outbound image + outbound chart
    ax5 = plt.subplot2grid((3, 3), (2, 0), colspan=2, rowspan=1)
    ax5.imshow(outbound_img_rgb)
    ax5.set_title("Outbound Area", fontsize=14)
    ax5.axis('off')
    
    ax6 = plt.subplot2grid((3, 3), (2, 2), colspan=1, rowspan=1)
    bars = ax6.bar(categories, outbound_values, color=['lightgreen', 'lightyellow', 'lightcoral'])
    ax6.set_ylim(0, 1)
    ax6.set_title("Outbound Traffic Prediction", fontsize=14)
    
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle(f"Traffic Analysis - {timestamp}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

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
    
    # Convert filename to timestamp
    timestamp = extract_timestamp(filename)
    
    # Crop inbound and outbound parts
    inbound_img = crop_image(img, INBOUND_CROP)
    outbound_img = crop_image(img, OUTBOUND_CROP)
    
    # Preprocess images
    inbound_img_processed = preprocess_image(inbound_img)
    outbound_img_processed = preprocess_image(outbound_img)
    
    # Predict using models
    inbound_pred = inbound_model.predict(inbound_img_processed, verbose=0)[0]
    outbound_pred = outbound_model.predict(outbound_img_processed, verbose=0)[0]
    
    # Store results
    predictions.append([timestamp, *inbound_pred, *outbound_pred])
    
    # Print predictions
    print(f"{timestamp} Predictions:")
    print(f"  Inbound - Low: {inbound_pred[0]:.2f}, Medium: {inbound_pred[1]:.2f}, High: {inbound_pred[2]:.2f}")
    print(f"  Outbound - Low: {outbound_pred[0]:.2f}, Medium: {outbound_pred[1]:.2f}, High: {outbound_pred[2]:.2f}")
    
    # Display charts
    # display_prediction_charts(img, inbound_img, outbound_img, inbound_pred, outbound_pred, timestamp)

# Save predictions to CSV
columns = ["Timestamp", "low_inbound", "medium_inbound", "high_inbound", "low_outbound", "medium_outbound", "high_outbound"]
df = pd.DataFrame(predictions, columns=columns)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Predictions saved to {OUTPUT_CSV}")
