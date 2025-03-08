import os
import cv2
import numpy as np
import tensorflow as tf

# Define constants
IMG_SIZE = (128, 128)
MODEL_PATH = "model/traffic_classification.keras"
TEST_IMAGE_PATH = "TESTING\dataset\camera_1740607989476.jpg"  # Change this to the actual test image path
OUTPUT_CSV = "prediction_results.csv"

# Load model
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully.")

# Preprocess image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.resize(img, IMG_SIZE) / 255.0  # Normalize
    return np.expand_dims(img, axis=0)  # Add batch dimension

# Predict function
def predict(image_path):
    img = preprocess_image(image_path)
    inbound_probs, outbound_probs = model.predict(img)
    
    # Get predicted classes
    inbound_class = np.argmax(inbound_probs)
    outbound_class = np.argmax(outbound_probs)
    
    # Convert to probability format for CSV output
    result = {
        "Filename": os.path.basename(image_path),
        "low_inbound": float(inbound_probs[0][0]),
        "medium_inbound": float(inbound_probs[0][1]),
        "high_inbound": float(inbound_probs[0][2]),
        "low_outbound": float(outbound_probs[0][0]),
        "medium_outbound": float(outbound_probs[0][1]),
        "high_outbound": float(outbound_probs[0][2])
    }
    return result

# Run prediction
prediction = predict(TEST_IMAGE_PATH)

# Save to CSV
import csv
with open(OUTPUT_CSV, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=prediction.keys())
    writer.writeheader()
    writer.writerow(prediction)

print(f"✅ Prediction saved to {OUTPUT_CSV}")
