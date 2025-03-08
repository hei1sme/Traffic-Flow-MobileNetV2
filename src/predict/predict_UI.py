import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# Define image properties
IMG_SIZE = (128, 128)
DATASET_DIR = r"C:\Users\Le Nguyen Gia Hung\Dropbox\Codes\.PROJECTS\mini-projects\Slicing_images\TESTING\dataset"
MODEL_DIR = r"C:\Users\Le Nguyen Gia Hung\Dropbox\Codes\.PROJECTS\mini-projects\Slicing_images\src\Training_model\model" 
MODEL_PATH = os.path.join(MODEL_DIR, "traffic_classification.keras")
OUTPUT_CSV = "predictions.csv"

# Check if model exists
if not os.path.exists(MODEL_PATH):
    print(f"Model not found at {MODEL_PATH}")
    print("Current directory:", os.getcwd())
    print("Available models:", os.listdir(MODEL_DIR) if os.path.exists(MODEL_DIR) else "Model directory not found")
    raise FileNotFoundError(f"Model file not found. Make sure to train the model first.")

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)
print(f"✅ Model loaded from {MODEL_PATH}")
model.summary()

# Check if dataset directory exists
if not os.path.exists(DATASET_DIR):
    print(f"Dataset directory not found at {DATASET_DIR}")
    print("Current directory:", os.getcwd())
    raise FileNotFoundError(f"Dataset directory not found. Please check the path.")

# List image files
image_files = os.listdir(DATASET_DIR)
print(f"Found {len(image_files)} files in dataset directory")

# Filter out non-image files and CSV files
valid_extensions = ['.jpg', '.jpeg', '.png']
image_files = [f for f in image_files if any(f.lower().endswith(ext) for ext in valid_extensions)]
print(f"Found {len(image_files)} image files in dataset directory")

# Prediction function
def predict_image(image_path, temperature=1.0):
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠ Warning: Image not found or couldn't be read -> {image_path}")
        return None
    
    # Preprocess image
    img = cv2.resize(img, IMG_SIZE) / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    # Make prediction - now returns two separate outputs
    inbound_pred, outbound_pred = model.predict(img, verbose=0)
    
    # Get first (and only) batch result
    inbound_pred = inbound_pred[0]  
    outbound_pred = outbound_pred[0]
    
    # Apply temperature scaling if needed
    if temperature != 1.0:
        # Process inbound predictions
        inbound_pred = np.exp(np.log(inbound_pred + 1e-10) / temperature)
        inbound_pred = inbound_pred / np.sum(inbound_pred)
        
        # Process outbound predictions
        outbound_pred = np.exp(np.log(outbound_pred + 1e-10) / temperature)
        outbound_pred = outbound_pred / np.sum(outbound_pred)
    
    # Round to 1 decimal place
    inbound_pred = np.round(inbound_pred, 1)
    outbound_pred = np.round(outbound_pred, 1)
    
    # Ensure they sum to 1.0 after rounding
    if np.sum(inbound_pred) != 1.0:
        inbound_pred = inbound_pred / np.sum(inbound_pred)
        inbound_pred = np.round(inbound_pred, 1)
        # Fix any rounding issues
        if np.sum(inbound_pred) > 1.0:
            idx = np.argmax(inbound_pred)
            inbound_pred[idx] -= (np.sum(inbound_pred) - 1.0)
        elif np.sum(inbound_pred) < 1.0:
            idx = np.argmax(inbound_pred)
            inbound_pred[idx] += (1.0 - np.sum(inbound_pred))
    
    if np.sum(outbound_pred) != 1.0:
        outbound_pred = outbound_pred / np.sum(outbound_pred)
        outbound_pred = np.round(outbound_pred, 1)
        # Fix any rounding issues
        if np.sum(outbound_pred) > 1.0:
            idx = np.argmax(outbound_pred)
            outbound_pred[idx] -= (np.sum(outbound_pred) - 1.0)
        elif np.sum(outbound_pred) < 1.0:
            idx = np.argmax(outbound_pred)
            outbound_pred[idx] += (1.0 - np.sum(outbound_pred))
    
    # Return both predictions
    return {
        'inbound': inbound_pred,
        'outbound': outbound_pred
    }

# Run predictions and save to CSV
results = []
for i, filename in enumerate(image_files):
    if i >= 100:  # Limit to 100 predictions to avoid processing all images
        break
        
    filepath = os.path.join(DATASET_DIR, filename)
    print(f"Processing {i+1}/{min(100, len(image_files))}: {filename}...")
    prediction = predict_image(filepath, temperature=2.0)  # Adjust temperature as needed
    
    if prediction is not None:
        # Format output with separate inbound and outbound probabilities
        result_row = {
            'filename': filename,
            'low_inbound': prediction['inbound'][0],
            'medium_inbound': prediction['inbound'][1],
            'high_inbound': prediction['inbound'][2],
            'low_outbound': prediction['outbound'][0],
            'medium_outbound': prediction['outbound'][1],
            'high_outbound': prediction['outbound'][2]
        }
        results.append(result_row)
        print(f"  Prediction: {result_row}")

if not results:
    print("No valid predictions were made. Please check your image files.")
else:
    # Create DataFrame
    df = pd.DataFrame(results)

    # Ensure the probabilities are balanced and varied
    print("\nSample statistics:")
    print(df.describe())
    
    # Check that probabilities sum to 1 for each direction
    df['inbound_sum'] = df[['low_inbound', 'medium_inbound', 'high_inbound']].sum(axis=1)
    df['outbound_sum'] = df[['low_outbound', 'medium_outbound', 'high_outbound']].sum(axis=1)
    
    print("\nVerifying probability sums:")
    print("Inbound sums:", df['inbound_sum'].describe())
    print("Outbound sums:", df['outbound_sum'].describe())
    
    # Remove verification columns before saving
    df = df.drop(columns=['inbound_sum', 'outbound_sum'])
    
    # Save predictions to CSV - using tab separator
    df.to_csv(OUTPUT_CSV, sep='\t', index=False, float_format='%.1f')
    print(f"✅ Predictions saved at {OUTPUT_CSV}")

    # Also print a sample of the output
    print("\nSample of predictions:")
    print(df.head())
    
    # Visualize some predictions
    plt.figure(figsize=(12, 6))
    
    # Inbound predictions
    plt.subplot(1, 2, 1)
    df[['low_inbound', 'medium_inbound', 'high_inbound']].mean().plot(kind='bar')
    plt.title('Average Inbound Predictions')
    plt.ylim(0, 1)
    
    # Outbound predictions
    plt.subplot(1, 2, 2)
    df[['low_outbound', 'medium_outbound', 'high_outbound']].mean().plot(kind='bar')
    plt.title('Average Outbound Predictions')
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('prediction_distribution.png')
    plt.show()
    
    print("Prediction distribution plot saved as 'prediction_distribution.png'")
    
    # Visualize individual predictions for a few samples
    num_samples = min(5, len(df))
    plt.figure(figsize=(15, num_samples*3))
    
    for i in range(num_samples):
        # Inbound predictions
        plt.subplot(num_samples, 2, i*2+1)
        plt.bar(['Low', 'Medium', 'High'], 
                [df.iloc[i]['low_inbound'], df.iloc[i]['medium_inbound'], df.iloc[i]['high_inbound']])
        plt.title(f'Inbound Prediction - {df.iloc[i]["filename"]}')
        plt.ylim(0, 1)
        
        # Outbound predictions
        plt.subplot(num_samples, 2, i*2+2)
        plt.bar(['Low', 'Medium', 'High'], 
                [df.iloc[i]['low_outbound'], df.iloc[i]['medium_outbound'], df.iloc[i]['high_outbound']])
        plt.title(f'Outbound Prediction - {df.iloc[i]["filename"]}')
        plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('sample_predictions.png')
    plt.show()
    
    print("Sample predictions plot saved as 'sample_predictions.png'")