import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2

# Define image properties
IMG_SIZE = (128, 128)
DATASET_DIR = "dataset"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "traffic_classification.keras")

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Label mapping based on filename patterns
LABELS_INBOUND = {
    "low_in_": [1, 0, 0],
    "mid_in_": [0, 1, 0],
    "high_in_": [0, 0, 1],
}
LABELS_OUTBOUND = {
    "low_out_": [1, 0, 0],
    "mid_out_": [0, 1, 0],
    "high_out_": [0, 0, 1]
}

# Load dataset
images, labels_inbound, labels_outbound = [], [], []
for filename in os.listdir(DATASET_DIR):
    filepath = os.path.join(DATASET_DIR, filename)
    img = cv2.imread(filepath)
    if img is None:
        print(f"⚠ Warning: Image not found or corrupted -> {filepath}")
        continue
    img = cv2.resize(img, IMG_SIZE) / 255.0  # Normalize
    images.append(img)
    
    # Assign labels
    label_inbound = [0, 0, 0]  # low, medium, high for inbound
    label_outbound = [0, 0, 0]  # low, medium, high for outbound
    
    for key in LABELS_INBOUND:
        if key in filename:
            label_inbound = LABELS_INBOUND[key]
            break
    for key in LABELS_OUTBOUND:
        if key in filename:
            label_outbound = LABELS_OUTBOUND[key]
            break
    
    labels_inbound.append(label_inbound)
    labels_outbound.append(label_outbound)

# Convert to NumPy arrays
images = np.array(images, dtype=np.float32)
labels_inbound = np.array(labels_inbound, dtype=np.float32)
labels_outbound = np.array(labels_outbound, dtype=np.float32)

# Train-test split
X_train, X_val, y_train_in, y_val_in, y_train_out, y_val_out = train_test_split(
    images, labels_inbound, labels_outbound, test_size=0.2, random_state=42
)

# Create data augmentation pipeline
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomBrightness(0.1),
    layers.RandomContrast(0.1)
])

# Build the model using transfer learning
# Use MobileNetV2 as the base model (smaller and faster than other models)
base_model = MobileNetV2(
    input_shape=(128, 128, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze the base model to prevent it from being trained
base_model.trainable = False

# Build the complete model
inputs = keras.Input(shape=(128, 128, 3))
# Apply data augmentation only during training
x = data_augmentation(inputs)
# Pre-process input for MobileNetV2
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
# Feed input to the base model
x = base_model(x, training=False)
# Add global average pooling
x = layers.GlobalAveragePooling2D()(x)
# Add dropout for regularization
x = layers.Dropout(0.3)(x)
# Add a dense layer
x = layers.Dense(128, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)

# Two separate outputs
output_inbound = layers.Dense(3, activation='softmax', name='inbound')(x)
output_outbound = layers.Dense(3, activation='softmax', name='outbound')(x)

# Create the model
model = keras.Model(inputs=inputs, outputs=[output_inbound, output_outbound])

# Compile the model with weighted losses
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss={'inbound': 'categorical_crossentropy', 'outbound': 'categorical_crossentropy'},
    loss_weights={'inbound': 1.0, 'outbound': 1.0},  # Adjust these weights if needed
    metrics={'inbound': 'accuracy', 'outbound': 'accuracy'}
)

# Set up callbacks
callbacks = [
    # Early stopping to prevent overfitting
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    ),
    # Reduce learning rate when training plateaus
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    ),
    # Save the best model
    keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        save_best_only=True,
        monitor='val_loss'
    ),
    # TensorBoard logging for visualization
    keras.callbacks.TensorBoard(
        log_dir='./logs',
        histogram_freq=1
    )
]

# Train the model
history = model.fit(
    X_train, 
    {'inbound': y_train_in, 'outbound': y_train_out},
    validation_data=(X_val, {'inbound': y_val_in, 'outbound': y_val_out}),
    epochs=100,  # Set higher, early stopping will prevent overfitting
    batch_size=32,  # Increased batch size
    callbacks=callbacks
)

# Fine-tuning phase (optional)
print("Starting fine-tuning phase...")
# Unfreeze the top layers of the base model
base_model.trainable = True
# Freeze all the layers except the last 30
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Recompile the model with a lower learning rate
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # Lower learning rate for fine-tuning
    loss={'inbound': 'categorical_crossentropy', 'outbound': 'categorical_crossentropy'},
    loss_weights={'inbound': 1.0, 'outbound': 1.0},
    metrics={'inbound': 'accuracy', 'outbound': 'accuracy'}
)

# Continue training with fine-tuning
history_fine = model.fit(
    X_train, 
    {'inbound': y_train_in, 'outbound': y_train_out},
    validation_data=(X_val, {'inbound': y_val_in, 'outbound': y_val_out}),
    epochs=50,
    batch_size=16,  # Smaller batch size for fine-tuning
    callbacks=callbacks
)

# Save the final model
model.save(MODEL_PATH.replace('.keras', '_final.keras'))
print(f"✅ Final model saved at {MODEL_PATH.replace('.keras', '_final.keras')}")

# Evaluate the model on validation data
val_results = model.evaluate(
    X_val, 
    {'inbound': y_val_in, 'outbound': y_val_out},
    verbose=1
)

print("Validation Results:")
print(f"Loss: {val_results[0]}")
print(f"Inbound Loss: {val_results[1]}, Inbound Accuracy: {val_results[3]}")
print(f"Outbound Loss: {val_results[2]}, Outbound Accuracy: {val_results[4]}")

# Generate and display predictions visualization (optional)
def visualize_predictions(model, X, y_true_in, y_true_out, num_samples=5):
    import matplotlib.pyplot as plt
    import random
    
    indices = random.sample(range(len(X)), num_samples)
    
    predictions = model.predict(X[indices])
    pred_inbound, pred_outbound = predictions
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples * 5))
    
    classes_in = ["Low", "Medium", "High"]
    classes_out = ["Low", "Medium", "High"]
    
    for i, idx in enumerate(indices):
        # Display the image
        axes[i, 0].imshow(X[idx])
        axes[i, 0].set_title(f"Sample {idx}")
        axes[i, 0].axis('off')
        
        # Display inbound predictions
        true_in = np.argmax(y_true_in[idx])
        pred_in = np.argmax(pred_inbound[i])
        color_in = 'green' if true_in == pred_in else 'red'
        
        axes[i, 1].bar(classes_in, pred_inbound[i])
        axes[i, 1].set_title(f"Inbound: True={classes_in[true_in]}, Pred={classes_in[pred_in]}", color=color_in)
        
        # Display outbound predictions
        true_out = np.argmax(y_true_out[idx])
        pred_out = np.argmax(pred_outbound[i])
        color_out = 'green' if true_out == pred_out else 'red'
        
        axes[i, 2].bar(classes_out, pred_outbound[i])
        axes[i, 2].set_title(f"Outbound: True={classes_out[true_out]}, Pred={classes_out[pred_out]}", color=color_out)
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "prediction_visualization.png"))
    print(f"✅ Prediction visualization saved at {os.path.join(MODEL_DIR, 'prediction_visualization.png')}")

# Uncomment to run visualization
# visualize_predictions(model, X_val, y_val_in, y_val_out)