import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2

# Define image properties
IMG_SIZE = (128, 128)
DATASET_DIR = ".DATA/dataset"
MODEL_DIR = "./model"
INBOUND_MODEL_PATH = os.path.join(MODEL_DIR, "inbound_classification.keras")
OUTBOUND_MODEL_PATH = os.path.join(MODEL_DIR, "outbound_classification.keras")
LABELS_PATH = os.path.join(DATASET_DIR, "labels.csv")

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Load labels from CSV file
labels_df = pd.read_csv(LABELS_PATH)
print(f"Loaded {len(labels_df)} label entries from CSV")

# Create separate datasets for inbound and outbound
inbound_images = []
outbound_images = []
inbound_labels = []
outbound_labels = []

for index, row in labels_df.iterrows():
    image_id = row['image_id']
    
    # Check if image is inbound or outbound based on filename
    is_inbound = "_in_" in image_id.lower()
    is_outbound = "_out_" in image_id.lower()
    
    filepath = os.path.join(DATASET_DIR, image_id)
    
    # Check if image exists
    if not os.path.exists(filepath):
        print(f"⚠ Warning: Image not found -> {filepath}")
        continue
    
    # Load and preprocess image
    img = cv2.imread(filepath)
    if img is None:
        print(f"⚠ Warning: Image corrupted -> {filepath}")
        continue
        
    img = cv2.resize(img, IMG_SIZE) / 255.0  # Normalize
    
    # Add to appropriate dataset
    if is_inbound:
        inbound_images.append(img)
        inbound_label = [row['low_inbound'], row['medium_inbound'], row['high_inbound']]
        inbound_labels.append(inbound_label)
    elif is_outbound:
        outbound_images.append(img)
        outbound_label = [row['low_outbound'], row['medium_outbound'], row['high_outbound']]
        outbound_labels.append(outbound_label)
    else:
        print(f"⚠ Warning: Unable to determine if image is inbound or outbound: {image_id}")

# Convert to NumPy arrays
inbound_images = np.array(inbound_images, dtype=np.float32) if inbound_images else np.array([])
outbound_images = np.array(outbound_images, dtype=np.float32) if outbound_images else np.array([])
inbound_labels = np.array(inbound_labels, dtype=np.float32) if inbound_labels else np.array([])
outbound_labels = np.array(outbound_labels, dtype=np.float32) if outbound_labels else np.array([])

print(f"Loaded {len(inbound_images)} inbound images and {len(outbound_images)} outbound images")

# Check if we have enough data for both models
if len(inbound_images) < 10:
    print("⚠ Warning: Not enough inbound images found. Inbound model will not be trained.")

if len(outbound_images) < 10:
    print("⚠ Warning: Not enough outbound images found. Outbound model will not be trained.")

# Only proceed with analyzing and training if we have data
if len(inbound_images) >= 10:
    print(f"Inbound labels shape: {inbound_labels.shape}")
    
    # Analyze label distribution for inbound
    print("\nInbound Label Distribution:")
    print("Low:", np.sum(inbound_labels[:, 0]), 
          "Medium:", np.sum(inbound_labels[:, 1]), 
          "High:", np.sum(inbound_labels[:, 2]))
    
    # Calculate class weights for inbound
    inbound_weights = {
        0: len(inbound_images) / (3 * np.sum(inbound_labels[:, 0])) if np.sum(inbound_labels[:, 0]) > 0 else 1.0,
        1: len(inbound_images) / (3 * np.sum(inbound_labels[:, 1])) if np.sum(inbound_labels[:, 1]) > 0 else 1.0,
        2: len(inbound_images) / (3 * np.sum(inbound_labels[:, 2])) if np.sum(inbound_labels[:, 2]) > 0 else 1.0
    }
    print("Inbound Class Weights:", inbound_weights)

if len(outbound_images) >= 10:
    print(f"Outbound labels shape: {outbound_labels.shape}")
    
    # Analyze label distribution for outbound
    print("\nOutbound Label Distribution:")
    print("Low:", np.sum(outbound_labels[:, 0]), 
          "Medium:", np.sum(outbound_labels[:, 1]), 
          "High:", np.sum(outbound_labels[:, 2]))
    
    # Calculate class weights for outbound
    outbound_weights = {
        0: len(outbound_images) / (3 * np.sum(outbound_labels[:, 0])) if np.sum(outbound_labels[:, 0]) > 0 else 1.0,
        1: len(outbound_images) / (3 * np.sum(outbound_labels[:, 1])) if np.sum(outbound_labels[:, 1]) > 0 else 1.0,
        2: len(outbound_images) / (3 * np.sum(outbound_labels[:, 2])) if np.sum(outbound_labels[:, 2]) > 0 else 1.0
    }
    print("Outbound Class Weights:", outbound_weights)

# Function to create data augmentation pipeline
def create_data_augmentation():
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomBrightness(0.1),
        layers.RandomContrast(0.1)
    ])

# Custom weighted loss function
def weighted_categorical_crossentropy(class_weights):
    def loss(y_true, y_pred):
        # Scale predictions so that they sum to 1
        y_pred /= tf.reduce_sum(y_pred, axis=-1, keepdims=True)
        # Clip to prevent NaN's and Inf's
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        # Calculate loss with weights
        loss = y_true * tf.math.log(y_pred)
        # Apply class weights
        weight_vector = tf.constant([class_weights[0], class_weights[1], class_weights[2]], dtype=tf.float32)
        loss = loss * weight_vector
        return -tf.reduce_sum(loss, axis=-1)
    return loss

# Function to build a single traffic classification model
def build_traffic_model(class_weights, model_name):
    # Use MobileNetV2 as the base model
    base_model = MobileNetV2(
        input_shape=(128, 128, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model initially
    base_model.trainable = False
    
    # Build the model
    inputs = keras.Input(shape=(128, 128, 3))
    # Apply data augmentation only during training
    x = create_data_augmentation()(inputs)
    # Pre-process input for MobileNetV2
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    # Feed input to the base model
    x = base_model(x, training=False)
    # Add global average pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Add classification layers
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(3, activation='softmax')(x)
    
    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)
    
    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=weighted_categorical_crossentropy(class_weights),
        metrics=['accuracy']
    )
    
    return model, base_model

# Function to train a model with fine-tuning
def train_with_fine_tuning(model, base_model, X_train, y_train, X_val, y_val, 
                          class_weights, model_path, model_name):
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
            model_path,
            save_best_only=True,
            monitor='val_loss'
        ),
        # TensorBoard logging for visualization
        keras.callbacks.TensorBoard(
            log_dir=f'./logs/{model_name}',
            histogram_freq=1
        )
    ]
    
    # Initial training phase
    print(f"\n--- Initial training phase for {model_name} model ---")
    history = model.fit(
        X_train, 
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks
    )
    
    # Fine-tuning phase
    print(f"\n--- Fine-tuning phase for {model_name} model ---")
    # Unfreeze the top layers of the base model
    base_model.trainable = True
    # Freeze all the layers except the last 30
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    # Recompile the model with a lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss=weighted_categorical_crossentropy(class_weights),
        metrics=['accuracy']
    )
    
    # Continue training with fine-tuning
    history_fine = model.fit(
        X_train, 
        y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=16,
        callbacks=callbacks
    )
    
    # Combine histories for plotting
    combined_history = {}
    for key in history.history:
        combined_history[key] = history.history[key] + history_fine.history[key]
    
    # Save the final model
    final_model_path = model_path.replace('.keras', '_final.keras')
    model.save(final_model_path)
    print(f"✅ Final {model_name} model saved at {final_model_path}")
    
    return model, combined_history

# Function to evaluate and visualize model performance
def evaluate_model(model, X_val, y_val, model_name, class_names=['Low', 'Medium', 'High']):
    # Evaluate the model
    val_results = model.evaluate(X_val, y_val, verbose=1)
    print(f"\n{model_name} Validation Results:")
    print(f"Loss: {val_results[0]}, Accuracy: {val_results[1]}")
    
    # Generate predictions
    predictions = model.predict(X_val)
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_val, axis=1)
    
    # Classification report
    from sklearn.metrics import classification_report, confusion_matrix
    print(f"\n{model_name} Classification Report:")
    print(classification_report(true_classes, pred_classes, target_names=class_names))
    
    # Plot confusion matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    cm = confusion_matrix(true_classes, pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_filename = os.path.join(MODEL_DIR, f'{model_name.lower()}_confusion.png')
    plt.savefig(cm_filename)
    print(f"✅ Saved confusion matrix to {cm_filename}")
    
    # Visualize sample predictions
    indices = np.random.choice(len(X_val), min(5, len(X_val)), replace=False)
    fig, axes = plt.subplots(len(indices), 2, figsize=(12, 4*len(indices)))
    
    # Handle the case where we have only one sample
    if len(indices) == 1:
        axes = np.array([axes])
    
    for i, idx in enumerate(indices):
        # Display the image
        axes[i, 0].imshow(X_val[idx])
        axes[i, 0].set_title(f"{model_name} Sample {idx}")
        axes[i, 0].axis('off')
        
        # Display predictions
        true_class = np.argmax(y_val[idx])
        pred_class = np.argmax(predictions[idx])
        color = 'green' if true_class == pred_class else 'red'
        
        axes[i, 1].bar(class_names, predictions[idx])
        axes[i, 1].set_title(f"True={class_names[true_class]}, Pred={class_names[pred_class]}", color=color)
    
    plt.tight_layout()
    vis_filename = os.path.join(MODEL_DIR, f"{model_name.lower()}_predictions.png")
    plt.savefig(vis_filename)
    print(f"✅ Prediction visualization saved at {vis_filename}")
    
    return predictions, true_classes, pred_classes

# Plot training history
def plot_training_history(history, model_name):
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Train')
    plt.plot(history['val_accuracy'], label='Validation')
    plt.title(f'{model_name} Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title(f'{model_name} Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    history_filename = os.path.join(MODEL_DIR, f'{model_name.lower()}_history.png')
    plt.savefig(history_filename)
    print(f"✅ Saved training history plot to {history_filename}")

# Only train and evaluate models if we have enough data
if len(inbound_images) >= 10:
    # Train-test split for inbound data
    X_inbound_train, X_inbound_val, y_inbound_train, y_inbound_val = train_test_split(
        inbound_images, inbound_labels, test_size=0.2, random_state=42,
        stratify=np.argmax(inbound_labels, axis=1)
    )
    
    # Build and train inbound model
    print("\n==== TRAINING INBOUND TRAFFIC MODEL ====")
    inbound_model, inbound_base_model = build_traffic_model(inbound_weights, "Inbound")
    inbound_model, inbound_history = train_with_fine_tuning(
        inbound_model, 
        inbound_base_model,
        X_inbound_train, 
        y_inbound_train, 
        X_inbound_val, 
        y_inbound_val,
        inbound_weights,
        INBOUND_MODEL_PATH,
        "Inbound"
    )
    
    # Evaluate inbound model
    print("\n==== EVALUATING INBOUND TRAFFIC MODEL ====")
    inbound_preds, inbound_true, inbound_pred = evaluate_model(
        inbound_model, X_inbound_val, y_inbound_val, "Inbound"
    )
    
    # Plot inbound training history
    plot_training_history(inbound_history, "Inbound")
    print("Inbound model training and evaluation completed!")

if len(outbound_images) >= 10:
    # Train-test split for outbound data
    X_outbound_train, X_outbound_val, y_outbound_train, y_outbound_val = train_test_split(
        outbound_images, outbound_labels, test_size=0.2, random_state=42,
        stratify=np.argmax(outbound_labels, axis=1)
    )
    
    # Build and train outbound model
    print("\n==== TRAINING OUTBOUND TRAFFIC MODEL ====")
    outbound_model, outbound_base_model = build_traffic_model(outbound_weights, "Outbound")
    outbound_model, outbound_history = train_with_fine_tuning(
        outbound_model, 
        outbound_base_model,
        X_outbound_train, 
        y_outbound_train, 
        X_outbound_val, 
        y_outbound_val,
        outbound_weights,
        OUTBOUND_MODEL_PATH,
        "Outbound"
    )
    
    # Evaluate outbound model
    print("\n==== EVALUATING OUTBOUND TRAFFIC MODEL ====")
    outbound_preds, outbound_true, outbound_pred = evaluate_model(
        outbound_model, X_outbound_val, y_outbound_val, "Outbound"
    )
    
    # Plot outbound training history
    plot_training_history(outbound_history, "Outbound")
    print("Outbound model training and evaluation completed!")

print("\nOverall training and evaluation process completed!")