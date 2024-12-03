import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Constants
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 16  # Adjusted for small datasets
CLASS_LABELS = ['Normal', 'Pneumonia-Bacterial', 'Pneumonia-Viral', 'COVID-19']
FEEDBACK_DIR = 'feedback_data'
MODEL_PATH = 'pneumonia_classifier_final.keras'
UPDATED_MODEL_PATH = 'pneumonia_classifier_updated.keras'
RETRAIN_THRESHOLD = 2  # Set to 2 for testing

def retrain_model():
    print("Starting retraining with feedback data...")

    # Ensure feedback data exists
    if not os.path.exists(FEEDBACK_DIR):
        print("No feedback data found. Retraining aborted.")
        return

    # Ensure feedback directories for all classes exist
    for cls in CLASS_LABELS:
        class_dir = os.path.join(FEEDBACK_DIR, cls)
        os.makedirs(class_dir, exist_ok=True)

    # Count total feedback images
    total_feedback_images = sum(
        len(files) for _, _, files in os.walk(FEEDBACK_DIR) if files
    )
    print(f"Total feedback images: {total_feedback_images}")

    if total_feedback_images < RETRAIN_THRESHOLD:
        print(f"Not enough feedback data for retraining (Threshold: {RETRAIN_THRESHOLD}). Retraining aborted.")
        return

    # Data augmentation with validation split
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2  # 20% of data for validation
    )

    train_generator = datagen.flow_from_directory(
        FEEDBACK_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    validation_generator = datagen.flow_from_directory(
        FEEDBACK_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=True
    )

    # Check if there are enough samples for training
    if train_generator.samples < 1 or validation_generator.samples < 1:
        print("Not enough feedback data for retraining. Retraining aborted.")
        return

    # Load the existing model
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Unfreeze the last few layers for fine-tuning
    for layer in model.layers[-10:]:
        layer.trainable = True

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    checkpoint = ModelCheckpoint(
        UPDATED_MODEL_PATH,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    )
    earlystop = EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        verbose=1,
        restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,
        verbose=1,
        min_lr=1e-6
    )
    callbacks_list = [checkpoint, earlystop, reduce_lr]

    # Retrain the model
    epochs = 5  # Adjust as needed
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        callbacks=callbacks_list
    )

    # Save the updated model
    model.save(UPDATED_MODEL_PATH)
    print(f"Retraining completed. Updated model saved as '{UPDATED_MODEL_PATH}'.")

    # Clear feedback data after retraining
    shutil.rmtree(FEEDBACK_DIR)
    os.makedirs(FEEDBACK_DIR, exist_ok=True)
    print("Feedback data cleared after retraining.")

# Function to count feedback images
def count_feedback_images():
    total = 0
    for cls in CLASS_LABELS:
        class_dir = os.path.join(FEEDBACK_DIR, cls)
        if os.path.exists(class_dir):
            total += len([
                file for file in os.listdir(class_dir)
                if file.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
    return total
