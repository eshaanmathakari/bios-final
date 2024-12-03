import os
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import random

# Constants
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 16  # Adjusted for small datasets
CLASS_LABELS = ['Normal', 'Pneumonia-Bacterial', 'Pneumonia-Viral', 'COVID-19']
FEEDBACK_DIR = 'feedback_data'
MODEL_PATH = 'pneumonia_classifier_final.keras'
UPDATED_MODEL_PATH = 'pneumonia_classifier_updated.keras'
TRAIN_DIR = 'train_data'  # Directory containing the original training data
RETRAIN_THRESHOLD = 5  # Set to 5 as per your requirement
SUBSET_SIZE_PER_CLASS = 100  # Number of images per class from the old data to use

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
    total_feedback_images = count_feedback_images()
    print(f"Total feedback images: {total_feedback_images}")

    if total_feedback_images < RETRAIN_THRESHOLD:
        print(f"Not enough feedback data for retraining (Threshold: {RETRAIN_THRESHOLD}). Retraining aborted.")
        return

    # Prepare augmented training directory
    AUGMENTED_TRAIN_DIR = 'augmented_train_data'
    if os.path.exists(AUGMENTED_TRAIN_DIR):
        shutil.rmtree(AUGMENTED_TRAIN_DIR)
    os.makedirs(AUGMENTED_TRAIN_DIR, exist_ok=True)

    # Copy feedback data into augmented training directory
    for cls in CLASS_LABELS:
        feedback_class_dir = os.path.join(FEEDBACK_DIR, cls)
        augmented_class_dir = os.path.join(AUGMENTED_TRAIN_DIR, cls)
        os.makedirs(augmented_class_dir, exist_ok=True)

        # Copy feedback images
        if os.path.exists(feedback_class_dir):
            for img_name in os.listdir(feedback_class_dir):
                src_path = os.path.join(feedback_class_dir, img_name)
                dest_path = os.path.join(augmented_class_dir, img_name)
                shutil.copy(src_path, dest_path)
            print(f"Feedback data for class '{cls}' added to augmented training data.")

        # Add a subset of old training data
        original_class_dir = os.path.join(TRAIN_DIR, cls)
        if os.path.exists(original_class_dir):
            all_images = [
                img for img in os.listdir(original_class_dir)
                if img.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            subset_images = random.sample(all_images, min(SUBSET_SIZE_PER_CLASS, len(all_images)))
            for img_name in subset_images:
                src_path = os.path.join(original_class_dir, img_name)
                dest_path = os.path.join(augmented_class_dir, img_name)
                shutil.copy(src_path, dest_path)
            print(f"Subset of original data for class '{cls}' added to augmented training data.")

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
        AUGMENTED_TRAIN_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    validation_generator = datagen.flow_from_directory(
        AUGMENTED_TRAIN_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=True
    )

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

    # Clean up augmented training data
    shutil.rmtree(AUGMENTED_TRAIN_DIR)
