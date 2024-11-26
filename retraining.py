# retraining.py

import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Define image size and batch size
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32

# Define class labels
CLASS_LABELS = ['Normal', 'Pneumonia-Bacterial', 'Pneumonia-Viral', 'COVID-19']

# Load the trained model
try:
    model = load_model('pneumonia_classifier_final.keras')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Unfreeze the last few layers for fine-tuning
for layer in model.layers[-10:]:
    layer.trainable = True

# Recompile the model with a low learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Define paths
FEEDBACK_DIR = 'feedback_data'
TRAIN_DIR = 'train_data'  # Original training data directory
AUGMENTED_TRAIN_DIR = 'augmented_train_data'
VALIDATION_DIR = 'validation_data'
TEST_DIR = 'test_data'

# Create augmented training directory by combining original training data and feedback data
if os.path.exists(AUGMENTED_TRAIN_DIR):
    shutil.rmtree(AUGMENTED_TRAIN_DIR)
shutil.copytree(TRAIN_DIR, AUGMENTED_TRAIN_DIR)

# Copy feedback data into augmented training directory
for cls in CLASS_LABELS:
    feedback_class_dir = os.path.join(FEEDBACK_DIR, cls)
    augmented_class_dir = os.path.join(AUGMENTED_TRAIN_DIR, cls)
    if os.path.exists(feedback_class_dir):
        # Ensure the class directory exists in augmented training data
        os.makedirs(augmented_class_dir, exist_ok=True)
        # Copy feedback images to the augmented training directory
        for img_name in os.listdir(feedback_class_dir):
            src_path = os.path.join(feedback_class_dir, img_name)
            dest_path = os.path.join(augmented_class_dir, img_name)
            shutil.copy(src_path, dest_path)
        print(f"Feedback data for class '{cls}' added to augmented training data.")
    else:
        print(f"No feedback data for class '{cls}'.")

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,
    fill_mode='nearest'
)

# No augmentation for validation and testing
val_test_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators
train_generator = train_datagen.flow_from_directory(
    AUGMENTED_TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

validation_generator = val_test_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Callbacks
checkpoint = ModelCheckpoint('pneumonia_classifier_updated.keras', monitor='val_accuracy', verbose=1,
                             save_best_only=True, mode='max')
earlystop = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3,
                              verbose=1, min_lr=1e-6)

callbacks_list = [checkpoint, earlystop, reduce_lr]

# Retrain the model
epochs = 5  # Adjust as needed

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=epochs,
    callbacks=callbacks_list
)

# Evaluate the updated model on the test set
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // BATCH_SIZE)
print(f'Test Accuracy after retraining: {test_acc:.2f}')

# Save the updated model
model.save('pneumonia_classifier_updated.keras')
print("Updated model saved as 'pneumonia_classifier_updated.keras'.")
