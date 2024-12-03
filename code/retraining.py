import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# Define image size and batch size
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32

# Debugging 
print("retraining.py TensorFlow version:", tf.__version__)

# Define class labels
CLASS_LABELS = ['Normal', 'Pneumonia-Bacterial', 'Pneumonia-Viral', 'COVID-19']

# Load the trained model
# try:
#     model = load_model('pneumonia_classifier_final.keras')
#     print("Model loaded successfully.")
# except Exception as e:
#     print(f"Error loading model: {e}")
#     exit(1)
model = tf.keras.models.load_model('pneumonia_classifier_final.keras')


# Unfreeze the last few layers for fine-tuning
for layer in model.layers[-10:]:
    layer.trainable = True

# Recompile the model with a low learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Load data code
import os
from PIL import Image
import glob

# Specify the path to the dataset in your local repository
dataset_path = '/Users/apple/Desktop/PG/SEM-3/BIOS-511/bios-final/data'

# Check if the directory exists
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"The specified path {dataset_path} does not exist.")

# List all image files in the directory (assuming jpg and png images)
image_files = glob.glob(os.path.join(dataset_path, '**/*.jpg'), recursive=True)
image_files += glob.glob(os.path.join(dataset_path, '**/*.png'), recursive=True)

# Check if images are found
if not image_files:
    print("No images found in the specified directory.")
else:
    print(f"Found {len(image_files)} images in the dataset.")

# Load and display a sample image to confirm the dataset is accessible
sample_image = Image.open(image_files[0])
sample_image.show()  # Opens the first image in your default image viewer

# Optional: Load all images (for further processing)
images = [Image.open(img_path) for img_path in image_files]

print("Dataset successfully loaded from local repository.")
dataset_path = '/Users/apple/Desktop/PG/SEM-3/BIOS-511/bios-final/data'
output_path = '/Users/apple/Desktop/PG/SEM-3/BIOS-511/bios-final/processed_data'

if not os.path.exists(output_path):
    os.makedirs(output_path)

# Augmenting data code
classes = ['Normal', 'Pneumonia-Bacterial', 'Pneumonia-Viral', 'COVID-19']
class_counts = {}
for cls in classes:
    class_dir = os.path.join(dataset_path, cls)
    class_counts[cls] = len(os.listdir(class_dir))

max_count = max(class_counts.values())
print("Class counts before balancing:", class_counts)

def augment_images(image_paths, num_new_images, save_dir):
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.05,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    num_generated = 0
    while num_generated < num_new_images:
        for img_path in image_paths:
            img = Image.open(img_path)  # Corrected line
            img = img.resize((224, 224))
            x = np.array(img)
            x = x.reshape((1,) + x.shape)

            prefix = os.path.splitext(os.path.basename(img_path))[0]
            for batch in datagen.flow(x, batch_size=1, save_to_dir=save_dir, save_prefix=prefix, save_format='jpg'):
                num_generated += 1
                if num_generated >= num_new_images:
                    break
        if num_generated >= num_new_images:
            break

target_count = 3500  # Adjust as needed

for cls in classes:
    class_dir = os.path.join(dataset_path, cls)
    images = [os.path.join(class_dir, fname) for fname in os.listdir(class_dir)]
    current_count = len(images)
    save_dir = os.path.join(output_path, cls)
    os.makedirs(save_dir, exist_ok=True)
    
    # Copy existing images
    for img_path in images:
        shutil.copy(img_path, save_dir)
    
    # Augment images if needed
    if current_count < target_count:
        num_new_images = target_count - current_count
        augment_images(images, num_new_images, save_dir)
    elif current_count > target_count:
        # Randomly select images to match target_count
        images_to_keep = random.sample(images, target_count)
        for img_path in images:
            if img_path not in images_to_keep:
                os.remove(os.path.join(save_dir, os.path.basename(img_path)))
    
    print(f"Class {cls} balanced to {target_count} images.")

# connecting train, test, val data
train_dir = os.path.join(output_path, 'train')
val_dir = os.path.join(output_path, 'validation')
test_dir = os.path.join(output_path, 'test')

for cls in classes:
    class_dir = os.path.join(output_path, cls)
    images = [os.path.join(class_dir, fname) for fname in os.listdir(class_dir)]
    
    train_images, temp_images = train_test_split(images, test_size=0.3, random_state=42)
    val_images, test_images = train_test_split(temp_images, test_size=0.5, random_state=42)
    
    # Function to copy images to destination
    def copy_images(image_list, destination):
        dest_dir = os.path.join(destination, cls)
        os.makedirs(dest_dir, exist_ok=True)
        for img_path in image_list:
            shutil.move(img_path, os.path.join(dest_dir, os.path.basename(img_path)))
    
    copy_images(train_images, train_dir)
    copy_images(val_images, val_dir)
    copy_images(test_images, test_dir)


# Loading the original data for Train test split
img_height, img_width = 224, 224  # Input size for the model
batch_size = 32

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

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # Important for evaluation
)

# Define paths
FEEDBACK_DIR = 'feedback_data'
TRAIN_DIR = 'train_data' 
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
