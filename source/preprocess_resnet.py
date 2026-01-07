import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -----------------------------
# CONFIGURATION
# -----------------------------
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
DATASET_PATH = r"/Users/manishsshetty/Documents/VITASCAN/Data"

# -----------------------------
# DATA AUGMENTATION (TRAINING)
# -----------------------------
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2
)

# -----------------------------
# NO AUGMENTATION (VALIDATION)
# -----------------------------
val_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

# -----------------------------
# TRAIN DATA GENERATOR
# -----------------------------
train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

# -----------------------------
# VALIDATION DATA GENERATOR
# -----------------------------
val_generator = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

# -----------------------------
# CLASS LABELS
# -----------------------------
class_labels = train_generator.class_indices
print("Class Labels:", class_labels)
