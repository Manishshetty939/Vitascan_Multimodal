import os
import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------- CONFIG ----------------
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

DATASET_PATH = "/Users/manishsshetty/Documents/VITASCAN/Data"
FEATURE_DIR = "features"

RESNET_MODEL_PATH = "/Users/manishsshetty/Documents/VITASCAN/models/resnet50_finetuned_model_final.h5"
EFFICIENTNET_MODEL_PATH = "/Users/manishsshetty/Documents/VITASCAN/models/efficientnet_b0_finetuned_best.h5"

os.makedirs(FEATURE_DIR, exist_ok=True)

# ---------------- LOAD MODELS ----------------
resnet_model = load_model(RESNET_MODEL_PATH)
efficientnet_model = load_model(EFFICIENTNET_MODEL_PATH)

# ---------------- FEATURE EXTRACTORS ----------------
resnet_feature_extractor = Model(
    inputs=resnet_model.input,
    outputs=resnet_model.get_layer("global_average_pooling2d").output
)

efficientnet_feature_extractor = Model(
    inputs=efficientnet_model.input,
    outputs=efficientnet_model.get_layer("global_average_pooling2d").output
)

# ---------------- DATA GENERATOR ----------------
datagen = ImageDataGenerator(rescale=1.0 / 255)

generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# ---------------- FEATURE EXTRACTION ----------------
print("Extracting ResNet50 features...")
resnet_features = resnet_feature_extractor.predict(generator, verbose=1)

print("Extracting EfficientNet-B0 features...")
efficientnet_features = efficientnet_feature_extractor.predict(generator, verbose=1)

labels = generator.classes

# ---------------- SAVE FEATURES ----------------
np.save(os.path.join(FEATURE_DIR, "resnet50_features.npy"), resnet_features)
np.save(os.path.join(FEATURE_DIR, "efficientnet_features.npy"), efficientnet_features)
np.save(os.path.join(FEATURE_DIR, "labels.npy"), labels)

print("Feature extraction completed successfully.")
