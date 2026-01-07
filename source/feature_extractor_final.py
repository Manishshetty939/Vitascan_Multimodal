import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ---------------- PATHS ----------------
DATASET_PATH = "/Users/manishsshetty/Documents/VITASCAN/Data"
SPLIT_CSV = "/Users/manishsshetty/Documents/VITASCAN/data/split.csv"
SYMPTOM_CSV = "/Users/manishsshetty/Documents/VITASCAN/data/symptoms.csv"
FEATURE_DIR = "features"

RESNET_MODEL_PATH = "/Users/manishsshetty/Documents/VITASCAN/models/resnet50_finetuned_model_final.h5"
EFFICIENTNET_MODEL_PATH = "/Users/manishsshetty/Documents/VITASCAN/models/efficientnet_b0_finetuned_best.h5"

IMAGE_SIZE = (224, 224)
VALID_EXT = (".jpg", ".jpeg", ".png")

os.makedirs(FEATURE_DIR, exist_ok=True)

# ---------------- LOAD MODELS ----------------
resnet = load_model(RESNET_MODEL_PATH)
efficientnet = load_model(EFFICIENTNET_MODEL_PATH)

resnet_extractor = Model(
    inputs=resnet.input,
    outputs=resnet.get_layer("global_average_pooling2d").output
)

efficientnet_extractor = Model(
    inputs=efficientnet.input,
    outputs=efficientnet.get_layer("global_average_pooling2d").output
)

# ---------------- LOAD DATA ----------------
split_df = pd.read_csv(SPLIT_CSV)
symptom_df = pd.read_csv(SYMPTOM_CSV)

symptom_map = dict(zip(symptom_df["image_name"], symptom_df["symptoms"]))

label_map = {label: idx for idx, label in enumerate(sorted(split_df["label"].unique()))}

# ---------------- IMAGE PREPROCESS ----------------
def preprocess(img_path):
    img = load_img(img_path, target_size=IMAGE_SIZE)
    img = img_to_array(img) / 255.0
    return np.expand_dims(img, axis=0)

# ---------------- FEATURE EXTRACTION ----------------
def extract(split_type):
    res_feats, eff_feats, labels, names = [], [], [], []

    df = split_df[split_df["split"] == split_type]

    for _, row in df.iterrows():
        img_name = row["image_name"]
        label = row["label"]

        img_path = os.path.join(DATASET_PATH, label, img_name)
        if not img_path.lower().endswith(VALID_EXT):
            continue

        img = preprocess(img_path)

        res_feat = resnet_extractor.predict(img, verbose=0).flatten()
        eff_feat = efficientnet_extractor.predict(img, verbose=0).flatten()

        res_feats.append(res_feat)
        eff_feats.append(eff_feat)
        labels.append(label_map[label])
        names.append(img_name)

    return np.array(res_feats), np.array(eff_feats), np.array(labels), names

# ---------------- RUN ----------------
print("Extracting TRAIN features...")
res_train, eff_train, y_train, train_names = extract("train")

print("Extracting TEST features...")
res_test, eff_test, y_test, test_names = extract("test")

# ---------------- SAVE ----------------
np.save("features/resnet_train.npy", res_train)
np.save("features/resnet_test.npy", res_test)
np.save("features/efficientnet_train.npy", eff_train)
np.save("features/efficientnet_test.npy", eff_test)
np.save("features/y_train.npy", y_train)
np.save("features/y_test.npy", y_test)

print("✅ Feature extraction using split completed")
print("Train samples:", len(y_train))
print("Test samples:", len(y_test))
