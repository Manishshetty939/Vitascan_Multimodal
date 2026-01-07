import numpy as np
import pandas as pd

# ---------------- PATHS ----------------
SPLIT_CSV = "/Users/manishsshetty/Documents/VITASCAN/data/split.csv"
SYMPTOM_CSV = "/Users/manishsshetty/Documents/VITASCAN/data/symptoms.csv"
EMB_PATH = "features/symptom_embeddings.npy"
FEATURE_DIR = "features"

# ---------------- LOAD ----------------
split_df = pd.read_csv(SPLIT_CSV)
sym_df = pd.read_csv(SYMPTOM_CSV)
embeddings = np.load(EMB_PATH)

# Safety check
assert len(sym_df) == embeddings.shape[0], "Mismatch between symptoms and embeddings"

# Create index mapping
img_to_index = {img: i for i, img in enumerate(sym_df["image_name"])}

train_emb, test_emb = [], []

for _, row in split_df.iterrows():
    img = row["image_name"]
    split = row["split"]

    idx = img_to_index.get(img)
    if idx is None:
        continue

    if split == "train":
        train_emb.append(embeddings[idx])
    else:
        test_emb.append(embeddings[idx])

train_emb = np.array(train_emb)
test_emb = np.array(test_emb)

# ---------------- SAVE ----------------
np.save(f"{FEATURE_DIR}/symptoms_train.npy", train_emb)
np.save(f"{FEATURE_DIR}/symptoms_test.npy", test_emb)

print("✅ Symptom embeddings split successfully")
print("Train:", train_emb.shape)
print("Test :", test_emb.shape)
