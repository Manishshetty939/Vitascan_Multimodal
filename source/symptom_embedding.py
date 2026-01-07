import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ---------------- PATHS ----------------
SYMPTOM_CSV = "/Users/manishsshetty/Documents/VITASCAN/data/symptoms.csv"
FEATURE_DIR = "features"
OUTPUT_PATH = os.path.join(FEATURE_DIR, "symptom_embeddings.npy")

os.makedirs(FEATURE_DIR, exist_ok=True)

# ---------------- LOAD CSV ----------------
df = pd.read_csv(SYMPTOM_CSV)

texts = df["symptoms"].astype(str).tolist()
print("Total symptom samples:", len(texts))

# ---------------- LOAD MODEL ----------------
# Lightweight, stable, 384-dim
model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------- ENCODE ----------------
print("Encoding symptoms...")
embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=False
)

# ---------------- SAVE ----------------
np.save(OUTPUT_PATH, embeddings)

print("✅ Symptom embeddings generated successfully")
print("Embedding shape:", embeddings.shape)
print("Saved at:", OUTPUT_PATH)
