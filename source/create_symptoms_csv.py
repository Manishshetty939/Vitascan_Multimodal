import os
import csv
import random
import pandas as pd

# ---------------- PATHS ----------------
SPLIT_CSV = "/Users/manishsshetty/Documents/VITASCAN/data/split.csv"
OUTPUT_CSV = "/Users/manishsshetty/Documents/VITASCAN/data/symptoms.csv"

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
random.seed(42)

# ---------------- SYMPTOM POOLS ----------------
VITAMIN_SYMPTOMS = {
    "vitamin_a": [
        "night blindness",
        "dry eyes",
        "poor vision",
        "eye irritation",
        "dry rough skin",
        "frequent infections"
    ],
    "vitamin_b12": [
        "chronic fatigue",
        "weakness",
        "numbness in hands",
        "tingling sensation",
        "memory problems",
        "pale skin"
    ],
    "vitamin_c": [
        "bleeding gums",
        "slow wound healing",
        "frequent bruising",
        "joint pain",
        "muscle pain",
        "loose teeth"
    ],
    "vitamin_d": [
        "bone pain",
        "lower back pain",
        "muscle weakness",
        "muscle cramps",
        "difficulty walking",
        "bone fractures"
    ],
    "vitamin_e": [
        "poor coordination",
        "balance issues",
        "blurred vision",
        "muscle weakness",
        "peripheral neuropathy",
        "immune weakness"
    ]
}

COMMON_SYMPTOMS = [
    "fatigue",
    "low energy",
    "tiredness",
    "weak immunity",
    "difficulty concentrating"
]

NOISE_SYMPTOMS = [
    "headache",
    "sleep problems",
    "stress",
    "loss of appetite"
]

# ---------------- LOAD SPLIT ----------------
df = pd.read_csv(SPLIT_CSV)

rows = []

# ---------------- GENERATE SYMPTOMS ----------------
for _, row in df.iterrows():
    image = row["image_name"]
    label = row["label"].lower()

    if label not in VITAMIN_SYMPTOMS:
        continue

    core = random.sample(VITAMIN_SYMPTOMS[label], k=random.randint(2, 4))
    common = random.sample(COMMON_SYMPTOMS, k=random.randint(1, 2))
    noise = random.sample(NOISE_SYMPTOMS, k=random.randint(0, 1))

    symptoms = core + common + noise
    random.shuffle(symptoms)

    symptom_text = " ".join(symptoms)

    rows.append([image, label, symptom_text])

# ---------------- SAVE CSV ----------------
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_name", "label", "symptoms"])
    writer.writerows(rows)

print("✅ symptoms.csv created using split.csv")
print(f"Total samples: {len(rows)}")
print(f"Saved at: {OUTPUT_CSV}")
