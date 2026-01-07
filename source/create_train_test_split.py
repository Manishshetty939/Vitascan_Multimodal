import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_PATH = "/Users/manishsshetty/Documents/VITASCAN/Data"
OUTPUT_PATH = "/Users/manishsshetty/Documents/VITASCAN/data/split.csv"

rows = []

for label in os.listdir(DATASET_PATH):
    class_dir = os.path.join(DATASET_PATH, label)
    if not os.path.isdir(class_dir):
        continue

    for img in os.listdir(class_dir):
        if img.lower().endswith((".jpg", ".jpeg", ".png")):
            rows.append([img, label])

df = pd.DataFrame(rows, columns=["image_name", "label"])

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

train_df["split"] = "train"
test_df["split"] = "test"

final_df = pd.concat([train_df, test_df])
final_df.to_csv(OUTPUT_PATH, index=False)

print("✅ Split created")
print(final_df["split"].value_counts())
