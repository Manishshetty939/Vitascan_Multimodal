import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical

# ---------------- CONFIG ----------------
FEATURE_DIR = "features"
RESULTS_DIR = "/Users/manishsshetty/Documents/VITASCAN/results_saved/fusion_image"
MODEL_SAVE_PATH = "models/image_only_fusion_classifier.h5"

NUM_CLASSES = 5
EPOCHS = 30
BATCH_SIZE = 32
RANDOM_STATE = 42

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

# ---------------- LOAD FEATURES ----------------
resnet_features = np.load(os.path.join(FEATURE_DIR, "resnet50_features.npy"))
efficientnet_features = np.load(os.path.join(FEATURE_DIR, "efficientnet_features.npy"))
labels = np.load(os.path.join(FEATURE_DIR, "labels.npy"))

print("ResNet features shape:", resnet_features.shape)
print("EfficientNet features shape:", efficientnet_features.shape)
print("Labels shape:", labels.shape)

# ---------------- FEATURE FUSION ----------------
X = np.concatenate([resnet_features, efficientnet_features], axis=1)
y = to_categorical(labels, num_classes=NUM_CLASSES)

print("Fused feature shape:", X.shape)

# ---------------- TRAIN / TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=labels
)

# ---------------- MODEL ----------------
model = Sequential([
    Dense(512, activation="relu", input_shape=(X.shape[1],)),
    BatchNormalization(),
    Dropout(0.4),

    Dense(256, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),

    Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ---------------- TRAIN ----------------
history = model.fit(
    X_train,
    y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

# ---------------- EVALUATE ----------------
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# ---------------- SAVE CLASSIFICATION REPORT ----------------
report = classification_report(y_true, y_pred_classes)

with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

print("Classification report saved.")

# ---------------- SAVE CONFUSION MATRIX ----------------
cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix – Image Fusion Model")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
plt.close()

print("Confusion matrix saved.")

# ---------------- SAVE METRICS (JSON) ----------------
metrics = {
    "test_accuracy": float(test_acc),
    "test_loss": float(test_loss),
    "num_samples": int(len(labels)),
    "num_classes": NUM_CLASSES,
    "model": "ResNet50 + EfficientNet-B0 Feature Fusion"
}

with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

print("Metrics JSON saved.")

# ---------------- SAVE TRAINING CURVES ----------------
# Accuracy curve
plt.figure()
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "accuracy_curve.png"))
plt.close()

# Loss curve
plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training vs Validation Loss")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "loss_curve.png"))
plt.close()

print("Training curves saved.")

# ---------------- SAVE MODEL ----------------
model.save(MODEL_SAVE_PATH)
print(f"Fusion classifier saved at {MODEL_SAVE_PATH}")

print("\nALL RESULTS SUCCESSFULLY SAVED.")
