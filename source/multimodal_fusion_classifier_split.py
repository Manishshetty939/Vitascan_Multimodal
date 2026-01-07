import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ---------------- PATHS ----------------
FEATURE_DIR = "features"
RESULT_DIR = "/Users/manishsshetty/Documents/VITASCAN/results_saved/multimodal_final"
MODEL_PATH = "models/multimodal_fusion_classifier_split.h5"

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

# ---------------- LOAD FEATURES ----------------
res_train = np.load(f"{FEATURE_DIR}/resnet_train.npy")
eff_train = np.load(f"{FEATURE_DIR}/efficientnet_train.npy")
sym_train = np.load(f"{FEATURE_DIR}/symptoms_train.npy", allow_pickle=True)
y_train = np.load(f"{FEATURE_DIR}/y_train.npy")

res_test = np.load(f"{FEATURE_DIR}/resnet_test.npy")
eff_test = np.load(f"{FEATURE_DIR}/efficientnet_test.npy")
sym_test = np.load(f"{FEATURE_DIR}/symptoms_test.npy", allow_pickle=True)
y_test = np.load(f"{FEATURE_DIR}/y_test.npy")

# ---------------- FUSION ----------------
X_train = np.concatenate([res_train, eff_train, sym_train], axis=1)
X_test = np.concatenate([res_test, eff_test, sym_test], axis=1)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# ---------------- MODEL ----------------
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(1024, activation="relu"),
    BatchNormalization(),
    Dropout(0.5),

    Dense(512, activation="relu"),
    BatchNormalization(),
    Dropout(0.4),

    Dense(256, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),

    Dense(5, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ---------------- TRAIN ----------------
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=6,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=40,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# ---------------- EVALUATE ----------------
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n🎯 FINAL Test Accuracy: {test_acc:.4f}")

# ---------------- REPORT ----------------
y_pred = model.predict(X_test).argmax(axis=1)

report = classification_report(y_test, y_pred, output_dict=True)
cm = confusion_matrix(y_test, y_pred)

with open(f"{RESULT_DIR}/classification_report.json", "w") as f:
    json.dump(report, f, indent=4)

disp = ConfusionMatrixDisplay(cm)
disp.plot(cmap="Blues")
plt.title("Multimodal Confusion Matrix (Leak-Free)")
plt.savefig(f"{RESULT_DIR}/confusion_matrix.png")
plt.close()

# ---------------- CURVES ----------------
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.legend()
plt.title("Accuracy Curve")
plt.savefig(f"{RESULT_DIR}/accuracy_curve.png")
plt.close()

plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.legend()
plt.title("Loss Curve")
plt.savefig(f"{RESULT_DIR}/loss_curve.png")
plt.close()

# ---------------- SAVE MODEL ----------------
model.save(MODEL_PATH)

print("\n✅ FINAL MULTIMODAL TRAINING COMPLETE")
print("Model saved at:", MODEL_PATH)
print("Results saved in:", RESULT_DIR)
