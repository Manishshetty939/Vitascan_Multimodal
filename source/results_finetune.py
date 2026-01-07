import os
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# =====================================================
# CREATE RESULTS DIRECTORY
# =====================================================
RESULTS_DIR = "results_saved"
os.makedirs(RESULTS_DIR, exist_ok=True)

# =====================================================
# COMMON EVALUATION FUNCTION
# =====================================================
def evaluate_and_save(model_path, val_generator, model_name):
    print(f"\n================ {model_name} =================\n")

    model = tf.keras.models.load_model(model_path)

    val_generator.reset()
    preds = model.predict(val_generator, verbose=1)

    y_pred = np.argmax(preds, axis=1)
    y_true = val_generator.classes
    class_names = list(val_generator.class_indices.keys())

    # -------- Classification Report --------
    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )
    report_text = classification_report(
        y_true, y_pred, target_names=class_names
    )

    # Save TXT
    txt_path = os.path.join(RESULTS_DIR, f"{model_name}_classification_report.txt")
    with open(txt_path, "w") as f:
        f.write(report_text)

    # Save CSV
    df_report = pd.DataFrame(report_dict).transpose()
    csv_path = os.path.join(RESULTS_DIR, f"{model_name}_classification_report.csv")
    df_report.to_csv(csv_path)

    # -------- Confusion Matrix --------
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{model_name} – Confusion Matrix")

    img_path = os.path.join(RESULTS_DIR, f"{model_name}_confusion_matrix.png")
    plt.savefig(img_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved results for {model_name}")
    print(f"📄 {txt_path}")
    print(f"📊 {csv_path}")
    print(f"🖼 {img_path}")

# =====================================================
# EVALUATE RESNET50
# =====================================================
from preprocess_resnet import val_generator as resnet_val_gen

evaluate_and_save(
    model_path="/Users/manishsshetty/Documents/VITASCAN/models/resnet50_finetuned_model_final.h5",
    val_generator=resnet_val_gen,
    model_name="ResNet50"
)

# =====================================================
# EVALUATE EFFICIENTNET-B0
# =====================================================
from preprocess_efficientnet import val_generator as effnet_val_gen

evaluate_and_save(
    model_path="/Users/manishsshetty/Documents/VITASCAN/models/efficientnet_b0_finetuned_best.h5",
    val_generator=effnet_val_gen,
    model_name="EfficientNetB0"
)
