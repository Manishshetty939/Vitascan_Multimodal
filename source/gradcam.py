import os
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = r"/Users/manishsshetty/Documents/VITASCAN/models/efficientnet_b0_finetuned_best.h5"
IMAGE_PATH = r"/Users/manishsshetty/Documents/VITASCAN/Data/Vitamin_A/10WartsTransplantPt10061.jpg"
SAVE_DIR = r"/Users/manishsshetty/Documents/VITASCAN/results_saved/gradcam_results"
IMAGE_SIZE = (224, 224)
LAST_CONV_LAYER = "top_conv"

os.makedirs(SAVE_DIR, exist_ok=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = tf.keras.models.load_model(MODEL_PATH)

# -----------------------------
# LOAD & PREPROCESS IMAGE
# -----------------------------
img = tf.keras.preprocessing.image.load_img(
    IMAGE_PATH, target_size=IMAGE_SIZE
)
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

# -----------------------------
# GRAD-CAM MODEL
# -----------------------------
grad_model = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=[model.get_layer(LAST_CONV_LAYER).output, model.output]
)

# -----------------------------
# COMPUTE GRADIENTS
# -----------------------------
with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_array)
    predicted_class = tf.argmax(predictions[0])
    loss = predictions[:, predicted_class]

grads = tape.gradient(loss, conv_outputs)
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

# -----------------------------
# GENERATE HEATMAP
# -----------------------------
conv_outputs = conv_outputs[0]
heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
heatmap = tf.squeeze(heatmap)

heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap) + 1e-8

# -----------------------------
# OVERLAY HEATMAP
# -----------------------------
img_original = cv2.imread(IMAGE_PATH)
img_original = cv2.resize(img_original, IMAGE_SIZE)

heatmap_resized = cv2.resize(heatmap, IMAGE_SIZE)
heatmap_colored = np.uint8(255 * heatmap_resized)
heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)

overlay = cv2.addWeighted(img_original, 0.6, heatmap_colored, 0.4, 0)

# -----------------------------
# SAVE RESULTS
# -----------------------------
cv2.imwrite(os.path.join(SAVE_DIR, "original.jpg"), img_original)
cv2.imwrite(os.path.join(SAVE_DIR, "heatmap.jpg"), heatmap_colored)
cv2.imwrite(os.path.join(SAVE_DIR, "overlay.jpg"), overlay)

# -----------------------------
# DISPLAY (OPTIONAL)
# -----------------------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(heatmap_resized, cmap="jet")
plt.title("Grad-CAM Heatmap")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.title("Overlay")
plt.axis("off")

plt.tight_layout()
plt.show()

print("✅ Grad-CAM saved to:", SAVE_DIR)
print("✅ Predicted class index:", predicted_class.numpy())
