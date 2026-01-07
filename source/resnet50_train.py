import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from preprocess_resnet import train_generator, val_generator

# -----------------------------
# LOAD PRETRAINED RESNET50
# -----------------------------
base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

# -----------------------------
# CUSTOM CLASSIFIER HEAD
# -----------------------------
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(5, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# -----------------------------
# COMPILE MODEL
# -----------------------------
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# CALLBACKS
# -----------------------------
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ModelCheckpoint(
        "models/resnet50_image_model.h5",
        monitor="val_accuracy",
        save_best_only=True
    )
]

# -----------------------------
# TRAIN MODEL
# -----------------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=25,
    callbacks=callbacks
)

# -----------------------------
# SAVE FINAL MODEL
# -----------------------------
model.save("models/resnet50_image_model_final.h5")
print("✅ ResNet50 model training complete and saved.")
