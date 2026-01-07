import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ✅ IMPORTANT: use EfficientNet preprocessing
from preprocess_efficientnet import train_generator, val_generator


# -----------------------------
# LOAD BASE MODEL
# -----------------------------
base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# -----------------------------
# FREEZE BASE MODEL (initially)
# -----------------------------
for layer in base_model.layers:
    layer.trainable = False

# -----------------------------
# CUSTOM CLASSIFIER
# -----------------------------
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(5, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# -----------------------------
# COMPILE
# -----------------------------
model.compile(
    optimizer=Adam(learning_rate=1e-4),
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
        "models/efficientnet_b0_best.h5",
        monitor="val_accuracy",
        save_best_only=True
    )
]

# -----------------------------
# TRAIN
# -----------------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15,
    callbacks=callbacks
)

model.save("models/efficientnet_b0_final.h5")
print("✅ EfficientNet-B0 training complete.")
