import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from preprocess_resnet import train_generator, val_generator

# -----------------------------
# LOAD BASE MODEL
# -----------------------------
base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# -----------------------------
# FREEZE ALL LAYERS FIRST
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
# LOAD BASELINE WEIGHTS (OPTIONAL BUT GOOD)
# -----------------------------
model.load_weights("models/resnet50_image_model_final.h5")

# -----------------------------
# UNFREEZE TOP LAYERS
# -----------------------------
for layer in base_model.layers[-40:]:
    layer.trainable = True

# -----------------------------
# COMPILE WITH LOW LR
# -----------------------------
model.compile(
    optimizer=Adam(learning_rate=1e-5),
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
        "models/resnet50_finetuned_model.h5",
        monitor="val_accuracy",
        save_best_only=True
    )
]

# -----------------------------
# FINE-TUNE MODEL
# -----------------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15,
    callbacks=callbacks
)

model.save("models/resnet50_finetuned_model_final.h5")
print("✅ ResNet50 fine-tuning complete.")
