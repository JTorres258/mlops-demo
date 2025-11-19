import tensorflow as tf
from tensorflow.keras import layers, Model
from pathlib import Path
import mlflow
import numpy as np

from app.train.dataset import get_dataset, prepare_dataset
from app.train.config import load_config

physical_devices = tf.config.list_physical_devices("GPU")
for gpu_instance in physical_devices:
    tf.config.experimental.set_memory_growth(gpu_instance, True)

# Retrieve config
cfg = load_config("./configs/static_config.yml")

if cfg.train.mixed_precision:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

# Set random seed
tf.keras.utils.set_random_seed(cfg.train.seed)

# Make tracking DB path absolute so it's always consistent
tracking_db_path = Path("mlflow.db").resolve()
mlflow.set_tracking_uri(f"sqlite:///{tracking_db_path}")

# Set or create experiment
experiment_name = getattr(cfg, "experiment_name", None) or "dogs-main-training"
mlflow.set_experiment(experiment_name)

# Enable autologging BEFORE model creation & fit
mlflow.keras.autolog(
    log_models=False,  # set True if you want model artifacts
)

# Dataset
raw_train, _ = get_dataset(split="train[:90%]")
raw_val, _ = get_dataset(split="train[90%:]")

# Preprocessing
ds_train = prepare_dataset(raw_train, cfg, training=True)
ds_val = prepare_dataset(raw_val, cfg, training=False)

# Backbone model
arch = cfg.model.architecture
ModelClass = getattr(tf.keras.applications, arch)

backbone = ModelClass(
    include_top=False,
    weights=cfg.model.weights,
    input_shape=cfg.data.input_size + (3,),
)

if cfg.model.freeze_backbone:
    for layer in backbone.layers:
        layer.trainable = False

# Head layers
x = layers.GlobalAveragePooling2D()(backbone.output)
x = layers.Dense(512, activation="relu")(x)
x = layers.Dropout(cfg.model.drop_rate)(x)
predictions = layers.Dense(cfg.model.num_classes, activation="softmax")(x)

# Joint backone + head
model = Model(inputs=backbone.input, outputs=predictions, name="cnn_with_top")

# Compile model
optimizer_class = getattr(tf.keras.optimizers, cfg.train.optimizer)
optimizer = optimizer_class(learning_rate=cfg.train.learning_rate)

loss_class = getattr(tf.keras.losses, cfg.train.loss)
loss = loss_class(from_logits=False, label_smoothing=0.1)

model.compile(optimizer=optimizer, loss=loss, metrics=cfg.train.metrics)

# Save the model and Callbacks
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)
best_model_path = models_dir / "dogs_best_model.keras"

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=False,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        best_model_path,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
    ),
]

# Training
with mlflow.start_run(run_name=f"{arch}-full_training"):

    # Save the exact config used
    mlflow.log_artifact("./configs/static_config.yml", artifact_path="config")

    # --- Train (autolog will record losses/metrics per epoch) ---
    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=cfg.train.epochs,
        verbose=2,
        callbacks=callbacks,
    )

    # Compute best epoch
    best_epoch_idx = int(np.argmin(history.history["val_loss"]))  # 0-based
    best_epoch = best_epoch_idx + 1  # 1-based

    # Log to MLflow
    mlflow.log_metric("best_epoch", best_epoch)  # human-friendly
    mlflow.log_metric("best_epoch_idx", best_epoch_idx)  # technical

    for name, values in history.history.items():
        mlflow.log_metric(f"best_{name}", float(values[best_epoch_idx]))
