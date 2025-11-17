import os
import tensorflow as tf
from tensorflow.keras import layers, Model
from pathlib import Path
import mlflow

from app.train.dataset import get_dataset, prepare_dataset
from app.train.config import load_config

physical_devices = tf.config.list_physical_devices('GPU') 
for gpu_instance in physical_devices: 
    tf.config.experimental.set_memory_growth(gpu_instance, True)

# Retrieve config
cfg = load_config("./configs/static_config.yml")

if cfg.train.mixed_precision:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Set random seed
tf.keras.utils.set_random_seed(cfg.train.seed)

# Make tracking DB path absolute so it's always consistent
tracking_db_path = Path("mlflow.db").resolve()
mlflow.set_tracking_uri(f"sqlite:///{tracking_db_path}")

# Set or create experiment
experiment_name = getattr(cfg, "experiment_name", None) or "dogs-finetune"
mlflow.set_experiment(experiment_name)

# Enable autologging BEFORE model creation & fit
mlflow.keras.autolog(
    log_models=True,          # set True if you want model artifacts
)

# Dataset
raw_train, ds_info = get_dataset(split="train[:90%]")
raw_val, ds_info = get_dataset(split="train[90%:]")
raw_test, ds_info = get_dataset(split="test")

# Preprocessing
ds_train = prepare_dataset(raw_train, cfg, training=True)
ds_val = prepare_dataset(raw_val, cfg, training=False)
ds_test = prepare_dataset(raw_test, cfg, training=False)

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
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(cfg.model.drop_rate)(x)
predictions = layers.Dense(cfg.model.num_classes, activation='softmax')(x)

# Joint backone + head
model = Model(inputs=backbone.input, outputs=predictions, name = "cnn_with_top")

# Compile model
optimizer_class = getattr(tf.keras.optimizers, cfg.train.optimizer)
optimizer = optimizer_class(learning_rate=cfg.train.learning_rate)

loss_class = getattr(tf.keras.losses, cfg.train.loss)
loss = loss_class(from_logits=False, label_smoothing=0.1)

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=cfg.train.metrics
)

# Save the model
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
    ),
]

# Wrap training in an MLflow run
run_name = f"{arch}-freeze={cfg.model.freeze_backbone}-lr={cfg.train.learning_rate}"

with mlflow.start_run(run_name=run_name):

    # Optional sanity check
    print("MLflow tracking URI:", mlflow.get_tracking_uri())
    print("MLflow experiment:", mlflow.get_experiment(mlflow.active_run().info.experiment_id).name)

    # Save the exact config used
    mlflow.log_artifact("./config.yml", artifact_path="config")

    # --- Train (autolog will record losses/metrics per epoch) ---
    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=cfg.train.epochs,
        verbose=2,
        callbacks=callbacks
    )