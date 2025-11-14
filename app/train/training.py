import os
import tensorflow as tf
from tensorflow.keras import layers, Model
from pathlib import Path

from app.train.dataset import get_dataset, prepare_dataset
from app.train.config import load_config

physical_devices = tf.config.list_physical_devices('GPU') 
for gpu_instance in physical_devices: 
    tf.config.experimental.set_memory_growth(gpu_instance, True)

# Retrieve config
cfg = load_config("./configs/config_test.yml")

tf.keras.utils.set_random_seed(cfg.train.seed)

# Dataset
ds_raw, ds_info = get_dataset(split="train")

# Preprocessing
ds = prepare_dataset(ds_raw, cfg, training=True)

# Split in train/val
data_size = tf.data.experimental.cardinality(ds).numpy()
val_size = int(0.1 * data_size)

raw_train = ds.shuffle(data_size, seed=cfg.train.seed, reshuffle_each_iteration=False)
ds_val = raw_train.take(val_size)
ds_train = raw_train.skip(val_size)

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
inputs = layers.Input(shape = backbone.output_shape[1:], name = "input")

x = layers.GlobalMaxPooling2D()(inputs)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(cfg.model.drop_rate)(x)
predictions = layers.Dense(cfg.model.num_classes, activation='softmax')(x)

top_model = Model(inputs = inputs, outputs = predictions, name = "cnn_top")

x = backbone.output

# Joint backone + head
predictions = top_model(x)
model = Model(inputs=backbone.input, outputs=predictions, name = "cnn_with_top")

# Compile model
optimizer_class = getattr(tf.keras.optimizers, cfg.train.optimizer.capitalize())
optimizer = optimizer_class(learning_rate=cfg.train.learning_rate)

model.compile(
    optimizer=optimizer,
    loss=cfg.train.loss,
    metrics=cfg.train.metrics
)

# Fit the model
if cfg.train.mixed_precision:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

Path(cfg.train.checkpoint_dir).mkdir(parents=True, exist_ok=True)
ckpt_path = os.path.join(cfg.train.checkpoint_dir, "ckpt-{epoch:02d}-{val_loss:.4f}.keras")
callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
        ),
    ]

model.fit(
    ds_train,
    validation_data=ds_val,
    epochs=cfg.train.epochs,
    verbose=1,
    callbacks=callbacks
)

# Save final model with best weights
best_model_path = max(Path(cfg.train.checkpoint_dir).glob("ckpt-*.keras"))
model = tf.keras.models.load_model(best_model_path)

Path(cfg.train.model_dir).mkdir(parents=True, exist_ok=True)
model.save(os.path.join(cfg.train.model_dir, "model.keras"))