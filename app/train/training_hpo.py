import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from pathlib import Path
import mlflow
import optuna
from copy import deepcopy

from app.train.dataset import get_dataset, prepare_dataset
from app.train.config import load_config, ExperimentConfig

from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_integer("n_trials", 20, "Number of Optuna trials to run.")
flags.DEFINE_string("study_name", "default-study", "Optuna study name.")
flags.DEFINE_string("optuna_db", "optuna_default.db", "Path to Optuna DB file.")

physical_devices = tf.config.list_physical_devices('GPU') 
for gpu_instance in physical_devices: 
    tf.config.experimental.set_memory_growth(gpu_instance, True)

# Global dataset variables (to be initialized in main)
base_cfg = load_config("./configs/base_config.yml")  # or your current config


def log_trial_config(cfg, trial):
    """
    Log the trial-specific config as a YAML file in MLflow artifacts.
    No permanent file is created in your repo.
    """
    # Turn cfg into a plain dict
    if hasattr(cfg, "model_dump"):           # Pydantic v2
        cfg_dict = cfg.model_dump(mode="python")
    elif hasattr(cfg, "dict"):               # Pydantic v1
        cfg_dict = cfg.dict()
    else:
        # Fallback: assume it's already something dict-like
        cfg_dict = dict(cfg)

    # Let MLflow handle the file creation + storage
    mlflow.log_dict(
        cfg_dict,
        artifact_file=f"./config/config_trial_{trial.number}.yaml",
    )


def make_trial_config(trial):
    # Deep copy so we don't mutate the global base
    cfg = deepcopy(base_cfg)  # or base_cfg.model_copy(deep=True) for Pydantic

    # --- Override fields with Optuna suggestions ---

    # Learning rate
    cfg.train.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)

    # Dropout
    # cfg.model.drop_rate = trial.suggest_float("drop_rate", 0.2, 0.6)

    # Freeze backbone or not
    # cfg.model.freeze_backbone = trial.suggest_categorical("freeze_backbone", [True, False])

    # Optionally fewer epochs for HPO
    # cfg.train.epochs = trial.suggest_int("epochs", 8, 30)

    # You can also tune data/augment stuff:
    # cfg.data.resize = trial.suggest_categorical("resize", [True, False])
    # cfg.augment.brightness = trial.suggest_float("brightness", 0.0, 0.3)

    return cfg


def set_model(cfg: ExperimentConfig) -> Model:
    """Create and compile a Keras model based on the provided configuration.
    Args:
        cfg: An ExperimentConfig object containing all model parameters.
    Returns:
        A compiled Keras Model.
    """

    # Backbone model
    ModelClass = getattr(tf.keras.applications, cfg.model.architecture)

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

    return model


def train_from_config(cfg: ExperimentConfig,
                      raw_data: dict[str, tf.data.Dataset]
                      ) -> float:
    """Train a model based on the provided configuration.
    Args:
        cfg: An ExperimentConfig object containing all training parameters.
    Returns:
        The best validation loss after training.
    """

    # Preprocessing
    ds_train = prepare_dataset(raw_data["train"], cfg, training=True)
    ds_val = prepare_dataset(raw_data["val"], cfg, training=False)
    ds_test = prepare_dataset(raw_data["test"], cfg, training=False)

    # Create the model
    model = set_model(cfg)

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
    
    # --- Train (autolog will record losses/metrics per epoch) ---
    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=cfg.train.epochs,
        verbose=2,
        callbacks=callbacks
    )

    return history
    

def objective(trial: optuna.Trial,
              raw_data: dict[str, tf.data.Dataset]):

    cfg = make_trial_config(trial)

    run_name = f"trial-{trial.number}"

    with mlflow.start_run(run_name=run_name, nested=True):
        # Save the exact config used
        hist = train_from_config(cfg, raw_data)

        val_losses = hist.history["val_loss"]
        best_epoch_idx = int(np.argmin(val_losses))
        best_epoch = best_epoch_idx + 1

        best_val_loss = float(val_losses[best_epoch_idx])

        best_metrics = {
            name: float(values[best_epoch_idx])
            for name, values in hist.history.items()
        }

        # For Optuna / MLflow:
        for k, v in best_metrics.items():
            trial.set_user_attr(f"{k}", v)
            mlflow.log_metric(f"{k}", v)

        mlflow.log_metric("best_val_loss", best_val_loss)

    return best_val_loss  # if you optimize val_loss


def main(argv):

    if base_cfg.train.mixed_precision:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

    # Set random seed
    tf.keras.utils.set_random_seed(base_cfg.train.seed)

    # Make tracking DB path absolute so it's always consistent
    tracking_db_path = Path("mlflow.db").resolve()
    mlflow.set_tracking_uri(f"sqlite:///{tracking_db_path}")

    # Set or create experiment
    experiment_name = getattr(base_cfg, "experiment_name", None) or "dogs-finetune-optuna"
    mlflow.set_experiment(experiment_name)

    # Enable autologging BEFORE model creation & fit
    mlflow.keras.autolog(
        log_models=False, # set True if you want model artifacts
    )

    # Dataset
    raw_train, _ = get_dataset(split="train[:90%]")
    raw_val, _ = get_dataset(split="train[90%:]")
    raw_test, _ = get_dataset(split="test")
    raw_data = {
        "train": raw_train,
        "val": raw_val,
        "test": raw_test,
        }

    # Create or load Optuna DB
    optuna_db_path = Path(FLAGS.optuna_db).resolve()

    study = optuna.create_study(
        study_name=FLAGS.study_name,
        storage=f"sqlite:///{optuna_db_path}",
        direction="minimize",
        load_if_exists=True,
    )

    study.optimize( #timeout=3600 for running trials during 1 hour.
        lambda trial: objective(trial, raw_data),
        n_trials=FLAGS.n_trials,
    )

    # Save summary
    best_trial = study.best_trial
    summary = {
        "study_name": FLAGS.study_name,
        "n_trials": FLAGS.n_trials,
        "trial_number": best_trial.number,
        "best_params": best_trial.params,
        "best_metrics": best_trial.user_attrs,
    }

    # Save JSON outside mlflow
    config_path = Path("configs") / f"{FLAGS.study_name}_best_params.json"
    config_path.parent.mkdir(exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(summary, f, indent=2)

    with mlflow.start_run(run_name="hpo_summary"):
        # Log best parameters
        mlflow.log_dict(
            study.best_trial.params,
            artifact_file="hpo/best_params.json"
        )

        # Log best objective value
        mlflow.log_metric("hpo_best_value", study.best_value)

        # Log the Optuna database for full reproducibility
        mlflow.log_artifact(str(optuna_db_path), artifact_path="hpo")

        print("Best params:", study.best_trial.params)
        print("Best value:", study.best_value)
    

if __name__ == "__main__":
    app.run(main)