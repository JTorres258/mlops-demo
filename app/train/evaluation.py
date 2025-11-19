import os
import tensorflow as tf
from tensorflow.keras import layers, Model
from pathlib import Path
import mlflow
from sklearn.metrics import confusion_matrix, matthews_corrcoef, classification_report

from app.train.dataset import get_dataset, prepare_dataset
from app.train.config import load_config
from app.train.utils import plot_confusion_matrix, plot_ROC, plot_precision_recall_curve, save_per_class_metrics


# -------------------------------------------------------------------
# TensorFlow / GPU setup
# -------------------------------------------------------------------
physical_devices = tf.config.list_physical_devices("GPU")
for gpu_instance in physical_devices:
    tf.config.experimental.set_memory_growth(gpu_instance, True)

# -------------------------------------------------------------------
# Config & random seed
# -------------------------------------------------------------------
cfg = load_config("./configs/static_config.yml")
tf.keras.utils.set_random_seed(cfg.train.seed)

# -------------------------------------------------------------------
# MLflow setup
# -------------------------------------------------------------------
tracking_db_path = Path("mlflow.db").resolve()
mlflow.set_tracking_uri(f"sqlite:///{tracking_db_path}")

experiment_name = getattr(cfg, "experiment_name", None) or "dogs-main-training"
mlflow.set_experiment(experiment_name)

# -------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------
raw_test, ds_info = get_dataset(split="test")
ds_test = prepare_dataset(raw_test, cfg, training=False)

# Make dataset for predict (only images)
ds_test_images = ds_test.map(lambda x, y: x)

# Collect true labels (for metrics)
y_true_list = []
for _, labels in ds_test:
    y_true_list.append(labels)

y_true_onehot = tf.concat(y_true_list, axis=0).numpy()  # One-hot
y_true = tf.argmax(y_true_onehot, axis=-1)

# Read list of dog breeds
list_breed = []
with open("/home/vscode/tensorflow_datasets/stanford_dogs/0.2.0/label.labels.txt", "r") as f:
    for line in f:
        breed = line.strip().split("-", 1)[1]  # split only once
        list_breed.append(breed)

# -------------------------------------------------------------------
# Load model
# -------------------------------------------------------------------
models_dir = Path("models")
best_model_path = models_dir / "dogs_best_model.keras"
assert best_model_path.exists(), f"Model not found at {best_model_path}"

model = tf.keras.models.load_model(best_model_path)
arch = cfg.model.architecture
print(f"Loaded best model from {best_model_path}")


# -------------------------------------------------------------------
# Evaluation + MLflow logging
# -------------------------------------------------------------------
report_path = Path("outputs").resolve()
report_path.mkdir(exist_ok=True)

with mlflow.start_run(run_name=f"{arch}-test_evaluation"):

    # --------- PREDICTIONS ---------    
    y_prob = model.predict(ds_test_images)
    y_pred = tf.argmax(y_prob, axis=-1)

    # --------- BASIC METRICS ---------  
    test_acc = tf.reduce_mean(
        tf.cast(tf.equal(y_true, y_pred), tf.float32)
    ).numpy()
    print("Test accuracy:", test_acc)
    mlflow.log_metric("test_accuracy", float(test_acc))

    loss_class = getattr(tf.keras.losses, cfg.train.loss)
    loss_fn = loss_class(from_logits=False, label_smoothing=0.1)
    test_loss = loss_fn(y_true_onehot, y_prob).numpy()
    print("Test loss:", test_loss)
    mlflow.log_metric("test_loss", float(test_loss))

    # Convert index to label
    pred_label_list = [list_breed[i] for i in y_pred]
    true_label_list = [list_breed[i] for i in y_true]

    # Classification report and MCC
    report_str = classification_report(true_label_list, pred_label_list,
                                labels=list_breed, digits=4)

    mcc_value = matthews_corrcoef(true_label_list, pred_label_list)

    summary_txt = (
        "------------------- Report -------------------\n"
        + report_str
        + "\n---------------------------------------------\n"
        + f"MCC: {mcc_value:.6f}\n"
    )

    summary_path = report_path / "classification_report.txt"
    with open(summary_path, "w") as f:
        f.write(summary_txt)

    # --------- CONFUSION MATRIX ---------    
    cm_path = report_path / "ConfMatrix.png" 

    ConfusionMatrix = confusion_matrix(true_label_list, pred_label_list,
                                        labels=list_breed)
    plot_confusion_matrix(ConfusionMatrix, list_breed,
                            title='Confusion matrix',
                            cmap='Blues',
                            path_save=cm_path)

    # --------- ROC AND PR CURVES ---------
    roc_path = report_path / "ROC.png"  
    pr_path = report_path / "PrecRecall.png"

    plot_ROC(y_true_onehot, y_prob, roc_path)
    plot_precision_recall_curve(y_true_onehot, y_prob, pr_path)

    # --------- PER-CLASS METRICS ---------  
    json_path = report_path / 'per_class_metrics.json'  
    save_per_class_metrics(y_true_onehot, y_prob, list_breed, 
                json_path)

    # --------- LOG ARTIFACTS ---------    
    mlflow.log_artifact(str(summary_path), artifact_path="evaluation")
    mlflow.log_artifact(str(cm_path), artifact_path="evaluation")
    mlflow.log_artifact(str(roc_path), artifact_path="evaluation")
    mlflow.log_artifact(str(pr_path), artifact_path="evaluation")
    mlflow.log_artifact(str(json_path), artifact_path="evaluation")
