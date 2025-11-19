import numpy as np
import matplotlib.pyplot as plt
import itertools
import json

from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)


def plot_confusion_matrix(
    cm: np.array,
    classes: list,
    title: str = "Confusion matrix",
    cmap: object = plt.cm.YlGn,
    add_values: bool = True,
    path_save: str = None,
):
    """
    This function prints and plots the confusion matrix (cm). Normalization
    can be applied by setting 'normalize=True'.

    Parameters:
    ----------
    cm : np.array
        Confusion matrix.
    classes : list
        Data labels.
    normalize : bool, optional
        Apply or not normalization. The default is False.
    title : str, optional
        Title of the image. The default is 'Confusion matrix'.
    cmap : object, optional
        Color map. The default is 'plt.cm.YlGn'.
    path_save : str, optional
        Path for saving the image in .png format. The defaul is None.

    Returns
    -------
    None

    """

    n_classes = len(classes)

    # ----- Adaptive sizes -----
    # Base size per class (inches)
    base_size = max(0.4, 10 / n_classes)  # keeps large datasets readable
    fig_size = (n_classes * base_size, n_classes * base_size)

    # Adaptive fonts
    tick_font = max(6, 20 - n_classes * 0.3)    # decreases as classes increase
    title_font = max(10, 24 - n_classes * 0.2)
    cell_font = max(6, 16 - n_classes * 0.25)

    plt.figure(figsize=fig_size, dpi=144, facecolor="w", edgecolor="k")

    # Normalize
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm_norm.max() / 2.0

    plt.imshow(cm_norm, interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    plt.title(title, fontsize=title_font)
    plt.colorbar()

    tick_marks = np.arange(n_classes)
    plt.xticks(tick_marks, classes, rotation=90, fontsize=tick_font)
    plt.yticks(tick_marks, classes, fontsize=tick_font)

    # Add values
    if add_values:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                j,
                i,
                "%.2f\n(%d)" % (cm_norm[i, j], cm[i, j]),
                ha="center",
                va="center",
                fontsize=cell_font,
                color="white" if cm_norm[i, j] > thresh else "black",
            )

    plt.ylabel("True Label", fontsize=tick_font)
    plt.xlabel("Predicted Label", fontsize=tick_font)
    plt.tight_layout()

    if path_save:
        plt.savefig(path_save, bbox_inches="tight")


def plot_precision_recall_curve(y_true_onehot, y_prob, path_save):
    """
    y_true_onehot: (n_samples, n_classes)
    y_prob:        (n_samples, n_classes)
    class_names:   list of class names
    path_save:     path for saving the PR figure
    """
    n_classes = y_true_onehot.shape[1]

    # ----- per-class curves & AP -----
    precisions = []
    recalls = []
    ap_per_class = []

    for i in range(n_classes):
        p_i, r_i, _ = precision_recall_curve(y_true_onehot[:, i], y_prob[:, i])
        # recall is already sorted ascending, but make it unique for interp
        r_unique, idx = np.unique(r_i, return_index=True)
        p_unique = p_i[idx]

        precisions.append(p_unique)
        recalls.append(r_unique)

        ap_per_class.append(
            average_precision_score(y_true_onehot[:, i], y_prob[:, i])
        )

    # ----- micro-average -----
    p_micro, r_micro, _ = precision_recall_curve(
        y_true_onehot.ravel(), y_prob.ravel()
    )
    ap_micro = average_precision_score(
        y_true_onehot.ravel(), y_prob.ravel()
    )

    # ----- macro-average curve -----
    recall_grid = np.linspace(0, 1, 200)
    mean_precision = np.zeros_like(recall_grid)

    for r_i, p_i in zip(recalls, precisions):
        mean_precision += np.interp(recall_grid, r_i, p_i)

    mean_precision /= n_classes
    ap_macro = float(np.mean(ap_per_class))

    # ----- plot -----
    plt.figure(figsize=(7, 7))
    plt.plot(r_micro, p_micro, label=f"micro-average (AP = {ap_micro:.3f})")
    plt.plot(recall_grid, mean_precision, label=f"macro-average (AP = {ap_macro:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Multi-class Precision–Recall (micro & macro)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="upper right")
    plt.tight_layout()

    if path_save is not None:
        plt.savefig(path_save, bbox_inches="tight")
    plt.close()


def plot_ROC(y_true_onehot, y_prob, path_save):
    """
    y_true_onehot: (n_samples, n_classes) one-hot true labels
    y_prob:        (n_samples, n_classes) predicted probabilities
    path_save:     path for saving the ROC figure
    """
    n_classes = y_true_onehot.shape[1]

    # ----- micro-average -----
    fpr_micro, tpr_micro, _ = roc_curve(
        y_true_onehot.ravel(), y_prob.ravel()
    )
    roc_auc_micro = auc(fpr_micro, tpr_micro)

    # ----- per-class ROC for macro -----
    fpr = []
    tpr = []
    roc_auc = []

    for i in range(n_classes):
        fpr_i, tpr_i, _ = roc_curve(y_true_onehot[:, i], y_prob[:, i])
        fpr.append(fpr_i)
        tpr.append(tpr_i)
        roc_auc.append(auc(fpr_i, tpr_i))

    # ----- macro-average -----
    all_fpr = np.unique(np.concatenate(fpr))
    mean_tpr = np.zeros_like(all_fpr)

    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_classes
    roc_auc_macro = auc(all_fpr, mean_tpr)

    # ----- plot only micro + macro -----
    plt.figure(figsize=(7, 7))
    plt.plot(
        fpr_micro,
        tpr_micro,
        label=f"micro-average (AUC = {roc_auc_micro:.3f})",
    )
    plt.plot(
        all_fpr,
        mean_tpr,
        label=f"macro-average (AUC = {roc_auc_macro:.3f})",
    )

    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multi-class ROC (micro & macro)")
    plt.legend(loc="lower right")
    plt.tight_layout()

    if path_save is not None:
        plt.savefig(path_save, bbox_inches="tight")
    plt.close()


def save_per_class_metrics(y_true_onehot, y_prob, class_names, path_json):
    """
    Saves ONLY per-class:
      - roc_auc
      - average_precision
      - support (# samples in this class)

    As a readable JSON file.
    """

    n_classes = y_true_onehot.shape[1]

    roc_auc_list = []
    ap_list = []
    support = y_true_onehot.sum(axis=0)

    for i in range(n_classes):
        # ROC-AUC
        fpr_i, tpr_i, _ = roc_curve(y_true_onehot[:, i], y_prob[:, i])
        roc_auc_list.append(auc(fpr_i, tpr_i))
        
        # Avg. Precision
        ap_list.append(average_precision_score(y_true_onehot[:, i], y_prob[:, i]))

    # Build per-class objects
    per_class = []
    for idx, cname in enumerate(class_names):
        per_class.append(
            {
                "index": int(idx),
                "class": cname,
                "support": int(support[idx]),
                "roc_auc": float(roc_auc_list[idx]),
                "average_precision": float(ap_list[idx]),
            }
        )

    # Final JSON
    data = {
        "n_classes": int(n_classes),
        "per_class": per_class
    }

    with open(path_json, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)