"""Publication-quality Matplotlib and Seaborn evaluation visualizations."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve
from sklearn.preprocessing import label_binarize

sns.set_theme(style="whitegrid", font_scale=0.95)


def plot_training_histories(
    histories: dict[str, dict[str, list[float]]],
    out_path: Path,
) -> Path:
    """Plot training/validation loss and accuracy curves for MLP, 1D-CNN, and Autoencoder."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("SMART IDS v3 — Training & Validation Convergence Histories", fontsize=14, fontweight="bold")

    configs = [
        ("MLP", "loss", "val_loss", "MLP — Categorical Loss", axes[0, 0]),
        ("CNN", "loss", "val_loss", "1D-CNN — Categorical Loss", axes[0, 1]),
        ("Autoencoder", "loss", "val_loss", "Autoencoder — Reconstruction MSE", axes[0, 2]),
        ("MLP", "accuracy", "val_accuracy", "MLP — Classification Accuracy", axes[1, 0]),
        ("CNN", "accuracy", "val_accuracy", "1D-CNN — Classification Accuracy", axes[1, 1]),
    ]

    for m_name, tr_k, val_k, title, ax in configs:
        hist = histories.get(m_name, {})
        if tr_k in hist and val_k in hist:
            epochs = np.arange(1, len(hist[tr_k]) + 1)
            ax.plot(epochs, hist[tr_k], label="Train", linewidth=2, color="#2563eb")
            ax.plot(epochs, hist[val_k], label="Validation", linewidth=2, linestyle="--", color="#f97316")
            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(tr_k.replace("_", " ").title())
            ax.legend()
        else:
            ax.text(0.5, 0.5, f"No {m_name} history", ha="center", va="center")
            ax.set_title(title)

    axes[1, 2].axis("off")
    if "Autoencoder" in histories and "loss" in histories["Autoencoder"]:
        ae_h = histories["Autoencoder"]
        best_e = int(np.argmin(ae_h["val_loss"])) + 1
        summary_txt = (
            "Autoencoder Training Summary\n"
            f"Epochs run: {len(ae_h['loss'])}\n"
            f"Best Epoch: {best_e}\n"
            f"Best Train MSE: {ae_h['loss'][best_e - 1]:.5f}\n"
            f"Best Val MSE: {ae_h['val_loss'][best_e - 1]:.5f}\n"
            "Trained strictly on BENIGN flows."
        )
        axes[1, 2].text(0.1, 0.5, summary_txt, fontsize=11, family="monospace", va="center")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_confusion_matrices(
    cm_norm_dict: dict[str, np.ndarray],
    class_names: list[str],
    out_path: Path,
) -> Path:
    """Render the 2x2 grid of row-normalized confusion matrices."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    fig.suptitle("SMART IDS v3 — Normalized Confusion Matrices", fontsize=14, fontweight="bold")

    order = [
        ("MLP", class_names, "MLP Confusion Matrix", axes[0, 0]),
        ("CNN", class_names, "1D-CNN Confusion Matrix", axes[0, 1]),
        ("XGBoost", class_names, "XGBoost Confusion Matrix", axes[1, 0]),
        ("Autoencoder", ["Benign", "Attack"], "Autoencoder Anomaly Matrix (Binary)", axes[1, 1]),
    ]
    for name, labels, title, ax in order:
        mat = cm_norm_dict.get(name)
        if mat is None:
            ax.text(0.5, 0.5, f"{name} unavailable", ha="center", va="center")
            ax.set_title(title)
            continue
        arr = np.asarray(mat, dtype=np.float64)
        sns.heatmap(
            arr,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
            xticklabels=labels,
            yticklabels=labels,
            annot_kws={"size": 7 if len(labels) > 4 else 10},
            ax=ax,
        )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_roc_curves(
    y_test: np.ndarray,
    supervised_probas: dict[str, np.ndarray],
    y_test_binary: np.ndarray,
    ae_errors: np.ndarray,
    class_names: list[str],
    out_path: Path,
    baseline_anomaly_scores: dict[str, np.ndarray] | None = None,
) -> Path:
    """Render one-vs-rest multi-class ROC curves and binary Autoencoder ROC curve."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("SMART IDS v3 — ROC Curves & AUC Diagnostics", fontsize=14, fontweight="bold")

    n_classes = len(class_names)
    y_bin = label_binarize(y_test, classes=list(range(n_classes)))
    if n_classes == 2 and y_bin.shape[1] == 1:
        y_bin = np.hstack([1 - y_bin, y_bin])

    slots = [
        ("MLP", axes[0, 0]),
        ("CNN", axes[0, 1]),
        ("XGBoost", axes[1, 0]),
    ]
    all_fpr = np.linspace(0.0, 1.0, 100)

    for m_name, ax in slots:
        probs = supervised_probas.get(m_name)
        if probs is None:
            ax.text(0.5, 0.5, f"{m_name} unavailable", ha="center", va="center")
            continue
        mean_tpr = np.zeros_like(all_fpr)
        valid = 0
        for i in range(min(n_classes, probs.shape[1])):
            if len(np.unique(y_bin[:, i])) > 1:
                fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
                c_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, lw=1.4, alpha=0.8, label=f"{class_names[i]} (AUC={c_auc:.3f})")
                mean_tpr += np.interp(all_fpr, fpr, tpr)
                valid += 1
        if valid > 0:
            mean_tpr /= valid
            macro_auc = auc(all_fpr, mean_tpr)
            ax.plot(all_fpr, mean_tpr, "k--", lw=2.3, label=f"Macro-avg (AUC={macro_auc:.3f})")
        ax.plot([0, 1], [0, 1], "k:", alpha=0.5)
        ax.set_title(f"{m_name} — One-vs-Rest ROC", fontweight="bold")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(fontsize=7.5, loc="lower right")

    # Anomaly Detectors Binary ROC (Autoencoder + Classical Baselines)
    ax_ae = axes[1, 1]
    if len(np.unique(y_test_binary)) > 1:
        fpr, tpr, _ = roc_curve(y_test_binary, ae_errors)
        ae_auc = auc(fpr, tpr)
        ax_ae.plot(fpr, tpr, lw=2.3, color="crimson", label=f"Autoencoder AUC = {ae_auc:.3f}")
        for b_name, b_scores in (baseline_anomaly_scores or {}).items():
            b_fpr, b_tpr, _ = roc_curve(y_test_binary, b_scores)
            b_auc = auc(b_fpr, b_tpr)
            ax_ae.plot(b_fpr, b_tpr, lw=1.7, ls="--", label=f"{b_name} AUC = {b_auc:.3f}")
    ax_ae.plot([0, 1], [0, 1], "k:", alpha=0.5)
    ax_ae.set_title("Unsupervised Anomaly — Binary ROC", fontweight="bold")
    ax_ae.set_xlabel("False Positive Rate")
    ax_ae.set_ylabel("True Positive Rate")
    ax_ae.legend(loc="lower right")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_precision_recall_curves(
    y_test: np.ndarray,
    supervised_probas: dict[str, np.ndarray],
    y_test_binary: np.ndarray,
    ae_errors: np.ndarray,
    class_names: list[str],
    out_path: Path,
    baseline_anomaly_scores: dict[str, np.ndarray] | None = None,
) -> Path:
    """Render Precision-Recall curves across supervised models and Autoencoder."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "SMART IDS v3 — Precision-Recall Curves (Imbalance Sensitive)", fontsize=14, fontweight="bold"
    )

    n_classes = len(class_names)
    y_bin = label_binarize(y_test, classes=list(range(n_classes)))
    for m_name, ax in [("MLP", axes[0, 0]), ("CNN", axes[0, 1]), ("XGBoost", axes[1, 0])]:
        probs = supervised_probas.get(m_name)
        if probs is None:
            continue
        for i in range(min(n_classes, probs.shape[1])):
            if len(np.unique(y_bin[:, i])) > 1:
                prec, rec, _ = precision_recall_curve(y_bin[:, i], probs[:, i])
                ap = average_precision_score(y_bin[:, i], probs[:, i])
                ax.plot(rec, prec, lw=1.4, label=f"{class_names[i]} (AP={ap:.3f})")
        ax.set_title(f"{m_name} — Precision-Recall", fontweight="bold")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend(fontsize=7.5, loc="lower left")

    ax_ae = axes[1, 1]
    if len(np.unique(y_test_binary)) > 1:
        prec, rec, _ = precision_recall_curve(y_test_binary, ae_errors)
        ap = average_precision_score(y_test_binary, ae_errors)
        ax_ae.plot(rec, prec, lw=2.3, color="#0ea5e9", label=f"Autoencoder AP = {ap:.3f}")
        for b_name, b_scores in (baseline_anomaly_scores or {}).items():
            b_prec, b_rec, _ = precision_recall_curve(y_test_binary, b_scores)
            b_ap = average_precision_score(y_test_binary, b_scores)
            ax_ae.plot(b_rec, b_prec, lw=1.7, ls="--", label=f"{b_name} AP = {b_ap:.3f}")
    ax_ae.set_title("Unsupervised Anomaly — Binary Precision-Recall", fontweight="bold")
    ax_ae.set_xlabel("Recall")
    ax_ae.set_ylabel("Precision")
    ax_ae.legend(loc="lower left")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_autoencoder_diagnostics(
    recon_errors_test: np.ndarray,
    y_test_binary: np.ndarray,
    threshold: float,
    percentile_sweep: list[dict[str, float]],
    out_path: Path,
) -> Path:
    """Plot benign vs attack MSE distributions and percentile threshold trade-offs."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    benign_err = np.clip(recon_errors_test[y_test_binary == 0], 1e-6, None)
    attack_err = np.clip(recon_errors_test[y_test_binary == 1], 1e-6, None)

    axes[0].hist(np.log10(benign_err), bins=60, alpha=0.65, density=True, label="Benign", color="#22c55e")
    if len(attack_err) > 0:
        axes[0].hist(np.log10(attack_err), bins=60, alpha=0.6, density=True, label="Attack", color="#ef4444")
    axes[0].axvline(
        np.log10(max(threshold, 1e-6)),
        color="#0f172a",
        linestyle="--",
        lw=2,
        label=f"95th pct Threshold ({threshold:.3f})",
    )
    axes[0].set_title("Reconstruction MSE Distribution (log10 scale)", fontweight="bold")
    axes[0].set_xlabel("log10(Reconstruction MSE)")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    if percentile_sweep:
        df_s = pd.DataFrame(percentile_sweep)
        axes[1].plot(df_s["percentile"], df_s["precision"], marker="o", label="Precision", lw=2)
        axes[1].plot(df_s["percentile"], df_s["recall"], marker="s", label="Recall", lw=2)
        axes[1].plot(df_s["percentile"], df_s["f1"], marker="^", label="F1-Score", lw=2)
        axes[1].plot(df_s["percentile"], df_s["fpr"], marker="x", linestyle="--", label="FPR", lw=1.8)
        axes[1].set_title("Autoencoder Threshold Percentile Trade-Off", fontweight="bold")
        axes[1].set_xlabel("Benign Calibration Percentile")
        axes[1].set_ylabel("Score")
        axes[1].set_ylim(0.0, 1.05)
        axes[1].legend()

    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_class_distributions(
    pre_counts: dict[str, int],
    post_counts: dict[str, int],
    out_path: Path,
) -> Path:
    """Plot pre-SMOTE vs post-SMOTE training class distributions."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    classes = list(pre_counts.keys())
    pre_vals = [pre_counts[c] for c in classes]
    post_vals = [post_counts.get(c, 0) for c in classes]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    axes[0].barh(classes, pre_vals, color="#38bdf8")
    axes[0].set_title("Training Class Distribution (Before SMOTE)", fontweight="bold")
    axes[0].set_xlabel("Sample Count")

    axes[1].barh(classes, post_vals, color="#22c55e")
    axes[1].set_title("Training Class Distribution (After SMOTE)", fontweight="bold")
    axes[1].set_xlabel("Sample Count")

    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path
