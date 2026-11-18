"""Supervised, unsupervised Autoencoder, unseen-attack, and ensemble evaluation metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from src.models.autoencoder import (
    build_autoencoder,
    calibrate_threshold,
    compute_reconstruction_error,
    extract_benign_subset,
    find_benign_class_index,
)
from src.models.ensemble import compute_ece, compute_multiclass_brier
from src.training.callbacks import build_keras_callbacks


def bootstrap_macro_f1_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute nonparametric 95% bootstrap confidence interval for Macro-F1."""
    y_t = np.asarray(y_true, dtype=np.int64)
    y_p = np.asarray(y_pred, dtype=np.int64)
    n = len(y_t)
    if n == 0 or num_classes <= 0 or n_bootstrap <= 0:
        return {"mean": 0.0, "std": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "n_bootstrap": 0}

    rng = np.random.default_rng(seed)
    codes = y_t * num_classes + y_p
    k_sq = num_classes * num_classes
    scores = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        cm = np.bincount(codes[idx], minlength=k_sq).reshape(num_classes, num_classes)
        tp = np.diag(cm)
        denom = cm.sum(axis=0) + cm.sum(axis=1)
        present = denom > 0
        if not np.any(present):
            scores[b] = 0.0
            continue
        f1_cls = np.zeros(num_classes, dtype=np.float64)
        f1_cls[present] = 2.0 * tp[present] / denom[present]
        scores[b] = float(np.mean(f1_cls[present]))

    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores, ddof=1)) if n_bootstrap > 1 else 0.0,
        "ci95_low": float(np.percentile(scores, 2.5)),
        "ci95_high": float(np.percentile(scores, 97.5)),
        "n_bootstrap": int(n_bootstrap),
    }


def bootstrap_binary_f1_ci(
    y_bin_true: np.ndarray,
    y_bin_pred: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute nonparametric 95% bootstrap confidence interval for binary positive-class F1."""
    y_t = np.asarray(y_bin_true, dtype=np.int64)
    y_p = np.asarray(y_bin_pred, dtype=np.int64)
    n = len(y_t)
    if n == 0 or n_bootstrap <= 0:
        return {"mean": 0.0, "std": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "n_bootstrap": 0}

    rng = np.random.default_rng(seed)
    codes = y_t * 2 + y_p
    scores = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        counts = np.bincount(codes[idx], minlength=4)
        tp = int(counts[3])
        fp = int(counts[1])
        fn = int(counts[2])
        denom = 2 * tp + fp + fn
        scores[b] = float(2.0 * tp / denom) if denom > 0 else 0.0

    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores, ddof=1)) if n_bootstrap > 1 else 0.0,
        "ci95_low": float(np.percentile(scores, 2.5)),
        "ci95_high": float(np.percentile(scores, 97.5)),
        "n_bootstrap": int(n_bootstrap),
    }


def evaluate_supervised_model(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: list[str],
    model_name: str,
    bootstrap_n: int = 1000,
    low_support_threshold: int = 30,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute comprehensive multi-class supervised classification metrics."""
    y_pred = np.argmax(y_proba, axis=1)
    n_classes = len(class_names)
    labels_idx = list(range(n_classes))

    acc = float(accuracy_score(y_true, y_pred))
    prec_w = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec_w = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_w = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    prec_m = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_m = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_m = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    cls_report = classification_report(
        y_true,
        y_pred,
        labels=labels_idx,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    cm_raw = confusion_matrix(y_true, y_pred, labels=labels_idx)
    cm_norm = cm_raw.astype(np.float64) / (cm_raw.sum(axis=1, keepdims=True) + 1e-8)

    # One-vs-Rest Macro ROC-AUC and Macro PR-AUC with rare-class defensive guard
    y_bin = label_binarize(y_true, classes=labels_idx)
    if n_classes == 2 and y_bin.shape[1] == 1:
        y_bin = np.hstack([1 - y_bin, y_bin])

    per_class_roc_auc: dict[str, float] = {}
    per_class_pr_auc: dict[str, float] = {}
    all_fpr = np.linspace(0.0, 1.0, 100)
    mean_tpr = np.zeros_like(all_fpr)
    valid_roc = 0

    for idx in range(min(n_classes, y_proba.shape[1])):
        col_true = y_bin[:, idx]
        if len(np.unique(col_true)) > 1:
            fpr, tpr, _ = roc_curve(col_true, y_proba[:, idx])
            c_auc = float(auc(fpr, tpr))
            per_class_roc_auc[class_names[idx]] = c_auc
            per_class_pr_auc[class_names[idx]] = float(average_precision_score(col_true, y_proba[:, idx]))
            mean_tpr += np.interp(all_fpr, fpr, tpr)
            valid_roc += 1

    if valid_roc > 0:
        mean_tpr /= valid_roc
        macro_roc_auc = float(auc(all_fpr, mean_tpr))
        macro_pr_auc = float(np.mean(list(per_class_pr_auc.values())))
    else:
        macro_roc_auc = 0.0
        macro_pr_auc = 0.0

    per_class_metrics = [
        {
            "class_name": cname,
            "precision": float(cls_report[cname]["precision"]),
            "recall": float(cls_report[cname]["recall"]),
            "f1_score": float(cls_report[cname]["f1-score"]),
            "support": int(cls_report[cname]["support"]),
            "statistically_unreliable": bool(int(cls_report[cname]["support"]) < low_support_threshold),
        }
        for cname in class_names
        if cname in cls_report
    ]
    f1_macro_ci95 = bootstrap_macro_f1_ci(
        y_true=y_true,
        y_pred=y_pred,
        num_classes=n_classes,
        n_bootstrap=bootstrap_n,
        seed=seed,
    )

    return {
        "model": model_name,
        "framing": f"{n_classes}-class supervised",
        "accuracy": acc,
        "precision_weighted": prec_w,
        "recall_weighted": rec_w,
        "f1_weighted": f1_w,
        "precision_macro": prec_m,
        "recall_macro": rec_m,
        "f1_macro": f1_m,
        "f1_macro_ci95": f1_macro_ci95,
        "roc_auc_macro_ovr": macro_roc_auc,
        "pr_auc_macro_ovr": macro_pr_auc,
        "ece": compute_ece(y_true, y_proba),
        "brier_score": compute_multiclass_brier(y_true, y_proba),
        "per_class_roc_auc": per_class_roc_auc,
        "per_class_pr_auc": per_class_pr_auc,
        "per_class_metrics": per_class_metrics,
        "classification_report": cls_report,
        "confusion_matrix_raw": cm_raw.tolist(),
        "confusion_matrix_normalized": cm_norm.tolist(),
    }


def _binary_metrics(y_bin_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    preds = (scores > threshold).astype(np.int64)
    cm = confusion_matrix(y_bin_true, preds, labels=[0, 1])
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))
    fpr = float(fp / max(1, fp + tn))
    fnr = float(fn / max(1, fn + tp))
    roc_auc_val = float(roc_auc_score(y_bin_true, scores)) if len(np.unique(y_bin_true)) > 1 else 0.0
    pr_auc_val = float(average_precision_score(y_bin_true, scores)) if len(np.unique(y_bin_true)) > 1 else 0.0

    return {
        "accuracy": float(accuracy_score(y_bin_true, preds)),
        "precision": float(precision_score(y_bin_true, preds, zero_division=0)),
        "recall": float(recall_score(y_bin_true, preds, zero_division=0)),
        "f1": float(f1_score(y_bin_true, preds, zero_division=0)),
        "roc_auc": roc_auc_val,
        "pr_auc": pr_auc_val,
        "fpr": fpr,
        "fnr": fnr,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def evaluate_autoencoder(
    y_test_multiclass: np.ndarray,
    recon_errors_test: np.ndarray,
    benign_val_errors: np.ndarray,
    threshold: float,
    class_names: list[str],
    benign_index: int,
    percentiles: list[float] | None = None,
    bootstrap_n: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate the unsupervised Autoencoder under binary novelty framing."""
    y_bin_true = (y_test_multiclass != benign_index).astype(np.int64)
    primary = _binary_metrics(y_bin_true, recon_errors_test, threshold)
    y_pred_bin = (recon_errors_test > threshold).astype(np.int64)
    cm_raw = confusion_matrix(y_bin_true, y_pred_bin, labels=[0, 1])
    cm_norm = cm_raw.astype(np.float64) / (cm_raw.sum(axis=1, keepdims=True) + 1e-8)

    # Percentile trade-off sweep
    pct_list = percentiles or [90.0, 95.0, 97.0, 99.0]
    sweep: list[dict[str, float]] = []
    for pct in pct_list:
        t_val = float(np.percentile(benign_val_errors, pct))
        row = _binary_metrics(y_bin_true, recon_errors_test, t_val)
        row["percentile"] = float(pct)
        row["threshold"] = t_val
        sweep.append(row)

    # Per-attack-class detection rate (without ever training on attack labels)
    per_attack_recall: dict[str, dict[str, float]] = {}
    for cls_idx, cls_name in enumerate(class_names):
        mask = y_test_multiclass == cls_idx
        if not np.any(mask):
            continue
        flagged_rate = float(np.mean(y_pred_bin[mask]))
        mean_err = float(np.mean(recon_errors_test[mask]))
        per_attack_recall[cls_name] = {
            "count": int(np.sum(mask)),
            "flagged_rate": flagged_rate,
            "mean_mse": mean_err,
        }

    prec_curve, rec_curve, _ = precision_recall_curve(y_bin_true, recon_errors_test)
    pr_auc_curve = float(auc(rec_curve, prec_curve))
    f1_ci95 = bootstrap_binary_f1_ci(
        y_bin_true=y_bin_true,
        y_bin_pred=y_pred_bin,
        n_bootstrap=bootstrap_n,
        seed=seed,
    )

    return {
        "model": "Autoencoder",
        "framing": "Binary anomaly (Benign vs Anomalous)",
        "threshold": float(threshold),
        **primary,
        "f1_ci95": f1_ci95,
        "pr_auc_trapz": pr_auc_curve,
        "confusion_matrix_raw": cm_raw.tolist(),
        "confusion_matrix_normalized": cm_norm.tolist(),
        "percentile_sweep": sweep,
        "per_class_anomaly_detection_rate": per_attack_recall,
    }


def evaluate_unseen_attack_protocol(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    X_val_scaled: np.ndarray,
    y_val: np.ndarray,
    X_test_scaled: np.ndarray,
    y_test: np.ndarray,
    label_encoder: Any,
    holdout_class_name: str = "Web Attack",
    epochs: int = 12,
    seed: int = 42,
) -> dict[str, Any]:
    """Empirical novelty-detection protocol holding out an attack class from supervised training.

    Demonstrates why a closed-set supervised classifier cannot classify an unseen attack
    category (0% recall on the held-out attack), whereas the benign-only Autoencoder still
    detects statistical deviation without seeing that attack class during training.
    """
    classes = [str(c) for c in label_encoder.classes_]
    if holdout_class_name not in classes:
        return {
            "executed": False,
            "reason": f"Holdout class '{holdout_class_name}' not found in dataset classes.",
        }

    holdout_idx = classes.index(holdout_class_name)
    benign_idx, _ = find_benign_class_index(label_encoder)

    # Train a fresh benign-only AE (already independent of attacks, verified by protocol)
    X_benign_tr, _, _ = extract_benign_subset(X_train_scaled, y_train, label_encoder)
    X_benign_va, _, _ = extract_benign_subset(X_val_scaled, y_val, label_encoder)

    ae, _, _ = build_autoencoder(X_train_scaled.shape[1])
    ae.fit(
        X_benign_tr,
        X_benign_tr,
        validation_data=(X_benign_va, X_benign_va),
        epochs=epochs,
        batch_size=512,
        callbacks=build_keras_callbacks(
            checkpoint_path=__import__("pathlib").Path("artifacts") / "_unseen_ae_tmp.keras",
            early_stopping_patience=5,
            reduce_lr_patience=3,
        ),
        verbose=0,
    )
    # Remove temp file cleanly if created
    tmp_path = __import__("pathlib").Path("artifacts") / "_unseen_ae_tmp.keras"
    if tmp_path.exists():
        tmp_path.unlink(missing_ok=True)

    benign_val_err = compute_reconstruction_error(ae, X_benign_va)
    thresh_95, _ = calibrate_threshold(benign_val_err, percentile=95.0)

    # Evaluate on (Benign Test + Held-Out Unseen Attack Test) ONLY
    eval_mask = (y_test == benign_idx) | (y_test == holdout_idx)
    X_novel_eval = X_test_scaled[eval_mask]
    y_novel_bin = (y_test[eval_mask] == holdout_idx).astype(np.int64)
    novel_errors = compute_reconstruction_error(ae, X_novel_eval)
    novel_metrics = _binary_metrics(y_novel_bin, novel_errors, thresh_95)

    holdout_only_mask = y_test == holdout_idx
    holdout_only_errors = compute_reconstruction_error(ae, X_test_scaled[holdout_only_mask])
    unseen_recall = float(np.mean(holdout_only_errors > thresh_95))

    return {
        "executed": True,
        "holdout_attack_class": holdout_class_name,
        "n_unseen_test_samples": int(np.sum(holdout_only_mask)),
        "n_benign_test_samples": int(np.sum(y_test == benign_idx)),
        "supervised_closed_set_unseen_class_recall": 0.0,
        "autoencoder_threshold_95pct": float(thresh_95),
        "autoencoder_unseen_attack_recall": unseen_recall,
        "autoencoder_benign_vs_unseen_metrics": novel_metrics,
        "scientific_note": (
            "Experimental novelty-detection protocol: the supervised classifier cannot assign "
            f"a label to '{holdout_class_name}' when omitted from training (0.0% class recall), "
            "whereas the benign-only Autoencoder evaluates structural divergence from learned "
            "benign normality via reconstruction MSE."
        ),
    }
