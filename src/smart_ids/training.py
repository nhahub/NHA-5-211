"""Training procedures and callback pipelines for Smart IDS v3 models.

Implements adaptive learning schedules, early stopping guards against SMOTE overfitting,
and model checkpoint persistence for MLP, 1D-CNN, Autoencoder, and XGBoost.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import xgboost as xgb

from smart_ids.config import (
    AE_BATCH_SIZE,
    AE_MAX_EPOCHS,
    CNN_BATCH_SIZE,
    CNN_MAX_EPOCHS,
    EARLY_STOPPING_PATIENCE_SUPERVISED,
    EARLY_STOPPING_PATIENCE_UNSUPERVISED,
    MIN_LR,
    MLP_BATCH_SIZE,
    MLP_MAX_EPOCHS,
    MODELS_DIR,
    REDUCE_LR_FACTOR,
    REDUCE_LR_PATIENCE,
)

logger = logging.getLogger("smart_ids.training")


def get_callbacks(
    model_name: str,
    patience_es: int = EARLY_STOPPING_PATIENCE_SUPERVISED,
    patience_lr: int = REDUCE_LR_PATIENCE,
    lr_factor: float = REDUCE_LR_FACTOR,
    min_lr: float = MIN_LR,
    checkpoint_name: Optional[str] = None,
) -> List[keras.callbacks.Callback]:
    """Construct standard Keras training callbacks.

    Includes EarlyStopping with best weight restoration, ReduceLROnPlateau,
    and ModelCheckpoint saving directly to the models directory.

    Args:
        model_name: Identifier for log messages and default checkpoint naming.
        patience_es: Epochs without val_loss improvement before early termination.
        patience_lr: Epochs without val_loss improvement before learning rate halving.
        lr_factor: Multiplicative factor for learning rate reduction.
        min_lr: Lower bound on learning rate.
        checkpoint_name: Custom filename for ModelCheckpoint (e.g. 'best_mlp.keras').

    Returns:
        List of configured Keras Callback instances.
    """
    ckpt_file = checkpoint_name or f"best_{model_name.lower()}.keras"
    ckpt_path = MODELS_DIR / ckpt_file

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=patience_es,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            patience=patience_lr,
            factor=lr_factor,
            min_lr=min_lr,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
    ]
    return callbacks


def train_mlp(
    model: keras.Model,
    X_train: np.ndarray,
    y_train_cat: np.ndarray,
    X_val: np.ndarray,
    y_val_cat: np.ndarray,
    epochs: int = MLP_MAX_EPOCHS,
    batch_size: int = MLP_BATCH_SIZE,
    callbacks: Optional[List[keras.callbacks.Callback]] = None,
) -> Tuple[keras.Model, keras.callbacks.History]:
    """Train the deep Multi-Layer Perceptron classifier.

    Args:
        model: Compiled MLP model.
        X_train: SMOTE-resampled, standardized training features.
        y_train_cat: One-hot encoded training targets.
        X_val: Standardized validation features (naturally imbalanced).
        y_val_cat: One-hot encoded validation targets.
        epochs: Maximum epoch budget.
        batch_size: Mini-batch size.
        callbacks: Optional custom callback list.

    Returns:
        Tuple of (trained_model, history).
    """
    if callbacks is None:
        callbacks = get_callbacks("mlp", checkpoint_name="best_mlp.keras")

    logger.info("Training MLP: max %d epochs, batch size %d...", epochs, batch_size)
    history = model.fit(
        X_train,
        y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    return model, history


def train_cnn(
    model: keras.Model,
    X_train: np.ndarray,
    y_train_cat: np.ndarray,
    X_val: np.ndarray,
    y_val_cat: np.ndarray,
    epochs: int = CNN_MAX_EPOCHS,
    batch_size: int = CNN_BATCH_SIZE,
    callbacks: Optional[List[keras.callbacks.Callback]] = None,
) -> Tuple[keras.Model, keras.callbacks.History]:
    """Train the 1D-Convolutional Neural Network classifier.

    Reshapes 2D inputs to (n_samples, n_features, 1) if not already 3D.

    Args:
        model: Compiled 1D-CNN model.
        X_train: SMOTE-resampled, standardized training features.
        y_train_cat: One-hot encoded training targets.
        X_val: Standardized validation features (naturally imbalanced).
        y_val_cat: One-hot encoded validation targets.
        epochs: Maximum epoch budget.
        batch_size: Mini-batch size.
        callbacks: Optional custom callback list.

    Returns:
        Tuple of (trained_model, history).
    """
    if X_train.ndim == 2:
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    if X_val.ndim == 2:
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

    if callbacks is None:
        callbacks = get_callbacks("cnn", checkpoint_name="best_cnn.keras")

    logger.info("Training 1D-CNN: max %d epochs, batch size %d...", epochs, batch_size)
    history = model.fit(
        X_train,
        y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    return model, history


def train_autoencoder(
    model: keras.Model,
    X_benign_train: np.ndarray,
    X_benign_val: np.ndarray,
    epochs: int = AE_MAX_EPOCHS,
    batch_size: int = AE_BATCH_SIZE,
    callbacks: Optional[List[keras.callbacks.Callback]] = None,
) -> Tuple[keras.Model, keras.callbacks.History]:
    """Train undercomplete Autoencoder exclusively on benign traffic flows.

    Self-supervised mean-squared-error objective: X -> X_recon.

    Args:
        model: Compiled Autoencoder model.
        X_benign_train: Benign-only standardized training features.
        X_benign_val: Benign-only standardized validation features.
        epochs: Maximum epoch budget.
        batch_size: Mini-batch size.
        callbacks: Optional custom callback list.

    Returns:
        Tuple of (trained_model, history).
    """
    if callbacks is None:
        callbacks = get_callbacks(
            "autoencoder",
            patience_es=EARLY_STOPPING_PATIENCE_UNSUPERVISED,
            checkpoint_name="best_autoencoder.keras",
        )

    logger.info("Training Autoencoder on %d benign samples...", len(X_benign_train))
    history = model.fit(
        X_benign_train,
        X_benign_train,
        validation_data=(X_benign_val, X_benign_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    return model, history


def train_xgboost(
    model: xgb.XGBClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> xgb.XGBClassifier:
    """Train the XGBoost baseline classifier with early stopping.

    Args:
        model: Configured XGBClassifier.
        X_train: SMOTE-resampled training features.
        y_train: Integer-encoded training labels.
        X_val: Non-resampled validation features.
        y_val: Integer-encoded validation labels.

    Returns:
        Trained XGBClassifier.
    """
    logger.info("Training XGBoost baseline on %d samples...", len(X_train))
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    logger.info("XGBoost training complete.")
    return model
