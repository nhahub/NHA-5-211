"""Standardized Keras training callbacks for SMART IDS v3."""

from __future__ import annotations

from pathlib import Path

import keras


def build_keras_callbacks(
    checkpoint_path: Path | None = None,
    early_stopping_patience: int = 8,
    reduce_lr_patience: int = 4,
    reduce_lr_factor: float = 0.5,
    min_lr: float = 1e-6,
) -> list[keras.callbacks.Callback]:
    """Create EarlyStopping, ReduceLROnPlateau, and optional ModelCheckpoint callbacks."""
    callbacks: list[keras.callbacks.Callback] = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=reduce_lr_factor,
            patience=reduce_lr_patience,
            min_lr=min_lr,
            verbose=1,
        ),
    ]
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor="val_loss",
                save_best_only=True,
                verbose=0,
            )
        )
    return callbacks
