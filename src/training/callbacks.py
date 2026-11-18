"""Standardized Keras training callbacks for SMART IDS v3."""

from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any

import keras


def safe_save_keras_model(
    model: keras.Model,
    filepath: Path,
    weights_only: bool = False,
    retries: int = 5,
) -> None:
    """Save a Keras model atomically via a temporary .keras file to avoid Windows file locks."""
    target = Path(filepath)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f".{target.stem}.tmp.{os.getpid()}{target.suffix}")
    try:
        if weights_only:
            model.save_weights(tmp_path, overwrite=True)
        else:
            model.save(tmp_path, overwrite=True)
        for attempt in range(retries):
            try:
                os.replace(tmp_path, target)
                return
            except OSError:
                if attempt == retries - 1:
                    raise
                time.sleep(0.15 * (attempt + 1))
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


class SafeModelCheckpoint(keras.callbacks.ModelCheckpoint):
    """Atomic ModelCheckpoint resistant to transient OS/AV locks during overwrite."""

    def _save_model(self, epoch: int, batch: int | None, logs: dict[str, Any] | None) -> None:
        filepath = self._get_file_path(epoch, batch, logs)
        if not self._should_save_model(epoch, batch, logs, filepath):
            return
        safe_save_keras_model(self.model, Path(filepath), weights_only=bool(self.save_weights_only))


def build_keras_callbacks(
    checkpoint_path: Path | None = None,
    early_stopping_patience: int = 8,
    reduce_lr_patience: int = 4,
    reduce_lr_factor: float = 0.5,
    min_lr: float = 1e-6,
) -> list[keras.callbacks.Callback]:
    """Create EarlyStopping, ReduceLROnPlateau, and optional atomic SafeModelCheckpoint callbacks."""
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
            SafeModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor="val_loss",
                save_best_only=True,
                verbose=0,
            )
        )
    return callbacks
