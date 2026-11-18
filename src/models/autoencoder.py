"""Unsupervised undercomplete Autoencoder for novelty / zero-day anomaly tracking."""

from __future__ import annotations

from typing import Any

import keras
from keras import layers
import numpy as np

from src.config import AutoencoderConfig


def build_autoencoder(
    input_dim: int,
    cfg: AutoencoderConfig | None = None,
) -> tuple[keras.Model, keras.Model, keras.Model]:
    """Construct the symmetric 256->128->64->32->64->128->256 Autoencoder.

    Trained exclusively on BENIGN flows using Mean Squared Error reconstruction loss
    with a linear output layer matching z-score standardized unbounded inputs.
    """
    conf = cfg or AutoencoderConfig()

    # Encoder
    encoder_input = keras.Input(shape=(input_dim,), name="encoder_input")
    x = encoder_input
    for idx, units in enumerate(conf.encoder_units):
        x = layers.Dense(units, activation="relu", name=f"enc_dense_{idx + 1}")(x)
        if idx < len(conf.encoder_units) - 1:
            x = layers.BatchNormalization(name=f"enc_bn_{idx + 1}")(x)

    latent = layers.Dense(conf.latent_dim, activation="relu", name="latent_space")(x)
    encoder = keras.Model(encoder_input, latent, name="smart_ids_encoder")

    # Decoder (mirror of encoder)
    decoder_input = keras.Input(shape=(conf.latent_dim,), name="decoder_input")
    y = decoder_input
    for idx, units in enumerate(conf.decoder_units):
        y = layers.Dense(units, activation="relu", name=f"dec_dense_{idx + 1}")(y)
        if idx < len(conf.decoder_units) - 1:
            y = layers.BatchNormalization(name=f"dec_bn_{idx + 1}")(y)

    reconstruction = layers.Dense(input_dim, activation="linear", name="reconstruction")(y)
    decoder = keras.Model(decoder_input, reconstruction, name="smart_ids_decoder")

    # End-to-End Autoencoder
    ae_output = decoder(encoder(encoder_input))
    autoencoder = keras.Model(encoder_input, ae_output, name="smart_ids_autoencoder")
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=conf.learning_rate),
        loss="mse",
    )
    return autoencoder, encoder, decoder


def find_benign_class_index(label_encoder: Any) -> tuple[int, str]:
    """Dynamically locate the BENIGN class index inside the fitted LabelEncoder."""
    classes = [str(c) for c in label_encoder.classes_]
    for idx, name in enumerate(classes):
        if "benign" in name.lower():
            return idx, name
    return 0, classes[0]


def extract_benign_subset(
    X: np.ndarray,
    y: np.ndarray,
    label_encoder: Any,
) -> tuple[np.ndarray, int, str]:
    """Filter a feature matrix to strictly BENIGN rows without exposing attack samples."""
    benign_idx, benign_name = find_benign_class_index(label_encoder)
    mask = y == benign_idx
    return X[mask], benign_idx, benign_name


def compute_reconstruction_error(
    model: keras.Model,
    X: np.ndarray,
    batch_size: int = 2048,
) -> np.ndarray:
    """Compute per-sample MSE reconstruction error."""
    arr = np.asarray(X, dtype=np.float32)
    if len(arr) <= 128:
        X_recon = model(arr, training=False).numpy()
    else:
        X_recon = model.predict(arr, batch_size=batch_size, verbose=0)
    return np.mean(np.square(arr - X_recon), axis=1).astype(np.float64)


def calibrate_threshold(
    benign_val_errors: np.ndarray,
    percentile: float = 95.0,
) -> tuple[float, dict[str, float]]:
    """Calibrate binary anomaly threshold from benign validation reconstruction errors."""
    threshold = float(np.percentile(benign_val_errors, percentile))
    stats = {
        "mean": float(np.mean(benign_val_errors)),
        "std": float(np.std(benign_val_errors)),
        "median": float(np.median(benign_val_errors)),
        "p90": float(np.percentile(benign_val_errors, 90.0)),
        "p95": float(np.percentile(benign_val_errors, 95.0)),
        "p97": float(np.percentile(benign_val_errors, 97.0)),
        "p99": float(np.percentile(benign_val_errors, 99.0)),
        "max": float(np.max(benign_val_errors)),
    }
    return threshold, stats
