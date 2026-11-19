"""Unsupervised undercomplete Autoencoder for novelty / zero-day anomaly tracking."""

from __future__ import annotations

from typing import Any

import keras
from keras import layers, regularizers
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


@keras.saving.register_keras_serializable(package="smart_ids")
class VAESampling(layers.Layer):
    """Reparameterization sampling layer adding KL-divergence regularization loss during training."""

    def __init__(self, kl_weight: float = 0.01, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.kl_weight = float(kl_weight)

    def call(self, inputs: list[Any], training: bool | None = None) -> Any:
        z_mean, z_log_var = inputs
        z_log_var_c = keras.ops.clip(z_log_var, -10.0, 10.0)
        kl = -0.5 * keras.ops.mean(1.0 + z_log_var_c - keras.ops.square(z_mean) - keras.ops.exp(z_log_var_c))
        self.add_loss(self.kl_weight * kl)
        if training:
            eps = keras.random.normal(shape=keras.ops.shape(z_mean))
            return z_mean + keras.ops.exp(0.5 * z_log_var_c) * eps
        return z_mean

    def get_config(self) -> dict[str, Any]:
        cfg = super().get_config()
        cfg["kl_weight"] = self.kl_weight
        return cfg


def build_deep_regularized_autoencoder(
    input_dim: int,
    cfg: AutoencoderConfig | None = None,
) -> keras.Model:
    """Build a deeper benign-only denoising + L2/Dropout regularized Autoencoder."""
    conf = cfg or AutoencoderConfig()
    reg = regularizers.l2(conf.deep_reg_l2)

    inputs = keras.Input(shape=(input_dim,), name="deep_ae_input")
    x = layers.GaussianNoise(conf.deep_reg_noise_std, name="deep_ae_noise")(inputs)
    for idx, units in enumerate(conf.deep_reg_encoder_units, start=1):
        x = layers.Dense(units, kernel_regularizer=reg, use_bias=False, name=f"deep_enc_dense_{idx}")(x)
        x = layers.BatchNormalization(name=f"deep_enc_bn_{idx}")(x)
        x = layers.LeakyReLU(negative_slope=0.1, name=f"deep_enc_lrelu_{idx}")(x)
        if conf.deep_reg_dropout > 0.0:
            x = layers.Dropout(conf.deep_reg_dropout, name=f"deep_enc_drop_{idx}")(x)

    x = layers.Dense(
        conf.deep_reg_latent_dim,
        kernel_regularizer=reg,
        activation="linear",
        name="deep_latent",
    )(x)
    x = layers.LeakyReLU(negative_slope=0.1, name="deep_latent_act")(x)

    for idx, units in enumerate(reversed(conf.deep_reg_encoder_units), start=1):
        x = layers.Dense(units, kernel_regularizer=reg, use_bias=False, name=f"deep_dec_dense_{idx}")(x)
        x = layers.BatchNormalization(name=f"deep_dec_bn_{idx}")(x)
        x = layers.LeakyReLU(negative_slope=0.1, name=f"deep_dec_lrelu_{idx}")(x)

    reconstruction = layers.Dense(input_dim, activation="linear", name="deep_reconstruction")(x)
    model = keras.Model(inputs=inputs, outputs=reconstruction, name="smart_ids_deep_reg_ae")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=conf.learning_rate, clipnorm=1.0),
        loss="mse",
    )
    return model


def build_vae(
    input_dim: int,
    cfg: AutoencoderConfig | None = None,
) -> keras.Model:
    """Build a benign-only Variational Autoencoder (VAE) using MSE + KL regularization."""
    conf = cfg or AutoencoderConfig()
    reg = regularizers.l2(conf.vae_l2)

    inputs = keras.Input(shape=(input_dim,), name="vae_input")
    x = inputs
    for idx, units in enumerate(conf.vae_encoder_units, start=1):
        x = layers.Dense(units, kernel_regularizer=reg, name=f"vae_enc_dense_{idx}")(x)
        x = layers.BatchNormalization(name=f"vae_enc_bn_{idx}")(x)
        x = layers.LeakyReLU(negative_slope=0.1, name=f"vae_enc_lrelu_{idx}")(x)

    z_mean = layers.Dense(conf.vae_latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(conf.vae_latent_dim, name="z_log_var")(x)
    z = VAESampling(kl_weight=conf.vae_kl_weight, name="vae_sampling")([z_mean, z_log_var])

    y = z
    for idx, units in enumerate(reversed(conf.vae_encoder_units), start=1):
        y = layers.Dense(units, kernel_regularizer=reg, name=f"vae_dec_dense_{idx}")(y)
        y = layers.BatchNormalization(name=f"vae_dec_bn_{idx}")(y)
        y = layers.LeakyReLU(negative_slope=0.1, name=f"vae_dec_lrelu_{idx}")(y)

    reconstruction = layers.Dense(input_dim, activation="linear", name="vae_reconstruction")(y)
    vae = keras.Model(inputs=inputs, outputs=reconstruction, name="smart_ids_vae")
    vae.compile(
        optimizer=keras.optimizers.Adam(learning_rate=conf.learning_rate, clipnorm=1.0),
        loss="mse",
    )
    return vae


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
