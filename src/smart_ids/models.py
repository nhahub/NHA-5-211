"""Model architectures for Smart IDS v3.

Constructs the 4 complementary learning paradigms:
1. Deep Multi-Layer Perceptron (MLP) with LeakyReLU, BatchNorm, L2, and Dropout.
2. 1D-Convolutional Neural Network (1D-CNN) for localized statistical co-occurrence.
3. Undercomplete Autoencoder for unsupervised zero-day anomaly detection.
4. Extreme Gradient Boosting (XGBoost) baseline with regularized histogram splitting.
"""

from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import xgboost as xgb

from smart_ids.config import (
    AE_DECODER_UNITS,
    AE_ENCODER_UNITS,
    AE_LATENT_DIM,
    AE_LR,
    CNN_BATCH_SIZE,
    CNN_FILTERS_BLOCK1,
    CNN_FILTERS_BLOCK2,
    CNN_HEAD_UNITS,
    CNN_KERNEL_SIZE,
    CNN_L2,
    CNN_LR,
    MLP_CLIPNORM,
    MLP_DROPOUT,
    MLP_L2,
    MLP_LEAKY_ALPHA,
    MLP_LR,
    MLP_UNITS,
    SEED,
    XGB_COLSAMPLE_BYTREE,
    XGB_EARLY_STOPPING_ROUNDS,
    XGB_GAMMA,
    XGB_LEARNING_RATE,
    XGB_MAX_DEPTH,
    XGB_MIN_CHILD_WEIGHT,
    XGB_N_ESTIMATORS,
    XGB_REG_ALPHA,
    XGB_REG_LAMBDA,
    XGB_SUBSAMPLE,
    XGB_TREE_METHOD,
)


def build_mlp(
    input_dim: int,
    num_classes: int = 8,
    units: Tuple[int, ...] = tuple(MLP_UNITS),
    l2_reg: float = MLP_L2,
    leaky_alpha: float = MLP_LEAKY_ALPHA,
    dropout_rate: float = MLP_DROPOUT,
    learning_rate: float = MLP_LR,
    clipnorm: float = MLP_CLIPNORM,
) -> keras.Model:
    """Build and compile the deep Multi-Layer Perceptron classifier.

    Architecture: 4 Dense blocks (512-256-128-64), each with L2 weight decay,
    use_bias=False (absorbed by BatchNorm), LeakyReLU activation, and Dropout(0.4).
    Final softmax head with Adam optimizer and gradient clipping.

    Args:
        input_dim: Number of input features.
        num_classes: Number of target traffic classes (default 8).
        units: Sequence of hidden layer widths.
        l2_reg: L2 kernel regularization coefficient.
        leaky_alpha: Slope coefficient for negative values in LeakyReLU.
        dropout_rate: Dropout fraction.
        learning_rate: Adam optimizer learning rate.
        clipnorm: Gradient norm ceiling.

    Returns:
        Compiled Keras Model.
    """
    inputs = keras.Input(shape=(input_dim,), name="mlp_input")
    x = inputs

    for idx, u in enumerate(units):
        x = layers.Dense(
            u,
            use_bias=False,
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f"dense_block_{idx + 1}",
        )(x)
        x = layers.BatchNormalization(name=f"bn_{idx + 1}")(x)
        x = layers.LeakyReLU(negative_slope=leaky_alpha, name=f"leaky_relu_{idx + 1}")(x)
        x = layers.Dropout(dropout_rate, name=f"dropout_{idx + 1}")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="SmartIDS_MLP")

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=clipnorm)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn(
    input_dim: int,
    num_classes: int = 8,
    filters_b1: int = CNN_FILTERS_BLOCK1,
    filters_b2: int = CNN_FILTERS_BLOCK2,
    kernel_size: int = CNN_KERNEL_SIZE,
    l2_reg: float = CNN_L2,
    head_units: int = CNN_HEAD_UNITS,
    learning_rate: float = CNN_LR,
    clipnorm: float = MLP_CLIPNORM,
) -> keras.Model:
    """Build and compile the 1D-Convolutional Neural Network classifier.

    Reinterprets the feature vector as a 1D signal with singleton channel.
    Block 1: 2x Conv1D(64) + BN + ReLU + MaxPooling1D(2) + Dropout(0.3).
    Block 2: 2x Conv1D(128) + BN + ReLU + GlobalAveragePooling1D + Dropout(0.4).
    Classification Head: Dense(128) + BN + ReLU + Dropout(0.3) + Softmax.

    Args:
        input_dim: Number of input features.
        num_classes: Number of target traffic classes.
        filters_b1: Number of filters in block 1 convolutions.
        filters_b2: Number of filters in block 2 convolutions.
        kernel_size: Convolution kernel width.
        l2_reg: L2 kernel regularization coefficient.
        head_units: Neurons in pre-classification dense layer.
        learning_rate: Adam optimizer learning rate.
        clipnorm: Gradient norm ceiling.

    Returns:
        Compiled Keras Model.
    """
    inputs = keras.Input(shape=(input_dim, 1), name="cnn_input")

    # Block 1: 64 filters
    x = layers.Conv1D(
        filters_b1,
        kernel_size=kernel_size,
        padding="same",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="conv1d_1",
    )(inputs)
    x = layers.BatchNormalization(name="bn_c1")(x)
    x = layers.Activation("relu", name="relu_c1")(x)

    x = layers.Conv1D(filters_b1, kernel_size=kernel_size, padding="same", name="conv1d_2")(x)
    x = layers.BatchNormalization(name="bn_c2")(x)
    x = layers.Activation("relu", name="relu_c2")(x)
    x = layers.MaxPooling1D(pool_size=2, name="maxpool_1")(x)
    x = layers.Dropout(0.3, name="dropout_c1")(x)

    # Block 2: 128 filters
    x = layers.Conv1D(
        filters_b2,
        kernel_size=kernel_size,
        padding="same",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="conv1d_3",
    )(x)
    x = layers.BatchNormalization(name="bn_c3")(x)
    x = layers.Activation("relu", name="relu_c3")(x)

    x = layers.Conv1D(filters_b2, kernel_size=kernel_size, padding="same", name="conv1d_4")(x)
    x = layers.BatchNormalization(name="bn_c4")(x)
    x = layers.Activation("relu", name="relu_c4")(x)
    x = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
    x = layers.Dropout(0.4, name="dropout_c2")(x)

    # Dense Classification Head
    x = layers.Dense(
        head_units,
        kernel_regularizer=regularizers.l2(l2_reg),
        name="dense_head",
    )(x)
    x = layers.BatchNormalization(name="bn_head")(x)
    x = layers.Activation("relu", name="relu_head")(x)
    x = layers.Dropout(0.3, name="dropout_head")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="SmartIDS_1DCNN")

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=clipnorm)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_autoencoder(
    input_dim: int,
    latent_dim: int = AE_LATENT_DIM,
    encoder_units: Tuple[int, ...] = tuple(AE_ENCODER_UNITS),
    decoder_units: Tuple[int, ...] = tuple(AE_DECODER_UNITS),
    learning_rate: float = AE_LR,
) -> Tuple[keras.Model, keras.Model, keras.Model]:
    """Build undercomplete Autoencoder for zero-day anomaly detection.

    Trained exclusively on benign network traffic to reconstruct normal behavior.
    Encoder: 256 -> 128 -> 64 -> Latent Bottleneck (32).
    Decoder: 64 -> 128 -> 256 -> Linear output (input_dim).

    Args:
        input_dim: Number of input features.
        latent_dim: Bottleneck dimension (default 32).
        encoder_units: Widths of encoder layers.
        decoder_units: Widths of decoder layers.
        learning_rate: Adam optimizer learning rate.

    Returns:
        Tuple of (full_autoencoder, encoder_submodel, decoder_submodel).
    """
    # Encoder
    enc_input = keras.Input(shape=(input_dim,), name="encoder_input")
    x = enc_input
    for idx, u in enumerate(encoder_units):
        x = layers.Dense(u, activation="relu", name=f"enc_dense_{idx + 1}")(x)
        x = layers.BatchNormalization(name=f"enc_bn_{idx + 1}")(x)
    latent = layers.Dense(latent_dim, activation="relu", name="latent_space")(x)
    encoder = keras.Model(inputs=enc_input, outputs=latent, name="SmartIDS_Encoder")

    # Decoder
    dec_input = keras.Input(shape=(latent_dim,), name="decoder_input")
    y = dec_input
    for idx, u in enumerate(decoder_units):
        y = layers.Dense(u, activation="relu", name=f"dec_dense_{idx + 1}")(y)
        y = layers.BatchNormalization(name=f"dec_bn_{idx + 1}")(y)
    reconstruction = layers.Dense(input_dim, activation="linear", name="reconstruction")(y)
    decoder = keras.Model(inputs=dec_input, outputs=reconstruction, name="SmartIDS_Decoder")

    # Full Autoencoder
    ae_output = decoder(encoder(enc_input))
    autoencoder = keras.Model(inputs=enc_input, outputs=ae_output, name="SmartIDS_Autoencoder")

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    autoencoder.compile(optimizer=optimizer, loss="mse")

    return autoencoder, encoder, decoder


def build_xgboost(
    num_classes: int = 8,
    random_state: int = SEED,
) -> xgb.XGBClassifier:
    """Build the Extreme Gradient Boosting baseline classifier.

    Configured with conservative depth, subsampling, and histogram split finding.
    Compatible with newer XGBoost versions by setting early_stopping_rounds in the constructor.

    Args:
        num_classes: Number of target traffic classes.
        random_state: PRNG seed for reproducibility.

    Returns:
        Configured XGBClassifier instance.
    """
    model = xgb.XGBClassifier(
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE_BYTREE,
        min_child_weight=XGB_MIN_CHILD_WEIGHT,
        gamma=XGB_GAMMA,
        reg_alpha=XGB_REG_ALPHA,
        reg_lambda=XGB_REG_LAMBDA,
        tree_method=XGB_TREE_METHOD,
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
        random_state=random_state,
        n_jobs=4,
    )
    return model
