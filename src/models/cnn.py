"""One-Dimensional Convolutional Neural Network (1D-CNN) for tabular flow telemetry."""

from __future__ import annotations

import keras
from keras import layers, regularizers

from src.config import CNNConfig


def build_cnn1d(
    input_dim: int,
    num_classes: int,
    cfg: CNNConfig | None = None,
) -> keras.Model:
    """Build the two-block Functional API 1D-CNN described in Chapter 4.2.

    Architectural Note:
        Tabular CICFlowMeter feature ordering does not represent physical time-series
        progression; rather, Conv1D kernels of width 3 operate over locally grouped
        statistics (e.g., consecutive IAT and packet-length descriptors). Whether this
        inductive bias outperforms the MLP and XGBoost is evaluated empirically.
    """
    conf = cfg or CNNConfig()

    inputs = keras.Input(shape=(input_dim, 1), name="flow_sequence")

    # Block 1: 64 filters + MaxPooling1D
    x = layers.Conv1D(
        conf.block1_filters,
        kernel_size=conf.kernel_size,
        padding="same",
        kernel_regularizer=regularizers.l2(conf.l2_reg),
        name="cnn_b1_conv1",
    )(inputs)
    x = layers.BatchNormalization(name="cnn_b1_bn1")(x)
    x = layers.Activation("relu", name="cnn_b1_relu1")(x)
    x = layers.Conv1D(
        conf.block1_filters,
        kernel_size=conf.kernel_size,
        padding="same",
        name="cnn_b1_conv2",
    )(x)
    x = layers.BatchNormalization(name="cnn_b1_bn2")(x)
    x = layers.Activation("relu", name="cnn_b1_relu2")(x)
    x = layers.MaxPooling1D(pool_size=conf.pool_size, name="cnn_b1_pool")(x)
    x = layers.Dropout(conf.dropout_block1, name="cnn_b1_drop")(x)

    # Block 2: 128 filters + GlobalAveragePooling1D
    x = layers.Conv1D(
        conf.block2_filters,
        kernel_size=conf.kernel_size,
        padding="same",
        kernel_regularizer=regularizers.l2(conf.l2_reg),
        name="cnn_b2_conv1",
    )(x)
    x = layers.BatchNormalization(name="cnn_b2_bn1")(x)
    x = layers.Activation("relu", name="cnn_b2_relu1")(x)
    x = layers.Conv1D(
        conf.block2_filters,
        kernel_size=conf.kernel_size,
        padding="same",
        name="cnn_b2_conv2",
    )(x)
    x = layers.BatchNormalization(name="cnn_b2_bn2")(x)
    x = layers.Activation("relu", name="cnn_b2_relu2")(x)
    x = layers.GlobalAveragePooling1D(name="cnn_gap")(x)
    x = layers.Dropout(conf.dropout_block2, name="cnn_b2_drop")(x)

    # Dense Classification Head
    x = layers.Dense(
        conf.dense_units,
        kernel_regularizer=regularizers.l2(conf.l2_reg),
        name="cnn_head_dense",
    )(x)
    x = layers.BatchNormalization(name="cnn_head_bn")(x)
    x = layers.Activation("relu", name="cnn_head_relu")(x)
    x = layers.Dropout(conf.dropout_head, name="cnn_head_drop")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="smart_ids_cnn1d")

    optimizer = keras.optimizers.Adam(learning_rate=conf.learning_rate, clipnorm=conf.clipnorm)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
