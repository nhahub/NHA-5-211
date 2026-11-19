"""Multi-Layer Perceptron (MLP) baseline and configurable classifier."""

from __future__ import annotations

import keras
from keras import layers, regularizers

from src.config import MLPConfig
from src.models.losses import resolve_sparse_classification_loss


def build_mlp(
    input_dim: int,
    num_classes: int,
    cfg: MLPConfig | None = None,
    hidden_units: list[int] | None = None,
    dropout_rate: float | None = None,
    learning_rate: float | None = None,
    l2_reg: float | None = None,
    loss_type: str = "cross_entropy",
    label_smoothing: float = 0.05,
    focal_gamma: float = 2.0,
) -> keras.Model:
    """Build the 512->256->128->64 regularized feed-forward MLP classifier.

    Preserves the report's exact block ordering:
    Dense(use_bias=False, L2) -> BatchNormalization -> LeakyReLU -> Dropout.
    """
    conf = cfg or MLPConfig()
    units_seq = hidden_units if hidden_units is not None else conf.hidden_units
    drop = dropout_rate if dropout_rate is not None else conf.dropout_rate
    lr = learning_rate if learning_rate is not None else conf.learning_rate
    l2_val = l2_reg if l2_reg is not None else conf.l2_reg

    inputs = keras.Input(shape=(input_dim,), name="flow_features")
    x = inputs
    for idx, units in enumerate(units_seq, start=1):
        x = layers.Dense(
            units,
            kernel_regularizer=regularizers.l2(l2_val),
            use_bias=False,
            name=f"mlp_dense_{idx}",
        )(x)
        x = layers.BatchNormalization(name=f"mlp_bn_{idx}")(x)
        x = layers.LeakyReLU(negative_slope=conf.leaky_relu_slope, name=f"mlp_lrelu_{idx}")(x)
        x = layers.Dropout(drop, name=f"mlp_drop_{idx}")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="smart_ids_mlp")

    optimizer = keras.optimizers.Adam(learning_rate=lr, clipnorm=conf.clipnorm)
    model.compile(
        optimizer=optimizer,
        loss=resolve_sparse_classification_loss(
            loss_type=loss_type,
            num_classes=num_classes,
            label_smoothing=label_smoothing,
            focal_gamma=focal_gamma,
        ),
        metrics=["accuracy"],
    )
    return model
