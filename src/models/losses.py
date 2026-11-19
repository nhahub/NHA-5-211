"""Custom serializable sparse-target classification losses for neural IDS models."""

from __future__ import annotations

from typing import Any

import keras


@keras.saving.register_keras_serializable(package="smart_ids")
class SparseLabelSmoothingCrossentropy(keras.losses.Loss):
    """Categorical cross-entropy with label smoothing accepting integer class targets."""

    def __init__(
        self,
        num_classes: int,
        label_smoothing: float = 0.05,
        name: str = "sparse_label_smoothing_ce",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.num_classes = int(num_classes)
        self.label_smoothing = float(label_smoothing)

    def call(self, y_true: Any, y_pred: Any) -> Any:
        y_int = keras.ops.cast(keras.ops.reshape(y_true, (-1,)), "int32")
        y_oh = keras.ops.one_hot(y_int, self.num_classes)
        y_oh = keras.ops.cast(y_oh, y_pred.dtype)
        eps = self.label_smoothing
        y_smooth = y_oh * (1.0 - eps) + (eps / float(max(self.num_classes, 1)))
        probs = keras.ops.clip(y_pred, 1e-7, 1.0 - 1e-7)
        return -keras.ops.sum(y_smooth * keras.ops.log(probs), axis=-1)

    def get_config(self) -> dict[str, Any]:
        cfg = super().get_config()
        cfg.update({"num_classes": self.num_classes, "label_smoothing": self.label_smoothing})
        return cfg


@keras.saving.register_keras_serializable(package="smart_ids")
class SparseCategoricalFocalLoss(keras.losses.Loss):
    """Multi-class focal loss down-weighting well-classified examples via ``(1 - p_t)^gamma``."""

    def __init__(
        self,
        num_classes: int,
        gamma: float = 2.0,
        name: str = "sparse_categorical_focal",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.num_classes = int(num_classes)
        self.gamma = float(gamma)

    def call(self, y_true: Any, y_pred: Any) -> Any:
        y_int = keras.ops.cast(keras.ops.reshape(y_true, (-1,)), "int32")
        y_oh = keras.ops.one_hot(y_int, self.num_classes)
        y_oh = keras.ops.cast(y_oh, y_pred.dtype)
        probs = keras.ops.clip(y_pred, 1e-7, 1.0 - 1e-7)
        p_t = keras.ops.sum(y_oh * probs, axis=-1)
        modulating = keras.ops.power(1.0 - p_t, self.gamma)
        return -modulating * keras.ops.log(p_t)

    def get_config(self) -> dict[str, Any]:
        cfg = super().get_config()
        cfg.update({"num_classes": self.num_classes, "gamma": self.gamma})
        return cfg


def resolve_sparse_classification_loss(
    loss_type: str,
    num_classes: int,
    label_smoothing: float = 0.05,
    focal_gamma: float = 2.0,
) -> str | keras.losses.Loss:
    """Resolve config loss identifier into Keras sparse classification loss."""
    norm = str(loss_type).strip().lower()
    if norm == "label_smoothing":
        return SparseLabelSmoothingCrossentropy(num_classes=num_classes, label_smoothing=label_smoothing)
    if norm == "focal_loss":
        return SparseCategoricalFocalLoss(num_classes=num_classes, gamma=focal_gamma)
    return "sparse_categorical_crossentropy"
