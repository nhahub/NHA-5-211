"""Reproducibility and intelligent GPU/CPU hardware configuration."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import os
import random
import subprocess

import numpy as np

from src.utils.logging import get_logger

logger = get_logger("smart_ids.reproducibility")


@dataclass(frozen=True)
class HardwareProfile:
    """Detected hardware acceleration capabilities."""

    tf_gpus: list[str]
    tf_device: str
    xgb_device: str
    cuda_driver_detected: bool
    notes: str


def set_global_seed(seed: int = 42) -> None:
    """Set deterministic seeds across Python, NumPy, and TensorFlow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("Could not set TensorFlow random seed: %s", exc)


def _detect_nvidia_smi() -> bool:
    """Check whether an NVIDIA GPU driver is accessible via nvidia-smi."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return proc.returncode == 0 and bool(proc.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def _probe_xgboost_cuda() -> bool:
    """Probe whether XGBoost can actually execute on CUDA in this environment."""
    try:
        import xgboost as xgb

        x_probe = np.array([[0.0, 1.0], [1.0, 0.0], [0.2, 0.8], [0.8, 0.2]], dtype=np.float32)
        y_probe = np.array([0, 1, 0, 1], dtype=np.int32)
        clf = xgb.XGBClassifier(
            n_estimators=2,
            max_depth=2,
            tree_method="hist",
            device="cuda",
        )
        clf.fit(x_probe, y_probe)
        return True
    except Exception:
        return False


def configure_hardware(
    use_gpu: str | bool = "auto",
    memory_growth: bool = True,
    mixed_precision: bool = False,
) -> HardwareProfile:
    """Configure TensorFlow and XGBoost compute devices with safe CPU fallback."""
    want_gpu = (
        str(use_gpu).lower() in {"auto", "true", "1", "yes"} if not isinstance(use_gpu, bool) else use_gpu
    )
    force_cpu = str(use_gpu).lower() in {"false", "0", "no"}

    cuda_present = _detect_nvidia_smi()
    tf_gpus: list[str] = []
    tf_device = "CPU"
    xgb_device = "cpu"
    notes_parts: list[str] = []

    if force_cpu or not want_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        notes_parts.append("GPU explicitly disabled by configuration; running on CPU.")
    else:
        try:
            import tensorflow as tf

            physical_gpus = tf.config.list_physical_devices("GPU")
            if physical_gpus:
                for gpu in physical_gpus:
                    if memory_growth:
                        with contextlib.suppress(RuntimeError):
                            tf.config.experimental.set_memory_growth(gpu, True)
                    tf_gpus.append(gpu.name)
                tf_device = f"GPU ({len(tf_gpus)}x)"
                if mixed_precision:
                    tf.keras.mixed_precision.set_global_policy("mixed_float16")
                    notes_parts.append("TensorFlow mixed_float16 enabled.")
            else:
                notes_parts.append(
                    "TensorFlow GPU not available on native Windows (>=2.11 CPU-only) or no CUDA runtime; using CPU."
                )
        except Exception as exc:
            notes_parts.append(f"TensorFlow device probe failed ({exc}); using CPU.")

        if cuda_present and _probe_xgboost_cuda():
            xgb_device = "cuda"
            notes_parts.append("XGBoost CUDA acceleration verified and enabled.")
        else:
            xgb_device = "cpu"
            notes_parts.append("XGBoost falling back to multi-core CPU hist.")

    profile = HardwareProfile(
        tf_gpus=tf_gpus,
        tf_device=tf_device,
        xgb_device=xgb_device,
        cuda_driver_detected=cuda_present,
        notes=" ".join(notes_parts),
    )
    logger.info(
        "Hardware profile -> TF: %s | XGBoost: %s | NVIDIA driver: %s | %s",
        profile.tf_device,
        profile.xgb_device,
        profile.cuda_driver_detected,
        profile.notes,
    )
    return profile
