"""Optional ONNX export and numerical parity verification for Keras models (MLP, 1D-CNN)."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import platform
import sys
import tempfile
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_app_config


def check_onnx_environment() -> dict[str, Any]:
    """Inspect installed ONNX tooling and runtime versions."""
    packages = ("onnx", "tf2onnx", "onnxruntime")
    available = {pkg: importlib.util.find_spec(pkg) is not None for pkg in packages}
    missing = [pkg for pkg, ok in available.items() if not ok]

    details: dict[str, Any] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "available": available,
        "missing": missing,
    }
    try:
        import keras
        import tensorflow as tf

        details["tensorflow"] = tf.__version__
        details["keras"] = keras.__version__
    except Exception as exc:  # pragma: no cover - environment dependent
        details["tensorflow_error"] = str(exc)

    return details


def _infer_n_features(models_dir: Path) -> int:
    feature_path = models_dir / "feature_names.json"
    if feature_path.exists():
        data = json.loads(feature_path.read_text(encoding="utf-8"))
        if isinstance(data, list) and data:
            return len(data)
    raise FileNotFoundError(f"Could not infer feature count from '{feature_path}'.")


def _export_single_keras_model(
    model_path: Path,
    onnx_path: Path,
    model_key: str,
    n_features: int,
    opset: int,
    verify: bool,
) -> dict[str, Any]:
    import tensorflow as tf
    import tf2onnx

    model = tf.keras.models.load_model(model_path, compile=False)
    input_shape = (1, n_features, 1) if model_key == "cnn" else (1, n_features)
    dynamic_shape = (None, n_features, 1) if model_key == "cnn" else (None, n_features)
    spec = (tf.TensorSpec(dynamic_shape, tf.float32, name="input"),)

    conversion_method = "from_keras"
    try:
        model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=opset)
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        with onnx_path.open("wb") as fh:
            fh.write(model_proto.SerializeToString())
    except Exception:
        # Keras 3.x Functional/Sequential models export reliably via SavedModel intermediate format.
        conversion_method = "from_saved_model"
        with tempfile.TemporaryDirectory() as tmp_dir:
            saved_dir = Path(tmp_dir) / f"{model_key}_saved_model"
            model.export(saved_dir, format="tf_saved_model")
            onnx_path.parent.mkdir(parents=True, exist_ok=True)
            tf2onnx.convert.from_saved_model(
                str(saved_dir),
                opset=opset,
                output_path=str(onnx_path),
            )

    result: dict[str, Any] = {
        "model": model_key,
        "source": str(model_path),
        "onnx_path": str(onnx_path),
        "opset": opset,
        "conversion_method": conversion_method,
        "status": "exported",
    }

    if verify:
        if importlib.util.find_spec("onnxruntime") is None:
            result["verification"] = "skipped_no_onnxruntime"
        else:
            import onnxruntime as ort

            rng = np.random.default_rng(42)
            sample = rng.normal(size=(8, *input_shape[1:])).astype(np.float32)
            keras_out = np.asarray(model(sample, training=False), dtype=np.float32)
            sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            input_name = sess.get_inputs()[0].name
            ort_out = np.asarray(sess.run(None, {input_name: sample})[0], dtype=np.float32)
            result["max_abs_diff"] = float(np.max(np.abs(keras_out - ort_out)))
            result["verification"] = "passed" if result["max_abs_diff"] < 1e-4 else "mismatch"

    return result


def export_models_to_onnx(
    models_dir: Path,
    out_dir: Path,
    targets: list[str],
    opset: int = 17,
    verify: bool = False,
) -> dict[str, Any]:
    """Export target Keras models to ONNX or report dependency/compatibility diagnostics."""
    env = check_onnx_environment()
    required = ["onnx", "tf2onnx"]
    missing_req = [pkg for pkg in required if not env["available"].get(pkg, False)]

    if missing_req:
        return {
            "status": "skipped_missing_dependencies",
            "reason": (
                f"Optional ONNX packages missing ({', '.join(missing_req)}). "
                "tf2onnx and onnx are excluded from requirements.txt to preserve clean "
                "Python 3.13 / TensorFlow 2.21 / NumPy 2.x wheel compatibility."
            ),
            "environment": env,
            "exports": [],
        }

    n_features = _infer_n_features(models_dir)
    filename_map = {
        "mlp": "best_mlp.keras",
        "cnn": "best_cnn.keras",
        "autoencoder": "best_autoencoder.keras",
    }

    exports: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for target in targets:
        fname = filename_map.get(target)
        if fname is None:
            errors.append({"model": target, "error": f"Unsupported target '{target}'."})
            continue
        model_path = models_dir / fname
        if not model_path.exists():
            errors.append({"model": target, "error": f"Model checkpoint not found: {model_path}"})
            continue
        try:
            exports.append(
                _export_single_keras_model(
                    model_path=model_path,
                    onnx_path=out_dir / f"{target}_model.onnx",
                    model_key=target,
                    n_features=n_features,
                    opset=opset,
                    verify=verify,
                )
            )
        except Exception as exc:  # pragma: no cover - requires optional onnx stack
            errors.append({"model": target, "error": f"{type(exc).__name__}: {exc}"})

    status = "exported" if exports and not errors else ("partial" if exports else "failed")
    return {
        "status": status,
        "environment": env,
        "exports": exports,
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    cfg = get_app_config()
    default_models_dir = Path(cfg.paths.models_dir)
    parser = argparse.ArgumentParser(description="Export Keras IDS models (MLP/CNN) to ONNX format.")
    parser.add_argument("--models-dir", type=Path, default=default_models_dir)
    parser.add_argument("--out-dir", type=Path, default=default_models_dir / "onnx")
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["mlp", "cnn"],
        choices=["mlp", "cnn", "autoencoder"],
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--verify", action="store_true", help="Verify parity with onnxruntime.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero exit status if ONNX dependencies are missing or export fails.",
    )
    args = parser.parse_args(argv)

    summary = export_models_to_onnx(
        models_dir=args.models_dir,
        out_dir=args.out_dir,
        targets=list(args.targets),
        opset=int(args.opset),
        verify=bool(args.verify),
    )
    print(json.dumps(summary, indent=2))
    if summary["status"] == "exported":
        return 0
    if summary["status"] == "skipped_missing_dependencies" and not args.strict:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
