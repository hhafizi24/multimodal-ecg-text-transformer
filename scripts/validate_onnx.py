"""Validate FP32 and INT8 ONNX artifacts and record validation results."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from src.training.evaluate import metrics_from_logits


def get_io_schema(onnx_path: str) -> dict:
    model = onnx.load(onnx_path)

    def describe(value_infos):
        return [
            {
                "name": v.name,
                "shape": [d.dim_param if d.dim_param else d.dim_value for d in v.type.tensor_type.shape.dim],
            }
            for v in value_infos
        ]

    return {
        "opset_version": model.opset_import[0].version,
        "inputs": describe(model.graph.input),
        "outputs": describe(model.graph.output),
    }


def run_onnx_session(
    onnx_path: str,
    signals: np.ndarray,
    text_embeddings: np.ndarray,
    text_available: np.ndarray,
    batch_size: int = 64,
) -> np.ndarray:
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    all_logits = []

    for i in range(0, len(signals), batch_size):
        outputs = session.run(
            None,
            {
                "signal": signals[i : i + batch_size].astype(np.float32),
                "text_embedding": text_embeddings[i : i + batch_size].astype(np.float32),
                "text_available": text_available[i : i + batch_size].astype(bool),
            },
        )
        all_logits.append(outputs[0])

    return np.concatenate(all_logits, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--fusion-cache-dir", required=True)
    parser.add_argument("--fp32-path", required=True)
    parser.add_argument("--int8-path", required=True)
    parser.add_argument("--f1-tolerance", type=float, default=0.01)
    parser.add_argument("--auc-tolerance", type=float, default=0.01)
    parser.add_argument("--output-path", default="results/onnx/validation.json")
    args = parser.parse_args()

    onnx.checker.check_model(onnx.load(args.fp32_path))
    onnx.checker.check_model(onnx.load(args.int8_path))
    print("Both ONNX graphs passed onnx.checker structural validation.")

    signals = np.load(Path(args.processed_dir) / "signals.npy")
    labels = np.load(Path(args.processed_dir) / "labels.npy")
    text_embeddings = np.load(Path(args.fusion_cache_dir) / "text_embeddings.npy")

    n = len(labels)
    assert len(signals) == n == len(text_embeddings), "Mismatched array lengths."

    dummy_signal = np.random.randn(4, 1000, 12).astype(np.float32)
    dummy_text = np.random.randn(4, text_embeddings.shape[1]).astype(np.float32)
    patterns = {
        "all_present": np.ones(4, dtype=bool),
        "all_absent": np.zeros(4, dtype=bool),
        "mixed": np.array([True, False, True, False]),
    }
    for name, avail in patterns.items():
        logits = run_onnx_session(args.int8_path, dummy_signal, dummy_text, avail)
        assert logits.shape == (4, 5), f"INT8 [{name}]: unexpected shape {logits.shape}"
        assert np.isfinite(logits).all(), f"INT8 [{name}]: contains non-finite values"
    print("INT8 sanity checks passed: finite logits, correct shape, all text-availability patterns.")

    text_available_full = np.ones(n, dtype=bool)
    fp32_logits = run_onnx_session(args.fp32_path, signals, text_embeddings, text_available_full)
    int8_logits = run_onnx_session(args.int8_path, signals, text_embeddings, text_available_full)

    labels_t = torch.from_numpy(labels).long()
    fp32_metrics = metrics_from_logits(torch.from_numpy(fp32_logits).float(), labels_t)
    int8_metrics = metrics_from_logits(torch.from_numpy(int8_logits).float(), labels_t)

    f1_delta = fp32_metrics["macro_f1"] - int8_metrics["macro_f1"]
    auc_delta = fp32_metrics["macro_auc"] - int8_metrics["macro_auc"]

    accepted = (
        f1_delta <= args.f1_tolerance
        and auc_delta <= args.auc_tolerance
    )

    print(f"FP32: macro F1={fp32_metrics['macro_f1']:.4f}, macro AUC={fp32_metrics['macro_auc']:.4f}")
    print(f"INT8: macro F1={int8_metrics['macro_f1']:.4f}, macro AUC={int8_metrics['macro_auc']:.4f}")
    print(f"Delta: F1={f1_delta:+.4f}, AUC={auc_delta:+.4f}")

    if not accepted:
        raise ValueError(
            f"INT8 regression exceeds tolerance (F1 delta {f1_delta:.4f}, AUC delta {auc_delta:.4f})."
        )
    print("INT8 model accepted: within tolerance of FP32.")

    schema = get_io_schema(args.fp32_path)
    summary = {
        "onnx_opset_version": schema["opset_version"],
        "onnx_package_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
        "input_schema": schema["inputs"],
        "output_schema": schema["outputs"],
        "fp32_onnx_checker_passed": True,
        "int8_onnx_checker_passed": True,
        "fp32_macro_f1": fp32_metrics["macro_f1"],
        "fp32_macro_auc": fp32_metrics["macro_auc"],
        "int8_macro_f1": int8_metrics["macro_f1"],
        "int8_macro_auc": int8_metrics["macro_auc"],
        "f1_delta": f1_delta,
        "auc_delta": auc_delta,
        "f1_tolerance": args.f1_tolerance,
        "auc_tolerance": args.auc_tolerance,
        "accepted": accepted,
        "fp32_artifact_size_mb": Path(args.fp32_path).stat().st_size / (1024 * 1024),
        "int8_artifact_size_mb": Path(args.int8_path).stat().st_size / (1024 * 1024),
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved validation summary → {output_path}")


if __name__ == "__main__":
    main()