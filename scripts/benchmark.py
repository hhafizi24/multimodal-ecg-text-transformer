"""
Benchmark inference latency, memory, and artifact size across model runtimes.
"""

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import psutil

from configs.config import BenchmarkConfig


def load_sample(processed_dir: str, fusion_cache_dir: str, index: int = 0) -> tuple:
    signals = np.load(Path(processed_dir) / "signals.npy", mmap_mode="r")
    text_embeddings = np.load(Path(fusion_cache_dir) / "text_embeddings.npy", mmap_mode="r")

    signal = np.array(
        signals[index : index + 1],
        dtype=np.float32,
        copy=True,
    )
    text_embedding = np.array(
        text_embeddings[index : index + 1],
        dtype=np.float32,
        copy=True,
    )
    text_available = np.ones(1, dtype=bool)
    return signal, text_embedding, text_available


def summarize_latencies(times_ms: list) -> dict:
    arr = np.array(times_ms)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
    }


def benchmark_pytorch(checkpoint_path, signal, text_embedding, text_available, warmup_runs, timed_runs, num_threads) -> dict:
    import torch

    from scripts.export_onnx import load_exportable_model

    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(1)

    model, _ = load_exportable_model(checkpoint_path)
    model.eval()

    signal_t = torch.from_numpy(signal)
    text_t = torch.from_numpy(text_embedding)
    avail_t = torch.from_numpy(text_available)

    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / "exportable_state_dict.pt"
        torch.save(model.state_dict(), state_path)
        artifact_size_mb = state_path.stat().st_size / (1024 * 1024)

    with torch.inference_mode():
        for _ in range(warmup_runs):
            model(signal_t, text_t, avail_t)

        times_ms = []
        for _ in range(timed_runs):
            start = time.perf_counter()
            model(signal_t, text_t, avail_t)
            times_ms.append((time.perf_counter() - start) * 1000)

    return {
        "artifact_size_mb": artifact_size_mb,
        "latency": summarize_latencies(times_ms),
    }


def benchmark_onnx(onnx_path, signal, text_embedding, text_available, warmup_runs, timed_runs, num_threads) -> dict:
    import onnxruntime as ort

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = num_threads
    session_options.inter_op_num_threads = 1
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    session = ort.InferenceSession(onnx_path, sess_options=session_options, providers=["CPUExecutionProvider"])
    inputs = {"signal": signal, "text_embedding": text_embedding, "text_available": text_available}

    for _ in range(warmup_runs):
        session.run(None, inputs)

    times_ms = []
    for _ in range(timed_runs):
        start = time.perf_counter()
        session.run(None, inputs)
        times_ms.append((time.perf_counter() - start) * 1000)

    return {
        "artifact_size_mb": Path(onnx_path).stat().st_size / (1024 * 1024),
        "latency": summarize_latencies(times_ms),
    }


def get_environment_info() -> dict:
    return {
        "hardware": platform.processor() or platform.machine(),
        "os": f"{platform.system()} {platform.release()}",
        "python_version": platform.python_version(),
        "torch_version": importlib.metadata.version("torch"),
        "onnx_version": importlib.metadata.version("onnx"),
        "onnxruntime_version": importlib.metadata.version("onnxruntime"),
    }


def build_command(backend: str, args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable, "-m", "scripts.benchmark",
        "--backend", backend,
        "--processed-dir", args.processed_dir,
        "--fusion-cache-dir", args.fusion_cache_dir,
        "--warmup-runs", str(args.warmup_runs),
        "--timed-runs", str(args.timed_runs),
        "--num-threads", str(args.num_threads),
    ]
    if backend == "pytorch":
        cmd += ["--checkpoint-path", args.checkpoint_path]
    elif backend == "onnx_fp32":
        cmd += ["--fp32-path", args.fp32_path]
    else:
        cmd += ["--int8-path", args.int8_path]
    return cmd


def run_backend_with_peak_rss(cmd: list[str], poll_interval: float = 0.01) -> dict:
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    ps_process = psutil.Process(process.pid)
    peak_rss_mb = 0.0

    while process.poll() is None:
        try:
            rss_mb = ps_process.memory_info().rss / (1024 * 1024)
            peak_rss_mb = max(peak_rss_mb, rss_mb)
        except psutil.NoSuchProcess:
            break
        time.sleep(poll_interval)

    stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"Backend command failed: {' '.join(cmd)}\n{stderr}")

    result = json.loads(stdout.strip().splitlines()[-1])
    result["peak_rss_mb"] = peak_rss_mb
    return result


def plot_benchmark_comparison(results: dict, output_path: str) -> None:
    import matplotlib.pyplot as plt

    backends = list(results.keys())
    display_names = {
        "pytorch": "PyTorch",
        "onnx_fp32": "ONNX FP32",
        "onnx_int8": "ONNX INT8",
    }
    labels = [display_names[b] for b in backends]

    p50 = [results[b]["latency"]["p50_ms"] for b in backends]
    p95 = [results[b]["latency"]["p95_ms"] for b in backends]
    peak_rss = [results[b]["peak_rss_mb"] for b in backends]
    artifact_size = [results[b]["artifact_size_mb"] for b in backends]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    x = list(range(len(backends)))
    width = 0.35

    p50_bars = axes[0].bar(
        [i - width / 2 for i in x],
        p50,
        width,
        label="p50",
    )
    p95_bars = axes[0].bar(
        [i + width / 2 for i in x],
        p95,
        width,
        label="p95",
    )
    axes[0].bar_label(p50_bars, fmt="%.2f", padding=3)
    axes[0].bar_label(p95_bars, fmt="%.2f", padding=3)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Latency (ms)")
    axes[0].set_title("Latency")
    axes[0].set_ylim(0, max(p50 + p95) * 1.15)
    axes[0].legend()

    memory_bars = axes[1].bar(x, peak_rss)
    axes[1].bar_label(memory_bars, fmt="%.1f", padding=3)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Peak RSS (MB)")
    axes[1].set_title("Peak Memory")
    axes[1].set_ylim(0, max(peak_rss) * 1.12)

    size_bars = axes[2].bar(x, artifact_size)
    axes[2].bar_label(size_bars, fmt="%.2f", padding=3)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)
    axes[2].set_ylabel("Size (MB)")
    axes[2].set_title("Artifact Size")
    axes[2].set_ylim(0, max(artifact_size) * 1.12)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_suite(args: argparse.Namespace) -> None:
    results = {}
    for backend in ("pytorch", "onnx_fp32", "onnx_int8"):
        print(f"Running {backend}...")
        results[backend] = run_backend_with_peak_rss(build_command(backend, args))

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "scope": "fusion_exported_inference_subgraph",
        "protocol": {
            "sample_split": "validation",
            "batch_size": 1,
            "warmup_runs": args.warmup_runs,
            "timed_runs": args.timed_runs,
            "num_threads": args.num_threads,
        },
        "model_backends": results,
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved benchmark results → {output_path}")

    plot_benchmark_comparison(results, args.figure_path)
    print(f"Saved comparison figure → {args.figure_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["pytorch", "onnx_fp32", "onnx_int8"])
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--fusion-cache-dir", required=True)
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--fp32-path")
    parser.add_argument("--int8-path")
    parser.add_argument("--warmup-runs", type=int, default=BenchmarkConfig.n_warmup_runs)
    parser.add_argument("--timed-runs", type=int, default=BenchmarkConfig.n_benchmark_runs)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--output-path", default="results/benchmark.json")
    parser.add_argument("--figure-path", default="results/figures/benchmark_comparison.png")
    args = parser.parse_args()

    if args.backend is None:
        run_suite(args)
        return

    signal, text_embedding, text_available = load_sample(args.processed_dir, args.fusion_cache_dir)

    if args.backend == "pytorch":
        result = benchmark_pytorch(
            args.checkpoint_path, signal, text_embedding, text_available,
            args.warmup_runs, args.timed_runs, args.num_threads,
        )
    elif args.backend == "onnx_fp32":
        result = benchmark_onnx(
            args.fp32_path, signal, text_embedding, text_available,
            args.warmup_runs, args.timed_runs, args.num_threads,
        )
    else:
        result = benchmark_onnx(
            args.int8_path, signal, text_embedding, text_available,
            args.warmup_runs, args.timed_runs, args.num_threads,
        )

    result["num_threads"] = args.num_threads
    result["environment"] = get_environment_info()
    print(json.dumps(result))


if __name__ == "__main__":
    main()