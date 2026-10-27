"""CIFAR-100 model complexity and inference benchmark (A4-2).

Place in: src/evaluation/cifar100_benchmark.py
Run at project root: python -m src.evaluation.cifar100_benchmark

Outputs: reports/raw/cifar100/benchmark/
    model_complexity.csv   # parameters, serialized weight size, Conv/Linear MACs
    layer_macs.csv         # Conv/Linear MACs by module
    cpu_latency.csv        # batch=1/64 CPU forward-only latency
    gpu_latency.csv        # batch=1/64 GPU forward-only latency
    benchmark_summary.csv  # joined overview
    benchmark_environment.json

Important: Uses synthetic FP32 inputs, excluding data loading/preprocessing and
checkpoint loading from latency measurements. Convolution/Linear-only MACs are
reported; FLOPs = 2*MACs by one particular convention, not a whole-model count.
"""

import csv
import io
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Reuse the same checkpoint names as the existing accuracy evaluation.
# Edit EXPERIMENTS in cifar100_evaluate.py if you rename runs/ folders.
from src.evaluation.cifar100_evaluate import EXPERIMENTS
from src.utils.constants import BASE


REPORTS_ROOT = BASE / "reports" / "raw" / "cifar100" / "benchmark"
INPUT_SHAPE = (3, 32, 32)
BATCH_SIZES = (1, 64)
GPU_WARMUP, GPU_REPEATS = 30, 200
CPU_WARMUP = 5
CPU_REPEATS = {1: 50, 64: 10}  # Keep CPU batch=64 reasonably fast.


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_model(model_class, save_dir: str) -> nn.Module:
    """Load only model weights; never benchmark an optimizer checkpoint."""
    path = BASE / "runs" / save_dir / "best.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Cannot find checkpoint: {path}")
    model = model_class(100)  # CIFAR-100
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.cpu().eval()


def count_parameters(model: nn.Module) -> dict:
    """Number of all and trainable model parameters (including biases)."""
    return {
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def measure_model_size(model: nn.Module) -> dict:
    """Serialized state_dict size; checkpoint optimizer/scheduler not counted."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    state_dict_bytes = buf.getbuffer().nbytes
    # Raw tensors include buffers (e.g. BN running means/vars), and omit zip headers.
    tensor_bytes = sum(t.numel() * t.element_size() for t in model.state_dict().values())
    return {
        "serialized_state_dict_mib": state_dict_bytes / (1024 ** 2),
        "raw_tensor_mib": tensor_bytes / (1024 ** 2),
    }


def measure_conv_linear_macs(model: nn.Module) -> tuple[int, list[dict]]:
    """Return batch=1 MACs and Conv2d/Linear breakdown using forward hooks."""
    entries = []
    handles = []

    def make_hook(name: str):
        def hook(layer: nn.Module, inputs, output):
            if isinstance(layer, nn.Conv2d):
                batch, out_channels, h, w = output.shape
                kh, kw = layer.kernel_size
                macs = batch * out_channels * h * w * (layer.in_channels // layer.groups) * kh * kw
            elif isinstance(layer, nn.Linear):
                # output.numel() counts all output elements, including batched tokens.
                macs = output.numel() * layer.in_features
            else:
                return
            entries.append({"layer": name, "type": type(layer).__name__, "macs": int(macs)})
        return hook

    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            handles.append(layer.register_forward_hook(make_hook(name)))

    try:
        with torch.inference_mode():
            model.eval()
            model(torch.randn(1, *INPUT_SHAPE, device="cpu"))
    finally:
        for handle in handles:
            handle.remove()

    return sum(item["macs"] for item in entries), entries


def describe_times(times_ms: list[float], batch: int) -> dict:
    mean = statistics.mean(times_ms)
    return {
        "batch_size": batch,
        "repeats": len(times_ms),
        "mean_ms": mean,
        "median_ms": statistics.median(times_ms),
        "p95_ms": float(np.percentile(times_ms, 95)),
        "throughput_images_per_sec": 1000 * batch / mean,
    }


def benchmark_cpu(model: nn.Module, batch: int) -> dict:
    """CPU forward only; input allocated before timing."""
    model = model.cpu().eval()
    x = torch.randn(batch, *INPUT_SHAPE, device="cpu", dtype=torch.float32)
    with torch.inference_mode():
        for _ in range(CPU_WARMUP):
            model(x)
        times = []
        for _ in range(CPU_REPEATS[batch]):
            start = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - start) * 1000)
    return describe_times(times, batch)


def benchmark_gpu(model: nn.Module, batch: int) -> dict:
    """CUDA event kernel/device time; excludes CPU dispatch and H2D copy."""
    device = torch.device("cuda:0")
    model = model.to(device).eval()
    x = torch.randn(batch, *INPUT_SHAPE, device=device, dtype=torch.float32)

    with torch.inference_mode():
        for _ in range(GPU_WARMUP):
            model(x)
        torch.cuda.synchronize(device)
        # Includes allocated model parameters + input + peak temporary tensors.
        torch.cuda.reset_peak_memory_stats(device)
        times = []
        for _ in range(GPU_REPEATS):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(x)
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
        peak_mib = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    result = describe_times(times, batch)
    result["peak_allocated_mib"] = peak_mib
    return result


def main():
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    complexity_rows = []
    layer_rows = []
    cpu_rows = []
    gpu_rows = []

    env = {
        "pytorch": torch.__version__,
        "python": platform.python_version(),
        "system": platform.platform(),
        "cpu_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "input_shape_per_image": list(INPUT_SHAPE),
        "precision": "fp32",
        "data": "random synthetic inputs; model forward only",
        "gpu_warmup": GPU_WARMUP,
        "gpu_repeats": GPU_REPEATS,
        "cpu_warmup": CPU_WARMUP,
        "cpu_repeats": CPU_REPEATS,
        "macs_scope": "Conv2d/Linear only; per image; FLOPs=2*MACs convention",
        "gpu_timing_scope": "CUDA event elapsed device time; not end-to-end latency",
    }
    (REPORTS_ROOT / "benchmark_environment.json").write_text(
        json.dumps(env, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    for name, (model_class, save_dir) in EXPERIMENTS.items():
        print(f"\n[{name}] Loading checkpoint: {save_dir}", flush=True)
        model = load_model(model_class, save_dir)
        complexity = count_parameters(model)
        complexity.update(measure_model_size(model))
        macs, layers = measure_conv_linear_macs(model)
        complexity.update({
            "model": name,
            "run_directory": save_dir,
            "macs_per_image": macs,
            "flops_per_image_approx": 2 * macs,
        })
        complexity_rows.append(complexity)
        layer_rows += [{"model": name, **row} for row in layers]
        print(f"  parameters={complexity['parameters']:,} | MACs={macs / 1e6:.2f}M | "
              f"state_dict={complexity['serialized_state_dict_mib']:.2f} MiB")

        for batch in BATCH_SIZES:
            cpu = benchmark_cpu(model, batch)
            cpu_rows.append({"model": name, **cpu})
            print(f"  CPU B={batch} mean={cpu['mean_ms']:.3f} ms, "
                  f"P95={cpu['p95_ms']:.3f} ms", flush=True)

        if torch.cuda.is_available():
            for batch in BATCH_SIZES:
                gpu = benchmark_gpu(model, batch)
                gpu_rows.append({"model": name, **gpu})
                print(f"  GPU B={batch} mean={gpu['mean_ms']:.3f} ms, "
                      f"P95={gpu['p95_ms']:.3f} ms, peak={gpu['peak_allocated_mib']:.1f} MiB", flush=True)
            model.cpu()
            torch.cuda.empty_cache()

        # Save progress even if a later model encounters an error.
        write_csv(REPORTS_ROOT / "model_complexity.csv", complexity_rows,
                  ["model", "run_directory", "parameters", "trainable_parameters",
                   "serialized_state_dict_mib", "raw_tensor_mib", "macs_per_image", "flops_per_image_approx"])
        write_csv(REPORTS_ROOT / "layer_macs.csv", layer_rows, ["model", "layer", "type", "macs"])
        write_csv(REPORTS_ROOT / "cpu_latency.csv", cpu_rows,
                  ["model", "batch_size", "repeats", "mean_ms", "median_ms", "p95_ms", "throughput_images_per_sec"])
        write_csv(REPORTS_ROOT / "gpu_latency.csv", gpu_rows,
                  ["model", "batch_size", "repeats", "mean_ms", "median_ms", "p95_ms",
                   "throughput_images_per_sec", "peak_allocated_mib"])
        del model

    # One row per (model, batch size), convenient for plotting/comparison.
    for_summary = []
    for row in complexity_rows:
        for batch in BATCH_SIZES:
            cpu = next(x for x in cpu_rows if x["model"] == row["model"] and x["batch_size"] == batch)
            gpu = next((x for x in gpu_rows if x["model"] == row["model"] and x["batch_size"] == batch), None)
            for_summary.append({
                "model": row["model"], "batch_size": batch,
                "parameters": row["parameters"],
                "model_size_mib": row["serialized_state_dict_mib"],
                "macs_million_per_image": row["macs_per_image"] / 1e6,
                "cpu_mean_ms": cpu["mean_ms"],
                "cpu_p95_ms": cpu["p95_ms"],
                "gpu_mean_ms": gpu["mean_ms"] if gpu else "",
                "gpu_p95_ms": gpu["p95_ms"] if gpu else "",
                "gpu_peak_allocated_mib": gpu["peak_allocated_mib"] if gpu else "",
            })
    write_csv(REPORTS_ROOT / "benchmark_summary.csv", for_summary,
              ["model", "batch_size", "parameters", "model_size_mib",
               "macs_million_per_image", "cpu_mean_ms", "cpu_p95_ms", "gpu_mean_ms",
               "gpu_p95_ms", "gpu_peak_allocated_mib"])
    print(f"\nDone. Reports -> {REPORTS_ROOT}")


if __name__ == "__main__":
    main()
