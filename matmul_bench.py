from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pyopencl as cl


TILE = 16


KERNEL_SOURCE = r"""
#define TILE 16

__kernel void matmul_tiled(
    __global const float *A,
    __global const float *B,
    __global float *C,
    const unsigned int M,
    const unsigned int K,
    const unsigned int N
) {
    const unsigned int row = get_global_id(1);
    const unsigned int col = get_global_id(0);

    const unsigned int local_row = get_local_id(1);
    const unsigned int local_col = get_local_id(0);

    __local float tile_a[TILE][TILE];
    __local float tile_b[TILE][TILE];

    float accumulator = 0.0f;

    const unsigned int tile_count = (K + TILE - 1) / TILE;

    for (unsigned int tile = 0; tile < tile_count; ++tile) {
        const unsigned int a_col = tile * TILE + local_col;
        const unsigned int b_row = tile * TILE + local_row;

        if (row < M && a_col < K) {
            tile_a[local_row][local_col] = A[row * K + a_col];
        } else {
            tile_a[local_row][local_col] = 0.0f;
        }

        if (b_row < K && col < N) {
            tile_b[local_row][local_col] = B[b_row * N + col];
        } else {
            tile_b[local_row][local_col] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned int inner = 0; inner < TILE; ++inner) {
            accumulator += (
                tile_a[local_row][inner]
                * tile_b[inner][local_col]
            );
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < M && col < N) {
        C[row * N + col] = accumulator;
    }
}
"""


def find_rx480() -> tuple[cl.Platform, cl.Device]:
    for platform in cl.get_platforms():
        if "rusticl" not in platform.name.lower():
            continue

        for device in platform.get_devices():
            is_gpu = bool(device.type & cl.device_type.GPU)

            if is_gpu and "radeon" in device.name.lower():
                return platform, device

    raise RuntimeError("Could not find RX 480 through Rusticl")


def rounded_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def benchmark_size(
    context: cl.Context,
    queue: cl.CommandQueue,
    kernel: cl.Kernel,
    size: int,
    rng: np.random.Generator,
) -> dict[str, float | int | bool]:
    print(f"\n=== {size} × {size} ===")

    a = rng.normal(
        0.0,
        0.25,
        size=(size, size),
    ).astype(np.float32)

    b = rng.normal(
        0.0,
        0.25,
        size=(size, size),
    ).astype(np.float32)

    gpu_result = np.empty((size, size), dtype=np.float32)

    flags = cl.mem_flags

    a_gpu = cl.Buffer(
        context,
        flags.READ_ONLY | flags.COPY_HOST_PTR,
        hostbuf=a,
    )
    b_gpu = cl.Buffer(
        context,
        flags.READ_ONLY | flags.COPY_HOST_PTR,
        hostbuf=b,
    )
    c_gpu = cl.Buffer(
        context,
        flags.WRITE_ONLY,
        gpu_result.nbytes,
    )

    kernel.set_args(
        a_gpu,
        b_gpu,
        c_gpu,
        np.uint32(size),
        np.uint32(size),
        np.uint32(size),
    )

    global_size = (
        rounded_up(size, TILE),
        rounded_up(size, TILE),
    )
    local_size = (TILE, TILE)

    # Compilation and cache warm-up.
    for _ in range(3):
        event = cl.enqueue_nd_range_kernel(
            queue,
            kernel,
            global_size,
            local_size,
        )
        event.wait()

    timings = []

    repetitions = 10 if size <= 512 else 5

    for _ in range(repetitions):
        event = cl.enqueue_nd_range_kernel(
            queue,
            kernel,
            global_size,
            local_size,
        )
        event.wait()

        seconds = (event.profile.end - event.profile.start) * 1e-9
        timings.append(seconds)

    cl.enqueue_copy(queue, gpu_result, c_gpu).wait()

    cpu_start = time.perf_counter()
    cpu_result = a @ b
    cpu_seconds = time.perf_counter() - cpu_start

    median_seconds = float(np.median(timings))

    operation_count = 2.0 * size * size * size
    gpu_gflops = operation_count / median_seconds / 1e9
    cpu_gflops = operation_count / cpu_seconds / 1e9

    absolute_error = np.abs(gpu_result - cpu_result)
    max_absolute_error = float(np.max(absolute_error))
    mean_absolute_error = float(np.mean(absolute_error))

    denominator = np.maximum(np.abs(cpu_result), 1e-5)
    max_relative_error = float(
        np.max(absolute_error / denominator)
    )

    accepted = bool(
        np.allclose(
            gpu_result,
            cpu_result,
            rtol=2e-3,
            atol=2e-3,
        )
    )

    print(f"GPU median:       {median_seconds * 1000:.3f} ms")
    print(f"GPU throughput:   {gpu_gflops:.2f} GFLOPS")
    print(f"CPU NumPy:        {cpu_seconds * 1000:.3f} ms")
    print(f"CPU throughput:   {cpu_gflops:.2f} GFLOPS")
    print(f"Maximum abs err:  {max_absolute_error:.8g}")
    print(f"Mean abs err:     {mean_absolute_error:.8g}")
    print(f"Maximum rel err:  {max_relative_error:.8g}")
    print(f"Verified:         {accepted}")

    return {
        "size": size,
        "gpu_median_ms": median_seconds * 1000,
        "gpu_gflops": gpu_gflops,
        "cpu_ms": cpu_seconds * 1000,
        "cpu_gflops": cpu_gflops,
        "max_absolute_error": max_absolute_error,
        "mean_absolute_error": mean_absolute_error,
        "max_relative_error": max_relative_error,
        "accepted": accepted,
    }


def main() -> int:
    platform, device = find_rx480()

    print(f"Platform: {platform.name}")
    print(f"Device:   {device.name}")
    print(f"Driver:   {device.driver_version}")

    context = cl.Context([device])
    queue = cl.CommandQueue(
        context,
        properties=cl.command_queue_properties.PROFILING_ENABLE,
    )

    program = cl.Program(context, KERNEL_SOURCE).build()
    kernel = cl.Kernel(program, "matmul_tiled")

    rng = np.random.default_rng(42)
    results = []

    for size in (128, 256, 512, 1024):
        results.append(
            benchmark_size(
                context=context,
                queue=queue,
                kernel=kernel,
                size=size,
                rng=rng,
            )
        )

    all_accepted = all(
        bool(result["accepted"])
        for result in results
    )

    receipt = {
        "benchmark": "rx480_verified_fp32_matmul",
        "platform": platform.name,
        "device": device.name,
        "driver": device.driver_version,
        "tile_size": TILE,
        "results": results,
        "all_accepted": all_accepted,
    }

    receipt_path = Path("matmul_receipt.json")
    receipt_path.write_text(
        json.dumps(receipt, indent=2),
        encoding="utf-8",
    )

    print(f"\nReceipt written to {receipt_path}")

    if not all_accepted:
        print("FAILED: at least one matrix result exceeded tolerance")
        return 1

    print("PASSED: all RX 480 matrix results were verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
