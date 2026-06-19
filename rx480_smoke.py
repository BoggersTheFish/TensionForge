from __future__ import annotations

import sys
import numpy as np
import pyopencl as cl


def find_rx480() -> tuple[cl.Platform, cl.Device]:
    for platform in cl.get_platforms():
        if "rusticl" not in platform.name.lower():
            continue

        for device in platform.get_devices():
            is_gpu = bool(device.type & cl.device_type.GPU)

            if is_gpu and "radeon" in device.name.lower():
                return platform, device

    raise RuntimeError("Could not find the RX 480 through Rusticl")


def main() -> int:
    platform, device = find_rx480()

    print(f"Platform:      {platform.name}")
    print(f"Device:        {device.name}")
    print(f"VRAM:          {device.global_mem_size / 1024**3:.2f} GiB")
    print(f"Max allocation:{device.max_mem_alloc_size / 1024**3:.2f} GiB")
    print(f"Compute units: {device.max_compute_units}")

    context = cl.Context([device])
    queue = cl.CommandQueue(
        context,
        properties=cl.command_queue_properties.PROFILING_ENABLE,
    )

    source = r"""
    __kernel void saxpy(
        const float alpha,
        __global const float *a,
        __global const float *b,
        __global float *out,
        const unsigned int n
    ) {
        const unsigned int i = get_global_id(0);

        if (i < n) {
            out[i] = alpha * a[i] + b[i];
        }
    }
    """

    program = cl.Program(context, source).build()
    kernel = cl.Kernel(program, "saxpy")

    count = 16 * 1024 * 1024
    rng = np.random.default_rng(42)

    a = rng.random(count, dtype=np.float32)
    b = rng.random(count, dtype=np.float32)
    result = np.empty_like(a)

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
    result_gpu = cl.Buffer(
        context,
        flags.WRITE_ONLY,
        result.nbytes,
    )

    kernel.set_args(
        np.float32(1.5),
        a_gpu,
        b_gpu,
        result_gpu,
        np.uint32(count),
    )

    local_size = 256
    global_size = (
        ((count + local_size - 1) // local_size) * local_size
    )

    # Warm-up
    event = cl.enqueue_nd_range_kernel(
        queue,
        kernel,
        (global_size,),
        (local_size,),
    )
    event.wait()

    timings = []

    for _ in range(10):
        event = cl.enqueue_nd_range_kernel(
            queue,
            kernel,
            (global_size,),
            (local_size,),
        )
        event.wait()

        seconds = (event.profile.end - event.profile.start) * 1e-9
        timings.append(seconds)

    cl.enqueue_copy(queue, result, result_gpu).wait()

    expected = np.float32(1.5) * a + b
    max_error = float(np.max(np.abs(result - expected)))

    median_seconds = float(np.median(timings))

    # Two input reads and one output write.
    bytes_processed = count * np.dtype(np.float32).itemsize * 3
    bandwidth_gbps = bytes_processed / median_seconds / 1e9

    print()
    print(f"Median kernel time: {median_seconds * 1000:.3f} ms")
    print(f"Approx. bandwidth:  {bandwidth_gbps:.2f} GB/s")
    print(f"Maximum error:      {max_error:.8g}")

    if not np.isfinite(max_error) or max_error > 1e-5:
        print("FAILED: GPU result exceeded tolerance")
        return 1

    print("PASSED: RX 480 produced a verified OpenCL result")
    return 0


if __name__ == "__main__":
    sys.exit(main())
