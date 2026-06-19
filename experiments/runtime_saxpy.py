from __future__ import annotations

import numpy as np

from tensionforge import TensionForgeRuntime
from tensionforge.ops import saxpy
from tensionforge.receipts import write_receipt


def run_experiment(
    *,
    count: int = 16 * 1024 * 1024,
    repetitions: int = 10,
    write_result: bool = True,
) -> dict[str, object]:
    runtime = TensionForgeRuntime(
        profiling=True,
    )

    rng = np.random.default_rng(42)

    a = rng.random(
        count,
        dtype=np.float32,
    )

    b = rng.random(
        count,
        dtype=np.float32,
    )

    result, operation = saxpy(
        runtime,
        alpha=1.5,
        a=a,
        b=b,
        repetitions=repetitions,
    )

    expected = (
        np.float32(1.5) * a + b
    )

    maximum_error = float(
        np.max(
            np.abs(
                result - expected
            )
        )
    )

    verified = bool(
        np.allclose(
            result,
            expected,
            rtol=1e-6,
            atol=1e-6,
        )
    )

    payload: dict[str, object] = {
        "milestone":
            "reusable_runtime_saxpy",
        "device": runtime.info.to_dict(),
        "operation": operation,
        "verification": {
            "maximum_absolute_error":
                maximum_error,
            "verified": verified,
        },
        "runtime": {
            "program_cache_entries":
                runtime.program_cache_size,
            "kernel_cache_entries":
                runtime.kernel_cache_size,
        },
    }

    receipt_path = None

    if write_result:
        receipt_path = write_receipt(
            "receipts/"
            "runtime_saxpy_receipt.json",
            payload,
        )

    print("=== TENSIONFORGE RUNTIME ===")
    print(
        f"Platform: {runtime.info.platform_name}"
    )
    print(
        f"Device:   {runtime.info.device_name}"
    )
    print(
        f"Driver:   {runtime.info.driver_version}"
    )
    print(
        f"VRAM:     "
        f"{runtime.info.global_memory_bytes / 1024**3:.2f} GiB"
    )
    print()

    print("=== REUSABLE SAXPY OPERATION ===")
    print(
        f"Elements:          "
        f"{operation['element_count']}"
    )
    print(
        f"Median kernel:     "
        f"{operation['median_kernel_ms']:.3f} ms"
    )
    print(
        f"Approx bandwidth:  "
        f"{operation['approximate_bandwidth_gbps']:.2f} GB/s"
    )
    print(
        f"Maximum error:     "
        f"{maximum_error:.8g}"
    )
    print(
        f"Program cache:     "
        f"{runtime.program_cache_size}"
    )
    print(
        f"Kernel cache:      "
        f"{runtime.kernel_cache_size}"
    )
    print(
        f"Verified:          {verified}"
    )

    if receipt_path is not None:
        print(f"Receipt:           {receipt_path}")

    if not verified:
        raise RuntimeError(
            "Reusable SAXPY verification failed"
        )

    print()
    print(
        "PASSED: reusable TensionForge runtime "
        "executed a verified RX 480 kernel."
    )

    return payload


if __name__ == "__main__":
    run_experiment()
