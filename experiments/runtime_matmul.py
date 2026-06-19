from __future__ import annotations

import numpy as np

from tensionforge import TensionForgeRuntime
from tensionforge.ops import matmul
from tensionforge.receipts import write_receipt


MATRIX_SIZES = (
    128,
    256,
    512,
    1024,
)


def main() -> None:
    runtime = TensionForgeRuntime(
        profiling=True,
    )

    rng = np.random.default_rng(42)

    results: list[dict[str, object]] = []

    print("=== TENSIONFORGE TILED MATMUL ===")
    print(
        f"Platform: {runtime.info.platform_name}"
    )
    print(
        f"Device:   {runtime.info.device_name}"
    )
    print(
        f"Driver:   {runtime.info.driver_version}"
    )
    print()

    all_verified = True

    for size in MATRIX_SIZES:
        a = rng.normal(
            size=(size, size),
        ).astype(np.float32)

        b = rng.normal(
            size=(size, size),
        ).astype(np.float32)

        output, metadata = matmul(
            runtime,
            a,
            b,
            repetitions=5,
            tile_size=16,
        )

        expected = a @ b

        maximum_error = float(
            np.max(
                np.abs(
                    output - expected
                )
            )
        )

        verified = bool(
            np.allclose(
                output,
                expected,
                rtol=2e-4,
                atol=2e-3,
            )
        )

        all_verified = (
            all_verified and verified
        )

        result = {
            **metadata,
            "maximum_absolute_error":
                maximum_error,
            "verified": verified,
        }

        results.append(result)

        print(
            f"{size:4d} x {size:4d} | "
            f"{metadata['median_kernel_ms']:8.3f} ms | "
            f"{metadata['gflops']:8.2f} GFLOPS | "
            f"error {maximum_error:.7g} | "
            f"verified {verified}"
        )

    payload = {
        "milestone":
            "reusable_runtime_tiled_matmul",
        "device": runtime.info.to_dict(),
        "configuration": {
            "precision": "float32",
            "tile_size": 16,
            "repetitions": 5,
        },
        "results": results,
        "runtime": {
            "program_cache_entries":
                runtime.program_cache_size,
            "kernel_cache_entries":
                runtime.kernel_cache_size,
        },
        "all_verified": all_verified,
    }

    receipt_path = write_receipt(
        "receipts/"
        "runtime_matmul_receipt.json",
        payload,
    )

    print()
    print(
        f"Program cache: "
        f"{runtime.program_cache_size}"
    )
    print(
        f"Kernel cache:  "
        f"{runtime.kernel_cache_size}"
    )
    print(f"Receipt:       {receipt_path}")

    if not all_verified:
        raise RuntimeError(
            "Reusable matrix multiplication "
            "verification failed"
        )

    print()
    print(
        "PASSED: reusable tiled FP32 matrix "
        "multiplication matched NumPy."
    )


if __name__ == "__main__":
    main()
