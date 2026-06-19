from __future__ import annotations

import numpy as np

from tensionforge import TensionForgeRuntime
from tensionforge.ops import linear
from tensionforge.receipts import write_receipt


CONFIGURATIONS = (
    (256, 256, 256),
    (512, 512, 256),
    (1024, 512, 512),
)


def main() -> None:
    runtime = TensionForgeRuntime(
        profiling=True,
    )

    rng = np.random.default_rng(42)

    results: list[dict[str, object]] = []
    all_verified = True

    print("=== TENSIONFORGE FUSED LINEAR ===")
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

    for (
        batch_size,
        input_features,
        output_features,
    ) in CONFIGURATIONS:
        inputs = rng.normal(
            size=(
                batch_size,
                input_features,
            ),
        ).astype(np.float32)

        weights = rng.normal(
            size=(
                input_features,
                output_features,
            ),
        ).astype(np.float32)

        bias = rng.normal(
            size=(output_features,),
        ).astype(np.float32)

        output, metadata = linear(
            runtime,
            inputs,
            weights,
            bias,
            repetitions=5,
            tile_size=16,
        )

        expected = inputs @ weights + bias

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

        results.append(
            {
                **metadata,
                "maximum_absolute_error":
                    maximum_error,
                "verified": verified,
            }
        )

        print(
            f"batch {batch_size:4d} | "
            f"in {input_features:4d} | "
            f"out {output_features:4d} | "
            f"{metadata['median_kernel_ms']:8.3f} ms | "
            f"{metadata['gflops']:8.2f} GFLOPS | "
            f"error {maximum_error:.7g} | "
            f"verified {verified}"
        )

    payload = {
        "milestone":
            "reusable_fused_linear_forward",
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
        "runtime_linear_receipt.json",
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
            "Reusable fused linear verification "
            "failed"
        )

    print()
    print(
        "PASSED: fused FP32 linear forward "
        "matched NumPy."
    )


if __name__ == "__main__":
    main()
