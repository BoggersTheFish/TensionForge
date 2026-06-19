from __future__ import annotations

import time

import numpy as np

from tensionforge import (
    DeviceTensor,
    TensionForgeRuntime,
)
from tensionforge.ops import (
    linear,
    linear_device,
)
from tensionforge.receipts import write_receipt


def main() -> None:
    runtime = TensionForgeRuntime(
        profiling=True,
    )

    rng = np.random.default_rng(42)

    batch_size = 1024
    input_features = 512
    output_features = 512
    repetitions = 100

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

    inputs_gpu = DeviceTensor.from_numpy(
        runtime,
        inputs,
    )

    weights_gpu = DeviceTensor.from_numpy(
        runtime,
        weights,
    )

    bias_gpu = DeviceTensor.from_numpy(
        runtime,
        bias,
    )

    output_gpu = DeviceTensor.empty(
        runtime,
        (
            batch_size,
            output_features,
        ),
        dtype=np.float32,
    )

    resident_started = time.perf_counter()

    output_gpu, resident_metadata = (
        linear_device(
            runtime,
            inputs_gpu,
            weights_gpu,
            bias_gpu,
            output=output_gpu,
            repetitions=repetitions,
        )
    )

    runtime.finish()

    resident_wall_seconds = (
        time.perf_counter()
        - resident_started
    )

    resident_output = output_gpu.to_numpy()

    host_started = time.perf_counter()

    host_output, host_metadata = linear(
        runtime,
        inputs,
        weights,
        bias,
        repetitions=1,
    )

    host_wall_seconds = (
        time.perf_counter()
        - host_started
    )

    expected = inputs @ weights + bias

    resident_error = float(
        np.max(
            np.abs(
                resident_output
                - expected
            )
        )
    )

    host_error = float(
        np.max(
            np.abs(
                host_output
                - expected
            )
        )
    )

    resident_verified = bool(
        np.allclose(
            resident_output,
            expected,
            rtol=2e-4,
            atol=2e-3,
        )
    )

    host_verified = bool(
        np.allclose(
            host_output,
            expected,
            rtol=2e-4,
            atol=2e-3,
        )
    )

    average_resident_wall_ms = (
        resident_wall_seconds
        * 1000.0
        / repetitions
    )

    print(
        "=== DEVICE-RESIDENT LINEAR ==="
    )
    print(
        f"Device: {runtime.info.device_name}"
    )
    print(
        f"Shape:  "
        f"{batch_size} x "
        f"{input_features} x "
        f"{output_features}"
    )
    print(
        f"Resident repetitions: "
        f"{repetitions}"
    )
    print()
    print(
        f"Resident median kernel: "
        f"{resident_metadata['median_kernel_ms']:.3f} ms"
    )
    print(
        f"Resident throughput:    "
        f"{resident_metadata['gflops']:.2f} GFLOPS"
    )
    print(
        f"Resident wall average:  "
        f"{average_resident_wall_ms:.3f} ms"
    )
    print(
        f"Host-wrapper wall time: "
        f"{host_wall_seconds * 1000.0:.3f} ms"
    )
    print(
        f"Resident max error:     "
        f"{resident_error:.7g}"
    )
    print(
        f"Host max error:         "
        f"{host_error:.7g}"
    )
    print(
        f"Resident verified:      "
        f"{resident_verified}"
    )
    print(
        f"Host verified:          "
        f"{host_verified}"
    )

    payload = {
        "milestone":
            "device_resident_tensor_and_linear",
        "device": runtime.info.to_dict(),
        "configuration": {
            "batch_size": batch_size,
            "input_features": input_features,
            "output_features":
                output_features,
            "resident_repetitions":
                repetitions,
            "precision": "float32",
        },
        "resident_execution": {
            **resident_metadata,
            "wall_seconds":
                resident_wall_seconds,
            "average_wall_ms":
                average_resident_wall_ms,
            "maximum_absolute_error":
                resident_error,
            "verified":
                resident_verified,
        },
        "host_wrapper_execution": {
            **host_metadata,
            "wall_seconds":
                host_wall_seconds,
            "maximum_absolute_error":
                host_error,
            "verified":
                host_verified,
        },
        "runtime": {
            "program_cache_entries":
                runtime.program_cache_size,
            "kernel_cache_entries":
                runtime.kernel_cache_size,
        },
        "all_verified": (
            resident_verified
            and host_verified
        ),
    }

    receipt_path = write_receipt(
        "receipts/"
        "device_resident_linear_receipt.json",
        payload,
    )

    print(f"Receipt: {receipt_path}")

    if not (
        resident_verified
        and host_verified
    ):
        raise RuntimeError(
            "Device-resident linear "
            "verification failed"
        )

    print()
    print(
        "PASSED: tensors and output buffers "
        "remained resident in RX 480 VRAM."
    )


if __name__ == "__main__":
    main()
