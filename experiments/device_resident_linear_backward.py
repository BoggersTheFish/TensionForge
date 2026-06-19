from __future__ import annotations

import numpy as np

from tensionforge import (
    DeviceTensor,
    TensionForgeRuntime,
)
from tensionforge.ops import linear_backward_device
from tensionforge.receipts import write_receipt


def main() -> None:
    runtime = TensionForgeRuntime(
        profiling=True,
    )

    rng = np.random.default_rng(42)

    batch_size = 1024
    input_features = 512
    output_features = 512
    repetitions = 20

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

    grad_output = rng.normal(
        size=(
            batch_size,
            output_features,
        ),
    ).astype(np.float32)

    inputs_gpu = DeviceTensor.from_numpy(
        runtime,
        inputs,
    )

    weights_gpu = DeviceTensor.from_numpy(
        runtime,
        weights,
    )

    grad_output_gpu = DeviceTensor.from_numpy(
        runtime,
        grad_output,
    )

    grad_input_gpu = DeviceTensor.empty(
        runtime,
        inputs.shape,
        dtype=np.float32,
    )

    grad_weights_gpu = DeviceTensor.empty(
        runtime,
        weights.shape,
        dtype=np.float32,
    )

    grad_bias_gpu = DeviceTensor.empty(
        runtime,
        (output_features,),
        dtype=np.float32,
    )

    (
        grad_input_gpu,
        grad_weights_gpu,
        grad_bias_gpu,
        metadata,
    ) = linear_backward_device(
        runtime,
        inputs_gpu,
        weights_gpu,
        grad_output_gpu,
        grad_input=grad_input_gpu,
        grad_weights=grad_weights_gpu,
        grad_bias=grad_bias_gpu,
        repetitions=repetitions,
    )

    actual_grad_input = (
        grad_input_gpu.to_numpy()
    )

    actual_grad_weights = (
        grad_weights_gpu.to_numpy()
    )

    actual_grad_bias = (
        grad_bias_gpu.to_numpy()
    )

    expected_grad_input = (
        grad_output @ weights.T
    ).astype(np.float32)

    expected_grad_weights = (
        inputs.T @ grad_output
    ).astype(np.float32)

    expected_grad_bias = grad_output.sum(
        axis=0,
        dtype=np.float32,
    )

    grad_input_error = float(
        np.max(
            np.abs(
                actual_grad_input
                - expected_grad_input
            )
        )
    )

    grad_weights_error = float(
        np.max(
            np.abs(
                actual_grad_weights
                - expected_grad_weights
            )
        )
    )

    grad_bias_error = float(
        np.max(
            np.abs(
                actual_grad_bias
                - expected_grad_bias
            )
        )
    )

    grad_input_verified = bool(
        np.allclose(
            actual_grad_input,
            expected_grad_input,
            rtol=3e-4,
            atol=3e-3,
        )
    )

    grad_weights_verified = bool(
        np.allclose(
            actual_grad_weights,
            expected_grad_weights,
            rtol=3e-4,
            atol=3e-3,
        )
    )

    grad_bias_verified = bool(
        np.allclose(
            actual_grad_bias,
            expected_grad_bias,
            rtol=3e-4,
            atol=3e-3,
        )
    )

    all_verified = (
        grad_input_verified
        and grad_weights_verified
        and grad_bias_verified
    )

    print(
        "=== DEVICE-RESIDENT LINEAR BACKWARD ==="
    )
    print(
        f"Device:       "
        f"{runtime.info.device_name}"
    )
    print(
        f"Batch:        {batch_size}"
    )
    print(
        f"Input:        {input_features}"
    )
    print(
        f"Output:       {output_features}"
    )
    print(
        f"Repetitions:  {repetitions}"
    )
    print()

    print(
        f"grad input:   "
        f"{metadata['grad_input_median_ms']:.3f} ms"
    )

    print(
        f"grad weights: "
        f"{metadata['grad_weights_median_ms']:.3f} ms"
    )

    print(
        f"grad bias:    "
        f"{metadata['grad_bias_median_ms']:.3f} ms"
    )

    print(
        f"combined:     "
        f"{metadata['combined_median_ms']:.3f} ms"
    )

    print(
        f"throughput:   "
        f"{metadata['approximate_matmul_gflops']:.2f} GFLOPS"
    )

    print(
        f"input error:  "
        f"{grad_input_error:.7g}"
    )

    print(
        f"weight error: "
        f"{grad_weights_error:.7g}"
    )

    print(
        f"bias error:   "
        f"{grad_bias_error:.7g}"
    )

    print(
        f"input valid:  "
        f"{grad_input_verified}"
    )

    print(
        f"weight valid: "
        f"{grad_weights_verified}"
    )

    print(
        f"bias valid:   "
        f"{grad_bias_verified}"
    )

    payload = {
        "milestone":
            "device_resident_linear_backward",
        "device": runtime.info.to_dict(),
        "configuration": {
            "batch_size": batch_size,
            "input_features": input_features,
            "output_features":
                output_features,
            "repetitions": repetitions,
            "precision": "float32",
        },
        "backward": {
            **metadata,
            "grad_input_max_error":
                grad_input_error,
            "grad_weights_max_error":
                grad_weights_error,
            "grad_bias_max_error":
                grad_bias_error,
            "grad_input_verified":
                grad_input_verified,
            "grad_weights_verified":
                grad_weights_verified,
            "grad_bias_verified":
                grad_bias_verified,
        },
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
        "device_resident_linear_backward_receipt.json",
        payload,
    )

    print()
    print(f"Receipt: {receipt_path}")

    if not all_verified:
        raise RuntimeError(
            "Device-resident linear backward "
            "verification failed"
        )

    print()
    print(
        "PASSED: input, weight, and bias "
        "gradients were calculated entirely "
        "in RX 480 VRAM."
    )


if __name__ == "__main__":
    main()
