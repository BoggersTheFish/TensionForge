from __future__ import annotations

import numpy as np

from tensionforge import (
    DeviceTensor,
    TensionForgeRuntime,
)
from tensionforge.ops import (
    sigmoid_device,
    tanh_device,
    tension_update_device,
)
from tensionforge.receipts import write_receipt


def main() -> None:
    runtime = TensionForgeRuntime(
        profiling=True,
    )

    rng = np.random.default_rng(42)

    element_count = 8 * 1024 * 1024
    shape = (element_count,)
    repetitions = 20

    values = rng.uniform(
        -6.0,
        6.0,
        size=shape,
    ).astype(np.float32)

    state = rng.normal(
        size=shape,
    ).astype(np.float32)

    proposal = rng.normal(
        size=shape,
    ).astype(np.float32)

    gate = rng.uniform(
        0.0,
        1.0,
        size=shape,
    ).astype(np.float32)

    values_gpu = DeviceTensor.from_numpy(
        runtime,
        values,
    )

    state_gpu = DeviceTensor.from_numpy(
        runtime,
        state,
    )

    proposal_gpu = DeviceTensor.from_numpy(
        runtime,
        proposal,
    )

    gate_gpu = DeviceTensor.from_numpy(
        runtime,
        gate,
    )

    tanh_output = DeviceTensor.empty(
        runtime,
        shape,
        dtype=np.float32,
    )

    sigmoid_output = DeviceTensor.empty(
        runtime,
        shape,
        dtype=np.float32,
    )

    tension_output = DeviceTensor.empty(
        runtime,
        shape,
        dtype=np.float32,
    )

    tanh_output, tanh_metadata = tanh_device(
        runtime,
        values_gpu,
        output=tanh_output,
        repetitions=repetitions,
    )

    sigmoid_output, sigmoid_metadata = (
        sigmoid_device(
            runtime,
            values_gpu,
            output=sigmoid_output,
            repetitions=repetitions,
        )
    )

    tension_output, tension_metadata = (
        tension_update_device(
            runtime,
            state_gpu,
            proposal_gpu,
            gate_gpu,
            output=tension_output,
            repetitions=repetitions,
        )
    )

    tanh_result = tanh_output.to_numpy()
    sigmoid_result = sigmoid_output.to_numpy()
    tension_result = tension_output.to_numpy()

    tanh_expected = np.tanh(
        values
    ).astype(np.float32)

    sigmoid_expected = (
        1.0
        / (
            1.0
            + np.exp(-values)
        )
    ).astype(np.float32)

    tension_expected = (
        state
        + gate
        * (
            proposal
            - state
        )
    ).astype(np.float32)

    tanh_error = float(
        np.max(
            np.abs(
                tanh_result
                - tanh_expected
            )
        )
    )

    sigmoid_error = float(
        np.max(
            np.abs(
                sigmoid_result
                - sigmoid_expected
            )
        )
    )

    tension_error = float(
        np.max(
            np.abs(
                tension_result
                - tension_expected
            )
        )
    )

    tanh_verified = bool(
        np.allclose(
            tanh_result,
            tanh_expected,
            rtol=2e-6,
            atol=2e-6,
        )
    )

    sigmoid_verified = bool(
        np.allclose(
            sigmoid_result,
            sigmoid_expected,
            rtol=2e-6,
            atol=2e-6,
        )
    )

    tension_verified = bool(
        np.allclose(
            tension_result,
            tension_expected,
            rtol=2e-6,
            atol=2e-6,
        )
    )

    all_verified = (
        tanh_verified
        and sigmoid_verified
        and tension_verified
    )

    print(
        "=== DEVICE-RESIDENT TENSION OPERATIONS ==="
    )
    print(
        f"Device:       "
        f"{runtime.info.device_name}"
    )
    print(
        f"Elements:     {element_count}"
    )
    print(
        f"Repetitions:  {repetitions}"
    )
    print()

    print(
        f"tanh     | "
        f"{tanh_metadata['median_kernel_ms']:.3f} ms | "
        f"{tanh_metadata['approximate_bandwidth_gbps']:.2f} GB/s | "
        f"error {tanh_error:.7g} | "
        f"verified {tanh_verified}"
    )

    print(
        f"sigmoid  | "
        f"{sigmoid_metadata['median_kernel_ms']:.3f} ms | "
        f"{sigmoid_metadata['approximate_bandwidth_gbps']:.2f} GB/s | "
        f"error {sigmoid_error:.7g} | "
        f"verified {sigmoid_verified}"
    )

    print(
        f"tension  | "
        f"{tension_metadata['median_kernel_ms']:.3f} ms | "
        f"{tension_metadata['approximate_bandwidth_gbps']:.2f} GB/s | "
        f"error {tension_error:.7g} | "
        f"verified {tension_verified}"
    )

    payload = {
        "milestone":
            "device_resident_tension_operations",
        "device": runtime.info.to_dict(),
        "configuration": {
            "element_count": element_count,
            "repetitions": repetitions,
            "precision": "float32",
        },
        "operations": {
            "tanh": {
                **tanh_metadata,
                "maximum_absolute_error":
                    tanh_error,
                "verified":
                    tanh_verified,
            },
            "sigmoid": {
                **sigmoid_metadata,
                "maximum_absolute_error":
                    sigmoid_error,
                "verified":
                    sigmoid_verified,
            },
            "tension_update": {
                **tension_metadata,
                "maximum_absolute_error":
                    tension_error,
                "verified":
                    tension_verified,
            },
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
        "device_resident_tension_ops_receipt.json",
        payload,
    )

    print()
    print(f"Receipt: {receipt_path}")

    if not all_verified:
        raise RuntimeError(
            "Device-resident tension operation "
            "verification failed"
        )

    print()
    print(
        "PASSED: reusable activations and causal "
        "tension updates remained in RX 480 VRAM."
    )


if __name__ == "__main__":
    main()
