from __future__ import annotations

import numpy as np

from tensionforge import (
    DeviceTensor,
    TensionForgeRuntime,
)
from tensionforge.ops import (
    sigmoid_backward_device,
    sigmoid_device,
    tanh_backward_device,
    tanh_device,
    tension_update_backward_device,
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

    grad_output = rng.normal(
        size=shape,
    ).astype(np.float32)

    values_gpu = DeviceTensor.from_numpy(
        runtime,
        values,
    )

    grad_output_gpu = DeviceTensor.from_numpy(
        runtime,
        grad_output,
    )

    tanh_output_gpu, _ = tanh_device(
        runtime,
        values_gpu,
    )

    sigmoid_output_gpu, _ = sigmoid_device(
        runtime,
        values_gpu,
    )

    tanh_grad_gpu, tanh_metadata = (
        tanh_backward_device(
            runtime,
            tanh_output_gpu,
            grad_output_gpu,
            repetitions=repetitions,
        )
    )

    sigmoid_grad_gpu, sigmoid_metadata = (
        sigmoid_backward_device(
            runtime,
            sigmoid_output_gpu,
            grad_output_gpu,
            repetitions=repetitions,
        )
    )

    (
        grad_state_gpu,
        grad_proposal_gpu,
        grad_gate_gpu,
        tension_metadata,
    ) = tension_update_backward_device(
        runtime,
        DeviceTensor.from_numpy(
            runtime,
            state,
        ),
        DeviceTensor.from_numpy(
            runtime,
            proposal,
        ),
        DeviceTensor.from_numpy(
            runtime,
            gate,
        ),
        grad_output_gpu,
        repetitions=repetitions,
    )

    tanh_output = np.tanh(
        values
    ).astype(np.float32)

    sigmoid_output = (
        1.0
        / (
            1.0
            + np.exp(-values)
        )
    ).astype(np.float32)

    expected_tanh_grad = (
        grad_output
        * (
            1.0
            - tanh_output
            * tanh_output
        )
    ).astype(np.float32)

    expected_sigmoid_grad = (
        grad_output
        * sigmoid_output
        * (
            1.0
            - sigmoid_output
        )
    ).astype(np.float32)

    expected_grad_state = (
        grad_output
        * (
            1.0
            - gate
        )
    ).astype(np.float32)

    expected_grad_proposal = (
        grad_output * gate
    ).astype(np.float32)

    expected_grad_gate = (
        grad_output
        * (
            proposal
            - state
        )
    ).astype(np.float32)

    errors = {
        "tanh": float(
            np.max(
                np.abs(
                    tanh_grad_gpu.to_numpy()
                    - expected_tanh_grad
                )
            )
        ),
        "sigmoid": float(
            np.max(
                np.abs(
                    sigmoid_grad_gpu.to_numpy()
                    - expected_sigmoid_grad
                )
            )
        ),
        "grad_state": float(
            np.max(
                np.abs(
                    grad_state_gpu.to_numpy()
                    - expected_grad_state
                )
            )
        ),
        "grad_proposal": float(
            np.max(
                np.abs(
                    grad_proposal_gpu.to_numpy()
                    - expected_grad_proposal
                )
            )
        ),
        "grad_gate": float(
            np.max(
                np.abs(
                    grad_gate_gpu.to_numpy()
                    - expected_grad_gate
                )
            )
        ),
    }

    verified = {
        "tanh": errors["tanh"] < 3e-6,
        "sigmoid": errors["sigmoid"] < 3e-6,
        "grad_state": errors["grad_state"] < 2e-6,
        "grad_proposal":
            errors["grad_proposal"] < 2e-6,
        "grad_gate": errors["grad_gate"] < 2e-6,
    }

    all_verified = all(
        verified.values()
    )

    print(
        "=== RECURRENT BACKWARD OPERATIONS ==="
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
        f"tanh backward     | "
        f"{tanh_metadata['median_kernel_ms']:.3f} ms | "
        f"{tanh_metadata['approximate_bandwidth_gbps']:.2f} GB/s | "
        f"error {errors['tanh']:.7g}"
    )

    print(
        f"sigmoid backward  | "
        f"{sigmoid_metadata['median_kernel_ms']:.3f} ms | "
        f"{sigmoid_metadata['approximate_bandwidth_gbps']:.2f} GB/s | "
        f"error {errors['sigmoid']:.7g}"
    )

    print(
        f"tension backward  | "
        f"{tension_metadata['median_kernel_ms']:.3f} ms | "
        f"{tension_metadata['approximate_bandwidth_gbps']:.2f} GB/s"
    )

    print(
        f"state error:        "
        f"{errors['grad_state']:.7g}"
    )

    print(
        f"proposal error:     "
        f"{errors['grad_proposal']:.7g}"
    )

    print(
        f"gate error:         "
        f"{errors['grad_gate']:.7g}"
    )

    print(
        f"All verified:       "
        f"{all_verified}"
    )

    payload = {
        "milestone":
            "device_resident_recurrent_backward",
        "device": runtime.info.to_dict(),
        "configuration": {
            "element_count": element_count,
            "repetitions": repetitions,
            "precision": "float32",
        },
        "operations": {
            "tanh_backward": {
                **tanh_metadata,
                "maximum_absolute_error":
                    errors["tanh"],
                "verified":
                    verified["tanh"],
            },
            "sigmoid_backward": {
                **sigmoid_metadata,
                "maximum_absolute_error":
                    errors["sigmoid"],
                "verified":
                    verified["sigmoid"],
            },
            "tension_backward": {
                **tension_metadata,
                "errors": {
                    "grad_state":
                        errors["grad_state"],
                    "grad_proposal":
                        errors["grad_proposal"],
                    "grad_gate":
                        errors["grad_gate"],
                },
                "verified": {
                    "grad_state":
                        verified["grad_state"],
                    "grad_proposal":
                        verified["grad_proposal"],
                    "grad_gate":
                        verified["grad_gate"],
                },
            },
        },
        "all_verified": all_verified,
    }

    receipt_path = write_receipt(
        "receipts/"
        "device_resident_recurrent_backward_receipt.json",
        payload,
    )

    print()
    print(f"Receipt: {receipt_path}")

    if not all_verified:
        raise RuntimeError(
            "Recurrent backward operation "
            "verification failed"
        )

    print()
    print(
        "PASSED: all backward operations needed "
        "for recurrent tension BPTT ran in "
        "RX 480 VRAM."
    )


if __name__ == "__main__":
    main()
