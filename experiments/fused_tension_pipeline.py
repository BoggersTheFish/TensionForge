from __future__ import annotations

import numpy as np

from tensionforge import (
    DeviceTensor,
    TensionForgeRuntime,
)
from tensionforge.ops import (
    fused_tension_linear_device,
    linear_device,
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

    batch_size = 1024
    feature_count = 1024
    hidden_size = 512
    repetitions = 30

    scale = np.float32(
        1.0 / np.sqrt(feature_count)
    )

    features = rng.normal(
        size=(batch_size, feature_count),
    ).astype(np.float32)

    state = rng.normal(
        size=(batch_size, hidden_size),
    ).astype(np.float32)

    proposal_weights = (
        rng.normal(
            size=(
                feature_count,
                hidden_size,
            ),
        ).astype(np.float32)
        * scale
    )

    gate_weights = (
        rng.normal(
            size=(
                feature_count,
                hidden_size,
            ),
        ).astype(np.float32)
        * scale
    )

    proposal_bias = rng.normal(
        scale=0.1,
        size=(hidden_size,),
    ).astype(np.float32)

    gate_bias = rng.normal(
        scale=0.1,
        size=(hidden_size,),
    ).astype(np.float32)

    features_gpu = DeviceTensor.from_numpy(
        runtime,
        features,
    )

    state_gpu = DeviceTensor.from_numpy(
        runtime,
        state,
    )

    proposal_weights_gpu = (
        DeviceTensor.from_numpy(
            runtime,
            proposal_weights,
        )
    )

    proposal_bias_gpu = (
        DeviceTensor.from_numpy(
            runtime,
            proposal_bias,
        )
    )

    gate_weights_gpu = DeviceTensor.from_numpy(
        runtime,
        gate_weights,
    )

    gate_bias_gpu = DeviceTensor.from_numpy(
        runtime,
        gate_bias,
    )

    fused_output_gpu = DeviceTensor.empty(
        runtime,
        state.shape,
        dtype=np.float32,
    )

    fused_output_gpu, fused_metadata = (
        fused_tension_linear_device(
            runtime,
            features_gpu,
            state_gpu,
            proposal_weights_gpu,
            proposal_bias_gpu,
            gate_weights_gpu,
            gate_bias_gpu,
            output=fused_output_gpu,
            repetitions=repetitions,
        )
    )

    proposal_logits_gpu = DeviceTensor.empty(
        runtime,
        state.shape,
        dtype=np.float32,
    )

    proposal_gpu = DeviceTensor.empty(
        runtime,
        state.shape,
        dtype=np.float32,
    )

    gate_logits_gpu = DeviceTensor.empty(
        runtime,
        state.shape,
        dtype=np.float32,
    )

    gate_gpu = DeviceTensor.empty(
        runtime,
        state.shape,
        dtype=np.float32,
    )

    unfused_output_gpu = DeviceTensor.empty(
        runtime,
        state.shape,
        dtype=np.float32,
    )

    proposal_logits_gpu, proposal_linear = (
        linear_device(
            runtime,
            features_gpu,
            proposal_weights_gpu,
            proposal_bias_gpu,
            output=proposal_logits_gpu,
            repetitions=repetitions,
        )
    )

    proposal_gpu, proposal_activation = (
        tanh_device(
            runtime,
            proposal_logits_gpu,
            output=proposal_gpu,
            repetitions=repetitions,
        )
    )

    gate_logits_gpu, gate_linear = (
        linear_device(
            runtime,
            features_gpu,
            gate_weights_gpu,
            gate_bias_gpu,
            output=gate_logits_gpu,
            repetitions=repetitions,
        )
    )

    gate_gpu, gate_activation = (
        sigmoid_device(
            runtime,
            gate_logits_gpu,
            output=gate_gpu,
            repetitions=repetitions,
        )
    )

    unfused_output_gpu, update_metadata = (
        tension_update_device(
            runtime,
            state_gpu,
            proposal_gpu,
            gate_gpu,
            output=unfused_output_gpu,
            repetitions=repetitions,
        )
    )

    fused_result = fused_output_gpu.to_numpy()
    unfused_result = unfused_output_gpu.to_numpy()

    proposal_expected = np.tanh(
        features @ proposal_weights
        + proposal_bias
    ).astype(np.float32)

    gate_expected = (
        1.0
        / (
            1.0
            + np.exp(
                -(
                    features @ gate_weights
                    + gate_bias
                )
            )
        )
    ).astype(np.float32)

    expected = (
        state
        + gate_expected
        * (
            proposal_expected
            - state
        )
    ).astype(np.float32)

    fused_error = float(
        np.max(
            np.abs(
                fused_result - expected
            )
        )
    )

    unfused_error = float(
        np.max(
            np.abs(
                unfused_result - expected
            )
        )
    )

    fused_vs_unfused_error = float(
        np.max(
            np.abs(
                fused_result
                - unfused_result
            )
        )
    )

    fused_verified = bool(
        np.allclose(
            fused_result,
            expected,
            rtol=5e-4,
            atol=5e-4,
        )
    )

    unfused_verified = bool(
        np.allclose(
            unfused_result,
            expected,
            rtol=5e-4,
            atol=5e-4,
        )
    )

    unfused_kernel_ms = float(
        proposal_linear["median_kernel_ms"]
        + proposal_activation[
            "median_kernel_ms"
        ]
        + gate_linear["median_kernel_ms"]
        + gate_activation[
            "median_kernel_ms"
        ]
        + update_metadata[
            "median_kernel_ms"
        ]
    )

    fused_kernel_ms = float(
        fused_metadata["median_kernel_ms"]
    )

    kernel_speedup = (
        unfused_kernel_ms
        / fused_kernel_ms
    )

    all_verified = (
        fused_verified
        and unfused_verified
    )

    print(
        "=== FUSED TENSION PIPELINE ==="
    )
    print(
        f"Device:       "
        f"{runtime.info.device_name}"
    )
    print(
        f"Batch:        {batch_size}"
    )
    print(
        f"Features:     {feature_count}"
    )
    print(
        f"Hidden:       {hidden_size}"
    )
    print(
        f"Repetitions:  {repetitions}"
    )
    print()

    print(
        f"Fused kernel:       "
        f"{fused_kernel_ms:.3f} ms"
    )

    print(
        f"Unfused kernel sum: "
        f"{unfused_kernel_ms:.3f} ms"
    )

    print(
        f"Kernel speedup:      "
        f"{kernel_speedup:.2f}x"
    )

    print(
        f"Launches reduced:    "
        f"5 to 1"
    )

    print(
        f"Fused throughput:    "
        f"{fused_metadata['approximate_linear_gflops']:.2f} GFLOPS"
    )

    print(
        f"Fused max error:     "
        f"{fused_error:.7g}"
    )

    print(
        f"Unfused max error:   "
        f"{unfused_error:.7g}"
    )

    print(
        f"Fused/unfused error: "
        f"{fused_vs_unfused_error:.7g}"
    )

    print(
        f"Fused verified:      "
        f"{fused_verified}"
    )

    print(
        f"Unfused verified:    "
        f"{unfused_verified}"
    )

    payload = {
        "milestone":
            "fused_tension_linear_pipeline",
        "device": runtime.info.to_dict(),
        "configuration": {
            "batch_size": batch_size,
            "feature_count": feature_count,
            "hidden_size": hidden_size,
            "repetitions": repetitions,
            "precision": "float32",
        },
        "fused": {
            **fused_metadata,
            "maximum_absolute_error":
                fused_error,
            "verified":
                fused_verified,
        },
        "unfused": {
            "proposal_linear":
                proposal_linear,
            "proposal_activation":
                proposal_activation,
            "gate_linear":
                gate_linear,
            "gate_activation":
                gate_activation,
            "tension_update":
                update_metadata,
            "combined_median_kernel_ms":
                unfused_kernel_ms,
            "maximum_absolute_error":
                unfused_error,
            "verified":
                unfused_verified,
        },
        "comparison": {
            "kernel_speedup":
                kernel_speedup,
            "launches_before": 5,
            "launches_after": 1,
            "fused_vs_unfused_max_error":
                fused_vs_unfused_error,
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
        "fused_tension_pipeline_receipt.json",
        payload,
    )

    print()
    print(f"Receipt: {receipt_path}")

    if not all_verified:
        raise RuntimeError(
            "Fused tension pipeline "
            "verification failed"
        )

    print()
    print(
        "PASSED: proposal, gate, activations, "
        "and tension update were fused into "
        "one verified RX 480 kernel."
    )


if __name__ == "__main__":
    main()
