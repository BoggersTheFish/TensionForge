from __future__ import annotations

import time

import numpy as np

from tensionforge import (
    DeviceTensor,
    TensionForgeRuntime,
)
from tensionforge.ops import (
    adamw_update_device,
    linear_backward_device,
    linear_device,
    mse_loss_grad_device,
)
from tensionforge.receipts import write_receipt


def read_loss(
    loss_terms: DeviceTensor,
) -> float:
    return float(
        np.sum(
            loss_terms.to_numpy(),
            dtype=np.float64,
        )
    )


def main() -> None:
    runtime = TensionForgeRuntime(
        profiling=True,
    )

    rng = np.random.default_rng(42)

    batch_size = 4096
    validation_size = 1024
    input_features = 64
    output_features = 32
    training_steps = 300
    learning_rate = 0.03

    inputs = rng.normal(
        size=(
            batch_size,
            input_features,
        ),
    ).astype(np.float32)

    true_weights = rng.normal(
        scale=0.25,
        size=(
            input_features,
            output_features,
        ),
    ).astype(np.float32)

    true_bias = rng.normal(
        scale=0.1,
        size=(output_features,),
    ).astype(np.float32)

    targets = (
        inputs @ true_weights
        + true_bias
    ).astype(np.float32)

    validation_inputs = rng.normal(
        size=(
            validation_size,
            input_features,
        ),
    ).astype(np.float32)

    validation_targets = (
        validation_inputs @ true_weights
        + true_bias
    ).astype(np.float32)

    initial_weights = rng.normal(
        scale=0.05,
        size=(
            input_features,
            output_features,
        ),
    ).astype(np.float32)

    initial_bias = np.zeros(
        (output_features,),
        dtype=np.float32,
    )

    inputs_gpu = DeviceTensor.from_numpy(
        runtime,
        inputs,
    )

    targets_gpu = DeviceTensor.from_numpy(
        runtime,
        targets,
    )

    weights_gpu = DeviceTensor.from_numpy(
        runtime,
        initial_weights,
    )

    bias_gpu = DeviceTensor.from_numpy(
        runtime,
        initial_bias,
    )

    predictions_gpu = DeviceTensor.empty(
        runtime,
        (
            batch_size,
            output_features,
        ),
        dtype=np.float32,
    )

    loss_terms_gpu = DeviceTensor.empty(
        runtime,
        predictions_gpu.shape,
        dtype=np.float32,
    )

    grad_prediction_gpu = DeviceTensor.empty(
        runtime,
        predictions_gpu.shape,
        dtype=np.float32,
    )

    grad_input_gpu = DeviceTensor.empty(
        runtime,
        inputs.shape,
        dtype=np.float32,
    )

    grad_weights_gpu = DeviceTensor.empty(
        runtime,
        weights_gpu.shape,
        dtype=np.float32,
    )

    grad_bias_gpu = DeviceTensor.empty(
        runtime,
        bias_gpu.shape,
        dtype=np.float32,
    )

    weight_first_gpu = DeviceTensor.from_numpy(
        runtime,
        np.zeros_like(initial_weights),
    )

    weight_second_gpu = DeviceTensor.from_numpy(
        runtime,
        np.zeros_like(initial_weights),
    )

    bias_first_gpu = DeviceTensor.from_numpy(
        runtime,
        np.zeros_like(initial_bias),
    )

    bias_second_gpu = DeviceTensor.from_numpy(
        runtime,
        np.zeros_like(initial_bias),
    )

    predictions_gpu, _ = linear_device(
        runtime,
        inputs_gpu,
        weights_gpu,
        bias_gpu,
        output=predictions_gpu,
    )

    (
        loss_terms_gpu,
        grad_prediction_gpu,
        _,
    ) = mse_loss_grad_device(
        runtime,
        predictions_gpu,
        targets_gpu,
        loss_terms=loss_terms_gpu,
        grad_prediction=grad_prediction_gpu,
    )

    initial_loss = read_loss(
        loss_terms_gpu
    )

    print(
        "=== DEVICE-RESIDENT LINEAR TRAINING ==="
    )
    print(
        f"Device:     "
        f"{runtime.info.device_name}"
    )
    print(
        f"Samples:    {batch_size}"
    )
    print(
        f"Input:      {input_features}"
    )
    print(
        f"Output:     {output_features}"
    )
    print(
        f"Parameters: "
        f"{initial_weights.size + initial_bias.size}"
    )
    print()

    print(
        f"Step    0 | loss "
        f"{initial_loss:.10f}"
    )

    report_steps = {
        1,
        10,
        25,
        50,
        100,
        200,
        training_steps,
    }

    last_forward: dict[str, object] = {}
    last_loss: dict[str, object] = {}
    last_backward: dict[str, object] = {}
    last_weight_update: dict[str, object] = {}
    last_bias_update: dict[str, object] = {}

    started_at = time.perf_counter()

    for step in range(
        1,
        training_steps + 1,
    ):
        (
            predictions_gpu,
            last_forward,
        ) = linear_device(
            runtime,
            inputs_gpu,
            weights_gpu,
            bias_gpu,
            output=predictions_gpu,
            repetitions=1,
        )

        (
            loss_terms_gpu,
            grad_prediction_gpu,
            last_loss,
        ) = mse_loss_grad_device(
            runtime,
            predictions_gpu,
            targets_gpu,
            loss_terms=loss_terms_gpu,
            grad_prediction=grad_prediction_gpu,
            repetitions=1,
        )

        (
            grad_input_gpu,
            grad_weights_gpu,
            grad_bias_gpu,
            last_backward,
        ) = linear_backward_device(
            runtime,
            inputs_gpu,
            weights_gpu,
            grad_prediction_gpu,
            grad_input=grad_input_gpu,
            grad_weights=grad_weights_gpu,
            grad_bias=grad_bias_gpu,
            repetitions=1,
        )

        last_weight_update = (
            adamw_update_device(
                runtime,
                weights_gpu,
                grad_weights_gpu,
                weight_first_gpu,
                weight_second_gpu,
                step=step,
                learning_rate=learning_rate,
            )
        )

        last_bias_update = (
            adamw_update_device(
                runtime,
                bias_gpu,
                grad_bias_gpu,
                bias_first_gpu,
                bias_second_gpu,
                step=step,
                learning_rate=learning_rate,
            )
        )

        if step in report_steps:
            step_loss = read_loss(
                loss_terms_gpu
            )

            print(
                f"Step {step:4d} | loss "
                f"{step_loss:.10f}"
            )

    runtime.finish()

    training_seconds = (
        time.perf_counter()
        - started_at
    )

    predictions_gpu, _ = linear_device(
        runtime,
        inputs_gpu,
        weights_gpu,
        bias_gpu,
        output=predictions_gpu,
    )

    (
        loss_terms_gpu,
        grad_prediction_gpu,
        _,
    ) = mse_loss_grad_device(
        runtime,
        predictions_gpu,
        targets_gpu,
        loss_terms=loss_terms_gpu,
        grad_prediction=grad_prediction_gpu,
    )

    final_loss = read_loss(
        loss_terms_gpu
    )

    learned_weights = weights_gpu.to_numpy()
    learned_bias = bias_gpu.to_numpy()

    weight_error = float(
        np.max(
            np.abs(
                learned_weights
                - true_weights
            )
        )
    )

    bias_error = float(
        np.max(
            np.abs(
                learned_bias
                - true_bias
            )
        )
    )

    validation_inputs_gpu = (
        DeviceTensor.from_numpy(
            runtime,
            validation_inputs,
        )
    )

    validation_output_gpu = (
        DeviceTensor.empty(
            runtime,
            (
                validation_size,
                output_features,
            ),
            dtype=np.float32,
        )
    )

    validation_output_gpu, _ = linear_device(
        runtime,
        validation_inputs_gpu,
        weights_gpu,
        bias_gpu,
        output=validation_output_gpu,
    )

    validation_predictions = (
        validation_output_gpu.to_numpy()
    )

    validation_loss = float(
        np.mean(
            (
                validation_predictions
                - validation_targets
            )
            ** 2,
            dtype=np.float64,
        )
    )

    loss_reduction = (
        initial_loss / final_loss
        if final_loss > 0.0
        else float("inf")
    )

    steps_per_second = (
        training_steps
        / training_seconds
    )

    passed = bool(
        final_loss < 1e-4
        and validation_loss < 1e-4
        and loss_reduction > 1000.0
        and weight_error < 5e-3
        and bias_error < 5e-3
    )

    print()
    print("=== RESULT ===")
    print(
        f"Initial loss:       "
        f"{initial_loss:.10f}"
    )
    print(
        f"Final train loss:   "
        f"{final_loss:.10f}"
    )
    print(
        f"Validation loss:    "
        f"{validation_loss:.10f}"
    )
    print(
        f"Loss reduction:     "
        f"{loss_reduction:.2f}x"
    )
    print(
        f"Weight max error:   "
        f"{weight_error:.8g}"
    )
    print(
        f"Bias max error:     "
        f"{bias_error:.8g}"
    )
    print(
        f"Training time:      "
        f"{training_seconds:.3f}s"
    )
    print(
        f"Steps per second:   "
        f"{steps_per_second:.2f}"
    )
    print(
        f"Passed:             {passed}"
    )

    payload = {
        "milestone":
            "composable_device_resident_training",
        "device": runtime.info.to_dict(),
        "configuration": {
            "batch_size": batch_size,
            "validation_size":
                validation_size,
            "input_features":
                input_features,
            "output_features":
                output_features,
            "parameter_count":
                int(
                    initial_weights.size
                    + initial_bias.size
                ),
            "training_steps":
                training_steps,
            "learning_rate":
                learning_rate,
            "precision": "float32",
        },
        "training": {
            "initial_loss":
                initial_loss,
            "final_loss":
                final_loss,
            "validation_loss":
                validation_loss,
            "loss_reduction_factor":
                loss_reduction,
            "weight_max_error":
                weight_error,
            "bias_max_error":
                bias_error,
            "training_seconds":
                training_seconds,
            "steps_per_second":
                steps_per_second,
        },
        "last_operation_metrics": {
            "forward": last_forward,
            "loss": last_loss,
            "backward": last_backward,
            "weight_update":
                last_weight_update,
            "bias_update":
                last_bias_update,
        },
        "runtime": {
            "program_cache_entries":
                runtime.program_cache_size,
            "kernel_cache_entries":
                runtime.kernel_cache_size,
        },
        "passed": passed,
    }

    receipt_path = write_receipt(
        "receipts/"
        "device_resident_training_receipt.json",
        payload,
    )

    print(f"Receipt:            {receipt_path}")

    if not passed:
        raise RuntimeError(
            "Composable device-resident "
            "training did not meet thresholds"
        )

    print()
    print(
        "PASSED: TensionForge trained a model "
        "using reusable forward, loss, backward, "
        "and AdamW operations in RX 480 VRAM."
    )


if __name__ == "__main__":
    main()
