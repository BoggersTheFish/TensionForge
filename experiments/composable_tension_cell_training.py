from __future__ import annotations

import time

import numpy as np

from tensionforge import TensionForgeRuntime
from tensionforge.models import (
    ComposableTensionCell,
)
from tensionforge.receipts import write_receipt


def sigmoid(
    values: np.ndarray,
) -> np.ndarray:
    return (
        1.0
        / (
            1.0
            + np.exp(-values)
        )
    ).astype(np.float32)


def cpu_predict(
    inputs: np.ndarray,
    parameters: dict[str, np.ndarray],
) -> np.ndarray:
    batch_size = inputs.shape[0]

    hidden_size = (
        parameters[
            "proposal_bias"
        ].shape[0]
    )

    state = np.zeros(
        (
            batch_size,
            hidden_size,
        ),
        dtype=np.float32,
    )

    for time_index in range(
        inputs.shape[1]
    ):
        combined = np.concatenate(
            (
                inputs[:, time_index, :],
                state,
            ),
            axis=1,
        ).astype(np.float32)

        proposal = np.tanh(
            combined
            @ parameters[
                "proposal_weights"
            ]
            + parameters[
                "proposal_bias"
            ]
        ).astype(np.float32)

        gate = sigmoid(
            combined
            @ parameters[
                "gate_weights"
            ]
            + parameters[
                "gate_bias"
            ]
        )

        state = (
            state
            + gate
            * (
                proposal
                - state
            )
        ).astype(np.float32)

    return (
        state
        @ parameters[
            "readout_weights"
        ]
        + parameters[
            "readout_bias"
        ]
    ).astype(np.float32)


def make_delayed_recall_data(
    rng: np.random.Generator,
    *,
    sample_count: int,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = rng.uniform(
        -1.0,
        1.0,
        size=(
            sample_count,
            1,
        ),
    ).astype(np.float32)

    inputs = rng.normal(
        scale=0.12,
        size=(
            sample_count,
            sequence_length,
            2,
        ),
    ).astype(np.float32)

    inputs[:, :, 1] = 0.0
    inputs[:, 0, 0] = values[:, 0]
    inputs[:, -1, 1] = 1.0

    targets = values.copy()

    return inputs, targets


def make_parameters(
    rng: np.random.Generator,
    *,
    input_size: int,
    hidden_size: int,
    output_size: int,
) -> dict[str, np.ndarray]:
    combined_size = (
        input_size + hidden_size
    )

    recurrent_scale = np.float32(
        1.0 / np.sqrt(combined_size)
    )

    readout_scale = np.float32(
        1.0 / np.sqrt(hidden_size)
    )

    return {
        "proposal_weights": (
            rng.normal(
                size=(
                    combined_size,
                    hidden_size,
                ),
            ).astype(np.float32)
            * recurrent_scale
            * np.float32(0.7)
        ),
        "proposal_bias": np.zeros(
            (hidden_size,),
            dtype=np.float32,
        ),
        "gate_weights": (
            rng.normal(
                size=(
                    combined_size,
                    hidden_size,
                ),
            ).astype(np.float32)
            * recurrent_scale
            * np.float32(0.5)
        ),
        "gate_bias": np.full(
            (hidden_size,),
            -1.0,
            dtype=np.float32,
        ),
        "readout_weights": (
            rng.normal(
                size=(
                    hidden_size,
                    output_size,
                ),
            ).astype(np.float32)
            * readout_scale
            * np.float32(0.5)
        ),
        "readout_bias": np.zeros(
            (output_size,),
            dtype=np.float32,
        ),
    }


def main() -> None:
    runtime = TensionForgeRuntime(
        profiling=True,
    )

    rng = np.random.default_rng(42)

    batch_size = 512
    validation_size = 256
    sequence_length = 8
    input_size = 2
    hidden_size = 16
    output_size = 1
    training_steps = 300
    learning_rate = 0.01

    training_inputs, training_targets = (
        make_delayed_recall_data(
            rng,
            sample_count=batch_size,
            sequence_length=sequence_length,
        )
    )

    validation_inputs, validation_targets = (
        make_delayed_recall_data(
            rng,
            sample_count=validation_size,
            sequence_length=sequence_length,
        )
    )

    initial_parameters = make_parameters(
        rng,
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
    )

    model = ComposableTensionCell(
        runtime,
        batch_size=batch_size,
        sequence_length=sequence_length,
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        parameters=initial_parameters,
    )

    model.load_batch(
        training_inputs,
        training_targets,
    )

    model.forward()

    initial_loss = model.loss()

    print(
        "=== COMPOSABLE TENSION CELL TRAINING ==="
    )
    print(
        f"Device:       "
        f"{runtime.info.device_name}"
    )
    print(
        f"Samples:      {batch_size}"
    )
    print(
        f"Sequence:     {sequence_length}"
    )
    print(
        f"Input size:   {input_size}"
    )
    print(
        f"Hidden size:  {hidden_size}"
    )
    print(
        f"Parameters:   "
        f"{model.parameter_count}"
    )
    print(
        f"Steps:        {training_steps}"
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

    last_forward = {}
    last_backward = {}
    last_update = {}

    started_at = time.perf_counter()

    for step in range(
        1,
        training_steps + 1,
    ):
        last_forward = model.forward()
        last_backward = model.backward()

        last_update = model.update(
            step=step,
            learning_rate=learning_rate,
        )

        if step in report_steps:
            model.forward()
            step_loss = model.loss()

            print(
                f"Step {step:4d} | loss "
                f"{step_loss:.10f}"
            )

    runtime.finish()

    training_seconds = (
        time.perf_counter()
        - started_at
    )

    model.forward()

    final_loss = model.loss()

    gpu_predictions = (
        model.prediction_numpy()
    )

    learned_parameters = (
        model.parameter_numpy()
    )

    cpu_training_predictions = cpu_predict(
        training_inputs,
        learned_parameters,
    )

    cpu_gpu_max_error = float(
        np.max(
            np.abs(
                cpu_training_predictions
                - gpu_predictions
            )
        )
    )

    validation_predictions = cpu_predict(
        validation_inputs,
        learned_parameters,
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
        final_loss < 0.01
        and validation_loss < 0.02
        and loss_reduction > 25.0
        and cpu_gpu_max_error < 2e-3
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
        f"CPU/GPU max error:  "
        f"{cpu_gpu_max_error:.8g}"
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
            "composable_recurrent_tension_cell",
        "device": runtime.info.to_dict(),
        "configuration": {
            "batch_size": batch_size,
            "validation_size":
                validation_size,
            "sequence_length":
                sequence_length,
            "input_size": input_size,
            "hidden_size": hidden_size,
            "output_size": output_size,
            "parameter_count":
                model.parameter_count,
            "training_steps":
                training_steps,
            "learning_rate":
                learning_rate,
            "precision": "float32",
            "task": "delayed_recall",
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
            "cpu_gpu_max_error":
                cpu_gpu_max_error,
            "training_seconds":
                training_seconds,
            "steps_per_second":
                steps_per_second,
        },
        "composition": {
            "forward_operations": [
                "concatenate_rows_device",
                "linear_device",
                "tanh_device",
                "sigmoid_device",
                "tension_update_device",
                "mse_loss_grad_device",
            ],
            "backward_operations": [
                "tension_update_backward_device",
                "tanh_backward_device",
                "sigmoid_backward_device",
                "linear_backward_device",
                "add_inplace_device",
                "merge_recurrent_state_gradient_device",
            ],
            "optimizer":
                "adamw_update_device",
            "experiment_specific_monolithic_kernel":
                False,
        },
        "last_operation_metrics": {
            "forward": last_forward,
            "backward": last_backward,
            "update": last_update,
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
        "composable_tension_cell_training_receipt.json",
        payload,
    )

    print(
        f"Receipt:            {receipt_path}"
    )

    if not passed:
        raise RuntimeError(
            "Composable recurrent TensionCell "
            "did not meet validation thresholds"
        )

    print()
    print(
        "PASSED: a recurrent TensionCell was "
        "trained through full BPTT using only "
        "reusable TensionForge operations."
    )


if __name__ == "__main__":
    main()
