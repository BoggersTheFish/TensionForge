from __future__ import annotations

import numpy as np

from tensionforge import TensionForgeRuntime
from tensionforge.models import (
    ComposableTensionCell,
)


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


def cpu_forward_backward(
    inputs: np.ndarray,
    targets: np.ndarray,
    parameters: dict[str, np.ndarray],
) -> tuple[
    np.ndarray,
    float,
    dict[str, np.ndarray],
]:
    batch_size = inputs.shape[0]
    sequence_length = inputs.shape[1]
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

    states = [state]
    combined_features = []
    proposals = []
    gates = []

    for time_index in range(
        sequence_length
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

        combined_features.append(combined)
        proposals.append(proposal)
        gates.append(gate)
        states.append(state)

    prediction = (
        state
        @ parameters[
            "readout_weights"
        ]
        + parameters[
            "readout_bias"
        ]
    ).astype(np.float32)

    difference = prediction - targets

    loss = float(
        np.mean(
            difference * difference,
            dtype=np.float64,
        )
    )

    grad_prediction = (
        2.0
        * difference
        / difference.size
    ).astype(np.float32)

    gradients = {
        name: np.zeros_like(value)
        for name, value in parameters.items()
    }

    gradients["readout_weights"] = (
        states[-1].T
        @ grad_prediction
    ).astype(np.float32)

    gradients["readout_bias"] = (
        grad_prediction.sum(
            axis=0,
            dtype=np.float32,
        )
    )

    grad_state = (
        grad_prediction
        @ parameters[
            "readout_weights"
        ].T
    ).astype(np.float32)

    input_size = inputs.shape[2]

    for time_index in reversed(
        range(sequence_length)
    ):
        previous_state = states[time_index]
        proposal = proposals[time_index]
        gate = gates[time_index]
        combined = combined_features[
            time_index
        ]

        grad_previous_direct = (
            grad_state
            * (
                1.0
                - gate
            )
        ).astype(np.float32)

        grad_proposal = (
            grad_state * gate
        ).astype(np.float32)

        grad_gate = (
            grad_state
            * (
                proposal
                - previous_state
            )
        ).astype(np.float32)

        grad_proposal_logits = (
            grad_proposal
            * (
                1.0
                - proposal
                * proposal
            )
        ).astype(np.float32)

        grad_gate_logits = (
            grad_gate
            * gate
            * (
                1.0
                - gate
            )
        ).astype(np.float32)

        gradients[
            "proposal_weights"
        ] += (
            combined.T
            @ grad_proposal_logits
        ).astype(np.float32)

        gradients[
            "proposal_bias"
        ] += grad_proposal_logits.sum(
            axis=0,
            dtype=np.float32,
        )

        gradients[
            "gate_weights"
        ] += (
            combined.T
            @ grad_gate_logits
        ).astype(np.float32)

        gradients[
            "gate_bias"
        ] += grad_gate_logits.sum(
            axis=0,
            dtype=np.float32,
        )

        grad_combined = (
            grad_proposal_logits
            @ parameters[
                "proposal_weights"
            ].T
            + grad_gate_logits
            @ parameters[
                "gate_weights"
            ].T
        ).astype(np.float32)

        grad_state = (
            grad_previous_direct
            + grad_combined[
                :,
                input_size:,
            ]
        ).astype(np.float32)

    return prediction, loss, gradients


def test_composable_tension_cell_matches_numpy() -> None:
    runtime = TensionForgeRuntime(
        profiling=True,
    )

    rng = np.random.default_rng(801)

    batch_size = 7
    sequence_length = 4
    input_size = 2
    hidden_size = 5
    output_size = 1
    combined_size = (
        input_size + hidden_size
    )

    parameters = {
        "proposal_weights": (
            rng.normal(
                scale=0.15,
                size=(
                    combined_size,
                    hidden_size,
                ),
            ).astype(np.float32)
        ),
        "proposal_bias": (
            rng.normal(
                scale=0.05,
                size=(hidden_size,),
            ).astype(np.float32)
        ),
        "gate_weights": (
            rng.normal(
                scale=0.15,
                size=(
                    combined_size,
                    hidden_size,
                ),
            ).astype(np.float32)
        ),
        "gate_bias": (
            rng.normal(
                scale=0.05,
                size=(hidden_size,),
            ).astype(np.float32)
        ),
        "readout_weights": (
            rng.normal(
                scale=0.15,
                size=(
                    hidden_size,
                    output_size,
                ),
            ).astype(np.float32)
        ),
        "readout_bias": (
            rng.normal(
                scale=0.05,
                size=(output_size,),
            ).astype(np.float32)
        ),
    }

    inputs = rng.normal(
        scale=0.4,
        size=(
            batch_size,
            sequence_length,
            input_size,
        ),
    ).astype(np.float32)

    targets = rng.normal(
        scale=0.5,
        size=(
            batch_size,
            output_size,
        ),
    ).astype(np.float32)

    (
        expected_prediction,
        expected_loss,
        expected_gradients,
    ) = cpu_forward_backward(
        inputs,
        targets,
        parameters,
    )

    model = ComposableTensionCell(
        runtime,
        batch_size=batch_size,
        sequence_length=sequence_length,
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        parameters=parameters,
    )

    model.load_batch(
        inputs,
        targets,
    )

    model.forward()

    actual_prediction = (
        model.prediction_numpy()
    )

    actual_loss = model.loss()

    model.backward()

    actual_gradients = (
        model.gradient_numpy()
    )

    np.testing.assert_allclose(
        actual_prediction,
        expected_prediction,
        rtol=5e-4,
        atol=5e-4,
    )

    np.testing.assert_allclose(
        actual_loss,
        expected_loss,
        rtol=5e-4,
        atol=5e-5,
    )

    for name in expected_gradients:
        np.testing.assert_allclose(
            actual_gradients[name],
            expected_gradients[name],
            rtol=8e-4,
            atol=8e-4,
            err_msg=(
                f"Gradient mismatch for {name}"
            ),
        )
