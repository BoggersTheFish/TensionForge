from __future__ import annotations

from typing import Any

import numpy as np

from tensionforge.ops import (
    adamw_update_device,
    linear_backward_device,
    linear_device,
    mse_loss_grad_device,
    sigmoid_backward_device,
    sigmoid_device,
    tanh_backward_device,
    tanh_device,
    tension_update_backward_device,
    tension_update_device,
)
from tensionforge.ops.recurrent_support import (
    add_inplace_device,
    concatenate_rows_device,
    fill_device,
    merge_recurrent_state_gradient_device,
)
from tensionforge.runtime import TensionForgeRuntime
from tensionforge.tensor import DeviceTensor


PARAMETER_NAMES = (
    "proposal_weights",
    "proposal_bias",
    "gate_weights",
    "gate_bias",
    "readout_weights",
    "readout_bias",
)


class ComposableTensionCell:
    def __init__(
        self,
        runtime: TensionForgeRuntime,
        *,
        batch_size: int,
        sequence_length: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
        parameters: dict[str, np.ndarray],
    ) -> None:
        if batch_size < 1:
            raise ValueError(
                "batch_size must be positive"
            )

        if sequence_length < 1:
            raise ValueError(
                "sequence_length must be positive"
            )

        if input_size < 1:
            raise ValueError(
                "input_size must be positive"
            )

        if hidden_size < 1:
            raise ValueError(
                "hidden_size must be positive"
            )

        if output_size < 1:
            raise ValueError(
                "output_size must be positive"
            )

        self.runtime = runtime
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.combined_size = (
            input_size + hidden_size
        )

        expected_shapes = {
            "proposal_weights": (
                self.combined_size,
                hidden_size,
            ),
            "proposal_bias": (
                hidden_size,
            ),
            "gate_weights": (
                self.combined_size,
                hidden_size,
            ),
            "gate_bias": (
                hidden_size,
            ),
            "readout_weights": (
                hidden_size,
                output_size,
            ),
            "readout_bias": (
                output_size,
            ),
        }

        missing = [
            name
            for name in PARAMETER_NAMES
            if name not in parameters
        ]

        if missing:
            raise ValueError(
                "Missing parameters: "
                + ", ".join(missing)
            )

        normalised_parameters: dict[
            str,
            np.ndarray,
        ] = {}

        for name in PARAMETER_NAMES:
            value = np.ascontiguousarray(
                parameters[name],
                dtype=np.float32,
            )

            if value.shape != expected_shapes[name]:
                raise ValueError(
                    f"{name} has shape {value.shape}; "
                    f"expected {expected_shapes[name]}"
                )

            normalised_parameters[name] = value

        self.parameters = {
            name: DeviceTensor.from_numpy(
                runtime,
                normalised_parameters[name],
            )
            for name in PARAMETER_NAMES
        }

        self.gradients = {
            name: self._zeros(
                expected_shapes[name]
            )
            for name in PARAMETER_NAMES
        }

        self.first_moments = {
            name: self._zeros(
                expected_shapes[name]
            )
            for name in PARAMETER_NAMES
        }

        self.second_moments = {
            name: self._zeros(
                expected_shapes[name]
            )
            for name in PARAMETER_NAMES
        }

        self.input_steps = [
            self._empty(
                (
                    batch_size,
                    input_size,
                )
            )
            for _ in range(sequence_length)
        ]

        self.target = self._empty(
            (
                batch_size,
                output_size,
            )
        )

        self.states = [
            self._zeros(
                (
                    batch_size,
                    hidden_size,
                )
            )
        ]

        self.states.extend(
            self._empty(
                (
                    batch_size,
                    hidden_size,
                )
            )
            for _ in range(sequence_length)
        )

        self.combined_features = [
            self._empty(
                (
                    batch_size,
                    self.combined_size,
                )
            )
            for _ in range(sequence_length)
        ]

        self.proposal_logits = [
            self._empty(
                (
                    batch_size,
                    hidden_size,
                )
            )
            for _ in range(sequence_length)
        ]

        self.proposals = [
            self._empty(
                (
                    batch_size,
                    hidden_size,
                )
            )
            for _ in range(sequence_length)
        ]

        self.gate_logits = [
            self._empty(
                (
                    batch_size,
                    hidden_size,
                )
            )
            for _ in range(sequence_length)
        ]

        self.gates = [
            self._empty(
                (
                    batch_size,
                    hidden_size,
                )
            )
            for _ in range(sequence_length)
        ]

        self.prediction = self._empty(
            (
                batch_size,
                output_size,
            )
        )

        self.loss_terms = self._empty(
            self.prediction.shape
        )

        self.grad_prediction = self._empty(
            self.prediction.shape
        )

        self.grad_state_carry_a = self._empty(
            (
                batch_size,
                hidden_size,
            )
        )

        self.grad_state_carry_b = self._empty(
            (
                batch_size,
                hidden_size,
            )
        )

        self.grad_state_direct = self._empty(
            (
                batch_size,
                hidden_size,
            )
        )

        self.grad_proposal = self._empty(
            (
                batch_size,
                hidden_size,
            )
        )

        self.grad_gate = self._empty(
            (
                batch_size,
                hidden_size,
            )
        )

        self.grad_proposal_logits = self._empty(
            (
                batch_size,
                hidden_size,
            )
        )

        self.grad_gate_logits = self._empty(
            (
                batch_size,
                hidden_size,
            )
        )

        self.grad_features_proposal = self._empty(
            (
                batch_size,
                self.combined_size,
            )
        )

        self.grad_features_gate = self._empty(
            (
                batch_size,
                self.combined_size,
            )
        )

        self.step_proposal_weight_gradient = (
            self._empty(
                expected_shapes[
                    "proposal_weights"
                ]
            )
        )

        self.step_proposal_bias_gradient = (
            self._empty(
                expected_shapes[
                    "proposal_bias"
                ]
            )
        )

        self.step_gate_weight_gradient = (
            self._empty(
                expected_shapes[
                    "gate_weights"
                ]
            )
        )

        self.step_gate_bias_gradient = (
            self._empty(
                expected_shapes[
                    "gate_bias"
                ]
            )
        )

        self._batch_loaded = False
        self._forward_complete = False
        self._backward_complete = False

    def _empty(
        self,
        shape: tuple[int, ...],
    ) -> DeviceTensor:
        return DeviceTensor.empty(
            self.runtime,
            shape,
            dtype=np.float32,
        )

    def _zeros(
        self,
        shape: tuple[int, ...],
    ) -> DeviceTensor:
        return DeviceTensor.from_numpy(
            self.runtime,
            np.zeros(
                shape,
                dtype=np.float32,
            ),
        )

    @property
    def parameter_count(self) -> int:
        return sum(
            tensor.size
            for tensor in self.parameters.values()
        )

    def load_batch(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        resolved_inputs = np.asarray(
            inputs,
            dtype=np.float32,
        )

        resolved_targets = np.asarray(
            targets,
            dtype=np.float32,
        )

        expected_input_shape = (
            self.batch_size,
            self.sequence_length,
            self.input_size,
        )

        expected_target_shape = (
            self.batch_size,
            self.output_size,
        )

        if (
            resolved_inputs.shape
            != expected_input_shape
        ):
            raise ValueError(
                "inputs have shape "
                f"{resolved_inputs.shape}; expected "
                f"{expected_input_shape}"
            )

        if (
            resolved_targets.shape
            != expected_target_shape
        ):
            raise ValueError(
                "targets have shape "
                f"{resolved_targets.shape}; expected "
                f"{expected_target_shape}"
            )

        for time_index in range(
            self.sequence_length
        ):
            self.input_steps[
                time_index
            ].copy_from(
                np.ascontiguousarray(
                    resolved_inputs[
                        :,
                        time_index,
                        :,
                    ]
                )
            )

        self.target.copy_from(
            np.ascontiguousarray(
                resolved_targets
            )
        )

        self._batch_loaded = True
        self._forward_complete = False
        self._backward_complete = False

    def forward(self) -> dict[str, Any]:
        if not self._batch_loaded:
            raise RuntimeError(
                "load_batch must be called before "
                "forward"
            )

        step_metrics: list[
            dict[str, Any]
        ] = []

        for time_index in range(
            self.sequence_length
        ):
            combined, combine_metadata = (
                concatenate_rows_device(
                    self.runtime,
                    self.input_steps[
                        time_index
                    ],
                    self.states[
                        time_index
                    ],
                    output=self.combined_features[
                        time_index
                    ],
                )
            )

            proposal_logits, proposal_linear = (
                linear_device(
                    self.runtime,
                    combined,
                    self.parameters[
                        "proposal_weights"
                    ],
                    self.parameters[
                        "proposal_bias"
                    ],
                    output=self.proposal_logits[
                        time_index
                    ],
                )
            )

            proposal, proposal_activation = (
                tanh_device(
                    self.runtime,
                    proposal_logits,
                    output=self.proposals[
                        time_index
                    ],
                )
            )

            gate_logits, gate_linear = (
                linear_device(
                    self.runtime,
                    combined,
                    self.parameters[
                        "gate_weights"
                    ],
                    self.parameters[
                        "gate_bias"
                    ],
                    output=self.gate_logits[
                        time_index
                    ],
                )
            )

            gate, gate_activation = (
                sigmoid_device(
                    self.runtime,
                    gate_logits,
                    output=self.gates[
                        time_index
                    ],
                )
            )

            state, state_update = (
                tension_update_device(
                    self.runtime,
                    self.states[
                        time_index
                    ],
                    proposal,
                    gate,
                    output=self.states[
                        time_index + 1
                    ],
                )
            )

            step_metrics.append(
                {
                    "time_index": time_index,
                    "concatenate":
                        combine_metadata,
                    "proposal_linear":
                        proposal_linear,
                    "proposal_activation":
                        proposal_activation,
                    "gate_linear":
                        gate_linear,
                    "gate_activation":
                        gate_activation,
                    "state_update":
                        state_update,
                }
            )

        prediction, readout_metadata = (
            linear_device(
                self.runtime,
                self.states[-1],
                self.parameters[
                    "readout_weights"
                ],
                self.parameters[
                    "readout_bias"
                ],
                output=self.prediction,
            )
        )

        (
            self.loss_terms,
            self.grad_prediction,
            loss_metadata,
        ) = mse_loss_grad_device(
            self.runtime,
            prediction,
            self.target,
            loss_terms=self.loss_terms,
            grad_prediction=(
                self.grad_prediction
            ),
        )

        self._forward_complete = True
        self._backward_complete = False

        return {
            "operation":
                "composable_tension_cell_forward",
            "sequence_length":
                self.sequence_length,
            "step_metrics": step_metrics,
            "readout": readout_metadata,
            "loss": loss_metadata,
        }

    def _zero_parameter_gradients(
        self,
    ) -> None:
        for gradient in self.gradients.values():
            fill_device(
                self.runtime,
                gradient,
                0.0,
            )

    def backward(self) -> dict[str, Any]:
        if not self._forward_complete:
            raise RuntimeError(
                "forward must be called before "
                "backward"
            )

        self._zero_parameter_gradients()

        (
            self.grad_state_carry_a,
            self.gradients[
                "readout_weights"
            ],
            self.gradients[
                "readout_bias"
            ],
            readout_backward,
        ) = linear_backward_device(
            self.runtime,
            self.states[-1],
            self.parameters[
                "readout_weights"
            ],
            self.grad_prediction,
            grad_input=(
                self.grad_state_carry_a
            ),
            grad_weights=self.gradients[
                "readout_weights"
            ],
            grad_bias=self.gradients[
                "readout_bias"
            ],
        )

        current_state_gradient = (
            self.grad_state_carry_a
        )

        next_state_gradient = (
            self.grad_state_carry_b
        )

        reverse_metrics: list[
            dict[str, Any]
        ] = []

        for time_index in reversed(
            range(self.sequence_length)
        ):
            (
                self.grad_state_direct,
                self.grad_proposal,
                self.grad_gate,
                tension_backward,
            ) = tension_update_backward_device(
                self.runtime,
                self.states[time_index],
                self.proposals[time_index],
                self.gates[time_index],
                current_state_gradient,
                grad_state=(
                    self.grad_state_direct
                ),
                grad_proposal=(
                    self.grad_proposal
                ),
                grad_gate=self.grad_gate,
            )

            (
                self.grad_proposal_logits,
                proposal_backward,
            ) = tanh_backward_device(
                self.runtime,
                self.proposals[time_index],
                self.grad_proposal,
                grad_input=(
                    self.grad_proposal_logits
                ),
            )

            (
                self.grad_gate_logits,
                gate_backward,
            ) = sigmoid_backward_device(
                self.runtime,
                self.gates[time_index],
                self.grad_gate,
                grad_input=(
                    self.grad_gate_logits
                ),
            )

            (
                self.grad_features_proposal,
                self.step_proposal_weight_gradient,
                self.step_proposal_bias_gradient,
                proposal_linear_backward,
            ) = linear_backward_device(
                self.runtime,
                self.combined_features[
                    time_index
                ],
                self.parameters[
                    "proposal_weights"
                ],
                self.grad_proposal_logits,
                grad_input=(
                    self.grad_features_proposal
                ),
                grad_weights=(
                    self.step_proposal_weight_gradient
                ),
                grad_bias=(
                    self.step_proposal_bias_gradient
                ),
            )

            (
                self.grad_features_gate,
                self.step_gate_weight_gradient,
                self.step_gate_bias_gradient,
                gate_linear_backward,
            ) = linear_backward_device(
                self.runtime,
                self.combined_features[
                    time_index
                ],
                self.parameters[
                    "gate_weights"
                ],
                self.grad_gate_logits,
                grad_input=(
                    self.grad_features_gate
                ),
                grad_weights=(
                    self.step_gate_weight_gradient
                ),
                grad_bias=(
                    self.step_gate_bias_gradient
                ),
            )

            add_inplace_device(
                self.runtime,
                self.gradients[
                    "proposal_weights"
                ],
                self.step_proposal_weight_gradient,
            )

            add_inplace_device(
                self.runtime,
                self.gradients[
                    "proposal_bias"
                ],
                self.step_proposal_bias_gradient,
            )

            add_inplace_device(
                self.runtime,
                self.gradients[
                    "gate_weights"
                ],
                self.step_gate_weight_gradient,
            )

            add_inplace_device(
                self.runtime,
                self.gradients[
                    "gate_bias"
                ],
                self.step_gate_bias_gradient,
            )

            (
                next_state_gradient,
                merge_metadata,
            ) = (
                merge_recurrent_state_gradient_device(
                    self.runtime,
                    self.grad_features_proposal,
                    self.grad_features_gate,
                    self.grad_state_direct,
                    input_features=(
                        self.input_size
                    ),
                    output=next_state_gradient,
                )
            )

            reverse_metrics.append(
                {
                    "time_index": time_index,
                    "tension_backward":
                        tension_backward,
                    "proposal_backward":
                        proposal_backward,
                    "gate_backward":
                        gate_backward,
                    "proposal_linear_backward":
                        proposal_linear_backward,
                    "gate_linear_backward":
                        gate_linear_backward,
                    "merge_state_gradient":
                        merge_metadata,
                }
            )

            (
                current_state_gradient,
                next_state_gradient,
            ) = (
                next_state_gradient,
                current_state_gradient,
            )

        self._backward_complete = True

        return {
            "operation":
                "composable_tension_cell_backward",
            "sequence_length":
                self.sequence_length,
            "readout_backward":
                readout_backward,
            "reverse_metrics":
                reverse_metrics,
        }

    def update(
        self,
        *,
        step: int,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> dict[str, dict[str, Any]]:
        if not self._backward_complete:
            raise RuntimeError(
                "backward must be called before "
                "update"
            )

        metrics: dict[
            str,
            dict[str, Any],
        ] = {}

        for name in PARAMETER_NAMES:
            metrics[name] = (
                adamw_update_device(
                    self.runtime,
                    self.parameters[name],
                    self.gradients[name],
                    self.first_moments[name],
                    self.second_moments[name],
                    step=step,
                    learning_rate=learning_rate,
                    beta1=beta1,
                    beta2=beta2,
                    epsilon=epsilon,
                    weight_decay=weight_decay,
                )
            )

        self._forward_complete = False
        self._backward_complete = False

        return metrics

    def loss(self) -> float:
        if not self._forward_complete:
            raise RuntimeError(
                "forward must be called before loss"
            )

        return float(
            np.sum(
                self.loss_terms.to_numpy(),
                dtype=np.float64,
            )
        )

    def prediction_numpy(self) -> np.ndarray:
        if not self._forward_complete:
            raise RuntimeError(
                "forward must be called before "
                "reading predictions"
            )

        return self.prediction.to_numpy()

    def parameter_numpy(
        self,
    ) -> dict[str, np.ndarray]:
        return {
            name: tensor.to_numpy()
            for name, tensor
            in self.parameters.items()
        }

    def gradient_numpy(
        self,
    ) -> dict[str, np.ndarray]:
        if not self._backward_complete:
            raise RuntimeError(
                "backward must be called before "
                "reading gradients"
            )

        return {
            name: tensor.to_numpy()
            for name, tensor
            in self.gradients.items()
        }
