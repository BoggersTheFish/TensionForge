from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np

from tensionforge.ops import (
    add_tension_penalty_device,
    batched_dot_device,
    broadcast_rows_device,
    broadcast_table_row_device,
    concatenate_columns_device,
    gather_slots_device,
    gelu_device,
    layer_norm_device,
    linear_device,
    row_delta_norm_device,
    scale_device,
    scatter_slots_device,
    sigmoid_device,
    softmax_device,
    tanh_device,
    tension_update_broadcast_device,
    topk_device,
    weighted_sum_device,
    workspace_reduce_device,
)
from tensionforge.runtime import TensionForgeRuntime
from tensionforge.tensor import DeviceTensor


@dataclass(frozen=True)
class TenSonBridgeConfig:
    num_slots: int
    slot_dim: int
    embed_dim: int
    key_dim: int
    top_k: int
    microsteps: int
    max_microsteps: int
    proposal_hidden: int
    tension_hidden: int
    readout_hidden: int

    @classmethod
    def from_object(cls, config: object) -> "TenSonBridgeConfig":
        return cls(**{name: int(getattr(config, name)) for name in cls.__dataclass_fields__})


LINEAR_WEIGHTS = {
    "router.query_network.0.weight",
    "router.query_network.3.weight",
    "proposal.network.1.weight",
    "proposal.network.3.weight",
    "tension.network.1.weight",
    "tension.network.3.weight",
    "readout.query.0.weight",
    "readout.query.2.weight",
    "readout.output.1.weight",
    "readout.output.3.weight",
}


class TenSonForwardBridge:
    """Device-resident forward-only port of PyTorch TensionWorkspace."""

    def __init__(
        self,
        runtime: TensionForgeRuntime,
        config: TenSonBridgeConfig | object,
        parameters: Mapping[str, np.ndarray],
    ) -> None:
        self.runtime = runtime
        self.config = config if isinstance(config, TenSonBridgeConfig) else TenSonBridgeConfig.from_object(config)
        if self.config.microsteps < 1 or self.config.microsteps > self.config.max_microsteps:
            raise ValueError("microsteps must be in [1, max_microsteps]")
        if self.config.top_k < 1 or self.config.top_k > self.config.num_slots:
            raise ValueError("top_k must be in [1, num_slots]")

        self.parameters: dict[str, DeviceTensor] = {}
        for name, raw in parameters.items():
            value = np.asarray(raw, dtype=np.float32)
            if name in LINEAR_WEIGHTS:
                value = value.T
            self.parameters[name] = DeviceTensor.from_numpy(runtime, np.ascontiguousarray(value))
        self._require_parameters()
        keys = np.asarray(parameters["slot_keys"], dtype=np.float32)
        self.slot_keys_linear = DeviceTensor.from_numpy(runtime, np.ascontiguousarray(keys.T))
        self.route_bias = DeviceTensor.from_numpy(runtime, np.zeros(self.config.num_slots, dtype=np.float32))

    def _require_parameters(self) -> None:
        required = {
            "initial_workspace", "slot_keys", "summary_norm.weight", "summary_norm.bias",
            "router.query_network.0.weight", "router.query_network.0.bias",
            "router.query_network.1.weight", "router.query_network.1.bias",
            "router.query_network.3.weight", "router.query_network.3.bias",
            "proposal.microstep_embedding.weight", "proposal.network.0.weight", "proposal.network.0.bias",
            "proposal.network.1.weight", "proposal.network.1.bias", "proposal.network.3.weight", "proposal.network.3.bias",
            "tension.microstep_embedding.weight", "tension.network.0.weight", "tension.network.0.bias",
            "tension.network.1.weight", "tension.network.1.bias", "tension.network.3.weight", "tension.network.3.bias",
            "readout.query.0.weight", "readout.query.0.bias", "readout.query.2.weight", "readout.query.2.bias",
            "readout.output.0.weight", "readout.output.0.bias", "readout.output.1.weight", "readout.output.1.bias",
            "readout.output.3.weight", "readout.output.3.bias",
        }
        missing = sorted(required - self.parameters.keys())
        if missing:
            raise ValueError("Missing Ten-SON parameters: " + ", ".join(missing))

    @staticmethod
    def _view(tensor: DeviceTensor, shape: tuple[int, ...]) -> DeviceTensor:
        if int(np.prod(shape)) != tensor.size:
            raise ValueError("reshape changes the tensor element count")
        return DeviceTensor(tensor.runtime, shape, tensor.dtype, tensor.buffer)

    def _p(self, name: str) -> DeviceTensor:
        return self.parameters[name]

    def _linear(self, inputs: DeviceTensor, prefix: str, output_shape: tuple[int, ...]) -> DeviceTensor:
        flat = self._view(inputs, (inputs.size // inputs.shape[-1], inputs.shape[-1]))
        result, _ = linear_device(
            self.runtime, flat, self._p(prefix + ".weight"), self._p(prefix + ".bias"), repetitions=1
        )
        return self._view(result, output_shape)

    def initial_state(self, batch_size: int) -> DeviceTensor:
        initial = self._p("initial_workspace").to_numpy()
        # This is the only state initialization transfer; forward intermediates remain resident.
        return DeviceTensor.from_numpy(self.runtime, np.broadcast_to(initial, (batch_size,) + initial.shape).copy())

    def _summary(self, workspace: DeviceTensor) -> DeviceTensor:
        mean, maximum = workspace_reduce_device(self.runtime, workspace)
        combined = concatenate_columns_device(self.runtime, mean, maximum)
        return layer_norm_device(
            self.runtime, combined, self._p("summary_norm.weight"), self._p("summary_norm.bias")
        )

    def _proposal(
        self, selected_state: DeviceTensor, selected_keys: DeviceTensor,
        token: DeviceTensor, summary: DeviceTensor, microstep: int,
    ) -> DeviceTensor:
        batch, selected, slot_dim = selected_state.shape
        rows = batch * selected
        state2 = self._view(selected_state, (rows, slot_dim))
        keys2 = self._view(selected_keys, (rows, self.config.key_dim))
        token2 = broadcast_rows_device(self.runtime, token, selected)
        summary2 = broadcast_rows_device(self.runtime, summary, selected)
        step = broadcast_table_row_device(self.runtime, self._p("proposal.microstep_embedding.weight"), microstep, rows)
        features = concatenate_columns_device(self.runtime, state2, keys2)
        for addition in (token2, summary2, step):
            features = concatenate_columns_device(self.runtime, features, addition)
        normal = layer_norm_device(self.runtime, features, self._p("proposal.network.0.weight"), self._p("proposal.network.0.bias"))
        hidden = self._linear(normal, "proposal.network.1", (rows, self.config.proposal_hidden))
        hidden = gelu_device(self.runtime, hidden)
        result = self._linear(hidden, "proposal.network.3", (rows, slot_dim))
        return self._view(result, (batch, selected, slot_dim))

    def _tension(
        self, selected_state: DeviceTensor, proposal: DeviceTensor, selected_keys: DeviceTensor,
        token: DeviceTensor, summary: DeviceTensor, selected_scores: DeviceTensor, microstep: int,
    ) -> tuple[DeviceTensor, DeviceTensor]:
        batch, selected, slot_dim = selected_state.shape
        rows = batch * selected
        state2 = self._view(selected_state, (rows, slot_dim))
        proposal2 = self._view(proposal, (rows, slot_dim))
        keys2 = self._view(selected_keys, (rows, self.config.key_dim))
        token2 = broadcast_rows_device(self.runtime, token, selected)
        summary2 = broadcast_rows_device(self.runtime, summary, selected)
        score2 = self._view(selected_scores, (rows, 1))
        delta_norm = row_delta_norm_device(self.runtime, proposal2, state2)
        step = broadcast_table_row_device(self.runtime, self._p("tension.microstep_embedding.weight"), microstep, rows)
        features = concatenate_columns_device(self.runtime, state2, proposal2)
        for addition in (keys2, token2, summary2, score2, delta_norm, step):
            features = concatenate_columns_device(self.runtime, features, addition)
        normal = layer_norm_device(self.runtime, features, self._p("tension.network.0.weight"), self._p("tension.network.0.bias"))
        hidden = self._linear(normal, "tension.network.1", (rows, self.config.tension_hidden))
        hidden = gelu_device(self.runtime, hidden)
        logits = self._linear(hidden, "tension.network.3", (rows, 1))
        tension, _ = sigmoid_device(self.runtime, logits, repetitions=1)
        return self._view(logits, (batch, selected, 1)), self._view(tension, (batch, selected, 1))

    def _readout(self, workspace: DeviceTensor, token: DeviceTensor, slot_tension: DeviceTensor):
        mean, maximum = workspace_reduce_device(self.runtime, workspace)
        query_input = concatenate_columns_device(self.runtime, token, mean)
        query_input = concatenate_columns_device(self.runtime, query_input, maximum)
        query = self._linear(query_input, "readout.query.0", (workspace.shape[0], self.config.readout_hidden))
        query = gelu_device(self.runtime, query)
        query = self._linear(query, "readout.query.2", (workspace.shape[0], self.config.slot_dim))
        scores = batched_dot_device(self.runtime, query, workspace, scale=1.0 / math.sqrt(self.config.slot_dim))
        adjusted = add_tension_penalty_device(self.runtime, scores, slot_tension)
        weights = softmax_device(self.runtime, adjusted)
        context = weighted_sum_device(self.runtime, weights, workspace)
        context = layer_norm_device(
            self.runtime, context, self._p("readout.output.0.weight"), self._p("readout.output.0.bias")
        )
        readout = self._linear(context, "readout.output.1", (workspace.shape[0], self.config.readout_hidden))
        readout = gelu_device(self.runtime, readout)
        readout = self._linear(readout, "readout.output.3", (workspace.shape[0], self.config.readout_hidden))
        return readout, weights

    def forward_token(self, token_embedding: DeviceTensor, workspace: DeviceTensor):
        if token_embedding.runtime is not self.runtime or workspace.runtime is not self.runtime:
            raise ValueError("inputs belong to a different runtime")
        if token_embedding.dtype != np.dtype(np.float32) or workspace.dtype != np.dtype(np.float32):
            raise ValueError("Ten-SON bridge inputs must use float32")
        batch = token_embedding.shape[0]
        if token_embedding.shape != (batch, self.config.embed_dim):
            raise ValueError("token_embedding shape is incorrect")
        if workspace.shape != (batch, self.config.num_slots, self.config.slot_dim):
            raise ValueError("workspace shape is incorrect")

        input_summary = self._summary(workspace)
        route_input = concatenate_columns_device(self.runtime, token_embedding, input_summary)
        query = self._linear(route_input, "router.query_network.0", (batch, self.config.key_dim))
        query = layer_norm_device(
            self.runtime, query, self._p("router.query_network.1.weight"), self._p("router.query_network.1.bias")
        )
        query, _ = tanh_device(self.runtime, query, repetitions=1)
        query = self._linear(query, "router.query_network.3", (batch, self.config.key_dim))
        routing_scores, _ = linear_device(self.runtime, query, self.slot_keys_linear, self.route_bias, repetitions=1)
        routing_scores = scale_device(self.runtime, routing_scores, 1.0 / math.sqrt(self.config.key_dim))
        routing_probabilities = softmax_device(self.runtime, routing_scores)
        selected_scores, selected_indices = topk_device(self.runtime, routing_scores, self.config.top_k)
        selected_keys = gather_slots_device(self.runtime, self._p("slot_keys"), selected_indices, shared=True)
        slot_tension = DeviceTensor.from_numpy(
            self.runtime, np.zeros((batch, self.config.num_slots, 1), dtype=np.float32)
        )

        diagnostics: dict[str, object] = {
            "input_summary": input_summary,
            "routing_scores": routing_scores,
            "routing_probabilities": routing_probabilities,
            "selected_indices": selected_indices,
            "selected_scores": selected_scores,
            "selected_states": [], "proposals": [], "pre_tensions": [],
            "updated_states": [], "workspaces": [], "post_tensions": [],
        }
        for microstep in range(self.config.microsteps):
            selected_state = gather_slots_device(self.runtime, workspace, selected_indices)
            summary = self._summary(workspace)
            proposal = self._proposal(selected_state, selected_keys, token_embedding, summary, microstep)
            _, tension = self._tension(
                selected_state, proposal, selected_keys, token_embedding, summary, selected_scores, microstep
            )
            rows = batch * self.config.top_k
            updated2 = tension_update_broadcast_device(
                self.runtime,
                self._view(selected_state, (rows, self.config.slot_dim)),
                self._view(proposal, (rows, self.config.slot_dim)),
                self._view(tension, (rows, 1)),
            )
            updated = self._view(updated2, selected_state.shape)
            workspace = scatter_slots_device(self.runtime, workspace, selected_indices, updated)
            slot_tension = scatter_slots_device(self.runtime, slot_tension, selected_indices, tension)
            post_summary = self._summary(workspace)
            _, post_tension = self._tension(
                updated, proposal, selected_keys, token_embedding, post_summary, selected_scores, microstep
            )
            diagnostics["selected_states"].append(selected_state)
            diagnostics["proposals"].append(proposal)
            diagnostics["pre_tensions"].append(tension)
            diagnostics["updated_states"].append(updated)
            diagnostics["workspaces"].append(workspace)
            diagnostics["post_tensions"].append(post_tension)

        readout, readout_weights = self._readout(workspace, token_embedding, slot_tension)
        diagnostics["slot_tension"] = slot_tension
        diagnostics["readout_weights"] = readout_weights
        diagnostics["readout_vector"] = readout
        return workspace, readout, diagnostics
