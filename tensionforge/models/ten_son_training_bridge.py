from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Mapping

import numpy as np

from tensionforge.models.ten_son_bridge import LINEAR_WEIGHTS, TenSonBridgeConfig
from tensionforge.ops import (
    add_device, add_tension_penalty_device, batched_dot_backward_device,
    adamw_update_device,
    batched_dot_device, broadcast_rows_backward_device, broadcast_rows_device,
    broadcast_table_row_device, concatenate_backward_device, concatenate_columns_device,
    clip_gradients_device,
    gather_backward_device, gather_slots_device, gelu_backward_device, gelu_device,
    embedding_backward_device, embedding_forward_device,
    layer_norm_backward_device, layer_norm_device, linear_backward_device, linear_device,
    row_delta_norm_backward_device, row_delta_norm_device, route_scores_backward_device,
    route_scores_device, scale_device, scatter_backward_device, scatter_slots_device,
    sigmoid_backward_device, sigmoid_device, softmax_device, table_row_backward_device,
    tanh_backward_device, tanh_device, tension_penalty_backward_device,
    tension_update_broadcast_backward_device, tension_update_broadcast_device,
    topk_backward_device, topk_device, weighted_sum_backward_device,
    weighted_sum_device, workspace_fill_device, workspace_reduce_backward_device,
    workspace_reduce_device,
)
from tensionforge.runtime import TensionForgeRuntime
from tensionforge.tensor import DeviceTensor


@dataclass(eq=False)
class TrainingValue:
    tensor: DeviceTensor
    name: str = ""
    grad: DeviceTensor | None = None


class WorkspaceBackwardProgram:
    """A bounded reverse program containing only operations in TensionWorkspace."""

    def __init__(self, runtime: TensionForgeRuntime):
        self.runtime = runtime
        self.records: list[tuple[TrainingValue, Callable[[DeviceTensor], list[tuple[TrainingValue, DeviceTensor]]]]] = []

    @staticmethod
    def tensor_view(tensor: DeviceTensor, shape: tuple[int, ...]) -> DeviceTensor:
        if int(np.prod(shape)) != tensor.size:
            raise ValueError("reshape changes element count")
        return DeviceTensor(tensor.runtime, shape, tensor.dtype, tensor.buffer)

    def op(self, tensor, parents, backward, name=""):
        out = TrainingValue(tensor, name)
        self.records.append((out, lambda g: list(zip(parents, backward(g)))))
        return out

    def reshape(self, x, shape):
        out = TrainingValue(self.tensor_view(x.tensor, shape))
        self.records.append((out, lambda g: [(x, self.tensor_view(g, x.tensor.shape))]))
        return out

    def concat(self, a, b):
        y = concatenate_columns_device(self.runtime, a.tensor, b.tensor)
        return self.op(y, [a, b], lambda g: concatenate_backward_device(self.runtime, g, a.tensor.shape[1]))

    def broadcast(self, x, repeats):
        y = broadcast_rows_device(self.runtime, x.tensor, repeats)
        return self.op(y, [x], lambda g: [broadcast_rows_backward_device(self.runtime, g, x.tensor.shape[0], repeats)])

    def table_row(self, x, row, rows):
        y = broadcast_table_row_device(self.runtime, x.tensor, row, rows)
        return self.op(y, [x], lambda g: [table_row_backward_device(self.runtime, g, x.tensor.shape, row)])

    def linear(self, x, w, b):
        y, _ = linear_device(self.runtime, x.tensor, w.tensor, b.tensor, repetitions=1)
        def back(g):
            gx, gw, gb, _ = linear_backward_device(self.runtime, x.tensor, w.tensor, g, repetitions=1)
            return gx, gw, gb
        return self.op(y, [x, w, b], back)

    def layer_norm(self, x, w, b):
        y = layer_norm_device(self.runtime, x.tensor, w.tensor, b.tensor)
        return self.op(y, [x, w, b], lambda g: layer_norm_backward_device(self.runtime, x.tensor, w.tensor, g))

    def gelu(self, x):
        return self.op(gelu_device(self.runtime, x.tensor), [x], lambda g: [gelu_backward_device(self.runtime, x.tensor, g)])

    def tanh(self, x):
        y, _ = tanh_device(self.runtime, x.tensor, repetitions=1)
        return self.op(y, [x], lambda g: [tanh_backward_device(self.runtime, y, g, repetitions=1)[0]])

    def sigmoid(self, x):
        y, _ = sigmoid_device(self.runtime, x.tensor, repetitions=1)
        return self.op(y, [x], lambda g: [sigmoid_backward_device(self.runtime, y, g, repetitions=1)[0]])

    def reduce(self, x):
        mean, maximum = workspace_reduce_device(self.runtime, x.tensor)
        zero_mean = workspace_fill_device(self.runtime, mean.shape)
        zero_max = workspace_fill_device(self.runtime, maximum.shape)
        m = self.op(mean, [x], lambda g: [workspace_reduce_backward_device(self.runtime, x.tensor, g, zero_max)])
        q = self.op(maximum, [x], lambda g: [workspace_reduce_backward_device(self.runtime, x.tensor, zero_mean, g)])
        return m, q

    def softmax(self, x):
        from tensionforge.ops import softmax_backward_device
        y = softmax_device(self.runtime, x.tensor)
        return self.op(y, [x], lambda g: [softmax_backward_device(self.runtime, y, g)])

    def topk(self, x, k):
        values, indices = topk_device(self.runtime, x.tensor, k)
        return self.op(values, [x], lambda g: [topk_backward_device(self.runtime, g, indices, x.tensor.shape[1])]), indices

    def gather(self, x, indices, shared=False):
        y = gather_slots_device(self.runtime, x.tensor, indices, shared=shared)
        return self.op(y, [x], lambda g: [gather_backward_device(self.runtime, g, indices, x.tensor.shape, shared)])

    def scatter(self, x, indices, updates):
        y = scatter_slots_device(self.runtime, x.tensor, indices, updates.tensor)
        return self.op(y, [x, updates], lambda g: scatter_backward_device(self.runtime, g, indices))

    def norm(self, a, b):
        y = row_delta_norm_device(self.runtime, a.tensor, b.tensor)
        return self.op(y, [a, b], lambda g: row_delta_norm_backward_device(self.runtime, a.tensor, b.tensor, y, g))

    def tension_update(self, s, p, t):
        y = tension_update_broadcast_device(self.runtime, s.tensor, p.tensor, t.tensor)
        return self.op(y, [s, p, t], lambda g: tension_update_broadcast_backward_device(self.runtime, s.tensor, p.tensor, t.tensor, g))

    def route(self, q, k, scale):
        y = route_scores_device(self.runtime, q.tensor, k.tensor, scale)
        return self.op(y, [q, k], lambda g: route_scores_backward_device(self.runtime, q.tensor, k.tensor, g, scale))

    def dot(self, q, v, scale):
        y = batched_dot_device(self.runtime, q.tensor, v.tensor, scale=scale)
        return self.op(y, [q, v], lambda g: batched_dot_backward_device(self.runtime, q.tensor, v.tensor, g, scale))

    def penalty(self, s, t):
        y = add_tension_penalty_device(self.runtime, s.tensor, t.tensor)
        return self.op(y, [s, t], lambda g: tension_penalty_backward_device(self.runtime, t.tensor, g))

    def weighted(self, w, v):
        y = weighted_sum_device(self.runtime, w.tensor, v.tensor)
        return self.op(y, [w, v], lambda g: weighted_sum_backward_device(self.runtime, w.tensor, v.tensor, g))

    def embedding(self, ids, table):
        y = embedding_forward_device(self.runtime, ids, table.tensor)
        return self.op(y, [table], lambda g: [embedding_backward_device(self.runtime, ids, g, table.tensor.shape)])

    def backward(self, roots):
        for node, grad in roots:
            self.accumulate(node, grad)
        for node, backward in reversed(self.records):
            if node.grad is not None:
                for parent, grad in backward(node.grad):
                    self.accumulate(parent, grad)

    def accumulate(self, node, grad):
        if grad.shape != node.tensor.shape:
            grad = self.tensor_view(grad, node.tensor.shape)
        node.grad = grad if node.grad is None else add_device(self.runtime, node.grad, grad)


class TenSonTrainingBridge:
    """Explicit device-resident forward/backward program for the audited workspace."""

    def __init__(self, runtime: TensionForgeRuntime, config: TenSonBridgeConfig | object, parameters: Mapping[str, np.ndarray]):
        self.runtime = runtime
        self.config = config if isinstance(config, TenSonBridgeConfig) else TenSonBridgeConfig.from_object(config)
        self.program = WorkspaceBackwardProgram(runtime)
        self.parameters: dict[str, TrainingValue] = {}
        for original, raw in parameters.items():
            name = original.removeprefix("workspace.")
            value = np.asarray(raw, dtype=np.float32)
            if name in LINEAR_WEIGHTS or name == "head.weight":
                value = value.T
            self.parameters[name] = TrainingValue(DeviceTensor.from_numpy(runtime, np.ascontiguousarray(value)), name)
        self.tokens: list[TrainingValue] = []
        self.first_moments: dict[str, DeviceTensor] = {}
        self.second_moments: dict[str, DeviceTensor] = {}

    def p(self, name): return self.parameters[name]
    def view(self, x, shape): return self.program.reshape(x, shape)

    def linear(self, x, prefix, shape):
        flat = self.view(x, (x.tensor.size // x.tensor.shape[-1], x.tensor.shape[-1]))
        return self.view(self.program.linear(flat, self.p(prefix + ".weight"), self.p(prefix + ".bias")), shape)

    def summary(self, workspace):
        mean, maximum = self.program.reduce(workspace)
        return self.program.layer_norm(self.program.concat(mean, maximum), self.p("summary_norm.weight"), self.p("summary_norm.bias"))

    def initial_state(self, batch):
        parameter = self.p("initial_workspace")
        return self.view(self.program.broadcast(self.view(parameter, (1, parameter.tensor.size)), batch), (batch,) + parameter.tensor.shape)

    def begin_step(self):
        self.program = WorkspaceBackwardProgram(self.runtime)
        self.tokens = []
        for value in self.parameters.values():
            value.grad = None

    def initialize_optimizer_state(self):
        self.first_moments = {
            name: workspace_fill_device(self.runtime, value.tensor.shape)
            for name, value in self.parameters.items()
        }
        self.second_moments = {
            name: workspace_fill_device(self.runtime, value.tensor.shape)
            for name, value in self.parameters.items()
        }

    def reset_training_state(self, parameters: Mapping[str, np.ndarray]):
        for original, raw in parameters.items():
            name = original.removeprefix("workspace.")
            value = np.asarray(raw, dtype=np.float32)
            if name in LINEAR_WEIGHTS or name == "head.weight":
                value = value.T
            self.parameters[name].tensor.copy_from(np.ascontiguousarray(value))
        if not self.first_moments:
            self.initialize_optimizer_state()
        else:
            for tensor in (*self.first_moments.values(), *self.second_moments.values()):
                workspace_fill_device(self.runtime, tensor.shape, output=tensor)
        self.begin_step()

    def export_parameters(self):
        return {name: value.tensor.to_numpy() for name, value in self.parameters.items()}

    def export_optimizer_state(self):
        return {
            name: {"first_moment": self.first_moments[name].to_numpy(),
                   "second_moment": self.second_moments[name].to_numpy()}
            for name in self.parameters
        }

    def clip_gradients(self, max_norm=1.0):
        return clip_gradients_device(self.runtime, self.gradients().values(), max_norm)

    def optimizer_step(self, step, *, learning_rate=3e-4, beta1=.9, beta2=.999, epsilon=1e-8, weight_decay=.01):
        if not self.first_moments:
            self.initialize_optimizer_state()
        for name, value in self.parameters.items():
            if value.grad is None:
                continue
            adamw_update_device(
                self.runtime, value.tensor, value.grad,
                self.first_moments[name], self.second_moments[name], step=step,
                learning_rate=learning_rate, beta1=beta1, beta2=beta2,
                epsilon=epsilon, weight_decay=weight_decay,
            )

    def embed(self, token_ids: DeviceTensor):
        return self.program.embedding(token_ids, self.p("embedding.weight"))

    def classify(self, readout):
        return self.program.linear(readout, self.p("head.weight"), self.p("head.bias"))

    def proposal(self, state, keys, token, summary, step):
        batch, selected, dim = state.tensor.shape; rows = batch * selected
        features = self.program.concat(self.view(state, (rows, dim)), self.view(keys, (rows, self.config.key_dim)))
        for value in (self.program.broadcast(token, selected), self.program.broadcast(summary, selected), self.program.table_row(self.p("proposal.microstep_embedding.weight"), step, rows)):
            features = self.program.concat(features, value)
        value = self.program.layer_norm(features, self.p("proposal.network.0.weight"), self.p("proposal.network.0.bias"))
        value = self.program.gelu(self.linear(value, "proposal.network.1", (rows, self.config.proposal_hidden)))
        return self.view(self.linear(value, "proposal.network.3", (rows, dim)), (batch, selected, dim))

    def tension(self, state, proposal, keys, token, summary, scores, step):
        batch, selected, dim = state.tensor.shape; rows = batch * selected
        state2 = self.view(state, (rows, dim)); proposal2 = self.view(proposal, (rows, dim))
        features = self.program.concat(state2, proposal2)
        for value in (self.view(keys, (rows, self.config.key_dim)), self.program.broadcast(token, selected), self.program.broadcast(summary, selected), self.view(scores, (rows, 1)), self.program.norm(proposal2, state2), self.program.table_row(self.p("tension.microstep_embedding.weight"), step, rows)):
            features = self.program.concat(features, value)
        value = self.program.layer_norm(features, self.p("tension.network.0.weight"), self.p("tension.network.0.bias"))
        value = self.program.gelu(self.linear(value, "tension.network.1", (rows, self.config.tension_hidden)))
        return self.view(self.program.sigmoid(self.linear(value, "tension.network.3", (rows, 1))), (batch, selected, 1))

    def readout(self, workspace, token, slot_tension):
        mean, maximum = self.program.reduce(workspace)
        query = self.program.concat(self.program.concat(token, mean), maximum)
        query = self.program.gelu(self.linear(query, "readout.query.0", (workspace.tensor.shape[0], self.config.readout_hidden)))
        query = self.linear(query, "readout.query.2", (workspace.tensor.shape[0], self.config.slot_dim))
        scores = self.program.dot(query, workspace, 1 / math.sqrt(self.config.slot_dim))
        weights = self.program.softmax(self.program.penalty(scores, self.view(slot_tension, scores.tensor.shape)))
        context = self.program.weighted(weights, workspace)
        value = self.program.layer_norm(context, self.p("readout.output.0.weight"), self.p("readout.output.0.bias"))
        value = self.program.gelu(self.linear(value, "readout.output.1", (workspace.tensor.shape[0], self.config.readout_hidden)))
        return self.linear(value, "readout.output.3", (workspace.tensor.shape[0], self.config.readout_hidden)), weights

    def forward_token(self, token_tensor: DeviceTensor | TrainingValue, workspace: TrainingValue | DeviceTensor):
        token = token_tensor if isinstance(token_tensor, TrainingValue) else TrainingValue(token_tensor, "token")
        self.tokens.append(token)
        workspace = workspace if isinstance(workspace, TrainingValue) else TrainingValue(workspace, "workspace_input")
        batch = token.tensor.shape[0]
        query = self.linear(self.program.concat(token, self.summary(workspace)), "router.query_network.0", (batch, self.config.key_dim))
        query = self.program.tanh(self.program.layer_norm(query, self.p("router.query_network.1.weight"), self.p("router.query_network.1.bias")))
        query = self.linear(query, "router.query_network.3", (batch, self.config.key_dim))
        scores = self.program.route(query, self.p("slot_keys"), 1 / math.sqrt(self.config.key_dim))
        selected_scores, indices = self.program.topk(scores, self.config.top_k)
        selected_keys = self.program.gather(self.p("slot_keys"), indices, True)
        slot_tension = TrainingValue(workspace_fill_device(self.runtime, (batch, self.config.num_slots, 1)), "slot_tension_zero")
        pre_tensions = []
        for step in range(self.config.microsteps):
            state = self.program.gather(workspace, indices)
            proposal = self.proposal(state, selected_keys, token, self.summary(workspace), step)
            tension = self.tension(state, proposal, selected_keys, token, self.summary(workspace), selected_scores, step)
            rows = batch * self.config.top_k
            updated = self.view(self.program.tension_update(self.view(state, (rows, self.config.slot_dim)), self.view(proposal, (rows, self.config.slot_dim)), self.view(tension, (rows, 1))), state.tensor.shape)
            workspace = self.program.scatter(workspace, indices, updated)
            slot_tension = self.program.scatter(slot_tension, indices, tension)
            pre_tensions.append(tension)
        readout, weights = self.readout(workspace, token, slot_tension)
        diagnostics = {"selected_indices": indices, "routing_scores": scores.tensor, "routing_probabilities": softmax_device(self.runtime, scores.tensor), "pre_tension": pre_tensions, "slot_tension": slot_tension.tensor, "readout_weights": weights.tensor}
        return workspace, readout, diagnostics

    def backward(self, roots): self.program.backward(roots)
    def gradients(self): return {name: value.grad for name, value in self.parameters.items()}
