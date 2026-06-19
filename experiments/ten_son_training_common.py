from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.ten_son_forward_parity import SOURCE, _load_torch_source, development_config
from experiments.ten_son_training_step_parity import arr, dn, full_parameters
from tensionforge.models.ten_son_training_bridge import TenSonTrainingBridge
from tensionforge.ops import cross_entropy_device, training_auxiliary_loss_device


@dataclass(frozen=True)
class HostBatch:
    inputs: np.ndarray
    targets: np.ndarray
    loss_mask: np.ndarray
    event_mask: np.ndarray
    sha256: str


@dataclass(frozen=True)
class DeviceBatch:
    token_ids: tuple[object, ...]
    supervised_targets: object
    supervised_position: int
    host: HostBatch


def load_reference_modules():
    torch, ModelConfig, _ = _load_torch_source()
    sys.path.insert(0, str(SOURCE))
    from tension_lm.config import TaskConfig, TrainConfig
    from tension_lm.model.tension_model import TensionModel
    from tension_lm.tasks.base import Batch
    from tension_lm.tasks.delayed_recall import DelayedRecallTask
    from tension_lm.training.losses import slot_usage_balance_loss, tension_balance_loss
    return torch, ModelConfig, TaskConfig, TrainConfig, TensionModel, Batch, DelayedRecallTask, slot_usage_balance_loss, tension_balance_loss


def benchmark_configuration(name: str):
    torch, ModelConfig, TaskConfig, TrainConfig, *_ = load_reference_modules()
    del torch
    if name == "development":
        model = development_config(ModelConfig)
        task = TaskConfig(name="delayed_recall", vocab_size=8, seq_len=8, delay=4)
        batch_size = 2
    elif name == "cpu_validation":
        source = json.loads((SOURCE / "experiments" / "milestone1_v1.json").read_text())
        values = source["model"]
        task_values = source["tasks"]["delayed_recall"]
        model = ModelConfig(vocab_size=task_values["vocab_size"], output_size=task_values["vocab_size"], **values)
        task = TaskConfig(name="delayed_recall", **task_values)
        batch_size = source["training"]["batch_size"]
    else:
        raise ValueError(f"unknown configuration: {name}")
    train = TrainConfig(
        task=task, model=model, seed=6201, batch_size=batch_size, steps=20,
        learning_rate=3e-4, weight_decay=.01, grad_clip_norm=1.0,
        tension_balance_weight=.001, slot_usage_weight=.001, device="cpu",
    )
    return model, task, train


def generate_host_batches(train_config, count: int, seed: int) -> list[HostBatch]:
    torch, *_, DelayedRecallTask, _, _ = load_reference_modules()
    task = DelayedRecallTask(train_config.task.vocab_size, train_config.task.delay)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    batches = []
    for _ in range(count):
        batch = task.generate_batch(train_config.batch_size, train_config.task.seq_len, "cpu", generator)
        arrays = (
            batch.inputs.detach().numpy().astype(np.int32),
            batch.targets.detach().numpy().astype(np.int32),
            batch.loss_mask.detach().numpy().astype(np.bool_),
            batch.event_mask.detach().numpy().astype(np.bool_),
        )
        digest = hashlib.sha256(b"".join(np.ascontiguousarray(value).tobytes() for value in arrays)).hexdigest()
        batches.append(HostBatch(*arrays, digest))
    return batches


def torch_batch(host: HostBatch):
    torch, _, _, _, _, Batch, *_ = load_reference_modules()
    return Batch(
        inputs=torch.from_numpy(host.inputs.astype(np.int64)),
        targets=torch.from_numpy(host.targets.astype(np.int64)),
        loss_mask=torch.from_numpy(host.loss_mask),
        event_mask=torch.from_numpy(host.event_mask),
    )


def upload_batch(runtime, host: HostBatch) -> DeviceBatch:
    positions = np.flatnonzero(host.loss_mask[0])
    if positions.size != 1 or not np.all(host.loss_mask[:, positions[0]]):
        raise ValueError("delayed-recall bridge expects one shared supervised position")
    position = int(positions[0])
    return DeviceBatch(
        tuple(dn(runtime, host.inputs[:, step], np.int32) for step in range(host.inputs.shape[1])),
        dn(runtime, host.targets[:, position], np.int32), position, host,
    )


def build_models(model_config, initial_seed: int, runtime):
    torch, _, _, _, TensionModel, *_ = load_reference_modules()
    torch.manual_seed(initial_seed)
    model = TensionModel(model_config).train()
    initial = full_parameters(model)
    bridge = TenSonTrainingBridge(runtime, model_config, initial)
    bridge.initialize_optimizer_state()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    return model, optimizer, bridge, initial


def torch_compute(model, host: HostBatch, train_config, *, backward: bool):
    torch, *_, slot_usage_balance_loss, tension_balance_loss = load_reference_modules()
    batch = torch_batch(host)
    if backward:
        model.zero_grad(set_to_none=True)
    context = torch.enable_grad() if backward else torch.no_grad()
    with context:
        output = model(batch.inputs, return_diagnostics=True)
        logits = output["logits"]
        mask = batch.loss_mask.bool()
        task_loss = torch.nn.functional.cross_entropy(logits[mask], batch.targets[mask])
        diagnostics = output["diagnostics"]
        balance = tension_balance_loss(diagnostics["pre_tension"])
        usage = slot_usage_balance_loss(diagnostics["selected_indices"], model.config.num_slots)
        loss = task_loss + train_config.tension_balance_weight * balance + train_config.slot_usage_weight * usage
    if backward:
        loss.backward()
    return loss, output, batch


def forge_compute(bridge: TenSonTrainingBridge, batch: DeviceBatch, *, backward: bool, train_config=None):
    bridge.begin_step()
    workspace = bridge.initial_state(batch.host.inputs.shape[0])
    logits = []
    diagnostics = []
    for token_ids in batch.token_ids:
        embedding = bridge.embed(token_ids)
        workspace, readout, diagnostic = bridge.forward_token(embedding, workspace)
        logits.append(bridge.classify(readout))
        diagnostics.append(diagnostic)
    losses, grad_logits = cross_entropy_device(
        bridge.runtime, logits[batch.supervised_position].tensor, batch.supervised_targets,
    )
    if train_config is not None:
        training_auxiliary_loss_device(
            bridge.runtime,
            [tension.tensor for diagnostic in diagnostics for tension in diagnostic["pre_tension"]],
            [diagnostic["selected_indices"] for diagnostic in diagnostics],
            train_config.model.num_slots,
            train_config.tension_balance_weight,
            train_config.slot_usage_weight,
        )
    if backward:
        bridge.backward([(logits[batch.supervised_position], grad_logits)])
    return losses, logits, workspace, diagnostics


def forge_loss_value(losses, diagnostics, train_config):
    values = losses.to_numpy()
    pre = np.concatenate([
        tension.tensor.to_numpy().ravel()
        for diagnostic in diagnostics for tension in diagnostic["pre_tension"]
    ])
    balance = max(.05 - float(pre.mean()), 0.0) + max(float(pre.mean()) - .95, 0.0)
    selected = np.concatenate([diagnostic["selected_indices"].to_numpy().ravel() for diagnostic in diagnostics])
    counts = np.bincount(selected, minlength=train_config.model.num_slots).astype(np.float32)
    probabilities = counts / max(float(counts.sum()), 1.0)
    usage = float(np.mean((probabilities - 1.0 / train_config.model.num_slots) ** 2))
    return float(values.sum()) + train_config.tension_balance_weight * balance + train_config.slot_usage_weight * usage
