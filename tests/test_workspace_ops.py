from __future__ import annotations

import numpy as np

from tensionforge import DeviceTensor, TensionForgeRuntime
from tensionforge.ops import (
    add_tension_penalty_device,
    batched_dot_device,
    broadcast_rows_device,
    broadcast_table_row_device,
    concatenate_columns_device,
    gather_slots_device,
    gelu_device,
    layer_norm_device,
    row_delta_norm_device,
    scale_device,
    scatter_slots_device,
    softmax_device,
    tension_update_broadcast_device,
    topk_device,
    weighted_sum_device,
    workspace_reduce_device,
)


def _runtime() -> TensionForgeRuntime:
    return TensionForgeRuntime(profiling=False)


def test_workspace_mean_and_max_match_numpy() -> None:
    runtime = _runtime()
    rng = np.random.default_rng(300)
    values = rng.normal(size=(3, 7, 5)).astype(np.float32)
    mean, maximum = workspace_reduce_device(runtime, DeviceTensor.from_numpy(runtime, values))
    np.testing.assert_allclose(mean.to_numpy(), values.mean(axis=1), rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(maximum.to_numpy(), values.max(axis=1))


def test_layer_norm_matches_numpy() -> None:
    runtime = _runtime()
    rng = np.random.default_rng(301)
    values = rng.normal(size=(4, 3, 11)).astype(np.float32)
    weight = rng.normal(size=11).astype(np.float32)
    bias = rng.normal(size=11).astype(np.float32)
    result = layer_norm_device(
        runtime,
        DeviceTensor.from_numpy(runtime, values),
        DeviceTensor.from_numpy(runtime, weight),
        DeviceTensor.from_numpy(runtime, bias),
    ).to_numpy()
    mean = values.mean(axis=-1, keepdims=True)
    variance = ((values - mean) ** 2).mean(axis=-1, keepdims=True)
    expected = (values - mean) / np.sqrt(variance + np.float32(1e-5)) * weight + bias
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


def test_softmax_matches_numpy() -> None:
    runtime = _runtime()
    rng = np.random.default_rng(302)
    values = rng.normal(size=(5, 13)).astype(np.float32)
    result = softmax_device(runtime, DeviceTensor.from_numpy(runtime, values)).to_numpy()
    shifted = values - values.max(axis=-1, keepdims=True)
    expected = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(result, expected, rtol=3e-6, atol=3e-6)


def test_deterministic_topk_matches_non_tied_reference() -> None:
    runtime = _runtime()
    values = np.array(
        [[0.2, -1.0, 3.5, 0.7, 2.1], [8.0, 4.0, -2.0, 7.0, 1.0]],
        dtype=np.float32,
    )
    selected, indices = topk_device(runtime, DeviceTensor.from_numpy(runtime, values), 3)
    expected_indices = np.argsort(-values, axis=-1)[:, :3].astype(np.int32)
    np.testing.assert_array_equal(indices.to_numpy(), expected_indices)
    np.testing.assert_array_equal(selected.to_numpy(), np.take_along_axis(values, expected_indices, axis=1))


def test_batched_gather_and_scatter_match_numpy() -> None:
    runtime = _runtime()
    values = np.arange(2 * 5 * 3, dtype=np.float32).reshape(2, 5, 3)
    indices = np.array([[4, 1], [0, 3]], dtype=np.int32)
    gathered = gather_slots_device(
        runtime,
        DeviceTensor.from_numpy(runtime, values),
        DeviceTensor.from_numpy(runtime, indices),
    )
    expected_gather = np.take_along_axis(values, indices[:, :, None], axis=1)
    np.testing.assert_array_equal(gathered.to_numpy(), expected_gather)

    updates = (expected_gather + 100).astype(np.float32)
    scattered = scatter_slots_device(
        runtime,
        DeviceTensor.from_numpy(runtime, values),
        DeviceTensor.from_numpy(runtime, indices),
        DeviceTensor.from_numpy(runtime, updates),
    ).to_numpy()
    expected_scatter = values.copy()
    for batch in range(2):
        expected_scatter[batch, indices[batch]] = updates[batch]
    np.testing.assert_array_equal(scattered, expected_scatter)


def test_shared_gather_concatenation_and_broadcasts_match_numpy() -> None:
    runtime = _runtime()
    shared = np.arange(15, dtype=np.float32).reshape(5, 3)
    indices = np.array([[4, 1], [0, 3]], dtype=np.int32)
    gathered = gather_slots_device(
        runtime, DeviceTensor.from_numpy(runtime, shared),
        DeviceTensor.from_numpy(runtime, indices), shared=True,
    ).to_numpy()
    np.testing.assert_array_equal(gathered, shared[indices])

    left = np.arange(6, dtype=np.float32).reshape(2, 3)
    right = np.arange(4, dtype=np.float32).reshape(2, 2)
    joined = concatenate_columns_device(
        runtime, DeviceTensor.from_numpy(runtime, left), DeviceTensor.from_numpy(runtime, right)
    ).to_numpy()
    np.testing.assert_array_equal(joined, np.concatenate([left, right], axis=1))
    repeated = broadcast_rows_device(runtime, DeviceTensor.from_numpy(runtime, left), 3).to_numpy()
    np.testing.assert_array_equal(repeated, np.repeat(left[:, None, :], 3, axis=1).reshape(6, 3))
    table = np.arange(12, dtype=np.float32).reshape(4, 3)
    table_rows = broadcast_table_row_device(
        runtime, DeviceTensor.from_numpy(runtime, table), 2, 5
    ).to_numpy()
    np.testing.assert_array_equal(table_rows, np.broadcast_to(table[2], (5, 3)))


def test_gelu_scale_delta_norm_and_broadcast_update_match_numpy() -> None:
    runtime = _runtime()
    rng = np.random.default_rng(303)
    state = rng.normal(size=(7, 9)).astype(np.float32)
    proposal = rng.normal(size=(7, 9)).astype(np.float32)
    gate = rng.uniform(size=(7, 1)).astype(np.float32)
    state_device = DeviceTensor.from_numpy(runtime, state)
    proposal_device = DeviceTensor.from_numpy(runtime, proposal)
    scaled = scale_device(runtime, state_device, 0.125).to_numpy()
    np.testing.assert_allclose(scaled, state * np.float32(0.125), rtol=1e-7, atol=1e-7)
    norms = row_delta_norm_device(runtime, proposal_device, state_device).to_numpy()
    np.testing.assert_allclose(norms, np.linalg.norm(proposal - state, axis=1, keepdims=True), rtol=2e-6, atol=2e-6)
    updated = tension_update_broadcast_device(
        runtime, state_device, proposal_device, DeviceTensor.from_numpy(runtime, gate)
    ).to_numpy()
    np.testing.assert_allclose(updated, state + gate * (proposal - state), rtol=2e-6, atol=2e-6)

    values = rng.normal(size=(31,)).astype(np.float32)
    gelu = gelu_device(runtime, DeviceTensor.from_numpy(runtime, values)).to_numpy()
    # NumPy has no erf ufunc in the supported baseline; compare with PyTorch's exact GELU when available.
    try:
        import torch
    except ModuleNotFoundError:
        import math
        expected = np.array([0.5 * x * (1.0 + math.erf(float(x) / np.sqrt(2.0))) for x in values], dtype=np.float32)
    else:
        expected = torch.nn.functional.gelu(torch.from_numpy(values)).numpy()
    np.testing.assert_allclose(gelu, expected, rtol=3e-6, atol=3e-6)


def test_batched_dot_penalty_and_weighted_sum_match_numpy() -> None:
    runtime = _runtime()
    rng = np.random.default_rng(304)
    query = rng.normal(size=(3, 5)).astype(np.float32)
    slots = rng.normal(size=(3, 7, 5)).astype(np.float32)
    scores = batched_dot_device(
        runtime, DeviceTensor.from_numpy(runtime, query), DeviceTensor.from_numpy(runtime, slots), scale=0.25
    ).to_numpy()
    expected_scores = np.einsum("bd,bnd->bn", query, slots) * np.float32(0.25)
    np.testing.assert_allclose(scores, expected_scores, rtol=2e-6, atol=2e-6)
    tension = rng.uniform(-0.2, 1.2, size=(3, 7, 1)).astype(np.float32)
    adjusted = add_tension_penalty_device(
        runtime, DeviceTensor.from_numpy(runtime, scores), DeviceTensor.from_numpy(runtime, tension)
    ).to_numpy()
    expected_adjusted = scores + np.log1p(-np.clip(tension.squeeze(-1), 0.0, 0.999))
    np.testing.assert_allclose(adjusted, expected_adjusted, rtol=3e-6, atol=3e-6)
    weights = rng.uniform(size=(3, 7)).astype(np.float32)
    context = weighted_sum_device(
        runtime, DeviceTensor.from_numpy(runtime, weights), DeviceTensor.from_numpy(runtime, slots)
    ).to_numpy()
    np.testing.assert_allclose(context, np.einsum("bn,bnd->bd", weights, slots), rtol=3e-6, atol=3e-6)
