from __future__ import annotations

import numpy as np
import pytest

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


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return (
        1.0
        / (
            1.0
            + np.exp(-values)
        )
    ).astype(np.float32)


def test_fused_tension_matches_numpy() -> None:
    runtime = TensionForgeRuntime(
        profiling=True,
    )

    rng = np.random.default_rng(401)

    batch_size = 37
    feature_count = 29
    hidden_size = 23

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

    output_gpu, metadata = (
        fused_tension_linear_device(
            runtime,
            DeviceTensor.from_numpy(
                runtime,
                features,
            ),
            DeviceTensor.from_numpy(
                runtime,
                state,
            ),
            DeviceTensor.from_numpy(
                runtime,
                proposal_weights,
            ),
            DeviceTensor.from_numpy(
                runtime,
                proposal_bias,
            ),
            DeviceTensor.from_numpy(
                runtime,
                gate_weights,
            ),
            DeviceTensor.from_numpy(
                runtime,
                gate_bias,
            ),
            repetitions=3,
        )
    )

    proposal = np.tanh(
        features @ proposal_weights
        + proposal_bias
    ).astype(np.float32)

    gate = _sigmoid(
        features @ gate_weights
        + gate_bias
    )

    expected = (
        state
        + gate
        * (
            proposal
            - state
        )
    ).astype(np.float32)

    np.testing.assert_allclose(
        output_gpu.to_numpy(),
        expected,
        rtol=5e-4,
        atol=5e-4,
    )

    assert (
        metadata[
            "kernel_launches_per_iteration"
        ]
        == 1
    )

    assert (
        metadata[
            "unfused_launches_per_iteration"
        ]
        == 5
    )


def test_fused_matches_unfused_runtime() -> None:
    runtime = TensionForgeRuntime(
        profiling=True,
    )

    rng = np.random.default_rng(402)

    batch_size = 64
    feature_count = 48
    hidden_size = 32

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

    gate_weights_gpu = DeviceTensor.from_numpy(
        runtime,
        gate_weights,
    )

    proposal_bias_gpu = (
        DeviceTensor.from_numpy(
            runtime,
            proposal_bias,
        )
    )

    gate_bias_gpu = DeviceTensor.from_numpy(
        runtime,
        gate_bias,
    )

    proposal_logits_gpu, _ = linear_device(
        runtime,
        features_gpu,
        proposal_weights_gpu,
        proposal_bias_gpu,
    )

    proposal_gpu, _ = tanh_device(
        runtime,
        proposal_logits_gpu,
    )

    gate_logits_gpu, _ = linear_device(
        runtime,
        features_gpu,
        gate_weights_gpu,
        gate_bias_gpu,
    )

    gate_gpu, _ = sigmoid_device(
        runtime,
        gate_logits_gpu,
    )

    unfused_output_gpu, _ = (
        tension_update_device(
            runtime,
            state_gpu,
            proposal_gpu,
            gate_gpu,
        )
    )

    fused_output_gpu, _ = (
        fused_tension_linear_device(
            runtime,
            features_gpu,
            state_gpu,
            proposal_weights_gpu,
            proposal_bias_gpu,
            gate_weights_gpu,
            gate_bias_gpu,
        )
    )

    np.testing.assert_allclose(
        fused_output_gpu.to_numpy(),
        unfused_output_gpu.to_numpy(),
        rtol=5e-4,
        atol=5e-4,
    )


def test_fused_tension_reuses_output() -> None:
    runtime = TensionForgeRuntime(
        profiling=True,
    )

    batch_size = 32
    feature_count = 16
    hidden_size = 8

    features = DeviceTensor.from_numpy(
        runtime,
        np.ones(
            (batch_size, feature_count),
            dtype=np.float32,
        ),
    )

    state = DeviceTensor.from_numpy(
        runtime,
        np.zeros(
            (batch_size, hidden_size),
            dtype=np.float32,
        ),
    )

    weights = DeviceTensor.from_numpy(
        runtime,
        np.zeros(
            (feature_count, hidden_size),
            dtype=np.float32,
        ),
    )

    bias = DeviceTensor.from_numpy(
        runtime,
        np.zeros(
            (hidden_size,),
            dtype=np.float32,
        ),
    )

    output = DeviceTensor.empty(
        runtime,
        (batch_size, hidden_size),
        dtype=np.float32,
    )

    returned, metadata = (
        fused_tension_linear_device(
            runtime,
            features,
            state,
            weights,
            bias,
            weights,
            bias,
            output=output,
            repetitions=2,
        )
    )

    assert returned is output
    assert metadata["output_buffer_reused"] is True

    np.testing.assert_allclose(
        output.to_numpy(),
        np.zeros(
            (batch_size, hidden_size),
            dtype=np.float32,
        ),
        rtol=1e-6,
        atol=1e-6,
    )


def test_fused_tension_rejects_bad_weights() -> None:
    runtime = TensionForgeRuntime(
        profiling=False,
    )

    features = DeviceTensor.empty(
        runtime,
        (8, 16),
        dtype=np.float32,
    )

    state = DeviceTensor.empty(
        runtime,
        (8, 12),
        dtype=np.float32,
    )

    bad_weights = DeviceTensor.empty(
        runtime,
        (15, 12),
        dtype=np.float32,
    )

    bias = DeviceTensor.empty(
        runtime,
        (12,),
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="proposal_weights shape",
    ):
        fused_tension_linear_device(
            runtime,
            features,
            state,
            bad_weights,
            bias,
            bad_weights,
            bias,
        )
