from __future__ import annotations

import numpy as np
import pytest

from tensionforge import (
    DeviceTensor,
    TensionForgeRuntime,
)
from tensionforge.ops import linear_backward_device


def test_linear_backward_matches_numpy() -> None:
    runtime = TensionForgeRuntime(
        profiling=True,
    )

    rng = np.random.default_rng(501)

    batch_size = 37
    input_features = 29
    output_features = 23

    inputs = rng.normal(
        size=(
            batch_size,
            input_features,
        ),
    ).astype(np.float32)

    weights = rng.normal(
        size=(
            input_features,
            output_features,
        ),
    ).astype(np.float32)

    grad_output = rng.normal(
        size=(
            batch_size,
            output_features,
        ),
    ).astype(np.float32)

    (
        grad_input_gpu,
        grad_weights_gpu,
        grad_bias_gpu,
        metadata,
    ) = linear_backward_device(
        runtime,
        DeviceTensor.from_numpy(
            runtime,
            inputs,
        ),
        DeviceTensor.from_numpy(
            runtime,
            weights,
        ),
        DeviceTensor.from_numpy(
            runtime,
            grad_output,
        ),
        repetitions=3,
    )

    expected_grad_input = (
        grad_output @ weights.T
    ).astype(np.float32)

    expected_grad_weights = (
        inputs.T @ grad_output
    ).astype(np.float32)

    expected_grad_bias = grad_output.sum(
        axis=0,
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        grad_input_gpu.to_numpy(),
        expected_grad_input,
        rtol=3e-4,
        atol=3e-3,
    )

    np.testing.assert_allclose(
        grad_weights_gpu.to_numpy(),
        expected_grad_weights,
        rtol=3e-4,
        atol=3e-3,
    )

    np.testing.assert_allclose(
        grad_bias_gpu.to_numpy(),
        expected_grad_bias,
        rtol=3e-4,
        atol=3e-3,
    )

    assert (
        metadata["kernel_launches_per_backward"]
        == 3
    )

    assert (
        metadata[
            "host_transfers_during_repetitions"
        ]
        == 0
    )


def test_linear_backward_reuses_buffers() -> None:
    runtime = TensionForgeRuntime(
        profiling=True,
    )

    inputs = DeviceTensor.from_numpy(
        runtime,
        np.ones(
            (32, 16),
            dtype=np.float32,
        ),
    )

    weights = DeviceTensor.from_numpy(
        runtime,
        np.ones(
            (16, 8),
            dtype=np.float32,
        ),
    )

    grad_output = DeviceTensor.from_numpy(
        runtime,
        np.ones(
            (32, 8),
            dtype=np.float32,
        ),
    )

    grad_input = DeviceTensor.empty(
        runtime,
        (32, 16),
        dtype=np.float32,
    )

    grad_weights = DeviceTensor.empty(
        runtime,
        (16, 8),
        dtype=np.float32,
    )

    grad_bias = DeviceTensor.empty(
        runtime,
        (8,),
        dtype=np.float32,
    )

    (
        returned_input,
        returned_weights,
        returned_bias,
        metadata,
    ) = linear_backward_device(
        runtime,
        inputs,
        weights,
        grad_output,
        grad_input=grad_input,
        grad_weights=grad_weights,
        grad_bias=grad_bias,
        repetitions=2,
    )

    assert returned_input is grad_input
    assert returned_weights is grad_weights
    assert returned_bias is grad_bias

    assert metadata["buffers_reused"] == {
        "grad_input": True,
        "grad_weights": True,
        "grad_bias": True,
    }

    np.testing.assert_allclose(
        grad_input.to_numpy(),
        np.full(
            (32, 16),
            8.0,
            dtype=np.float32,
        ),
        rtol=1e-6,
        atol=1e-6,
    )

    np.testing.assert_allclose(
        grad_weights.to_numpy(),
        np.full(
            (16, 8),
            32.0,
            dtype=np.float32,
        ),
        rtol=1e-6,
        atol=1e-6,
    )

    np.testing.assert_allclose(
        grad_bias.to_numpy(),
        np.full(
            (8,),
            32.0,
            dtype=np.float32,
        ),
        rtol=1e-6,
        atol=1e-6,
    )


def test_linear_backward_rejects_bad_gradient() -> None:
    runtime = TensionForgeRuntime(
        profiling=False,
    )

    inputs = DeviceTensor.empty(
        runtime,
        (8, 16),
        dtype=np.float32,
    )

    weights = DeviceTensor.empty(
        runtime,
        (16, 12),
        dtype=np.float32,
    )

    grad_output = DeviceTensor.empty(
        runtime,
        (8, 11),
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="grad_output shape",
    ):
        linear_backward_device(
            runtime,
            inputs,
            weights,
            grad_output,
        )
