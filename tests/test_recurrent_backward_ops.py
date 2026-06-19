from __future__ import annotations

import numpy as np
import pytest

from tensionforge import (
    DeviceTensor,
    TensionForgeRuntime,
)
from tensionforge.ops import (
    sigmoid_backward_device,
    sigmoid_device,
    tanh_backward_device,
    tanh_device,
    tension_update_backward_device,
)


def test_tanh_backward_matches_numpy() -> None:
    runtime = TensionForgeRuntime(
        profiling=True,
    )

    rng = np.random.default_rng(701)

    values = rng.uniform(
        -4.0,
        4.0,
        size=(257, 33),
    ).astype(np.float32)

    grad_output = rng.normal(
        size=values.shape,
    ).astype(np.float32)

    values_gpu = DeviceTensor.from_numpy(
        runtime,
        values,
    )

    output_gpu, _ = tanh_device(
        runtime,
        values_gpu,
    )

    grad_input_gpu, metadata = (
        tanh_backward_device(
            runtime,
            output_gpu,
            DeviceTensor.from_numpy(
                runtime,
                grad_output,
            ),
            repetitions=3,
        )
    )

    output = np.tanh(values).astype(np.float32)

    expected = (
        grad_output
        * (
            1.0
            - output
            * output
        )
    ).astype(np.float32)

    np.testing.assert_allclose(
        grad_input_gpu.to_numpy(),
        expected,
        rtol=3e-6,
        atol=3e-6,
    )

    assert (
        metadata["operation"]
        == "tanh_backward_device_fp32"
    )


def test_sigmoid_backward_matches_numpy() -> None:
    runtime = TensionForgeRuntime(
        profiling=True,
    )

    rng = np.random.default_rng(702)

    values = rng.uniform(
        -8.0,
        8.0,
        size=(257, 33),
    ).astype(np.float32)

    grad_output = rng.normal(
        size=values.shape,
    ).astype(np.float32)

    values_gpu = DeviceTensor.from_numpy(
        runtime,
        values,
    )

    output_gpu, _ = sigmoid_device(
        runtime,
        values_gpu,
    )

    grad_input_gpu, _ = (
        sigmoid_backward_device(
            runtime,
            output_gpu,
            DeviceTensor.from_numpy(
                runtime,
                grad_output,
            ),
            repetitions=3,
        )
    )

    output = (
        1.0
        / (
            1.0
            + np.exp(-values)
        )
    ).astype(np.float32)

    expected = (
        grad_output
        * output
        * (
            1.0
            - output
        )
    ).astype(np.float32)

    np.testing.assert_allclose(
        grad_input_gpu.to_numpy(),
        expected,
        rtol=3e-6,
        atol=3e-6,
    )


def test_tension_backward_matches_numpy() -> None:
    runtime = TensionForgeRuntime(
        profiling=True,
    )

    rng = np.random.default_rng(703)

    shape = (257, 33)

    state = rng.normal(
        size=shape,
    ).astype(np.float32)

    proposal = rng.normal(
        size=shape,
    ).astype(np.float32)

    gate = rng.uniform(
        0.0,
        1.0,
        size=shape,
    ).astype(np.float32)

    grad_output = rng.normal(
        size=shape,
    ).astype(np.float32)

    (
        grad_state_gpu,
        grad_proposal_gpu,
        grad_gate_gpu,
        metadata,
    ) = tension_update_backward_device(
        runtime,
        DeviceTensor.from_numpy(
            runtime,
            state,
        ),
        DeviceTensor.from_numpy(
            runtime,
            proposal,
        ),
        DeviceTensor.from_numpy(
            runtime,
            gate,
        ),
        DeviceTensor.from_numpy(
            runtime,
            grad_output,
        ),
        repetitions=3,
    )

    expected_grad_state = (
        grad_output
        * (
            1.0
            - gate
        )
    ).astype(np.float32)

    expected_grad_proposal = (
        grad_output * gate
    ).astype(np.float32)

    expected_grad_gate = (
        grad_output
        * (
            proposal
            - state
        )
    ).astype(np.float32)

    np.testing.assert_allclose(
        grad_state_gpu.to_numpy(),
        expected_grad_state,
        rtol=2e-6,
        atol=2e-6,
    )

    np.testing.assert_allclose(
        grad_proposal_gpu.to_numpy(),
        expected_grad_proposal,
        rtol=2e-6,
        atol=2e-6,
    )

    np.testing.assert_allclose(
        grad_gate_gpu.to_numpy(),
        expected_grad_gate,
        rtol=2e-6,
        atol=2e-6,
    )

    assert (
        metadata["operation"]
        == "tension_update_backward_device_fp32"
    )


def test_tension_backward_reuses_buffers() -> None:
    runtime = TensionForgeRuntime(
        profiling=True,
    )

    shape = (32, 16)

    state = DeviceTensor.from_numpy(
        runtime,
        np.zeros(
            shape,
            dtype=np.float32,
        ),
    )

    proposal = DeviceTensor.from_numpy(
        runtime,
        np.ones(
            shape,
            dtype=np.float32,
        ),
    )

    gate = DeviceTensor.from_numpy(
        runtime,
        np.full(
            shape,
            0.25,
            dtype=np.float32,
        ),
    )

    grad_output = DeviceTensor.from_numpy(
        runtime,
        np.ones(
            shape,
            dtype=np.float32,
        ),
    )

    grad_state = DeviceTensor.empty(
        runtime,
        shape,
        dtype=np.float32,
    )

    grad_proposal = DeviceTensor.empty(
        runtime,
        shape,
        dtype=np.float32,
    )

    grad_gate = DeviceTensor.empty(
        runtime,
        shape,
        dtype=np.float32,
    )

    (
        returned_state,
        returned_proposal,
        returned_gate,
        metadata,
    ) = tension_update_backward_device(
        runtime,
        state,
        proposal,
        gate,
        grad_output,
        grad_state=grad_state,
        grad_proposal=grad_proposal,
        grad_gate=grad_gate,
        repetitions=2,
    )

    assert returned_state is grad_state
    assert returned_proposal is grad_proposal
    assert returned_gate is grad_gate

    assert metadata["buffers_reused"] == {
        "grad_state": True,
        "grad_proposal": True,
        "grad_gate": True,
    }

    np.testing.assert_allclose(
        grad_state.to_numpy(),
        np.full(
            shape,
            0.75,
            dtype=np.float32,
        ),
        rtol=1e-6,
        atol=1e-6,
    )

    np.testing.assert_allclose(
        grad_proposal.to_numpy(),
        np.full(
            shape,
            0.25,
            dtype=np.float32,
        ),
        rtol=1e-6,
        atol=1e-6,
    )

    np.testing.assert_allclose(
        grad_gate.to_numpy(),
        np.ones(
            shape,
            dtype=np.float32,
        ),
        rtol=1e-6,
        atol=1e-6,
    )


def test_backward_rejects_bad_shape() -> None:
    runtime = TensionForgeRuntime(
        profiling=False,
    )

    output = DeviceTensor.empty(
        runtime,
        (8, 8),
        dtype=np.float32,
    )

    grad_output = DeviceTensor.empty(
        runtime,
        (8, 7),
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="shapes must match",
    ):
        tanh_backward_device(
            runtime,
            output,
            grad_output,
        )
