from __future__ import annotations

import numpy as np
import pytest

from tensionforge import (
    DeviceTensor,
    TensionForgeRuntime,
)
from tensionforge.ops import tension_update_device


def test_tension_update_matches_numpy() -> None:
    runtime = TensionForgeRuntime(
        profiling=True,
    )

    rng = np.random.default_rng(103)

    state = rng.normal(
        size=(512, 64),
    ).astype(np.float32)

    proposal = rng.normal(
        size=(512, 64),
    ).astype(np.float32)

    gate = rng.uniform(
        0.0,
        1.0,
        size=(512, 64),
    ).astype(np.float32)

    state_gpu = DeviceTensor.from_numpy(
        runtime,
        state,
    )

    proposal_gpu = DeviceTensor.from_numpy(
        runtime,
        proposal,
    )

    gate_gpu = DeviceTensor.from_numpy(
        runtime,
        gate,
    )

    output_gpu, metadata = (
        tension_update_device(
            runtime,
            state_gpu,
            proposal_gpu,
            gate_gpu,
            repetitions=3,
        )
    )

    result = output_gpu.to_numpy()

    expected = (
        state
        + gate
        * (
            proposal
            - state
        )
    ).astype(np.float32)

    np.testing.assert_allclose(
        result,
        expected,
        rtol=2e-6,
        atol=2e-6,
    )

    assert (
        metadata["operation"]
        == "tension_update_device_fp32"
    )


def test_tension_update_reuses_output() -> None:
    runtime = TensionForgeRuntime(
        profiling=True,
    )

    shape = (64, 32)

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

    output = DeviceTensor.empty(
        runtime,
        shape,
        dtype=np.float32,
    )

    returned, metadata = (
        tension_update_device(
            runtime,
            state,
            proposal,
            gate,
            output=output,
            repetitions=2,
        )
    )

    assert returned is output
    assert metadata["output_buffer_reused"] is True

    np.testing.assert_allclose(
        output.to_numpy(),
        np.full(
            shape,
            0.25,
            dtype=np.float32,
        ),
        rtol=1e-6,
        atol=1e-6,
    )


def test_tension_update_rejects_bad_shape() -> None:
    runtime = TensionForgeRuntime(
        profiling=False,
    )

    state = DeviceTensor.empty(
        runtime,
        (8, 8),
        dtype=np.float32,
    )

    proposal = DeviceTensor.empty(
        runtime,
        (8, 7),
        dtype=np.float32,
    )

    gate = DeviceTensor.empty(
        runtime,
        (8, 8),
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="proposal shape",
    ):
        tension_update_device(
            runtime,
            state,
            proposal,
            gate,
        )
