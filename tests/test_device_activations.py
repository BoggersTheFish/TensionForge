from __future__ import annotations

import numpy as np

from tensionforge import (
    DeviceTensor,
    TensionForgeRuntime,
)
from tensionforge.ops import (
    sigmoid_device,
    tanh_device,
)


def test_tanh_device_matches_numpy() -> None:
    runtime = TensionForgeRuntime(
        profiling=True,
    )

    rng = np.random.default_rng(101)

    values = rng.uniform(
        -4.0,
        4.0,
        size=(512, 64),
    ).astype(np.float32)

    values_gpu = DeviceTensor.from_numpy(
        runtime,
        values,
    )

    output_gpu, metadata = tanh_device(
        runtime,
        values_gpu,
        repetitions=3,
    )

    result = output_gpu.to_numpy()
    expected = np.tanh(values).astype(
        np.float32
    )

    np.testing.assert_allclose(
        result,
        expected,
        rtol=2e-6,
        atol=2e-6,
    )

    assert (
        metadata["operation"]
        == "tanh_device_fp32"
    )

    assert (
        metadata[
            "host_transfers_during_repetitions"
        ]
        == 0
    )


def test_sigmoid_device_matches_numpy() -> None:
    runtime = TensionForgeRuntime(
        profiling=True,
    )

    rng = np.random.default_rng(102)

    values = rng.uniform(
        -8.0,
        8.0,
        size=(512, 64),
    ).astype(np.float32)

    values_gpu = DeviceTensor.from_numpy(
        runtime,
        values,
    )

    reusable_output = DeviceTensor.empty(
        runtime,
        values.shape,
        dtype=np.float32,
    )

    output_gpu, metadata = sigmoid_device(
        runtime,
        values_gpu,
        output=reusable_output,
        repetitions=3,
    )

    result = output_gpu.to_numpy()

    expected = (
        1.0
        / (
            1.0
            + np.exp(-values)
        )
    ).astype(np.float32)

    np.testing.assert_allclose(
        result,
        expected,
        rtol=2e-6,
        atol=2e-6,
    )

    assert output_gpu is reusable_output
    assert metadata["output_buffer_reused"] is True
