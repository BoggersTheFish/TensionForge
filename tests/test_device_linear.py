from __future__ import annotations

import numpy as np

from tensionforge import (
    DeviceTensor,
    TensionForgeRuntime,
)
from tensionforge.ops import linear_device


def test_device_linear_matches_numpy() -> None:
    runtime = TensionForgeRuntime(
        profiling=True,
    )

    rng = np.random.default_rng(2026)

    inputs = rng.normal(
        size=(130, 70),
    ).astype(np.float32)

    weights = rng.normal(
        size=(70, 96),
    ).astype(np.float32)

    bias = rng.normal(
        size=(96,),
    ).astype(np.float32)

    inputs_gpu = DeviceTensor.from_numpy(
        runtime,
        inputs,
    )

    weights_gpu = DeviceTensor.from_numpy(
        runtime,
        weights,
    )

    bias_gpu = DeviceTensor.from_numpy(
        runtime,
        bias,
    )

    output_gpu, metadata = linear_device(
        runtime,
        inputs_gpu,
        weights_gpu,
        bias_gpu,
        repetitions=3,
    )

    result = output_gpu.to_numpy()
    expected = inputs @ weights + bias

    np.testing.assert_allclose(
        result,
        expected,
        rtol=2e-4,
        atol=2e-3,
    )

    assert (
        metadata["operation"]
        == "linear_forward_device_fp32"
    )

    assert (
        metadata[
            "host_transfers_during_repetitions"
        ]
        == 0
    )


def test_device_linear_reuses_output_buffer() -> None:
    runtime = TensionForgeRuntime(
        profiling=True,
    )

    inputs = DeviceTensor.from_numpy(
        runtime,
        np.ones(
            (64, 64),
            dtype=np.float32,
        ),
    )

    weights = DeviceTensor.from_numpy(
        runtime,
        np.eye(
            64,
            dtype=np.float32,
        ),
    )

    bias = DeviceTensor.from_numpy(
        runtime,
        np.zeros(
            64,
            dtype=np.float32,
        ),
    )

    output = DeviceTensor.empty(
        runtime,
        (64, 64),
        dtype=np.float32,
    )

    returned_output, _ = linear_device(
        runtime,
        inputs,
        weights,
        bias,
        output=output,
        repetitions=2,
    )

    first_program_count = (
        runtime.program_cache_size
    )

    first_kernel_count = (
        runtime.kernel_cache_size
    )

    returned_again, _ = linear_device(
        runtime,
        inputs,
        weights,
        bias,
        output=output,
        repetitions=2,
    )

    assert returned_output is output
    assert returned_again is output

    assert runtime.program_cache_size == (
        first_program_count
    )

    assert runtime.kernel_cache_size == (
        first_kernel_count
    )

    np.testing.assert_allclose(
        output.to_numpy(),
        np.ones(
            (64, 64),
            dtype=np.float32,
        ),
        rtol=1e-6,
        atol=1e-6,
    )
