from __future__ import annotations

import numpy as np
import pytest

from tensionforge import TensionForgeRuntime
from tensionforge.ops import linear


def test_linear_matches_numpy() -> None:
    runtime = TensionForgeRuntime(
        profiling=True,
    )

    rng = np.random.default_rng(789)

    inputs = rng.normal(
        size=(130, 70),
    ).astype(np.float32)

    weights = rng.normal(
        size=(70, 96),
    ).astype(np.float32)

    bias = rng.normal(
        size=(96,),
    ).astype(np.float32)

    result, metadata = linear(
        runtime,
        inputs,
        weights,
        bias,
        repetitions=2,
        tile_size=16,
    )

    expected = inputs @ weights + bias

    np.testing.assert_allclose(
        result,
        expected,
        rtol=2e-4,
        atol=2e-3,
    )

    assert result.shape == (130, 96)

    assert (
        metadata["operation"]
        == "linear_forward_fp32"
    )

    assert metadata["gflops"] > 0.0
    assert runtime.program_cache_size == 1
    assert runtime.kernel_cache_size == 1


def test_linear_reuses_compiled_kernel() -> None:
    runtime = TensionForgeRuntime(
        profiling=True,
    )

    inputs = np.ones(
        (64, 64),
        dtype=np.float32,
    )

    weights = np.ones(
        (64, 64),
        dtype=np.float32,
    )

    bias = np.zeros(
        (64,),
        dtype=np.float32,
    )

    linear(
        runtime,
        inputs,
        weights,
        bias,
        tile_size=16,
    )

    initial_program_count = (
        runtime.program_cache_size
    )

    initial_kernel_count = (
        runtime.kernel_cache_size
    )

    linear(
        runtime,
        inputs,
        weights,
        bias,
        tile_size=16,
    )

    assert runtime.program_cache_size == (
        initial_program_count
    )

    assert runtime.kernel_cache_size == (
        initial_kernel_count
    )


def test_linear_rejects_bad_bias_shape() -> None:
    runtime = TensionForgeRuntime(
        profiling=False,
    )

    inputs = np.zeros(
        (8, 7),
        dtype=np.float32,
    )

    weights = np.zeros(
        (7, 6),
        dtype=np.float32,
    )

    bias = np.zeros(
        (5,),
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="Bias shape",
    ):
        linear(
            runtime,
            inputs,
            weights,
            bias,
        )
