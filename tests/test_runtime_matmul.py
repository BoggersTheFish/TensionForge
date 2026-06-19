from __future__ import annotations

import numpy as np
import pytest

from tensionforge import TensionForgeRuntime
from tensionforge.ops import matmul


def test_matmul_matches_numpy() -> None:
    runtime = TensionForgeRuntime(
        profiling=True,
    )

    rng = np.random.default_rng(456)

    a = rng.normal(
        size=(130, 70),
    ).astype(np.float32)

    b = rng.normal(
        size=(70, 96),
    ).astype(np.float32)

    result, metadata = matmul(
        runtime,
        a,
        b,
        repetitions=2,
        tile_size=16,
    )

    expected = a @ b

    np.testing.assert_allclose(
        result,
        expected,
        rtol=2e-4,
        atol=2e-3,
    )

    assert result.shape == (130, 96)
    assert metadata["tile_size"] == 16
    assert metadata["gflops"] > 0.0
    assert runtime.program_cache_size == 1
    assert runtime.kernel_cache_size == 1


def test_matmul_reuses_compiled_kernel() -> None:
    runtime = TensionForgeRuntime(
        profiling=True,
    )

    a = np.ones(
        (64, 64),
        dtype=np.float32,
    )

    b = np.ones(
        (64, 64),
        dtype=np.float32,
    )

    matmul(
        runtime,
        a,
        b,
        tile_size=16,
    )

    initial_programs = (
        runtime.program_cache_size
    )

    initial_kernels = (
        runtime.kernel_cache_size
    )

    matmul(
        runtime,
        a,
        b,
        tile_size=16,
    )

    assert runtime.program_cache_size == (
        initial_programs
    )

    assert runtime.kernel_cache_size == (
        initial_kernels
    )


def test_matmul_rejects_bad_shapes() -> None:
    runtime = TensionForgeRuntime(
        profiling=False,
    )

    a = np.zeros(
        (8, 7),
        dtype=np.float32,
    )

    b = np.zeros(
        (6, 8),
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="Incompatible matrix dimensions",
    ):
        matmul(runtime, a, b)
