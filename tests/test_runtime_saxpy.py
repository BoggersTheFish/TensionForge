from __future__ import annotations

import numpy as np

from tensionforge import TensionForgeRuntime
from tensionforge.ops import saxpy


def test_runtime_saxpy_matches_numpy() -> None:
    runtime = TensionForgeRuntime(
        profiling=True,
    )

    rng = np.random.default_rng(123)

    a = rng.normal(
        size=262_144,
    ).astype(np.float32)

    b = rng.normal(
        size=262_144,
    ).astype(np.float32)

    result, metadata = saxpy(
        runtime,
        alpha=0.75,
        a=a,
        b=b,
        repetitions=2,
    )

    expected = (
        np.float32(0.75) * a + b
    )

    np.testing.assert_allclose(
        result,
        expected,
        rtol=1e-6,
        atol=1e-6,
    )

    assert metadata["operation"] == "saxpy_fp32"
    assert runtime.program_cache_size == 1
    assert runtime.kernel_cache_size == 1


def test_runtime_reuses_compiled_kernel() -> None:
    runtime = TensionForgeRuntime(
        profiling=True,
    )

    a = np.arange(
        4096,
        dtype=np.float32,
    )

    b = np.ones_like(a)

    saxpy(
        runtime,
        alpha=2.0,
        a=a,
        b=b,
    )

    first_program_count = (
        runtime.program_cache_size
    )

    first_kernel_count = (
        runtime.kernel_cache_size
    )

    saxpy(
        runtime,
        alpha=3.0,
        a=a,
        b=b,
    )

    assert runtime.program_cache_size == (
        first_program_count
    )

    assert runtime.kernel_cache_size == (
        first_kernel_count
    )
