from __future__ import annotations

import numpy as np
import pytest

from tensionforge import (
    DeviceTensor,
    TensionForgeRuntime,
)


def test_device_tensor_roundtrip() -> None:
    runtime = TensionForgeRuntime(
        profiling=True,
    )

    source = np.arange(
        4096,
        dtype=np.float32,
    ).reshape(64, 64)

    tensor = DeviceTensor.from_numpy(
        runtime,
        source,
    )

    result = tensor.to_numpy()

    np.testing.assert_array_equal(
        result,
        source,
    )

    assert tensor.shape == (64, 64)
    assert tensor.ndim == 2
    assert tensor.size == 4096
    assert tensor.nbytes == source.nbytes


def test_device_tensor_copy_from() -> None:
    runtime = TensionForgeRuntime(
        profiling=True,
    )

    tensor = DeviceTensor.empty(
        runtime,
        (32, 16),
        dtype=np.float32,
    )

    source = np.full(
        (32, 16),
        7.5,
        dtype=np.float32,
    )

    tensor.copy_from(source)

    np.testing.assert_array_equal(
        tensor.to_numpy(),
        source,
    )


def test_device_tensor_rejects_bad_copy() -> None:
    runtime = TensionForgeRuntime(
        profiling=False,
    )

    tensor = DeviceTensor.empty(
        runtime,
        (8, 8),
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="Source shape",
    ):
        tensor.copy_from(
            np.zeros(
                (4, 4),
                dtype=np.float32,
            )
        )
