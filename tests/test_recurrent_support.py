from __future__ import annotations

import numpy as np

from tensionforge import (
    DeviceTensor,
    TensionForgeRuntime,
)
from tensionforge.ops import (
    add_inplace_device,
    concatenate_rows_device,
    fill_device,
    merge_recurrent_state_gradient_device,
)


def test_concatenate_and_accumulate() -> None:
    runtime = TensionForgeRuntime(
        profiling=True,
    )

    left = np.arange(
        12,
        dtype=np.float32,
    ).reshape(4, 3)

    right = np.arange(
        8,
        dtype=np.float32,
    ).reshape(4, 2)

    output, _ = concatenate_rows_device(
        runtime,
        DeviceTensor.from_numpy(
            runtime,
            left,
        ),
        DeviceTensor.from_numpy(
            runtime,
            right,
        ),
    )

    expected = np.concatenate(
        (left, right),
        axis=1,
    )

    np.testing.assert_array_equal(
        output.to_numpy(),
        expected,
    )

    fill_device(
        runtime,
        output,
        1.0,
    )

    add_inplace_device(
        runtime,
        output,
        DeviceTensor.from_numpy(
            runtime,
            np.full(
                output.shape,
                2.0,
                dtype=np.float32,
            ),
        ),
    )

    np.testing.assert_array_equal(
        output.to_numpy(),
        np.full(
            output.shape,
            3.0,
            dtype=np.float32,
        ),
    )


def test_merge_recurrent_state_gradient() -> None:
    runtime = TensionForgeRuntime(
        profiling=True,
    )

    batch_size = 3
    input_size = 2
    hidden_size = 4

    proposal_features = np.arange(
        batch_size
        * (
            input_size + hidden_size
        ),
        dtype=np.float32,
    ).reshape(
        batch_size,
        input_size + hidden_size,
    )

    gate_features = (
        proposal_features * 0.5
    ).astype(np.float32)

    direct = np.full(
        (
            batch_size,
            hidden_size,
        ),
        3.0,
        dtype=np.float32,
    )

    result, _ = (
        merge_recurrent_state_gradient_device(
            runtime,
            DeviceTensor.from_numpy(
                runtime,
                proposal_features,
            ),
            DeviceTensor.from_numpy(
                runtime,
                gate_features,
            ),
            DeviceTensor.from_numpy(
                runtime,
                direct,
            ),
            input_features=input_size,
        )
    )

    expected = (
        direct
        + proposal_features[
            :,
            input_size:,
        ]
        + gate_features[
            :,
            input_size:,
        ]
    )

    np.testing.assert_array_equal(
        result.to_numpy(),
        expected,
    )
