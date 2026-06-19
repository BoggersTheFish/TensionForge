from __future__ import annotations

import numpy as np
import pytest

from tensionforge import (
    DeviceTensor,
    TensionForgeRuntime,
)
from tensionforge.ops import (
    adamw_update_device,
    mse_loss_grad_device,
)


def test_mse_loss_and_gradient_match_numpy() -> None:
    runtime = TensionForgeRuntime(
        profiling=True,
    )

    rng = np.random.default_rng(601)

    prediction = rng.normal(
        size=(37, 23),
    ).astype(np.float32)

    target = rng.normal(
        size=(37, 23),
    ).astype(np.float32)

    (
        loss_terms_gpu,
        gradient_gpu,
        metadata,
    ) = mse_loss_grad_device(
        runtime,
        DeviceTensor.from_numpy(
            runtime,
            prediction,
        ),
        DeviceTensor.from_numpy(
            runtime,
            target,
        ),
        repetitions=3,
    )

    difference = prediction - target

    expected_loss_terms = (
        difference
        * difference
        / difference.size
    ).astype(np.float32)

    expected_gradient = (
        2.0
        * difference
        / difference.size
    ).astype(np.float32)

    np.testing.assert_allclose(
        loss_terms_gpu.to_numpy(),
        expected_loss_terms,
        rtol=2e-6,
        atol=2e-6,
    )

    np.testing.assert_allclose(
        gradient_gpu.to_numpy(),
        expected_gradient,
        rtol=2e-6,
        atol=2e-6,
    )

    assert (
        metadata["operation"]
        == "mse_loss_grad_device_fp32"
    )


def test_adamw_matches_numpy_for_two_steps() -> None:
    runtime = TensionForgeRuntime(
        profiling=True,
    )

    rng = np.random.default_rng(602)

    parameter = rng.normal(
        size=(128,),
    ).astype(np.float32)

    gradient1 = rng.normal(
        size=(128,),
    ).astype(np.float32)

    gradient2 = rng.normal(
        size=(128,),
    ).astype(np.float32)

    expected_parameter = parameter.copy()
    expected_first = np.zeros_like(parameter)
    expected_second = np.zeros_like(parameter)

    parameter_gpu = DeviceTensor.from_numpy(
        runtime,
        parameter,
    )

    gradient_gpu = DeviceTensor.from_numpy(
        runtime,
        gradient1,
    )

    first_gpu = DeviceTensor.from_numpy(
        runtime,
        np.zeros_like(parameter),
    )

    second_gpu = DeviceTensor.from_numpy(
        runtime,
        np.zeros_like(parameter),
    )

    learning_rate = 0.01
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    weight_decay = 0.01

    gradients = (
        gradient1,
        gradient2,
    )

    for step, gradient in enumerate(
        gradients,
        start=1,
    ):
        gradient_gpu.copy_from(gradient)

        expected_first = (
            beta1 * expected_first
            + (1.0 - beta1) * gradient
        ).astype(np.float32)

        expected_second = (
            beta2 * expected_second
            + (1.0 - beta2)
            * gradient
            * gradient
        ).astype(np.float32)

        corrected_first = (
            expected_first
            / (
                1.0
                - beta1 ** step
            )
        )

        corrected_second = (
            expected_second
            / (
                1.0
                - beta2 ** step
            )
        )

        expected_parameter = (
            expected_parameter
            * (
                1.0
                - learning_rate
                * weight_decay
            )
            - learning_rate
            * corrected_first
            / (
                np.sqrt(
                    corrected_second
                )
                + epsilon
            )
        ).astype(np.float32)

        adamw_update_device(
            runtime,
            parameter_gpu,
            gradient_gpu,
            first_gpu,
            second_gpu,
            step=step,
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            weight_decay=weight_decay,
        )

    np.testing.assert_allclose(
        parameter_gpu.to_numpy(),
        expected_parameter,
        rtol=3e-6,
        atol=3e-6,
    )

    np.testing.assert_allclose(
        first_gpu.to_numpy(),
        expected_first,
        rtol=3e-6,
        atol=3e-6,
    )

    np.testing.assert_allclose(
        second_gpu.to_numpy(),
        expected_second,
        rtol=3e-6,
        atol=3e-6,
    )


def test_adamw_rejects_step_zero() -> None:
    runtime = TensionForgeRuntime(
        profiling=False,
    )

    shape = (16,)

    parameter = DeviceTensor.empty(
        runtime,
        shape,
        dtype=np.float32,
    )

    gradient = DeviceTensor.empty(
        runtime,
        shape,
        dtype=np.float32,
    )

    first = DeviceTensor.empty(
        runtime,
        shape,
        dtype=np.float32,
    )

    second = DeviceTensor.empty(
        runtime,
        shape,
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="step",
    ):
        adamw_update_device(
            runtime,
            parameter,
            gradient,
            first,
            second,
            step=0,
            learning_rate=0.01,
        )
