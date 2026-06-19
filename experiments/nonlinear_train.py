from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pyopencl as cl


KERNEL_SOURCE = r"""
__kernel void linear_forward(
    __global const float *x,
    __global const float *weights,
    __global const float *bias,
    __global float *output,
    const unsigned int samples,
    const unsigned int inputs,
    const unsigned int outputs
) {
    const unsigned int output_index = get_global_id(0);
    const unsigned int sample_index = get_global_id(1);

    if (sample_index >= samples || output_index >= outputs) {
        return;
    }

    float total = bias[output_index];

    for (
        unsigned int input_index = 0;
        input_index < inputs;
        ++input_index
    ) {
        total +=
            x[sample_index * inputs + input_index]
            * weights[input_index * outputs + output_index];
    }

    output[sample_index * outputs + output_index] = total;
}


__kernel void tanh_forward(
    __global const float *input,
    __global float *output,
    const unsigned int count
) {
    const unsigned int index = get_global_id(0);

    if (index < count) {
        output[index] = tanh(input[index]);
    }
}


__kernel void calculate_error(
    __global const float *prediction,
    __global const float *target,
    __global float *error,
    const unsigned int count
) {
    const unsigned int index = get_global_id(0);

    if (index < count) {
        error[index] = prediction[index] - target[index];
    }
}


__kernel void linear_weight_gradient(
    __global const float *x,
    __global const float *error,
    __global float *weight_gradient,
    const unsigned int samples,
    const unsigned int inputs,
    const unsigned int outputs
) {
    const unsigned int input_index = get_global_id(0);
    const unsigned int output_index = get_global_id(1);

    if (input_index >= inputs || output_index >= outputs) {
        return;
    }

    float total = 0.0f;

    for (
        unsigned int sample_index = 0;
        sample_index < samples;
        ++sample_index
    ) {
        total +=
            x[sample_index * inputs + input_index]
            * error[sample_index * outputs + output_index];
    }

    weight_gradient[input_index * outputs + output_index] =
        total / (float)samples;
}


__kernel void linear_bias_gradient(
    __global const float *error,
    __global float *bias_gradient,
    const unsigned int samples,
    const unsigned int outputs
) {
    const unsigned int output_index = get_global_id(0);

    if (output_index >= outputs) {
        return;
    }

    float total = 0.0f;

    for (
        unsigned int sample_index = 0;
        sample_index < samples;
        ++sample_index
    ) {
        total += error[sample_index * outputs + output_index];
    }

    bias_gradient[output_index] = total / (float)samples;
}


__kernel void hidden_tanh_gradient(
    __global const float *output_error,
    __global const float *output_weights,
    __global const float *hidden_values,
    __global float *hidden_gradient,
    const unsigned int samples,
    const unsigned int hidden_size,
    const unsigned int outputs
) {
    const unsigned int hidden_index = get_global_id(0);
    const unsigned int sample_index = get_global_id(1);

    if (
        sample_index >= samples
        || hidden_index >= hidden_size
    ) {
        return;
    }

    float propagated = 0.0f;

    for (
        unsigned int output_index = 0;
        output_index < outputs;
        ++output_index
    ) {
        propagated +=
            output_error[
                sample_index * outputs + output_index
            ]
            * output_weights[
                hidden_index * outputs + output_index
            ];
    }

    const float hidden_value =
        hidden_values[
            sample_index * hidden_size + hidden_index
        ];

    hidden_gradient[
        sample_index * hidden_size + hidden_index
    ] = propagated * (1.0f - hidden_value * hidden_value);
}


__kernel void sgd_update(
    __global float *values,
    __global const float *gradients,
    const float learning_rate,
    const unsigned int count
) {
    const unsigned int index = get_global_id(0);

    if (index < count) {
        values[index] -= learning_rate * gradients[index];
    }
}
"""


def find_rx480() -> tuple[cl.Platform, cl.Device]:
    for platform in cl.get_platforms():
        if "rusticl" not in platform.name.lower():
            continue

        for device in platform.get_devices():
            is_gpu = bool(device.type & cl.device_type.GPU)

            if is_gpu and "radeon" in device.name.lower():
                return platform, device

    raise RuntimeError(
        "RX 480 not found through Rusticl. "
        "Set RUSTICL_ENABLE=radeonsi."
    )


def read_buffer(
    queue: cl.CommandQueue,
    destination: np.ndarray,
    source: cl.Buffer,
) -> np.ndarray:
    cl.enqueue_copy(queue, destination, source).wait()
    return destination


def max_error(
    gpu_value: np.ndarray,
    cpu_value: np.ndarray,
) -> float:
    return float(
        np.max(
            np.abs(
                gpu_value - cpu_value
            )
        )
    )


def main() -> int:
    samples = 4096
    inputs = 2
    hidden_size = 32
    outputs = 1
    steps = 750
    learning_rate = np.float32(0.2)

    rng = np.random.default_rng(42)

    # The model receives x0 and x1 and must learn x0 * x1.
    # A single linear layer cannot represent multiplication.
    x = rng.uniform(
        -1.0,
        1.0,
        size=(samples, inputs),
    ).astype(np.float32)

    target = (
        x[:, 0:1] * x[:, 1:2]
    ).astype(np.float32)

    weights_1 = rng.normal(
        0.0,
        0.5,
        size=(inputs, hidden_size),
    ).astype(np.float32)

    bias_1 = np.zeros(
        hidden_size,
        dtype=np.float32,
    )

    weights_2 = rng.normal(
        0.0,
        0.2,
        size=(hidden_size, outputs),
    ).astype(np.float32)

    bias_2 = np.zeros(
        outputs,
        dtype=np.float32,
    )

    initial_weights_1 = weights_1.copy()
    initial_bias_1 = bias_1.copy()
    initial_weights_2 = weights_2.copy()
    initial_bias_2 = bias_2.copy()

    platform, device = find_rx480()

    print("=== DEVICE ===")
    print(f"Platform: {platform.name}")
    print(f"Device:   {device.name}")
    print(f"Driver:   {device.driver_version}")
    print()

    context = cl.Context([device])
    queue = cl.CommandQueue(context)

    program = cl.Program(
        context,
        KERNEL_SOURCE,
    ).build()

    linear_1 = cl.Kernel(program, "linear_forward")
    activation = cl.Kernel(program, "tanh_forward")
    linear_2 = cl.Kernel(program, "linear_forward")
    calculate_error = cl.Kernel(
        program,
        "calculate_error",
    )

    output_weight_gradient = cl.Kernel(
        program,
        "linear_weight_gradient",
    )
    output_bias_gradient = cl.Kernel(
        program,
        "linear_bias_gradient",
    )

    hidden_gradient_kernel = cl.Kernel(
        program,
        "hidden_tanh_gradient",
    )

    input_weight_gradient = cl.Kernel(
        program,
        "linear_weight_gradient",
    )
    hidden_bias_gradient = cl.Kernel(
        program,
        "linear_bias_gradient",
    )

    update_weights_1 = cl.Kernel(
        program,
        "sgd_update",
    )
    update_bias_1 = cl.Kernel(
        program,
        "sgd_update",
    )
    update_weights_2 = cl.Kernel(
        program,
        "sgd_update",
    )
    update_bias_2 = cl.Kernel(
        program,
        "sgd_update",
    )

    flags = cl.mem_flags

    x_gpu = cl.Buffer(
        context,
        flags.READ_ONLY | flags.COPY_HOST_PTR,
        hostbuf=x,
    )

    target_gpu = cl.Buffer(
        context,
        flags.READ_ONLY | flags.COPY_HOST_PTR,
        hostbuf=target,
    )

    weights_1_gpu = cl.Buffer(
        context,
        flags.READ_WRITE | flags.COPY_HOST_PTR,
        hostbuf=weights_1,
    )

    bias_1_gpu = cl.Buffer(
        context,
        flags.READ_WRITE | flags.COPY_HOST_PTR,
        hostbuf=bias_1,
    )

    weights_2_gpu = cl.Buffer(
        context,
        flags.READ_WRITE | flags.COPY_HOST_PTR,
        hostbuf=weights_2,
    )

    bias_2_gpu = cl.Buffer(
        context,
        flags.READ_WRITE | flags.COPY_HOST_PTR,
        hostbuf=bias_2,
    )

    hidden_pre_gpu = cl.Buffer(
        context,
        flags.READ_WRITE,
        samples * hidden_size * 4,
    )

    hidden_gpu = cl.Buffer(
        context,
        flags.READ_WRITE,
        samples * hidden_size * 4,
    )

    prediction_gpu = cl.Buffer(
        context,
        flags.READ_WRITE,
        samples * outputs * 4,
    )

    error_gpu = cl.Buffer(
        context,
        flags.READ_WRITE,
        samples * outputs * 4,
    )

    hidden_gradient_gpu = cl.Buffer(
        context,
        flags.READ_WRITE,
        samples * hidden_size * 4,
    )

    weights_1_gradient_gpu = cl.Buffer(
        context,
        flags.READ_WRITE,
        weights_1.nbytes,
    )

    bias_1_gradient_gpu = cl.Buffer(
        context,
        flags.READ_WRITE,
        bias_1.nbytes,
    )

    weights_2_gradient_gpu = cl.Buffer(
        context,
        flags.READ_WRITE,
        weights_2.nbytes,
    )

    bias_2_gradient_gpu = cl.Buffer(
        context,
        flags.READ_WRITE,
        bias_2.nbytes,
    )

    linear_1.set_args(
        x_gpu,
        weights_1_gpu,
        bias_1_gpu,
        hidden_pre_gpu,
        np.uint32(samples),
        np.uint32(inputs),
        np.uint32(hidden_size),
    )

    activation.set_args(
        hidden_pre_gpu,
        hidden_gpu,
        np.uint32(samples * hidden_size),
    )

    linear_2.set_args(
        hidden_gpu,
        weights_2_gpu,
        bias_2_gpu,
        prediction_gpu,
        np.uint32(samples),
        np.uint32(hidden_size),
        np.uint32(outputs),
    )

    calculate_error.set_args(
        prediction_gpu,
        target_gpu,
        error_gpu,
        np.uint32(samples * outputs),
    )

    output_weight_gradient.set_args(
        hidden_gpu,
        error_gpu,
        weights_2_gradient_gpu,
        np.uint32(samples),
        np.uint32(hidden_size),
        np.uint32(outputs),
    )

    output_bias_gradient.set_args(
        error_gpu,
        bias_2_gradient_gpu,
        np.uint32(samples),
        np.uint32(outputs),
    )

    hidden_gradient_kernel.set_args(
        error_gpu,
        weights_2_gpu,
        hidden_gpu,
        hidden_gradient_gpu,
        np.uint32(samples),
        np.uint32(hidden_size),
        np.uint32(outputs),
    )

    input_weight_gradient.set_args(
        x_gpu,
        hidden_gradient_gpu,
        weights_1_gradient_gpu,
        np.uint32(samples),
        np.uint32(inputs),
        np.uint32(hidden_size),
    )

    hidden_bias_gradient.set_args(
        hidden_gradient_gpu,
        bias_1_gradient_gpu,
        np.uint32(samples),
        np.uint32(hidden_size),
    )

    update_weights_1.set_args(
        weights_1_gpu,
        weights_1_gradient_gpu,
        learning_rate,
        np.uint32(weights_1.size),
    )

    update_bias_1.set_args(
        bias_1_gpu,
        bias_1_gradient_gpu,
        learning_rate,
        np.uint32(bias_1.size),
    )

    update_weights_2.set_args(
        weights_2_gpu,
        weights_2_gradient_gpu,
        learning_rate,
        np.uint32(weights_2.size),
    )

    update_bias_2.set_args(
        bias_2_gpu,
        bias_2_gradient_gpu,
        learning_rate,
        np.uint32(bias_2.size),
    )

    def forward_and_backward() -> None:
        cl.enqueue_nd_range_kernel(
            queue,
            linear_1,
            (hidden_size, samples),
            None,
        )

        cl.enqueue_nd_range_kernel(
            queue,
            activation,
            (samples * hidden_size,),
            None,
        )

        cl.enqueue_nd_range_kernel(
            queue,
            linear_2,
            (outputs, samples),
            None,
        )

        cl.enqueue_nd_range_kernel(
            queue,
            calculate_error,
            (samples * outputs,),
            None,
        )

        cl.enqueue_nd_range_kernel(
            queue,
            output_weight_gradient,
            (hidden_size, outputs),
            None,
        )

        cl.enqueue_nd_range_kernel(
            queue,
            output_bias_gradient,
            (outputs,),
            None,
        )

        cl.enqueue_nd_range_kernel(
            queue,
            hidden_gradient_kernel,
            (hidden_size, samples),
            None,
        )

        cl.enqueue_nd_range_kernel(
            queue,
            input_weight_gradient,
            (inputs, hidden_size),
            None,
        )

        cl.enqueue_nd_range_kernel(
            queue,
            hidden_bias_gradient,
            (hidden_size,),
            None,
        )

        queue.finish()

    forward_and_backward()

    gpu_hidden = np.empty(
        (samples, hidden_size),
        dtype=np.float32,
    )
    gpu_prediction = np.empty_like(target)
    gpu_error = np.empty_like(target)
    gpu_hidden_gradient = np.empty_like(gpu_hidden)

    gpu_w1_gradient = np.empty_like(weights_1)
    gpu_b1_gradient = np.empty_like(bias_1)
    gpu_w2_gradient = np.empty_like(weights_2)
    gpu_b2_gradient = np.empty_like(bias_2)

    read_buffer(queue, gpu_hidden, hidden_gpu)
    read_buffer(queue, gpu_prediction, prediction_gpu)
    read_buffer(queue, gpu_error, error_gpu)

    read_buffer(
        queue,
        gpu_hidden_gradient,
        hidden_gradient_gpu,
    )
    read_buffer(
        queue,
        gpu_w1_gradient,
        weights_1_gradient_gpu,
    )
    read_buffer(
        queue,
        gpu_b1_gradient,
        bias_1_gradient_gpu,
    )
    read_buffer(
        queue,
        gpu_w2_gradient,
        weights_2_gradient_gpu,
    )
    read_buffer(
        queue,
        gpu_b2_gradient,
        bias_2_gradient_gpu,
    )

    cpu_hidden = np.tanh(
        x @ initial_weights_1 + initial_bias_1
    ).astype(np.float32)

    cpu_prediction = (
        cpu_hidden @ initial_weights_2
        + initial_bias_2
    ).astype(np.float32)

    cpu_error = (
        cpu_prediction - target
    ).astype(np.float32)

    cpu_w2_gradient = (
        cpu_hidden.T @ cpu_error / samples
    ).astype(np.float32)

    cpu_b2_gradient = (
        np.mean(cpu_error, axis=0)
    ).astype(np.float32)

    cpu_hidden_gradient = (
        (cpu_error @ initial_weights_2.T)
        * (1.0 - cpu_hidden * cpu_hidden)
    ).astype(np.float32)

    cpu_w1_gradient = (
        x.T @ cpu_hidden_gradient / samples
    ).astype(np.float32)

    cpu_b1_gradient = (
        np.mean(cpu_hidden_gradient, axis=0)
    ).astype(np.float32)

    verification = {
        "hidden_forward_error":
            max_error(gpu_hidden, cpu_hidden),
        "output_forward_error":
            max_error(gpu_prediction, cpu_prediction),
        "output_weight_gradient_error":
            max_error(gpu_w2_gradient, cpu_w2_gradient),
        "output_bias_gradient_error":
            max_error(gpu_b2_gradient, cpu_b2_gradient),
        "hidden_gradient_error":
            max_error(
                gpu_hidden_gradient,
                cpu_hidden_gradient,
            ),
        "input_weight_gradient_error":
            max_error(gpu_w1_gradient, cpu_w1_gradient),
        "hidden_bias_gradient_error":
            max_error(gpu_b1_gradient, cpu_b1_gradient),
    }

    all_verified = all(
        np.allclose(
            gpu_value,
            cpu_value,
            rtol=5e-3,
            atol=5e-4,
        )
        for gpu_value, cpu_value in (
            (gpu_hidden, cpu_hidden),
            (gpu_prediction, cpu_prediction),
            (gpu_w2_gradient, cpu_w2_gradient),
            (gpu_b2_gradient, cpu_b2_gradient),
            (
                gpu_hidden_gradient,
                cpu_hidden_gradient,
            ),
            (gpu_w1_gradient, cpu_w1_gradient),
            (gpu_b1_gradient, cpu_b1_gradient),
        )
    )

    initial_loss = float(
        0.5 * np.mean(gpu_error * gpu_error)
    )

    print("=== NUMERICAL VERIFICATION ===")

    for name, value in verification.items():
        print(f"{name}: {value:.8g}")

    print(f"All operations valid: {all_verified}")
    print()

    if not all_verified:
        print(
            "FAILED: GPU gradients did not match NumPy."
        )
        return 1

    print("=== NONLINEAR GPU TRAINING ===")
    print(f"Step    0 | loss {initial_loss:.10f}")

    report_steps = {
        1,
        10,
        25,
        50,
        100,
        250,
        500,
        750,
    }

    start_time = time.perf_counter()

    for step in range(1, steps + 1):
        cl.enqueue_nd_range_kernel(
            queue,
            update_weights_1,
            (weights_1.size,),
            None,
        )

        cl.enqueue_nd_range_kernel(
            queue,
            update_bias_1,
            (bias_1.size,),
            None,
        )

        cl.enqueue_nd_range_kernel(
            queue,
            update_weights_2,
            (weights_2.size,),
            None,
        )

        cl.enqueue_nd_range_kernel(
            queue,
            update_bias_2,
            (bias_2.size,),
            None,
        )

        forward_and_backward()

        if step in report_steps:
            read_buffer(
                queue,
                gpu_error,
                error_gpu,
            )

            current_loss = float(
                0.5
                * np.mean(
                    gpu_error * gpu_error
                )
            )

            print(
                f"Step {step:4d} | "
                f"loss {current_loss:.10f}"
            )

    training_seconds = (
        time.perf_counter() - start_time
    )

    read_buffer(
        queue,
        gpu_prediction,
        prediction_gpu,
    )
    read_buffer(
        queue,
        gpu_error,
        error_gpu,
    )

    final_loss = float(
        0.5 * np.mean(gpu_error * gpu_error)
    )

    mean_absolute_error = float(
        np.mean(
            np.abs(
                gpu_prediction - target
            )
        )
    )

    loss_reduction = (
        initial_loss / max(final_loss, 1e-30)
    )

    training_passed = bool(
        final_loss < 0.005
        and final_loss < initial_loss * 0.05
    )

    receipt = {
        "milestone": "verified_nonlinear_gpu_training",
        "task": "learn multiplication x0_times_x1",
        "device": device.name,
        "driver": device.driver_version,
        "configuration": {
            "samples": samples,
            "inputs": inputs,
            "hidden_size": hidden_size,
            "outputs": outputs,
            "steps": steps,
            "learning_rate": float(learning_rate),
        },
        "verification": {
            **verification,
            "all_operations_verified": all_verified,
        },
        "training": {
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "loss_reduction_factor":
                loss_reduction,
            "mean_absolute_error":
                mean_absolute_error,
            "training_seconds":
                training_seconds,
            "passed": training_passed,
        },
    }

    receipt_path = Path(
        "receipts/nonlinear_training_receipt.json"
    )

    receipt_path.write_text(
        json.dumps(receipt, indent=2),
        encoding="utf-8",
    )

    print()
    print("=== RESULT ===")
    print(f"Initial loss:      {initial_loss:.10f}")
    print(f"Final loss:        {final_loss:.10f}")
    print(f"Loss reduction:    {loss_reduction:.2f}x")
    print(f"Mean abs error:    {mean_absolute_error:.8f}")
    print(f"Training time:     {training_seconds:.3f}s")
    print(f"Receipt:           {receipt_path}")

    print()
    print("Example predictions:")

    for index in range(5):
        print(
            f"  {x[index, 0]: .3f} × "
            f"{x[index, 1]: .3f} = "
            f"{target[index, 0]: .3f} | "
            f"model: {gpu_prediction[index, 0]: .3f}"
        )

    if not training_passed:
        print()
        print(
            "FAILED: the nonlinear model did not "
            "reduce its loss enough."
        )
        return 1

    print()
    print(
        "PASSED: RX 480 learned a nonlinear "
        "function using verified custom backpropagation."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
