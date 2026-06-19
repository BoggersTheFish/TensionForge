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

    for (unsigned int input_index = 0;
         input_index < inputs;
         ++input_index) {
        total += (
            x[sample_index * inputs + input_index]
            * weights[input_index * outputs + output_index]
        );
    }

    output[sample_index * outputs + output_index] = total;
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

    for (unsigned int sample_index = 0;
         sample_index < samples;
         ++sample_index) {
        total += (
            x[sample_index * inputs + input_index]
            * error[sample_index * outputs + output_index]
        );
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

    for (unsigned int sample_index = 0;
         sample_index < samples;
         ++sample_index) {
        total += error[sample_index * outputs + output_index];
    }

    bias_gradient[output_index] = total / (float)samples;
}


__kernel void linear_input_gradient(
    __global const float *error,
    __global const float *weights,
    __global float *input_gradient,
    const unsigned int samples,
    const unsigned int inputs,
    const unsigned int outputs
) {
    const unsigned int input_index = get_global_id(0);
    const unsigned int sample_index = get_global_id(1);

    if (sample_index >= samples || input_index >= inputs) {
        return;
    }

    float total = 0.0f;

    for (unsigned int output_index = 0;
         output_index < outputs;
         ++output_index) {
        total += (
            error[sample_index * outputs + output_index]
            * weights[input_index * outputs + output_index]
        );
    }

    input_gradient[sample_index * inputs + input_index] = total;
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
        "Could not find the RX 480 through Rusticl. "
        "Make sure RUSTICL_ENABLE=radeonsi is set."
    )


def copy_from_gpu(
    queue: cl.CommandQueue,
    destination: np.ndarray,
    source: cl.Buffer,
) -> np.ndarray:
    cl.enqueue_copy(queue, destination, source).wait()
    return destination


def main() -> int:
    samples = 4096
    inputs = 32
    outputs = 8
    learning_rate = np.float32(0.1)
    training_steps = 200

    rng = np.random.default_rng(42)

    x = rng.normal(
        0.0,
        1.0,
        size=(samples, inputs),
    ).astype(np.float32)

    true_weights = rng.normal(
        0.0,
        0.5,
        size=(inputs, outputs),
    ).astype(np.float32)

    true_bias = rng.normal(
        0.0,
        0.2,
        size=(outputs,),
    ).astype(np.float32)

    target = (
        x @ true_weights + true_bias
    ).astype(np.float32)

    initial_weights = rng.normal(
        0.0,
        0.05,
        size=(inputs, outputs),
    ).astype(np.float32)

    initial_bias = np.zeros(
        outputs,
        dtype=np.float32,
    )

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

    forward_kernel = cl.Kernel(
        program,
        "linear_forward",
    )
    error_kernel = cl.Kernel(
        program,
        "calculate_error",
    )
    weight_gradient_kernel = cl.Kernel(
        program,
        "linear_weight_gradient",
    )
    bias_gradient_kernel = cl.Kernel(
        program,
        "linear_bias_gradient",
    )
    input_gradient_kernel = cl.Kernel(
        program,
        "linear_input_gradient",
    )
    update_weights_kernel = cl.Kernel(
        program,
        "sgd_update",
    )
    update_bias_kernel = cl.Kernel(
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
    weights_gpu = cl.Buffer(
        context,
        flags.READ_WRITE | flags.COPY_HOST_PTR,
        hostbuf=initial_weights,
    )
    bias_gpu = cl.Buffer(
        context,
        flags.READ_WRITE | flags.COPY_HOST_PTR,
        hostbuf=initial_bias,
    )

    prediction_gpu = cl.Buffer(
        context,
        flags.READ_WRITE,
        target.nbytes,
    )
    error_gpu = cl.Buffer(
        context,
        flags.READ_WRITE,
        target.nbytes,
    )
    weight_gradient_gpu = cl.Buffer(
        context,
        flags.READ_WRITE,
        initial_weights.nbytes,
    )
    bias_gradient_gpu = cl.Buffer(
        context,
        flags.READ_WRITE,
        initial_bias.nbytes,
    )
    input_gradient_gpu = cl.Buffer(
        context,
        flags.WRITE_ONLY,
        x.nbytes,
    )

    forward_kernel.set_args(
        x_gpu,
        weights_gpu,
        bias_gpu,
        prediction_gpu,
        np.uint32(samples),
        np.uint32(inputs),
        np.uint32(outputs),
    )

    error_kernel.set_args(
        prediction_gpu,
        target_gpu,
        error_gpu,
        np.uint32(samples * outputs),
    )

    weight_gradient_kernel.set_args(
        x_gpu,
        error_gpu,
        weight_gradient_gpu,
        np.uint32(samples),
        np.uint32(inputs),
        np.uint32(outputs),
    )

    bias_gradient_kernel.set_args(
        error_gpu,
        bias_gradient_gpu,
        np.uint32(samples),
        np.uint32(outputs),
    )

    input_gradient_kernel.set_args(
        error_gpu,
        weights_gpu,
        input_gradient_gpu,
        np.uint32(samples),
        np.uint32(inputs),
        np.uint32(outputs),
    )

    update_weights_kernel.set_args(
        weights_gpu,
        weight_gradient_gpu,
        learning_rate,
        np.uint32(inputs * outputs),
    )

    update_bias_kernel.set_args(
        bias_gpu,
        bias_gradient_gpu,
        learning_rate,
        np.uint32(outputs),
    )

    def calculate_forward_and_gradients(
        include_input_gradient: bool = False,
    ) -> None:
        cl.enqueue_nd_range_kernel(
            queue,
            forward_kernel,
            (outputs, samples),
            None,
        )

        cl.enqueue_nd_range_kernel(
            queue,
            error_kernel,
            (samples * outputs,),
            None,
        )

        cl.enqueue_nd_range_kernel(
            queue,
            weight_gradient_kernel,
            (inputs, outputs),
            None,
        )

        cl.enqueue_nd_range_kernel(
            queue,
            bias_gradient_kernel,
            (outputs,),
            None,
        )

        if include_input_gradient:
            cl.enqueue_nd_range_kernel(
                queue,
                input_gradient_kernel,
                (inputs, samples),
                None,
            )

        queue.finish()

    calculate_forward_and_gradients(
        include_input_gradient=True,
    )

    gpu_prediction = np.empty_like(target)
    gpu_error = np.empty_like(target)
    gpu_weight_gradient = np.empty_like(initial_weights)
    gpu_bias_gradient = np.empty_like(initial_bias)
    gpu_input_gradient = np.empty_like(x)

    copy_from_gpu(
        queue,
        gpu_prediction,
        prediction_gpu,
    )
    copy_from_gpu(
        queue,
        gpu_error,
        error_gpu,
    )
    copy_from_gpu(
        queue,
        gpu_weight_gradient,
        weight_gradient_gpu,
    )
    copy_from_gpu(
        queue,
        gpu_bias_gradient,
        bias_gradient_gpu,
    )
    copy_from_gpu(
        queue,
        gpu_input_gradient,
        input_gradient_gpu,
    )

    cpu_prediction = (
        x @ initial_weights + initial_bias
    ).astype(np.float32)

    cpu_error = (
        cpu_prediction - target
    ).astype(np.float32)

    cpu_weight_gradient = (
        x.T @ cpu_error / samples
    ).astype(np.float32)

    cpu_bias_gradient = (
        np.mean(cpu_error, axis=0)
    ).astype(np.float32)

    cpu_input_gradient = (
        cpu_error @ initial_weights.T
    ).astype(np.float32)

    prediction_error = float(
        np.max(
            np.abs(
                gpu_prediction - cpu_prediction
            )
        )
    )

    weight_gradient_error = float(
        np.max(
            np.abs(
                gpu_weight_gradient
                - cpu_weight_gradient
            )
        )
    )

    bias_gradient_error = float(
        np.max(
            np.abs(
                gpu_bias_gradient
                - cpu_bias_gradient
            )
        )
    )

    input_gradient_error = float(
        np.max(
            np.abs(
                gpu_input_gradient
                - cpu_input_gradient
            )
        )
    )

    forward_verified = bool(
        np.allclose(
            gpu_prediction,
            cpu_prediction,
            rtol=2e-3,
            atol=2e-3,
        )
    )

    weight_gradient_verified = bool(
        np.allclose(
            gpu_weight_gradient,
            cpu_weight_gradient,
            rtol=3e-3,
            atol=3e-3,
        )
    )

    bias_gradient_verified = bool(
        np.allclose(
            gpu_bias_gradient,
            cpu_bias_gradient,
            rtol=3e-3,
            atol=3e-3,
        )
    )

    input_gradient_verified = bool(
        np.allclose(
            gpu_input_gradient,
            cpu_input_gradient,
            rtol=3e-3,
            atol=3e-3,
        )
    )

    all_operations_verified = all(
        (
            forward_verified,
            weight_gradient_verified,
            bias_gradient_verified,
            input_gradient_verified,
        )
    )

    initial_loss = float(
        0.5
        * np.mean(
            np.sum(
                gpu_error * gpu_error,
                axis=1,
            )
        )
    )

    print("=== NUMERICAL VERIFICATION ===")
    print(
        f"Forward max error:     "
        f"{prediction_error:.8g}"
    )
    print(
        f"Weight gradient error: "
        f"{weight_gradient_error:.8g}"
    )
    print(
        f"Bias gradient error:   "
        f"{bias_gradient_error:.8g}"
    )
    print(
        f"Input gradient error:  "
        f"{input_gradient_error:.8g}"
    )
    print(
        f"All operations valid:  "
        f"{all_operations_verified}"
    )
    print()

    if not all_operations_verified:
        print(
            "FAILED: GPU forward/backward did not "
            "match the NumPy reference."
        )
        return 1

    print("=== GPU TRAINING ===")
    print(f"Step   0 | loss {initial_loss:.10f}")

    training_start = time.perf_counter()

    report_steps = {
        1,
        10,
        20,
        50,
        100,
        150,
        200,
    }

    current_loss = initial_loss

    for step in range(1, training_steps + 1):
        cl.enqueue_nd_range_kernel(
            queue,
            update_weights_kernel,
            (inputs * outputs,),
            None,
        )

        cl.enqueue_nd_range_kernel(
            queue,
            update_bias_kernel,
            (outputs,),
            None,
        )

        calculate_forward_and_gradients()

        if step in report_steps:
            copy_from_gpu(
                queue,
                gpu_error,
                error_gpu,
            )

            current_loss = float(
                0.5
                * np.mean(
                    np.sum(
                        gpu_error * gpu_error,
                        axis=1,
                    )
                )
            )

            print(
                f"Step {step:3d} | "
                f"loss {current_loss:.10f}"
            )

    queue.finish()
    training_seconds = (
        time.perf_counter() - training_start
    )

    final_weights = np.empty_like(initial_weights)
    final_bias = np.empty_like(initial_bias)

    copy_from_gpu(
        queue,
        final_weights,
        weights_gpu,
    )
    copy_from_gpu(
        queue,
        final_bias,
        bias_gpu,
    )
    copy_from_gpu(
        queue,
        gpu_error,
        error_gpu,
    )

    final_loss = float(
        0.5
        * np.mean(
            np.sum(
                gpu_error * gpu_error,
                axis=1,
            )
        )
    )

    weight_recovery_error = float(
        np.max(
            np.abs(
                final_weights - true_weights
            )
        )
    )

    bias_recovery_error = float(
        np.max(
            np.abs(
                final_bias - true_bias
            )
        )
    )

    loss_reduction = (
        initial_loss / max(final_loss, 1e-30)
    )

    training_passed = bool(
        final_loss < initial_loss * 1e-5
    )

    receipt = {
        "milestone": "verified_gpu_linear_training",
        "device": device.name,
        "driver": device.driver_version,
        "configuration": {
            "samples": samples,
            "inputs": inputs,
            "outputs": outputs,
            "learning_rate": float(
                learning_rate
            ),
            "training_steps": training_steps,
        },
        "verification": {
            "forward_max_error": prediction_error,
            "weight_gradient_max_error":
                weight_gradient_error,
            "bias_gradient_max_error":
                bias_gradient_error,
            "input_gradient_max_error":
                input_gradient_error,
            "all_operations_verified":
                all_operations_verified,
        },
        "training": {
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "loss_reduction_factor":
                loss_reduction,
            "training_seconds":
                training_seconds,
            "weight_recovery_max_error":
                weight_recovery_error,
            "bias_recovery_max_error":
                bias_recovery_error,
            "passed": training_passed,
        },
    }

    receipt_path = Path(
        "linear_training_receipt.json"
    )

    receipt_path.write_text(
        json.dumps(
            receipt,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=== RESULT ===")
    print(f"Initial loss:       {initial_loss:.10f}")
    print(f"Final loss:         {final_loss:.10f}")
    print(
        f"Loss reduction:     "
        f"{loss_reduction:.2f}x"
    )
    print(
        f"Training time:      "
        f"{training_seconds:.3f} seconds"
    )
    print(
        f"Weight recovery err:"
        f" {weight_recovery_error:.8g}"
    )
    print(
        f"Bias recovery err:  "
        f"{bias_recovery_error:.8g}"
    )
    print(
        f"Receipt:            "
        f"{receipt_path}"
    )

    if not training_passed:
        print(
            "FAILED: gradients were correct, but "
            "the loss did not fall enough."
        )
        return 1

    print()
    print(
        "PASSED: RX 480 completed verified "
        "forward propagation, backpropagation, "
        "and GPU training."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
