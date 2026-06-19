from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pyopencl as cl


SEQUENCE_LENGTH = 8
INPUT_SIZE = 2
HIDDEN_SIZE = 8

WP_OFFSET = 0
UP_OFFSET = WP_OFFSET + INPUT_SIZE * HIDDEN_SIZE
BP_OFFSET = UP_OFFSET + HIDDEN_SIZE * HIDDEN_SIZE

WG_OFFSET = BP_OFFSET + HIDDEN_SIZE
UG_OFFSET = WG_OFFSET + INPUT_SIZE * HIDDEN_SIZE
BG_OFFSET = UG_OFFSET + HIDDEN_SIZE * HIDDEN_SIZE

WO_OFFSET = BG_OFFSET + HIDDEN_SIZE
BO_OFFSET = WO_OFFSET + HIDDEN_SIZE

PARAMETER_COUNT = BO_OFFSET + 1


KERNEL_SOURCE = rf"""
#define SEQUENCE_LENGTH {SEQUENCE_LENGTH}
#define INPUT_SIZE {INPUT_SIZE}
#define HIDDEN_SIZE {HIDDEN_SIZE}

#define WP_OFFSET {WP_OFFSET}
#define UP_OFFSET {UP_OFFSET}
#define BP_OFFSET {BP_OFFSET}

#define WG_OFFSET {WG_OFFSET}
#define UG_OFFSET {UG_OFFSET}
#define BG_OFFSET {BG_OFFSET}

#define WO_OFFSET {WO_OFFSET}
#define BO_OFFSET {BO_OFFSET}

#define PARAMETER_COUNT {PARAMETER_COUNT}


__kernel void tension_forward_backward(
    __global const float *inputs,
    __global const float *targets,
    __global const float *parameters,
    __global float *predictions,
    __global float *errors,
    __global float *sample_gradients,
    const unsigned int samples
) {{
    const unsigned int sample_index = get_global_id(0);

    if (sample_index >= samples) {{
        return;
    }}

    float hidden[SEQUENCE_LENGTH + 1][HIDDEN_SIZE];
    float proposals[SEQUENCE_LENGTH][HIDDEN_SIZE];
    float gates[SEQUENCE_LENGTH][HIDDEN_SIZE];
    float gradient[PARAMETER_COUNT];

    for (
        unsigned int parameter_index = 0;
        parameter_index < PARAMETER_COUNT;
        ++parameter_index
    ) {{
        gradient[parameter_index] = 0.0f;
    }}

    for (
        unsigned int hidden_index = 0;
        hidden_index < HIDDEN_SIZE;
        ++hidden_index
    ) {{
        hidden[0][hidden_index] = 0.0f;
    }}

    /*
     * Forward recurrence:
     *
     * proposal = tanh(x Wp + h Up + bp)
     * gate     = sigmoid(x Wg + h Ug + bg)
     * new_h    = h + gate * (proposal - h)
     */
    for (
        unsigned int time_index = 0;
        time_index < SEQUENCE_LENGTH;
        ++time_index
    ) {{
        for (
            unsigned int output_index = 0;
            output_index < HIDDEN_SIZE;
            ++output_index
        ) {{
            float proposal_z =
                parameters[BP_OFFSET + output_index];

            float gate_z =
                parameters[BG_OFFSET + output_index];

            for (
                unsigned int input_index = 0;
                input_index < INPUT_SIZE;
                ++input_index
            ) {{
                const float input_value =
                    inputs[
                        (
                            sample_index
                            * SEQUENCE_LENGTH
                            + time_index
                        )
                        * INPUT_SIZE
                        + input_index
                    ];

                proposal_z +=
                    input_value
                    * parameters[
                        WP_OFFSET
                        + input_index * HIDDEN_SIZE
                        + output_index
                    ];

                gate_z +=
                    input_value
                    * parameters[
                        WG_OFFSET
                        + input_index * HIDDEN_SIZE
                        + output_index
                    ];
            }}

            for (
                unsigned int hidden_index = 0;
                hidden_index < HIDDEN_SIZE;
                ++hidden_index
            ) {{
                const float previous_hidden =
                    hidden[time_index][hidden_index];

                proposal_z +=
                    previous_hidden
                    * parameters[
                        UP_OFFSET
                        + hidden_index * HIDDEN_SIZE
                        + output_index
                    ];

                gate_z +=
                    previous_hidden
                    * parameters[
                        UG_OFFSET
                        + hidden_index * HIDDEN_SIZE
                        + output_index
                    ];
            }}

            gate_z = clamp(
                gate_z,
                -20.0f,
                20.0f
            );

            const float proposal =
                tanh(proposal_z);

            const float gate =
                1.0f / (1.0f + exp(-gate_z));

            const float previous_value =
                hidden[time_index][output_index];

            proposals[time_index][output_index] =
                proposal;

            gates[time_index][output_index] =
                gate;

            hidden[time_index + 1][output_index] =
                previous_value
                + gate * (
                    proposal - previous_value
                );
        }}
    }}

    float prediction = parameters[BO_OFFSET];

    for (
        unsigned int hidden_index = 0;
        hidden_index < HIDDEN_SIZE;
        ++hidden_index
    ) {{
        prediction +=
            hidden[SEQUENCE_LENGTH][hidden_index]
            * parameters[
                WO_OFFSET + hidden_index
            ];
    }}

    const float error =
        prediction - targets[sample_index];

    predictions[sample_index] = prediction;
    errors[sample_index] = error;

    gradient[BO_OFFSET] = error;

    float hidden_gradient[HIDDEN_SIZE];

    for (
        unsigned int hidden_index = 0;
        hidden_index < HIDDEN_SIZE;
        ++hidden_index
    ) {{
        gradient[WO_OFFSET + hidden_index] =
            error
            * hidden[
                SEQUENCE_LENGTH
            ][hidden_index];

        hidden_gradient[hidden_index] =
            error
            * parameters[
                WO_OFFSET + hidden_index
            ];
    }}

    /*
     * Backpropagation through time.
     */
    for (
        int time_index = SEQUENCE_LENGTH - 1;
        time_index >= 0;
        --time_index
    ) {{
        float previous_hidden_gradient[HIDDEN_SIZE];

        for (
            unsigned int hidden_index = 0;
            hidden_index < HIDDEN_SIZE;
            ++hidden_index
        ) {{
            previous_hidden_gradient[hidden_index] =
                0.0f;
        }}

        for (
            unsigned int output_index = 0;
            output_index < HIDDEN_SIZE;
            ++output_index
        ) {{
            const float previous_hidden =
                hidden[time_index][output_index];

            const float proposal =
                proposals[time_index][output_index];

            const float gate =
                gates[time_index][output_index];

            const float current_gradient =
                hidden_gradient[output_index];

            const float proposal_gradient =
                current_gradient * gate;

            const float gate_gradient =
                current_gradient
                * (proposal - previous_hidden);

            const float proposal_z_gradient =
                proposal_gradient
                * (1.0f - proposal * proposal);

            const float gate_z_gradient =
                gate_gradient
                * gate
                * (1.0f - gate);

            /*
             * Direct path:
             *
             * h_new = (1 - gate) * h_old
             *         + gate * proposal
             */
            previous_hidden_gradient[output_index] +=
                current_gradient
                * (1.0f - gate);

            gradient[BP_OFFSET + output_index] +=
                proposal_z_gradient;

            gradient[BG_OFFSET + output_index] +=
                gate_z_gradient;

            for (
                unsigned int input_index = 0;
                input_index < INPUT_SIZE;
                ++input_index
            ) {{
                const float input_value =
                    inputs[
                        (
                            sample_index
                            * SEQUENCE_LENGTH
                            + time_index
                        )
                        * INPUT_SIZE
                        + input_index
                    ];

                gradient[
                    WP_OFFSET
                    + input_index * HIDDEN_SIZE
                    + output_index
                ] +=
                    input_value
                    * proposal_z_gradient;

                gradient[
                    WG_OFFSET
                    + input_index * HIDDEN_SIZE
                    + output_index
                ] +=
                    input_value
                    * gate_z_gradient;
            }}

            for (
                unsigned int hidden_index = 0;
                hidden_index < HIDDEN_SIZE;
                ++hidden_index
            ) {{
                const float hidden_value =
                    hidden[
                        time_index
                    ][hidden_index];

                gradient[
                    UP_OFFSET
                    + hidden_index * HIDDEN_SIZE
                    + output_index
                ] +=
                    hidden_value
                    * proposal_z_gradient;

                gradient[
                    UG_OFFSET
                    + hidden_index * HIDDEN_SIZE
                    + output_index
                ] +=
                    hidden_value
                    * gate_z_gradient;

                previous_hidden_gradient[
                    hidden_index
                ] +=
                    proposal_z_gradient
                    * parameters[
                        UP_OFFSET
                        + hidden_index * HIDDEN_SIZE
                        + output_index
                    ]
                    + gate_z_gradient
                    * parameters[
                        UG_OFFSET
                        + hidden_index * HIDDEN_SIZE
                        + output_index
                    ];
            }}
        }}

        for (
            unsigned int hidden_index = 0;
            hidden_index < HIDDEN_SIZE;
            ++hidden_index
        ) {{
            hidden_gradient[hidden_index] =
                previous_hidden_gradient[
                    hidden_index
                ];
        }}
    }}

    for (
        unsigned int parameter_index = 0;
        parameter_index < PARAMETER_COUNT;
        ++parameter_index
    ) {{
        sample_gradients[
            sample_index * PARAMETER_COUNT
            + parameter_index
        ] = gradient[parameter_index];
    }}
}}


__kernel void reduce_mean_gradients(
    __global const float *sample_gradients,
    __global float *mean_gradients,
    const unsigned int samples
) {{
    const unsigned int parameter_index =
        get_global_id(0);

    if (parameter_index >= PARAMETER_COUNT) {{
        return;
    }}

    float total = 0.0f;

    for (
        unsigned int sample_index = 0;
        sample_index < samples;
        ++sample_index
    ) {{
        total +=
            sample_gradients[
                sample_index * PARAMETER_COUNT
                + parameter_index
            ];
    }}

    mean_gradients[parameter_index] =
        total / (float)samples;
}}


__kernel void adam_update(
    __global float *parameters,
    __global const float *gradients,
    __global float *first_moments,
    __global float *second_moments,
    const float learning_rate,
    const float beta_1,
    const float beta_2,
    const float epsilon,
    const unsigned int step,
    const unsigned int parameter_count
) {{
    const unsigned int parameter_index =
        get_global_id(0);

    if (parameter_index >= parameter_count) {{
        return;
    }}

    const float gradient =
        gradients[parameter_index];

    const float first_moment =
        beta_1 * first_moments[parameter_index]
        + (1.0f - beta_1) * gradient;

    const float second_moment =
        beta_2 * second_moments[parameter_index]
        + (1.0f - beta_2)
        * gradient * gradient;

    first_moments[parameter_index] =
        first_moment;

    second_moments[parameter_index] =
        second_moment;

    const float first_correction =
        1.0f - pow(
            beta_1,
            (float)step
        );

    const float second_correction =
        1.0f - pow(
            beta_2,
            (float)step
        );

    const float corrected_first =
        first_moment / first_correction;

    const float corrected_second =
        second_moment / second_correction;

    parameters[parameter_index] -=
        learning_rate
        * corrected_first
        / (
            sqrt(corrected_second)
            + epsilon
        );
}}
"""


def find_rx480() -> tuple[cl.Platform, cl.Device]:
    for platform in cl.get_platforms():
        if "rusticl" not in platform.name.lower():
            continue

        for device in platform.get_devices():
            is_gpu = bool(
                device.type & cl.device_type.GPU
            )

            if (
                is_gpu
                and "radeon" in device.name.lower()
            ):
                return platform, device

    raise RuntimeError(
        "RX 480 not found through Rusticl. "
        "Set RUSTICL_ENABLE=radeonsi."
    )


def unpack_parameters(
    parameters: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.float32,
]:
    wp = parameters[
        WP_OFFSET:UP_OFFSET
    ].reshape(INPUT_SIZE, HIDDEN_SIZE)

    up = parameters[
        UP_OFFSET:BP_OFFSET
    ].reshape(HIDDEN_SIZE, HIDDEN_SIZE)

    bp = parameters[
        BP_OFFSET:WG_OFFSET
    ]

    wg = parameters[
        WG_OFFSET:UG_OFFSET
    ].reshape(INPUT_SIZE, HIDDEN_SIZE)

    ug = parameters[
        UG_OFFSET:BG_OFFSET
    ].reshape(HIDDEN_SIZE, HIDDEN_SIZE)

    bg = parameters[
        BG_OFFSET:WO_OFFSET
    ]

    wo = parameters[
        WO_OFFSET:BO_OFFSET
    ]

    bo = np.float32(
        parameters[BO_OFFSET]
    )

    return wp, up, bp, wg, ug, bg, wo, bo


def numpy_reference(
    inputs: np.ndarray,
    targets: np.ndarray,
    parameters: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    samples = inputs.shape[0]

    (
        wp,
        up,
        bp,
        wg,
        ug,
        bg,
        wo,
        bo,
    ) = unpack_parameters(parameters)

    hidden = np.zeros(
        (
            samples,
            SEQUENCE_LENGTH + 1,
            HIDDEN_SIZE,
        ),
        dtype=np.float32,
    )

    proposals = np.zeros(
        (
            samples,
            SEQUENCE_LENGTH,
            HIDDEN_SIZE,
        ),
        dtype=np.float32,
    )

    gates = np.zeros_like(proposals)

    for time_index in range(SEQUENCE_LENGTH):
        proposal_z = (
            inputs[:, time_index] @ wp
            + hidden[:, time_index] @ up
            + bp
        )

        gate_z = (
            inputs[:, time_index] @ wg
            + hidden[:, time_index] @ ug
            + bg
        )

        proposal = np.tanh(
            proposal_z
        ).astype(np.float32)

        gate = (
            1.0
            / (
                1.0
                + np.exp(
                    -np.clip(
                        gate_z,
                        -20.0,
                        20.0,
                    )
                )
            )
        ).astype(np.float32)

        proposals[:, time_index] = proposal
        gates[:, time_index] = gate

        previous_hidden = hidden[:, time_index]

        hidden[:, time_index + 1] = (
            previous_hidden
            + gate
            * (
                proposal
                - previous_hidden
            )
        )

    predictions = (
        hidden[:, SEQUENCE_LENGTH] @ wo
        + bo
    ).astype(np.float32)

    errors = (
        predictions - targets
    ).astype(np.float32)

    wp_gradient = np.zeros_like(wp)
    up_gradient = np.zeros_like(up)
    bp_gradient = np.zeros_like(bp)

    wg_gradient = np.zeros_like(wg)
    ug_gradient = np.zeros_like(ug)
    bg_gradient = np.zeros_like(bg)

    wo_gradient = np.mean(
        hidden[:, SEQUENCE_LENGTH]
        * errors[:, None],
        axis=0,
    ).astype(np.float32)

    bo_gradient = np.float32(
        np.mean(errors)
    )

    hidden_gradient = (
        errors[:, None] * wo[None, :]
    ).astype(np.float32)

    for time_index in range(
        SEQUENCE_LENGTH - 1,
        -1,
        -1,
    ):
        previous_hidden = hidden[:, time_index]
        proposal = proposals[:, time_index]
        gate = gates[:, time_index]

        proposal_z_gradient = (
            hidden_gradient
            * gate
            * (
                1.0
                - proposal * proposal
            )
        ).astype(np.float32)

        gate_z_gradient = (
            hidden_gradient
            * (
                proposal
                - previous_hidden
            )
            * gate
            * (
                1.0 - gate
            )
        ).astype(np.float32)

        wp_gradient += (
            inputs[:, time_index].T
            @ proposal_z_gradient
            / samples
        ).astype(np.float32)

        up_gradient += (
            previous_hidden.T
            @ proposal_z_gradient
            / samples
        ).astype(np.float32)

        bp_gradient += np.mean(
            proposal_z_gradient,
            axis=0,
        ).astype(np.float32)

        wg_gradient += (
            inputs[:, time_index].T
            @ gate_z_gradient
            / samples
        ).astype(np.float32)

        ug_gradient += (
            previous_hidden.T
            @ gate_z_gradient
            / samples
        ).astype(np.float32)

        bg_gradient += np.mean(
            gate_z_gradient,
            axis=0,
        ).astype(np.float32)

        hidden_gradient = (
            hidden_gradient
            * (
                1.0 - gate
            )
            + proposal_z_gradient @ up.T
            + gate_z_gradient @ ug.T
        ).astype(np.float32)

    gradient = np.zeros(
        PARAMETER_COUNT,
        dtype=np.float32,
    )

    gradient[
        WP_OFFSET:UP_OFFSET
    ] = wp_gradient.reshape(-1)

    gradient[
        UP_OFFSET:BP_OFFSET
    ] = up_gradient.reshape(-1)

    gradient[
        BP_OFFSET:WG_OFFSET
    ] = bp_gradient

    gradient[
        WG_OFFSET:UG_OFFSET
    ] = wg_gradient.reshape(-1)

    gradient[
        UG_OFFSET:BG_OFFSET
    ] = ug_gradient.reshape(-1)

    gradient[
        BG_OFFSET:WO_OFFSET
    ] = bg_gradient

    gradient[
        WO_OFFSET:BO_OFFSET
    ] = wo_gradient

    gradient[BO_OFFSET] = bo_gradient

    return predictions, errors, gradient, gates


def read_buffer(
    queue: cl.CommandQueue,
    destination: np.ndarray,
    source: cl.Buffer,
) -> None:
    cl.enqueue_copy(
        queue,
        destination,
        source,
    ).wait()


def main() -> int:
    samples = 2048
    validation_samples = 512
    training_steps = 1000

    learning_rate = np.float32(0.01)
    beta_1 = np.float32(0.9)
    beta_2 = np.float32(0.999)
    epsilon = np.float32(1e-8)

    rng = np.random.default_rng(42)

    # Input channel 0 contains values.
    # Input channel 1 is a cue.
    #
    # The cue is active only on the first timestep.
    # The model must remember that first value while
    # ignoring seven later distractor values.
    inputs = rng.uniform(
        -1.0,
        1.0,
        size=(
            samples,
            SEQUENCE_LENGTH,
            INPUT_SIZE,
        ),
    ).astype(np.float32)

    targets = inputs[:, 0, 0].copy()

    inputs[:, :, 1] = 0.0
    inputs[:, 0, 1] = 1.0

    validation_inputs = rng.uniform(
        -1.0,
        1.0,
        size=(
            validation_samples,
            SEQUENCE_LENGTH,
            INPUT_SIZE,
        ),
    ).astype(np.float32)

    validation_targets = (
        validation_inputs[:, 0, 0].copy()
    )

    validation_inputs[:, :, 1] = 0.0
    validation_inputs[:, 0, 1] = 1.0

    parameters = np.zeros(
        PARAMETER_COUNT,
        dtype=np.float32,
    )

    parameters[
        WP_OFFSET:UP_OFFSET
    ] = rng.normal(
        0.0,
        0.25,
        size=INPUT_SIZE * HIDDEN_SIZE,
    ).astype(np.float32)

    parameters[
        UP_OFFSET:BP_OFFSET
    ] = rng.normal(
        0.0,
        0.15,
        size=HIDDEN_SIZE * HIDDEN_SIZE,
    ).astype(np.float32)

    parameters[
        WG_OFFSET:UG_OFFSET
    ] = rng.normal(
        0.0,
        0.25,
        size=INPUT_SIZE * HIDDEN_SIZE,
    ).astype(np.float32)

    parameters[
        UG_OFFSET:BG_OFFSET
    ] = rng.normal(
        0.0,
        0.15,
        size=HIDDEN_SIZE * HIDDEN_SIZE,
    ).astype(np.float32)

    parameters[
        WO_OFFSET:BO_OFFSET
    ] = rng.normal(
        0.0,
        0.25,
        size=HIDDEN_SIZE,
    ).astype(np.float32)

    initial_parameters = parameters.copy()

    platform, device = find_rx480()

    print("=== DEVICE ===")
    print(f"Platform: {platform.name}")
    print(f"Device:   {device.name}")
    print(f"Driver:   {device.driver_version}")
    print()

    print("=== CONFIGURATION ===")
    print(f"Sequence length: {SEQUENCE_LENGTH}")
    print(f"Input size:      {INPUT_SIZE}")
    print(f"Hidden size:     {HIDDEN_SIZE}")
    print(f"Parameters:      {PARAMETER_COUNT}")
    print(f"Samples:         {samples}")
    print()

    context = cl.Context([device])
    queue = cl.CommandQueue(context)

    program = cl.Program(
        context,
        KERNEL_SOURCE,
    ).build()

    forward_backward_kernel = cl.Kernel(
        program,
        "tension_forward_backward",
    )

    reduction_kernel = cl.Kernel(
        program,
        "reduce_mean_gradients",
    )

    adam_kernel = cl.Kernel(
        program,
        "adam_update",
    )

    flags = cl.mem_flags

    inputs_gpu = cl.Buffer(
        context,
        flags.READ_ONLY | flags.COPY_HOST_PTR,
        hostbuf=inputs,
    )

    targets_gpu = cl.Buffer(
        context,
        flags.READ_ONLY | flags.COPY_HOST_PTR,
        hostbuf=targets,
    )

    parameters_gpu = cl.Buffer(
        context,
        flags.READ_WRITE | flags.COPY_HOST_PTR,
        hostbuf=parameters,
    )

    predictions_gpu = cl.Buffer(
        context,
        flags.READ_WRITE,
        targets.nbytes,
    )

    errors_gpu = cl.Buffer(
        context,
        flags.READ_WRITE,
        targets.nbytes,
    )

    sample_gradients_gpu = cl.Buffer(
        context,
        flags.READ_WRITE,
        samples
        * PARAMETER_COUNT
        * np.dtype(np.float32).itemsize,
    )

    mean_gradients_gpu = cl.Buffer(
        context,
        flags.READ_WRITE,
        parameters.nbytes,
    )

    first_moments = np.zeros_like(parameters)
    second_moments = np.zeros_like(parameters)

    first_moments_gpu = cl.Buffer(
        context,
        flags.READ_WRITE | flags.COPY_HOST_PTR,
        hostbuf=first_moments,
    )

    second_moments_gpu = cl.Buffer(
        context,
        flags.READ_WRITE | flags.COPY_HOST_PTR,
        hostbuf=second_moments,
    )

    forward_backward_kernel.set_args(
        inputs_gpu,
        targets_gpu,
        parameters_gpu,
        predictions_gpu,
        errors_gpu,
        sample_gradients_gpu,
        np.uint32(samples),
    )

    reduction_kernel.set_args(
        sample_gradients_gpu,
        mean_gradients_gpu,
        np.uint32(samples),
    )

    def calculate_gradients() -> None:
        cl.enqueue_nd_range_kernel(
            queue,
            forward_backward_kernel,
            (samples,),
            None,
        )

        cl.enqueue_nd_range_kernel(
            queue,
            reduction_kernel,
            (PARAMETER_COUNT,),
            None,
        )

        queue.finish()

    calculate_gradients()

    gpu_predictions = np.empty_like(targets)
    gpu_errors = np.empty_like(targets)
    gpu_gradients = np.empty_like(parameters)

    read_buffer(
        queue,
        gpu_predictions,
        predictions_gpu,
    )

    read_buffer(
        queue,
        gpu_errors,
        errors_gpu,
    )

    read_buffer(
        queue,
        gpu_gradients,
        mean_gradients_gpu,
    )

    (
        cpu_predictions,
        cpu_errors,
        cpu_gradients,
        initial_gates,
    ) = numpy_reference(
        inputs,
        targets,
        initial_parameters,
    )

    prediction_max_error = float(
        np.max(
            np.abs(
                gpu_predictions
                - cpu_predictions
            )
        )
    )

    error_max_error = float(
        np.max(
            np.abs(
                gpu_errors - cpu_errors
            )
        )
    )

    gradient_max_error = float(
        np.max(
            np.abs(
                gpu_gradients
                - cpu_gradients
            )
        )
    )

    predictions_verified = bool(
        np.allclose(
            gpu_predictions,
            cpu_predictions,
            rtol=5e-3,
            atol=5e-4,
        )
    )

    gradients_verified = bool(
        np.allclose(
            gpu_gradients,
            cpu_gradients,
            rtol=5e-3,
            atol=5e-4,
        )
    )

    all_verified = bool(
        predictions_verified
        and gradients_verified
    )

    initial_loss = float(
        0.5
        * np.mean(
            gpu_errors * gpu_errors
        )
    )

    print("=== BPTT VERIFICATION ===")
    print(
        f"Prediction max error: "
        f"{prediction_max_error:.8g}"
    )
    print(
        f"Error max error:      "
        f"{error_max_error:.8g}"
    )
    print(
        f"Gradient max error:   "
        f"{gradient_max_error:.8g}"
    )
    print(
        f"All operations valid: "
        f"{all_verified}"
    )
    print()

    if not all_verified:
        print(
            "FAILED: recurrent GPU gradients "
            "did not match NumPy."
        )
        return 1

    print("=== RECURRENT TENSION TRAINING ===")
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
        1000,
    }

    training_start = time.perf_counter()

    for step in range(
        1,
        training_steps + 1,
    ):
        adam_kernel.set_args(
            parameters_gpu,
            mean_gradients_gpu,
            first_moments_gpu,
            second_moments_gpu,
            learning_rate,
            beta_1,
            beta_2,
            epsilon,
            np.uint32(step),
            np.uint32(PARAMETER_COUNT),
        )

        cl.enqueue_nd_range_kernel(
            queue,
            adam_kernel,
            (PARAMETER_COUNT,),
            None,
        )

        calculate_gradients()

        if step in report_steps:
            read_buffer(
                queue,
                gpu_errors,
                errors_gpu,
            )

            current_loss = float(
                0.5
                * np.mean(
                    gpu_errors
                    * gpu_errors
                )
            )

            print(
                f"Step {step:4d} | "
                f"loss {current_loss:.10f}"
            )

    training_seconds = (
        time.perf_counter()
        - training_start
    )

    final_parameters = np.empty_like(parameters)

    read_buffer(
        queue,
        final_parameters,
        parameters_gpu,
    )

    read_buffer(
        queue,
        gpu_predictions,
        predictions_gpu,
    )

    read_buffer(
        queue,
        gpu_errors,
        errors_gpu,
    )

    final_loss = float(
        0.5
        * np.mean(
            gpu_errors * gpu_errors
        )
    )

    training_mean_absolute_error = float(
        np.mean(
            np.abs(
                gpu_predictions
                - targets
            )
        )
    )

    (
        final_cpu_predictions,
        _,
        _,
        final_training_gates,
    ) = numpy_reference(
        inputs,
        targets,
        final_parameters,
    )

    final_gpu_cpu_error = float(
        np.max(
            np.abs(
                gpu_predictions
                - final_cpu_predictions
            )
        )
    )

    (
        validation_predictions,
        validation_errors,
        _,
        validation_gates,
    ) = numpy_reference(
        validation_inputs,
        validation_targets,
        final_parameters,
    )

    validation_loss = float(
        0.5
        * np.mean(
            validation_errors
            * validation_errors
        )
    )

    validation_mean_absolute_error = float(
        np.mean(
            np.abs(
                validation_predictions
                - validation_targets
            )
        )
    )

    cue_gate_mean = float(
        np.mean(
            validation_gates[:, 0]
        )
    )

    later_gate_mean = float(
        np.mean(
            validation_gates[:, 1:]
        )
    )

    loss_reduction = (
        initial_loss
        / max(final_loss, 1e-30)
    )

    training_passed = bool(
        final_loss < 0.005
        and validation_loss < 0.01
        and final_loss
        < initial_loss * 0.05
        and final_gpu_cpu_error < 0.001
    )

    receipt = {
        "milestone":
            "verified_recurrent_tension_cell",
        "task":
            "delayed_recall_with_distractors",
        "device": device.name,
        "driver": device.driver_version,
        "configuration": {
            "sequence_length":
                SEQUENCE_LENGTH,
            "input_size": INPUT_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "parameter_count":
                PARAMETER_COUNT,
            "training_samples": samples,
            "validation_samples":
                validation_samples,
            "training_steps":
                training_steps,
            "learning_rate":
                float(learning_rate),
        },
        "verification": {
            "prediction_max_error":
                prediction_max_error,
            "error_max_error":
                error_max_error,
            "gradient_max_error":
                gradient_max_error,
            "final_gpu_cpu_error":
                final_gpu_cpu_error,
            "all_operations_verified":
                all_verified,
        },
        "training": {
            "initial_loss":
                initial_loss,
            "final_loss":
                final_loss,
            "loss_reduction_factor":
                loss_reduction,
            "mean_absolute_error":
                training_mean_absolute_error,
            "training_seconds":
                training_seconds,
        },
        "validation": {
            "loss":
                validation_loss,
            "mean_absolute_error":
                validation_mean_absolute_error,
            "cue_gate_mean":
                cue_gate_mean,
            "later_gate_mean":
                later_gate_mean,
        },
        "passed": training_passed,
    }

    receipt_path = Path(
        "receipts/"
        "tension_cell_training_receipt.json"
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
    print(
        f"Initial loss:       "
        f"{initial_loss:.10f}"
    )
    print(
        f"Final train loss:   "
        f"{final_loss:.10f}"
    )
    print(
        f"Validation loss:    "
        f"{validation_loss:.10f}"
    )
    print(
        f"Loss reduction:     "
        f"{loss_reduction:.2f}x"
    )
    print(
        f"Train mean error:   "
        f"{training_mean_absolute_error:.8f}"
    )
    print(
        f"Validation error:   "
        f"{validation_mean_absolute_error:.8f}"
    )
    print(
        f"Training time:      "
        f"{training_seconds:.3f}s"
    )
    print(
        f"GPU/CPU final error:"
        f" {final_gpu_cpu_error:.8g}"
    )
    print(
        f"Cue gate mean:      "
        f"{cue_gate_mean:.6f}"
    )
    print(
        f"Later gate mean:    "
        f"{later_gate_mean:.6f}"
    )
    print(
        f"Receipt:            "
        f"{receipt_path}"
    )

    print()
    print("Validation examples:")

    for index in range(8):
        print(
            f"  remembered "
            f"{validation_targets[index]: .3f}"
            f" | model "
            f"{validation_predictions[index]: .3f}"
            f" | error "
            f"{validation_errors[index]: .3f}"
        )

    if not training_passed:
        print()
        print(
            "FAILED: recurrence worked, but "
            "the delayed-recall target was "
            "not learned strongly enough."
        )
        return 1

    print()
    print(
        "PASSED: RX 480 trained a verified "
        "recurrent TensionCell with full "
        "backpropagation through time."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
