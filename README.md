# TensionForge

A small verifier-first GPU training runtime built for legacy commodity hardware.

## Current hardware

- AMD Radeon RX 480 Gaming X
- 8 GiB VRAM
- Mesa Rusticl OpenCL backend
- Linux Mint
- No ROCm or CUDA dependency

## Verified milestones

### M0 — GPU execution

A handwritten OpenCL SAXPY kernel ran on the RX 480 and matched the CPU
reference with zero error.

Measured device-memory bandwidth:

- approximately 196 GB/s

### M1 — Matrix multiplication

A handwritten tiled FP32 matrix multiplication kernel reached approximately:

- 921 GFLOPS for a 1024 x 1024 matrix

All tested outputs matched NumPy within declared tolerances.

### M2 — GPU training

A complete linear model was trained using custom OpenCL kernels for:

- forward propagation
- prediction error
- weight gradients
- bias gradients
- input gradients
- SGD parameter updates

Results:

- initial loss: 38.4889106750
- final loss: effectively zero
- training steps: 200
- training time: approximately 0.597 seconds
- maximum weight recovery error: approximately 5.96e-7
- all GPU operations verified against NumPy

## Goal

Develop a narrow training runtime for TensionLM that uses:

- persistent GPU tensors
- CPU reference verification
- fixed-shape compiled kernels
- fused tension operations
- tension-gated compute allocation
- deterministic JSON receipts

## TensionForge v0.2 runtime foundation

TensionForge is being converted from standalone experiments into a reusable
OpenCL runtime.

The initial runtime foundation provides:

- central RX 480 and Rusticl device discovery;
- reusable OpenCL context and profiling queue;
- compiled program and kernel caching;
- checked GPU buffer allocation;
- CPU-to-GPU and GPU-to-CPU transfer helpers;
- reusable operation modules;
- deterministic JSON verification receipts.

The first migrated operation is FP32 SAXPY. It is available through the public
`TensionForgeRuntime` and `tensionforge.ops` APIs rather than embedding a full
runtime inside the experiment.

## Reusable tiled matrix multiplication

The verified tiled FP32 matrix multiplication kernel has been migrated into
the public TensionForge runtime.

The operation now provides:

- arbitrary two-dimensional matrix shapes;
- padded work-group dispatch for non-multiple dimensions;
- reusable OpenCL program and kernel caching;
- device work-group validation;
- profiling and calculated GFLOPS;
- NumPy parity tests;
- deterministic benchmark receipts.

## Fused linear forward operation

TensionForge provides a reusable fused FP32 linear operation:

    output = input @ weights + bias

Matrix multiplication and bias addition execute in one OpenCL kernel. This
reduces kernel launches and avoids a separate bias-addition pass.

The operation supports arbitrary batch, input-feature, and output-feature
dimensions, with NumPy parity tests, kernel profiling, compiled-kernel caching,
and deterministic benchmark receipts.

## Device-resident tensors

TensionForge now includes a DeviceTensor abstraction for persistent OpenCL
buffers.

DeviceTensor records shape, dtype, element count, byte size, runtime ownership,
and its underlying GPU buffer. Tensors can be uploaded once, reused across many
kernel executions, updated in place, and copied back only when verification or
external access is required.

The device-resident linear operation accepts DeviceTensor inputs, weights,
biases, and optional reusable output storage. Repeated execution performs no
CPU-to-GPU or GPU-to-CPU transfers between kernel launches.

## Device-resident tension operations

TensionForge provides reusable device-resident FP32 tanh and sigmoid
activations, plus the causal tension update used by tension-based recurrent
state systems.

The update computes:

    next_state = state + gate * (proposal - state)

Inputs and reusable outputs remain in GPU memory across repeated kernel
launches. NumPy references verify every operation, while receipts record
timings, approximate memory bandwidth, source hashes, and numerical errors.

## Fused tension pipeline

TensionForge includes a fused device-resident tension layer.

The kernel performs five logical operations in one launch:

    proposal = tanh(features @ proposal_weights + proposal_bias)
    gate = sigmoid(features @ gate_weights + gate_bias)
    next_state = state + gate * (proposal - state)

The unfused version requires two linear launches, two activation launches, and
one tension-update launch. The fused implementation reduces this from five
kernel launches to one while keeping features, weights, state, and output in
GPU memory.

NumPy and unfused-runtime references verify the fused output. Benchmark
receipts record numerical error, kernel timings, launch reduction, throughput,
and measured speedup.
