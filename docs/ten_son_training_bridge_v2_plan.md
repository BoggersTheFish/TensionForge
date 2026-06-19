# Ten-SON Training Bridge v2 extension plan

The v2 work extends the v1 implementation at TensionForge commit `c7ab822cf0fe97ccf467951c076d608939fc1a2c`; it does not introduce a second model port. The read-only reference is `/home/boggersthefish/BoggersSpace/TensionLM` at `b4976f6e5846d0df498f6ee82c919b3215914415`.

## Reused v1 substrate

- `experiments.ten_son_forward_parity._load_torch_source`, `development_config`, and `cpu_validation_config` construct the reference source and model dimensions.
- `experiments.ten_son_training_step_parity.dn`, `arr`, `full_parameters`, `internal_name`, `internal_array`, `metric`, and `aggregate` remain the parameter-conversion and comparison authority.
- `TenSonTrainingBridge.initial_state`, `embed`, `forward_token`, `classify`, and `backward` remain the only TensionForge model/training graph.
- `cross_entropy_device`, `scale_inplace_device`, `adamw_update_device`, and `workspace_fill_device` remain the audited loss, clipping/update building blocks.
- `DelayedRecallTask.generate_batch`, `DelayedRecallTask.loss`, the real diagnostic auxiliary losses, and PyTorch `AdamW` remain the task and optimizer references.

## Multi-step extension

`TenSonTrainingBridge` will gain bounded lifecycle methods that clear one reverse program while retaining device parameter buffers, export/reset parameters, retain/reset AdamW first/second moments, apply global clipping without a host readback, and perform an indexed AdamW step. The 20-step loop belongs in `experiments/ten_son_training_trajectory_parity.py`; it will call the existing bridge forward/backward methods once per synchronized update.

All 20 delayed-recall batches are generated once with one PyTorch generator, copied into immutable host arrays, and hashed before either implementation trains. PyTorch tensors and TensionForge device tensors are created from those exact arrays. Row 0 is the initial model evaluated on batch 0. Row `s` is the post-update model after update `s`, evaluated on batch `s-1`; therefore 20 batches produce 21 rows without a separately generated evaluation stream.

PyTorch and TensionForge start from one captured state dict. Both AdamW states start at zero and use step numbers 1 through 20. Parameter and moment comparisons occur after updates 1, 5, 10, 15, and 20; pre-clip gradient comparisons occur on updates 1, 10, and 20. Both paths use the real global norm clip and normal training mode.

## Benchmark design

`experiments/ten_son_cpu_gpu_benchmark.py` will reuse the same batch preparation and one-step functions. Cold start includes construction, OpenCL compilation, upload, forward, backward, clipping, and update. Steady-state pre-uploads batches, performs five warm-ups, resets parameters and moments, synchronizes immediately around the timed loop, and performs no tensor readback within that loop. Final loss and selected correctness parameters are read once after timing.

The development benchmark uses the existing small bridge configuration with batch 2, sequence length 8, and delayed-recall delay 4. The CPU scientific-validation benchmark reads `experiments/milestone1_v1.json` from Ten-SON and uses its exact model dimensions (`N=32,D=32,E=32,Q=16,K=6,P=T=R=64`), delayed-recall task (`vocab=16, sequence=20, delay=10`), and batch size 32. Timed counts are bounded to 50 development and 20 CPU-validation steps, with three repetitions each. Construction, compilation, conversion, batch generation, receipt creation, diagnostics, assertions, and readbacks are excluded from steady-state timing.
