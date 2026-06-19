from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.ten_son_forward_parity import SOURCE, _load_torch_source, cpu_validation_config
from experiments.ten_son_training_common import (
    build_models, forge_compute, forge_loss_value, generate_host_batches,
    load_reference_modules, torch_compute, upload_batch,
)
from experiments.ten_son_training_step_parity import arr, internal_array, internal_name, metric
from tensionforge.runtime import TensionForgeRuntime


RECEIPT = ROOT / "receipts" / "ten_son_training_trajectory_benchmark_receipt.json"
START_COMMIT = "c7ab822cf0fe97ccf467951c076d608939fc1a2c"
BATCH_SEED = 6200
MODEL_SEED = 6201
LOSS_ABS = 5e-5
LOSS_REL = 5e-4
FORWARD_ATOL = 7e-4
FORWARD_RTOL = 7e-4
STATE_ATOL = 5e-4
STATE_REL = 5e-4


def trajectory_config():
    _, ModelConfig, TaskConfig, TrainConfig, *_ = load_reference_modules()
    model = cpu_validation_config(ModelConfig)
    task = TaskConfig(name="delayed_recall", vocab_size=16, seq_len=14, delay=10)
    return TrainConfig(
        task=task, model=model, seed=MODEL_SEED, batch_size=2, steps=20,
        learning_rate=3e-4, weight_decay=.01, grad_clip_norm=1.0,
        tension_balance_weight=.001, slot_usage_weight=.001, device="cpu",
    )


def state_metric(name, reference, actual):
    ref = reference.astype(np.float64); act = actual.astype(np.float64); delta = act - ref
    maximum = float(np.max(np.abs(delta))); mean = float(np.mean(np.abs(delta)))
    relative = float(np.linalg.norm(delta) / (np.linalg.norm(ref) + 1e-30))
    return {"parameter_name": name, "shape": list(reference.shape), "maximum_absolute_error": maximum,
            "mean_absolute_error": mean, "relative_l2_error": relative,
            "passed": bool(maximum <= STATE_ATOL and relative <= STATE_REL)}


def aggregate_state(metrics):
    return {"worst_maximum_absolute_error": max(item["maximum_absolute_error"] for item in metrics),
            "worst_relative_l2_error": max(item["relative_l2_error"] for item in metrics),
            "matched_tensors": sum(item["passed"] for item in metrics),
            "failed_tensors": sum(not item["passed"] for item in metrics)}


def compare_parameters(model, bridge):
    metrics = [
        state_metric(name, arr(parameter), internal_array(name, bridge.parameters[internal_name(name)].tensor))
        for name, parameter in model.named_parameters()
    ]
    return metrics, aggregate_state(metrics)


def compare_moments(model, optimizer, bridge, key, forge_moments):
    metrics = []
    for name, parameter in model.named_parameters():
        reference = arr(optimizer.state[parameter][key])
        actual = internal_array(name, forge_moments[internal_name(name)])
        metrics.append(state_metric(name, reference, actual))
    return metrics, aggregate_state(metrics)


def compare_gradients(model, bridge):
    metrics = [
        metric(name, arr(parameter.grad), internal_array(name, bridge.gradients()[internal_name(name)]))
        for name, parameter in model.named_parameters()
    ]
    aggregate = {
        "worst_maximum_absolute_error": max(item["maximum_absolute_error"] for item in metrics),
        "worst_relative_l2_error": max(item["relative_l2_error"] for item in metrics if item["pytorch_gradient_maximum_absolute_value"] > 1e-8),
        "lowest_cosine_similarity": min(item["cosine_similarity"] for item in metrics if item["pytorch_gradient_maximum_absolute_value"] > 1e-8),
        "failed_parameters": sum(not item["passed"] for item in metrics),
    }
    return metrics, aggregate


def evaluation_row(step, model, bridge, host, device, train_config):
    torch_loss, output, _ = torch_compute(model, host, train_config, backward=False)
    forge_losses, forge_logits, forge_workspace, diagnostics = forge_compute(bridge, device, backward=False, train_config=train_config)
    forge_loss = forge_loss_value(forge_losses, diagnostics, train_config)
    torch_loss_value = float(torch_loss.detach())
    loss_abs = abs(torch_loss_value - forge_loss)
    loss_rel = loss_abs / max(abs(torch_loss_value), 1e-30)
    actual_logits = np.stack([value.tensor.to_numpy() for value in forge_logits], axis=1)
    reference_logits = arr(output["logits"])
    actual_workspace = forge_workspace.tensor.to_numpy(); reference_workspace = arr(output["workspace"])
    actual_selected = np.stack([item["selected_indices"].to_numpy() for item in diagnostics], axis=1)
    reference_selected = arr(output["diagnostics"]["selected_indices"]).astype(np.int32)
    logits_error = float(np.max(np.abs(reference_logits - actual_logits)))
    workspace_error = float(np.max(np.abs(reference_workspace - actual_workspace)))
    selected_equal = bool(np.array_equal(reference_selected, actual_selected))
    passed = (loss_abs <= LOSS_ABS and loss_rel <= LOSS_REL and selected_equal
              and np.allclose(reference_logits, actual_logits, rtol=FORWARD_RTOL, atol=FORWARD_ATOL)
              and np.allclose(reference_workspace, actual_workspace, rtol=FORWARD_RTOL, atol=FORWARD_ATOL))
    return {"step": step, "torch_loss": torch_loss_value, "tensionforge_loss": forge_loss,
            "loss_absolute_error": loss_abs, "loss_relative_error": loss_rel,
            "selected_index_equality": selected_equal, "logits_maximum_absolute_error": logits_error,
            "final_workspace_maximum_absolute_error": workspace_error, "passed": bool(passed)}


def run_trajectory(step_count=20):
    torch, *_ = _load_torch_source()
    train_config = trajectory_config()
    batches = generate_host_batches(train_config, step_count, BATCH_SEED)
    runtime = TensionForgeRuntime(profiling=False)
    model, optimizer, bridge, _ = build_models(train_config.model, MODEL_SEED, runtime)
    device_batches = [upload_batch(runtime, batch) for batch in batches]
    rows = [evaluation_row(0, model, bridge, batches[0], device_batches[0], train_config)]
    checkpoints = {}; first_failing_step = None
    if not rows[0]["passed"]: first_failing_step = 0
    for step, (host, device) in enumerate(zip(batches, device_batches), start=1):
        _, torch_output, _ = torch_compute(model, host, train_config, backward=True)
        _, _, _, forge_diagnostics = forge_compute(bridge, device, backward=True, train_config=train_config)
        balance_mean = float(torch_output["diagnostics"]["pre_tension"].mean().detach())
        if not .05 <= balance_mean <= .95:
            raise RuntimeError(f"tension balance gradient activated at step {step}")
        checkpoint = {}
        if step in {1, 10, 20} or step == step_count:
            checkpoint["gradient_metrics"], checkpoint["aggregate_gradient_metrics"] = compare_gradients(model, bridge)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip_norm)
        bridge.clip_gradients(train_config.grad_clip_norm)
        optimizer.step()
        bridge.optimizer_step(step, learning_rate=train_config.learning_rate, weight_decay=train_config.weight_decay)
        if step in {1, 5, 10, 15, 20} or step == step_count:
            checkpoint["parameter_metrics"], checkpoint["aggregate_parameter_metrics"] = compare_parameters(model, bridge)
            checkpoint["first_moment_metrics"], checkpoint["aggregate_first_moment_metrics"] = compare_moments(model, optimizer, bridge, "exp_avg", bridge.first_moments)
            checkpoint["second_moment_metrics"], checkpoint["aggregate_second_moment_metrics"] = compare_moments(model, optimizer, bridge, "exp_avg_sq", bridge.second_moments)
        if checkpoint:
            checkpoints[str(step)] = checkpoint
        row = evaluation_row(step, model, bridge, host, device, train_config)
        checkpoint_pass = all(
            value.get("failed_parameters", value.get("failed_tensors", 0)) == 0
            for key, value in checkpoint.items() if key.startswith("aggregate_")
        )
        row["passed"] = bool(row["passed"] and checkpoint_pass)
        rows.append(row)
        if not row["passed"] and first_failing_step is None:
            first_failing_step = step
            break
    passed = first_failing_step is None and len(rows) == step_count + 1
    return {"configuration": {"model": asdict(train_config.model), "task": asdict(train_config.task),
                              "batch_size": train_config.batch_size, "optimizer_steps": step_count},
            "deterministic_seeds": {"batch": BATCH_SEED, "model": MODEL_SEED},
            "batch_hashes": [batch.sha256 for batch in batches], "rows": rows,
            "checkpoints": checkpoints, "first_failing_step": first_failing_step,
            "passed": passed}, runtime


def git_sha(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def print_table(result):
    print("step torch_loss forge_loss loss_abs logits_max workspace_max selected result")
    for row in result["rows"]:
        print(f"{row['step']:>4} {row['torch_loss']:.8f} {row['tensionforge_loss']:.8f} "
              f"{row['loss_absolute_error']:.3e} {row['logits_maximum_absolute_error']:.3e} "
              f"{row['final_workspace_maximum_absolute_error']:.3e} "
              f"{str(row['selected_index_equality']):>8} {'PASS' if row['passed'] else 'FAIL'}")


def main():
    result, runtime = run_trajectory(20)
    print_table(result)
    receipt = {"schema_version": "ten_son_training_trajectory_benchmark_v2",
               "tensionforge_starting_commit_sha": START_COMMIT,
               "ten_son_source_commit_sha": git_sha(SOURCE),
               "trajectory": result, "benchmark": None,
               "parity_thresholds": {"loss_absolute": LOSS_ABS, "loss_relative": LOSS_REL,
                                     "forward_rtol": FORWARD_RTOL, "forward_atol": FORWARD_ATOL,
                                     "gradient_maximum_absolute": 3e-3, "gradient_relative_l2": 3e-3,
                                     "gradient_cosine": .999, "parameter_maximum_absolute": STATE_ATOL,
                                     "parameter_relative_l2": STATE_REL, "moment_maximum_absolute": STATE_ATOL,
                                     "moment_relative_l2": STATE_REL},
               "gpu_program_cache_size": runtime.program_cache_size,
               "gpu_kernel_cache_size": runtime.kernel_cache_size,
               "trajectory_passed": result["passed"], "benchmark_valid": None,
               "no_performance_optimization_implemented": True}
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"trajectory: {'PASS' if result['passed'] else 'FAIL'}")
    print(f"receipt: {RECEIPT}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
