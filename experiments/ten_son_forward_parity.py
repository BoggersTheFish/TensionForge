from __future__ import annotations

from dataclasses import asdict
import importlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

from tensionforge import DeviceTensor, TensionForgeRuntime
from tensionforge.models import TenSonForwardBridge


ROOT = Path(__file__).resolve().parents[1]
BOGGERSSPACE = ROOT.parent
SOURCE = BOGGERSSPACE / "TensionLM"
RECEIPT = ROOT / "receipts" / "ten_son_forward_parity_receipt.json"
ATOL = 5e-4
RTOL = 5e-4
NUMPY_SEED = 4100
TORCH_SEED = 4101


def _load_torch_source():
    if not SOURCE.is_dir():
        raise RuntimeError(f"Ten-SON source repository not found: {SOURCE}")
    sys.path.insert(0, str(SOURCE))
    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError:
        version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        shared = BOGGERSSPACE / ".venv" / "lib" / version / "site-packages"
        if shared.is_dir():
            sys.path.append(str(shared))
        try:
            torch = importlib.import_module("torch")
        except ModuleNotFoundError as exc:
            raise RuntimeError("PyTorch is unavailable; no package was installed") from exc
    config_module = importlib.import_module("tension_lm.config")
    workspace_module = importlib.import_module("tension_lm.model.workspace")
    return torch, config_module.ModelConfig, workspace_module.TensionWorkspace


def development_config(ModelConfig):
    return ModelConfig(
        vocab_size=8, output_size=8, num_slots=8, slot_dim=8, embed_dim=8,
        key_dim=4, top_k=3, microsteps=2, max_microsteps=4,
        proposal_hidden=16, tension_hidden=16, readout_hidden=16,
    )


def cpu_validation_config(ModelConfig):
    return ModelConfig(
        vocab_size=16, output_size=16, num_slots=32, slot_dim=32, embed_dim=32,
        key_dim=16, top_k=6, microsteps=2, max_microsteps=4,
        proposal_hidden=64, tension_hidden=64, readout_hidden=64,
    )


def _numpy(tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def trace_pytorch(model, token, workspace):
    torch = importlib.import_module("torch")
    batch = token.size(0)
    summary = model.summarize(workspace)
    route = model.router(token, summary, model.slot_keys, top_k=model.config.top_k)
    indices = route["selected_indices"]
    selected_scores = route["selected_scores"]
    selected_keys = model.slot_keys[indices]
    slot_tension = torch.zeros(batch, model.config.num_slots, 1, device=workspace.device)
    trace: dict[str, Any] = {
        "input_summary": _numpy(summary),
        "routing_scores": _numpy(route["scores"]),
        "routing_probabilities": _numpy(route["probabilities"]),
        "selected_indices": _numpy(indices),
        "selected_scores": _numpy(selected_scores),
        "selected_states": [], "proposals": [], "pre_tensions": [],
        "updated_states": [], "workspaces": [], "post_tensions": [],
    }
    for microstep in range(model.config.microsteps):
        expanded = indices.unsqueeze(-1).expand(-1, -1, model.config.slot_dim)
        selected_state = workspace.gather(dim=1, index=expanded)
        summary = model.summarize(workspace)
        proposal = model.proposal(selected_state, selected_keys, token, summary, microstep)
        _, tension = model.tension(
            selected_state, proposal, selected_keys, token, summary, selected_scores, microstep
        )
        updated, _ = model.cell(selected_state, proposal, tension)
        workspace = workspace.scatter(dim=1, index=expanded, src=updated)
        slot_tension = slot_tension.scatter(
            dim=1, index=indices.unsqueeze(-1).expand(-1, -1, 1), src=tension
        )
        post_summary = model.summarize(workspace)
        _, post_tension = model.tension(
            updated, proposal, selected_keys, token, post_summary, selected_scores, microstep
        )
        for key, value in (
            ("selected_states", selected_state), ("proposals", proposal),
            ("pre_tensions", tension), ("updated_states", updated),
            ("workspaces", workspace), ("post_tensions", post_tension),
        ):
            trace[key].append(_numpy(value))
    readout, weights = model.readout(workspace, token, slot_tension)
    trace["slot_tension"] = _numpy(slot_tension)
    trace["readout_weights"] = _numpy(weights)
    trace["readout_vector"] = _numpy(readout)
    return workspace, readout, trace


def _device_trace(trace: dict[str, Any]) -> dict[str, Any]:
    return {
        key: ([item.to_numpy() for item in value] if isinstance(value, list) else value.to_numpy())
        for key, value in trace.items()
    }


def _error(reference: np.ndarray, actual: np.ndarray) -> float:
    return float(np.max(np.abs(reference.astype(np.float64) - actual.astype(np.float64))))


def compare_trace(reference: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "selected_index_equality": bool(np.array_equal(reference["selected_indices"], actual["selected_indices"])),
        "summary_max_abs_error": _error(reference["input_summary"], actual["input_summary"]),
        "routing_score_max_abs_error": _error(reference["routing_scores"], actual["routing_scores"]),
        "routing_probability_max_abs_error": _error(reference["routing_probabilities"], actual["routing_probabilities"]),
        "selected_score_max_abs_error": _error(reference["selected_scores"], actual["selected_scores"]),
        "selected_state_max_abs_error_per_microstep": [],
        "proposal_max_abs_error_per_microstep": [],
        "pre_tension_max_abs_error_per_microstep": [],
        "updated_state_max_abs_error_per_microstep": [],
        "full_workspace_max_abs_error_per_microstep": [],
        "post_tension_max_abs_error_per_microstep": [],
        "slot_tension_max_abs_error": _error(reference["slot_tension"], actual["slot_tension"]),
        "readout_weight_max_abs_error": _error(reference["readout_weights"], actual["readout_weights"]),
        "readout_vector_max_abs_error": _error(reference["readout_vector"], actual["readout_vector"]),
    }
    mapping = {
        "selected_states": "selected_state_max_abs_error_per_microstep",
        "proposals": "proposal_max_abs_error_per_microstep",
        "pre_tensions": "pre_tension_max_abs_error_per_microstep",
        "updated_states": "updated_state_max_abs_error_per_microstep",
        "workspaces": "full_workspace_max_abs_error_per_microstep",
        "post_tensions": "post_tension_max_abs_error_per_microstep",
    }
    for source, destination in mapping.items():
        result[destination] = [_error(a, b) for a, b in zip(reference[source], actual[source], strict=True)]
    numeric = []
    for value in result.values():
        if isinstance(value, float):
            numeric.append(value)
        elif isinstance(value, list):
            numeric.extend(value)
    result["maximum_abs_error"] = max(numeric)
    result["passed"] = result["selected_index_equality"] and result["maximum_abs_error"] <= ATOL
    return result


def run_parity_case(config, *, sequence_length=4, batch_size=2, runtime=None):
    torch = importlib.import_module("torch")
    _, _, TensionWorkspace = _load_torch_source()
    np.random.seed(NUMPY_SEED)
    torch.manual_seed(TORCH_SEED)
    model = TensionWorkspace(config).eval()
    parameters = {name: _numpy(value).astype(np.float32) for name, value in model.state_dict().items()}
    runtime = runtime or TensionForgeRuntime(profiling=False)
    bridge = TenSonForwardBridge(runtime, config, parameters)
    rng = np.random.default_rng(NUMPY_SEED)
    initial = rng.normal(0.0, 0.2, size=(batch_size, config.num_slots, config.slot_dim)).astype(np.float32)
    tokens = rng.normal(0.0, 0.3, size=(sequence_length, batch_size, config.embed_dim)).astype(np.float32)
    torch_workspace = torch.from_numpy(initial.copy())
    forge_workspace = DeviceTensor.from_numpy(runtime, initial.copy())
    steps = []
    for step in range(sequence_length):
        token_torch = torch.from_numpy(tokens[step])
        with torch.no_grad():
            torch_workspace, _, reference = trace_pytorch(model, token_torch, torch_workspace)
        forge_workspace, _, forge_trace = bridge.forward_token(
            DeviceTensor.from_numpy(runtime, tokens[step]), forge_workspace
        )
        steps.append(compare_trace(reference, _device_trace(forge_trace)))
    final_error = _error(_numpy(torch_workspace), forge_workspace.to_numpy())
    result = {
        "steps": steps,
        "sequence_final_workspace_max_abs_error": final_error,
        "selected_index_equality": all(step["selected_index_equality"] for step in steps),
        "maximum_abs_error": max([final_error] + [step["maximum_abs_error"] for step in steps]),
    }
    result["passed"] = result["selected_index_equality"] and result["maximum_abs_error"] <= ATOL
    return result, model, bridge, runtime


def _sha(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def _device_info(runtime) -> dict[str, Any]:
    info = runtime.info
    return {name: getattr(info, name) for name in info.__dataclass_fields__}


def main() -> int:
    torch, ModelConfig, _ = _load_torch_source()
    configurations = {
        "development": development_config(ModelConfig),
        "cpu_validation": cpu_validation_config(ModelConfig),
    }
    runtime = TensionForgeRuntime(profiling=False)
    runs = {}
    parameter_counts = {}
    for name, config in configurations.items():
        result, model, _, runtime = run_parity_case(config, sequence_length=4, batch_size=2, runtime=runtime)
        runs[name] = result
        parameter_counts[name] = sum(parameter.numel() for parameter in model.parameters())
        print(f"{name}: selected_indices={result['selected_index_equality']} max_abs={result['maximum_abs_error']:.9g} final_workspace={result['sequence_final_workspace_max_abs_error']:.9g} passed={result['passed']}")
        for index, step in enumerate(result["steps"]):
            print(f"  token {index}: route={step['routing_score_max_abs_error']:.3g} probability={step['routing_probability_max_abs_error']:.3g} proposal={max(step['proposal_max_abs_error_per_microstep']):.3g} pre_tension={max(step['pre_tension_max_abs_error_per_microstep']):.3g} readout={step['readout_vector_max_abs_error']:.3g}")

    receipt = {
        "schema_version": "ten_son_forward_parity_v0",
        "ten_son_source_commit_sha": _sha(SOURCE),
        "tensionforge_commit_sha_before_work": "d63e21108b5d373ed4b0987c4d7a9365ca93cd2b",
        "model_configurations": {name: asdict(config) for name, config in configurations.items()},
        "deterministic_seeds": {"numpy": NUMPY_SEED, "torch": TORCH_SEED},
        "parameter_counts": parameter_counts,
        "device_information": _device_info(runtime),
        "parity_errors": runs,
        "selected_index_equality": all(run["selected_index_equality"] for run in runs.values()),
        "sequence_length": 4,
        "runtime_program_cache_size": runtime.program_cache_size,
        "runtime_kernel_cache_size": runtime.kernel_cache_size,
        "thresholds": {"rtol": RTOL, "atol": ATOL, "selected_indices": "exact"},
        "passed": all(run["passed"] for run in runs.values()),
        "training_or_backward_performed": False,
        "statement": "Forward parity only; no training or backward pass was performed.",
        "torch_version": torch.__version__,
    }
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"receipt: {RECEIPT}")
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
