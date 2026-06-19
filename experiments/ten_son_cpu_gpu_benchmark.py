from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.ten_son_training_common import (
    benchmark_configuration, build_models, forge_compute, forge_loss_value,
    generate_host_batches, load_reference_modules, torch_batch, upload_batch,
)
from experiments.ten_son_training_step_parity import arr, full_parameters, internal_array, internal_name
from experiments.ten_son_training_trajectory_parity import RECEIPT, MODEL_SEED, state_metric
from tensionforge.models.ten_son_bridge import LINEAR_WEIGHTS
from tensionforge.runtime import TensionForgeRuntime


WARMUP_STEPS = 5
REPETITIONS = 3
TIMED_STEPS = {"development": 50, "cpu_validation": 20}


def cpu_train_step(torch, model, optimizer, batch, train_config):
    model.zero_grad(set_to_none=True)
    output = model(batch.inputs, return_diagnostics=True)
    mask = batch.loss_mask.bool()
    loss = torch.nn.functional.cross_entropy(output["logits"][mask], batch.targets[mask])
    pre_tension = output["diagnostics"]["pre_tension"]
    mean_tension = pre_tension.mean()
    balance = torch.relu(torch.as_tensor(.05) - mean_tension) + torch.relu(mean_tension - torch.as_tensor(.95))
    selected = output["diagnostics"]["selected_indices"].reshape(-1)
    counts = torch.bincount(selected, minlength=model.config.num_slots).float()
    probabilities = counts / counts.sum().clamp_min(1.0)
    usage = torch.mean((probabilities - 1.0 / model.config.num_slots) ** 2)
    loss = loss + train_config.tension_balance_weight * balance + train_config.slot_usage_weight * usage
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip_norm)
    optimizer.step()
    return loss


def gpu_train_step(bridge, batch, train_config, step):
    losses, _, _, _ = forge_compute(bridge, batch, backward=True, train_config=train_config)
    bridge.clip_gradients(train_config.grad_clip_norm)
    bridge.optimizer_step(step, learning_rate=train_config.learning_rate, weight_decay=train_config.weight_decay)
    return losses


def cpu_eval_loss(torch, model, batch, train_config):
    with torch.no_grad():
        output = model(batch.inputs, return_diagnostics=True)
        mask = batch.loss_mask.bool()
        loss = torch.nn.functional.cross_entropy(output["logits"][mask], batch.targets[mask])
        pre_tension = output["diagnostics"]["pre_tension"]
        mean_tension = pre_tension.mean()
        balance = torch.relu(torch.as_tensor(.05) - mean_tension) + torch.relu(mean_tension - torch.as_tensor(.95))
        selected = output["diagnostics"]["selected_indices"].reshape(-1)
        counts = torch.bincount(selected, minlength=model.config.num_slots).float()
        probabilities = counts / counts.sum().clamp_min(1.0)
        usage = torch.mean((probabilities - 1.0 / model.config.num_slots) ** 2)
        return float(loss + train_config.tension_balance_weight * balance + train_config.slot_usage_weight * usage)


def gpu_eval_loss(bridge, batch, train_config):
    losses, _, _, diagnostics = forge_compute(bridge, batch, backward=False, train_config=train_config)
    return forge_loss_value(losses, diagnostics, train_config)


def cold_start_cpu(name, host):
    torch, _, _, _, TensionModel, *_ = load_reference_modules()
    model_config, _, train_config = benchmark_configuration(name)
    batch = torch_batch(host)
    torch.manual_seed(MODEL_SEED)
    started = time.perf_counter()
    model = TensionModel(model_config).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
    cpu_train_step(torch, model, optimizer, batch, train_config)
    elapsed = time.perf_counter() - started
    return elapsed


def cold_start_gpu(name, host):
    model_config, _, train_config = benchmark_configuration(name)
    torch, _, _, _, TensionModel, *_ = load_reference_modules()
    torch.manual_seed(MODEL_SEED)
    reference = TensionModel(model_config).train()
    started = time.perf_counter()
    initial = full_parameters(reference)
    runtime = TensionForgeRuntime(profiling=False)
    bridge = __import__("tensionforge.models", fromlist=["TenSonTrainingBridge"]).TenSonTrainingBridge(runtime, model_config, initial)
    bridge.initialize_optimizer_state()
    batch = upload_batch(runtime, host)
    gpu_train_step(bridge, batch, train_config, 1)
    runtime.finish()
    elapsed = time.perf_counter() - started
    return elapsed


def run_cpu_steady(name, hosts, steps, repetitions=REPETITIONS):
    torch, _, _, _, TensionModel, *_ = load_reference_modules()
    model_config, _, train_config = benchmark_configuration(name)
    torch.manual_seed(MODEL_SEED); template = TensionModel(model_config).train(); initial = template.state_dict()
    batches = [torch_batch(host) for host in hosts]
    warm_model = TensionModel(model_config).train(); warm_model.load_state_dict(initial)
    warm_optimizer = torch.optim.AdamW(warm_model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
    for index in range(WARMUP_STEPS): cpu_train_step(torch, warm_model, warm_optimizer, batches[index % len(batches)], train_config)
    runs=[]; first_final=None; validity=[]
    for repetition in range(repetitions):
        model=TensionModel(model_config).train();model.load_state_dict(initial)
        optimizer=torch.optim.AdamW(model.parameters(),lr=train_config.learning_rate,weight_decay=train_config.weight_decay)
        initial_loss=cpu_eval_loss(torch,model,batches[0],train_config)
        started=time.perf_counter()
        for index in range(steps):cpu_train_step(torch,model,optimizer,batches[index%len(batches)],train_config)
        elapsed=time.perf_counter()-started
        final_loss=cpu_eval_loss(torch,model,batches[0],train_config);validity.append(np.isfinite(final_loss) and final_loss<initial_loss);runs.append(elapsed)
        if repetition==0:first_final={name:arr(value) for name,value in model.named_parameters()}
    return summary(runs,steps),first_final,all(validity)


def run_cpu_reference(name, hosts, steps):
    torch, _, _, _, TensionModel, *_ = load_reference_modules()
    model_config, _, train_config = benchmark_configuration(name)
    torch.manual_seed(MODEL_SEED); model=TensionModel(model_config).train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=train_config.learning_rate,weight_decay=train_config.weight_decay)
    batches=[torch_batch(host) for host in hosts]
    for index in range(steps):cpu_train_step(torch,model,optimizer,batches[index%len(batches)],train_config)
    return ({name:arr(value) for name,value in model.named_parameters()},
            cpu_eval_loss(torch,model,batches[0],train_config))


def run_gpu_steady(name, hosts, steps, repetitions=REPETITIONS):
    model_config, _, train_config = benchmark_configuration(name)
    runtime=TensionForgeRuntime(profiling=False);_,_,bridge,initial=build_models(model_config,MODEL_SEED,runtime)
    batches=[upload_batch(runtime,host) for host in hosts]
    for index in range(WARMUP_STEPS):gpu_train_step(bridge,batches[index%len(batches)],train_config,index+1)
    runtime.finish();bridge.reset_training_state(initial)
    runs=[];validity=[];first_final=None;first_final_loss=None;counter_runs=[]
    for repetition in range(repetitions):
        bridge.reset_training_state(initial);initial_loss=gpu_eval_loss(bridge,batches[0],train_config)
        runtime.reset_counters();runtime.finish();started=time.perf_counter()
        for index in range(steps):gpu_train_step(bridge,batches[index%len(batches)],train_config,index+1)
        runtime.finish();elapsed=time.perf_counter()-started;counters=runtime.counters().copy()
        final_loss=gpu_eval_loss(bridge,batches[0],train_config);validity.append(np.isfinite(final_loss) and final_loss<initial_loss);runs.append(elapsed);counter_runs.append(counters)
        if repetition==0:
            first_final={key:value.tensor.to_numpy() for key,value in bridge.parameters.items()}
            first_final_loss=final_loss
    result=summary(runs,steps);result["counters_per_run"]=counter_runs;result["kernel_launches_per_step"]=statistics.median(x["kernel_launches"] for x in counter_runs)/steps
    result["host_to_device_bytes_per_step"]=statistics.median(x["host_to_device_bytes"] for x in counter_runs)/steps;result["device_to_host_bytes_per_step"]=statistics.median(x["device_to_host_bytes"] for x in counter_runs)/steps
    result["program_cache_size"]=runtime.program_cache_size;result["kernel_cache_size"]=runtime.kernel_cache_size
    return result,first_final,first_final_loss,all(validity),runtime


def summary(runs,steps):
    median=statistics.median(runs)
    return {"individual_run_seconds":runs,"median_total_seconds":median,"median_milliseconds_per_step":median*1000/steps,"steps_per_second":steps/median,"timed_steps":steps,"repetitions":len(runs)}


def cpu_details(torch):
    model="unknown"
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            model=line.split(":",1)[1].strip();break
    return {"model":model,"pytorch_version":torch.__version__,"thread_count":torch.get_num_threads(),
            "interop_thread_count":torch.get_num_interop_threads(),"parallel_backend":torch.__config__.parallel_info(),
            "process_affinity":sorted(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else None,
            "single_thread_forced":False}


def gpu_details(runtime):
    values={name:getattr(runtime.info,name) for name in runtime.info.__dataclass_fields__}
    values.update({"warmup_steps":WARMUP_STEPS,"parameters_moments_intermediates_device_resident":True})
    return values


def run_benchmark(repetitions=REPETITIONS, step_overrides=None):
    torch,*_ = load_reference_modules();results={};overall_valid=True
    for name in ("development","cpu_validation"):
        _,_,train_config=benchmark_configuration(name);steps=(step_overrides or TIMED_STEPS)[name]
        hosts=generate_host_batches(train_config,max(steps,WARMUP_STEPS),6300+(0 if name=="development" else 100))
        cold_cpu=cold_start_cpu(name,hosts[0]);cold_gpu=cold_start_gpu(name,hosts[0])
        cpu,_,cpu_valid=run_cpu_steady(name,hosts,steps,repetitions)
        reference_final,reference_loss=run_cpu_reference(name,hosts,steps)
        gpu,gpu_final,gpu_final_loss,gpu_valid,runtime=run_gpu_steady(name,hosts,steps,repetitions)
        comparisons=[]
        for parameter in ("embedding.weight","workspace.initial_workspace","head.weight"):
            actual=gpu_final[internal_name(parameter)]
            if internal_name(parameter) in LINEAR_WEIGHTS or parameter=="head.weight":actual=actual.T
            comparisons.append(state_metric(parameter,reference_final[parameter],actual))
        comparison_valid=all(item["passed"] for item in comparisons)
        final_loss_error=abs(reference_loss-gpu_final_loss)
        speedup=cpu["median_total_seconds"]/gpu["median_total_seconds"]
        results[name]={"configuration":{"model":asdict(train_config.model),"task":asdict(train_config.task),"batch_size":train_config.batch_size},
                       "batch_hashes":[host.sha256 for host in hosts],
                       "cold_start_seconds":{"pytorch_cpu":cold_cpu,"tensionforge_gpu":cold_gpu},"pytorch_cpu":cpu,"tensionforge_gpu":gpu,
                       "speedup":speedup,"first_repetition_final_parameter_checks":comparisons,
                       "untimed_reference_final_loss":reference_loss,"first_gpu_repetition_final_loss":gpu_final_loss,
                       "final_loss_absolute_error":final_loss_error,
                       "final_loss_decreased":{"pytorch_cpu":cpu_valid,"tensionforge_gpu":gpu_valid},
                       "valid":bool(cpu_valid and gpu_valid and comparison_valid and final_loss_error<=5e-5)}
        overall_valid &= results[name]["valid"]
    cpu_validation_speedup=results["cpu_validation"]["speedup"]
    return {"configurations":results,"cpu_details":cpu_details(torch),"gpu_details":gpu_details(runtime),
            "steady_state_exclusions":["model/runtime construction","OpenCL compilation","parameter conversion and reset","batch generation and upload","receipt generation","diagnostic readbacks","test assertions"],
            "cpu_validation_speedup_classification":classification(cpu_validation_speedup),
            "benchmark_valid":bool(overall_valid)},runtime


def classification(speedup):
    if speedup<1:return "currently slower than CPU"
    if speedup<3:return "correct but not yet compelling"
    if speedup<5:return "useful early acceleration"
    if speedup<10:return "successful runtime"
    return "strong acceleration result"


def main():
    benchmark,runtime=run_benchmark();receipt=json.loads(RECEIPT.read_text())
    if not receipt.get("trajectory_passed"):raise RuntimeError("trajectory receipt is missing or failed")
    receipt["benchmark"]=benchmark;receipt["benchmark_valid"]=benchmark["benchmark_valid"]
    receipt["gpu_program_cache_size"]=runtime.program_cache_size;receipt["gpu_kernel_cache_size"]=runtime.kernel_cache_size
    RECEIPT.write_text(json.dumps(receipt,indent=2)+"\n")
    for name,result in benchmark["configurations"].items():
        print(f"{name}: CPU {result['pytorch_cpu']['median_total_seconds']:.6f}s, GPU {result['tensionforge_gpu']['median_total_seconds']:.6f}s, speedup {result['speedup']:.3f}x, cold CPU/GPU {result['cold_start_seconds']['pytorch_cpu']:.6f}/{result['cold_start_seconds']['tensionforge_gpu']:.6f}s, valid={result['valid']}")
    speedup=benchmark["configurations"]["cpu_validation"]["speedup"]
    print(f"CPU-validation classification: {classification(speedup)}")
    return 0 if benchmark["benchmark_valid"] else 1


if __name__=="__main__":raise SystemExit(main())
