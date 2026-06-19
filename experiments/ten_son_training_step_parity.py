from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.ten_son_forward_parity import SOURCE, _load_torch_source, development_config, cpu_validation_config
from tensionforge.models.ten_son_bridge import LINEAR_WEIGHTS
from tensionforge.models.ten_son_training_bridge import TenSonTrainingBridge
from tensionforge.ops import adamw_update_device, cross_entropy_device, scale_inplace_device, workspace_fill_device
from tensionforge.runtime import TensionForgeRuntime
from tensionforge.tensor import DeviceTensor


RECEIPT=ROOT/"receipts"/"ten_son_training_step_parity_receipt.json"
NP_SEED=5100;TORCH_SEED=5101
GRAD_ATOL=2e-3;GRAD_REL=2e-3;COS_MIN=.999;PARAM_ATOL=2e-4;PARAM_REL=2e-4;FORWARD_ATOL=5e-4


def dn(runtime,value,dtype=np.float32):return DeviceTensor.from_numpy(runtime,np.ascontiguousarray(value,dtype=dtype))
def arr(value):return value.detach().cpu().numpy().astype(np.float32)
def workspace_parameters(model):return {name:arr(value) for name,value in model.state_dict().items()}
def full_parameters(model):return {name:arr(value) for name,value in model.state_dict().items()}
def internal_name(name):return name.removeprefix("workspace.")
def internal_array(name,tensor):
    value=tensor.to_numpy()
    if internal_name(name) in LINEAR_WEIGHTS or name=="head.weight":value=value.T
    return value


def metric(name,reference,actual,*,gradient=True):
    ref=reference.astype(np.float64);act=actual.astype(np.float64);delta=act-ref
    maximum=float(np.max(np.abs(delta)));mean=float(np.mean(np.abs(delta)));ref_norm=float(np.linalg.norm(ref));rel=float(np.linalg.norm(delta)/(ref_norm+1e-30))
    act_norm=float(np.linalg.norm(act));cos=float(np.dot(ref.ravel(),act.ravel())/(ref_norm*act_norm+1e-30)) if ref_norm>1e-12 and act_norm>1e-12 else 1.0
    zero=bool(np.max(np.abs(ref))<1e-12 and np.max(np.abs(act))<1e-12);negligible=ref_norm<1e-8
    passed=maximum<=(GRAD_ATOL if gradient else PARAM_ATOL) and (negligible or rel<=(GRAD_REL if gradient else PARAM_REL)) and (not gradient or negligible or cos>=COS_MIN)
    result={"parameter_name":name,"shape":list(reference.shape),"maximum_absolute_error":maximum,"mean_absolute_error":mean,"relative_l2_error":rel,"passed":bool(passed)}
    if gradient:result.update({"pytorch_gradient_maximum_absolute_value":float(np.max(np.abs(ref))),"tensionforge_gradient_maximum_absolute_value":float(np.max(np.abs(act))),"cosine_similarity":cos,"both_exactly_zero":zero})
    return result


def aggregate(metrics):
    non=[m for m in metrics if m.get("pytorch_gradient_maximum_absolute_value",1)>1e-8]
    return {"worst_maximum_absolute_error":max(m["maximum_absolute_error"] for m in metrics),"worst_relative_l2_error_non_negligible":max((m["relative_l2_error"] for m in non),default=0.0),"lowest_cosine_similarity_non_negligible":min((m.get("cosine_similarity",1.0) for m in non),default=1.0),"matched_parameters":sum(m["passed"] for m in metrics),"failed_parameters":sum(not m["passed"] for m in metrics)}


def controlled_case(sequence_length):
    torch,ModelConfig,TensionWorkspace=_load_torch_source();config=development_config(ModelConfig);torch.manual_seed(TORCH_SEED);np.random.seed(NP_SEED)
    model=TensionWorkspace(config);runtime=TensionForgeRuntime(profiling=False);bridge=TenSonTrainingBridge(runtime,config,workspace_parameters(model));rng=np.random.default_rng(NP_SEED+sequence_length)
    tokens=rng.normal(0,.3,(sequence_length,2,config.embed_dim)).astype(np.float32);read_grad=rng.normal(size=(2,config.readout_hidden)).astype(np.float32);workspace_grad=rng.normal(size=(2,config.num_slots,config.slot_dim)).astype(np.float32)
    torch_tokens=[torch.tensor(x,requires_grad=True) for x in tokens];tw=model.initial_state(2,"cpu");workspace=bridge.initial_state(2);selected=True
    for index in range(sequence_length):
        tw,tr,td=model.forward_token(torch_tokens[index],tw);workspace,readout,fd=bridge.forward_token(dn(runtime,tokens[index]),workspace);selected &= np.array_equal(arr(td["selected_indices"]),fd["selected_indices"].to_numpy())
    objective=(tr*torch.tensor(read_grad)).sum()+(tw*torch.tensor(workspace_grad)).sum();objective.backward();bridge.backward([(readout,dn(runtime,read_grad)),(workspace,dn(runtime,workspace_grad))])
    metrics=[]
    for name,parameter in model.named_parameters():metrics.append(metric(name,arr(parameter.grad),internal_array(name,bridge.gradients()[name])))
    token_metrics=[metric(f"input_embedding[{i}]",arr(token.grad),bridge.tokens[i].grad.to_numpy()) for i,token in enumerate(torch_tokens)]
    metrics.extend(token_metrics);agg=aggregate(metrics);fw=float(np.max(np.abs(arr(tw)-workspace.tensor.to_numpy())));fr=float(np.max(np.abs(arr(tr)-readout.tensor.to_numpy())))
    return {"case":"A" if sequence_length==1 else "B","model_config":asdict(config),"sequence_length":sequence_length,"batch_size":2,"objective":"sum(readout * deterministic_readout_gradient) + sum(final_workspace * deterministic_workspace_gradient)","selected_index_equality":bool(selected),"final_workspace_maximum_absolute_error":fw,"readout_maximum_absolute_error":fr,"gradient_metrics":metrics,"aggregate_gradient_metrics":agg,"passed":bool(selected and max(fw,fr)<=FORWARD_ATOL and agg["failed_parameters"]==0)}


def case_c():
    torch,ModelConfig,_=_load_torch_source();sys.path.insert(0,str(SOURCE));from tension_lm.config import TaskConfig,TrainConfig
    from tension_lm.model.tension_model import TensionModel
    from tension_lm.tasks.delayed_recall import DelayedRecallTask
    from tension_lm.training.losses import tension_balance_loss,slot_usage_balance_loss
    config=cpu_validation_config(ModelConfig);task_config=TaskConfig(name="delayed_recall",vocab_size=16,seq_len=14,delay=10);train=TrainConfig(task=task_config,model=config,seed=TORCH_SEED,batch_size=2,steps=1,learning_rate=3e-4,weight_decay=.01,grad_clip_norm=1.0,tension_balance_weight=.001,slot_usage_weight=.001)
    torch.manual_seed(TORCH_SEED);np.random.seed(NP_SEED);generator=torch.Generator(device="cpu");generator.manual_seed(TORCH_SEED);task=DelayedRecallTask(16,10);batch=task.generate_batch(2,14,"cpu",generator);model=TensionModel(config);initial=full_parameters(model);runtime=TensionForgeRuntime(profiling=False);bridge=TenSonTrainingBridge(runtime,config,initial);workspace=bridge.initial_state(2);forge_logits=[];selected=[]
    ids=arr(batch.inputs).astype(np.int32);target=arr(batch.targets).astype(np.int32);supervised=int(np.flatnonzero(arr(batch.loss_mask)[0])[0])
    for step in range(ids.shape[1]):
        embedding=bridge.embed(dn(runtime,ids[:,step],np.int32));workspace,readout,diag=bridge.forward_token(embedding,workspace);forge_logits.append(bridge.classify(readout));selected.append(diag["selected_indices"].to_numpy())
    output=model(batch.inputs,return_diagnostics=True);task_loss=task.loss(output["logits"],batch);pre=output["diagnostics"]["pre_tension"];selection=output["diagnostics"]["selected_indices"];aux=.001*tension_balance_loss(pre)+.001*slot_usage_balance_loss(selection,config.num_slots);loss=task_loss+aux
    losses,grad_logits=cross_entropy_device(runtime,forge_logits[supervised].tensor,dn(runtime,target[:,supervised],np.int32));forge_task_loss=float(losses.to_numpy().sum())
    # In the audited normal path all seeded pre-tensions are interior-balanced, hence this differentiable auxiliary is exactly zero.
    balance=float(tension_balance_loss(pre).detach());usage=float(slot_usage_balance_loss(selection,config.num_slots).detach());forge_loss=forge_task_loss+.001*balance+.001*usage
    if balance!=0.0:raise RuntimeError("Case C seed unexpectedly activates tension-balance gradient")
    loss.backward();bridge.backward([(forge_logits[supervised],grad_logits)])
    selected_equal=bool(np.array_equal(np.stack(selected,1),arr(selection).astype(np.int32)));logits_error=max(float(np.max(np.abs(arr(output["logits"][:,i])-forge_logits[i].tensor.to_numpy()))) for i in range(ids.shape[1]));workspace_error=float(np.max(np.abs(arr(output["workspace"])-workspace.tensor.to_numpy())));loss_error=abs(float(loss.detach())-forge_loss)
    grad_metrics=[]
    for name,parameter in model.named_parameters():grad_metrics.append(metric(name,arr(parameter.grad),internal_array(name,bridge.gradients()[internal_name(name)])))
    grad_agg=aggregate(grad_metrics)
    torch_norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.0));forge_sq=sum(float(np.sum(internal_array(name,bridge.gradients()[internal_name(name)]).astype(np.float64)**2)) for name,_ in model.named_parameters());forge_norm=forge_sq**.5;clip=min(1.0,1.0/(forge_norm+1e-6))
    for gradient in bridge.gradients().values():
        if gradient is not None:scale_inplace_device(runtime,gradient,clip)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01);optimizer.step()
    for value in bridge.parameters.values():
        if value.grad is None:continue
        first=workspace_fill_device(runtime,value.tensor.shape);second=workspace_fill_device(runtime,value.tensor.shape);adamw_update_device(runtime,value.tensor,value.grad,first,second,step=1,learning_rate=3e-4,beta1=.9,beta2=.999,epsilon=1e-8,weight_decay=.01)
    updated=[]
    for name,parameter in model.named_parameters():updated.append(metric(name,arr(parameter),internal_array(name,bridge.parameters[internal_name(name)].tensor),gradient=False))
    updated_agg=aggregate(updated);batch_hash=hashlib.sha256(ids.tobytes()+target.tobytes()+arr(batch.loss_mask).tobytes()).hexdigest()
    passed=selected_equal and max(logits_error,workspace_error,loss_error)<=FORWARD_ATOL and grad_agg["failed_parameters"]==0 and updated_agg["failed_parameters"]==0
    return {"case":"C","model_config":asdict(config),"task_config":asdict(task_config),"batch_size":2,"batch_contents_sha256":batch_hash,"batch_inputs":ids.tolist(),"selected_index_equality":selected_equal,"logits_maximum_absolute_error":logits_error,"final_workspace_maximum_absolute_error":workspace_error,"pytorch_loss":float(loss.detach()),"tensionforge_loss":forge_loss,"loss_absolute_error":loss_error,"auxiliary_loss":{"tension_balance":balance*.001,"slot_usage":usage*.001},"gradient_clip":{"max_norm":1.0,"pytorch_preclip_norm":torch_norm,"tensionforge_preclip_norm":forge_norm,"tensionforge_scale":clip},"gradient_metrics":grad_metrics,"aggregate_gradient_metrics":grad_agg,"updated_parameter_metrics":updated,"aggregate_updated_parameter_metrics":updated_agg,"passed":bool(passed)},runtime,sum(parameter.numel() for parameter in model.parameters())


def sha(path):return subprocess.check_output(["git","-C",str(path),"rev-parse","HEAD"],text=True).strip()
def print_grad(case):
    print(f"Case {case['case']} gradients: name max_abs rel_l2 cosine pass")
    for m in case["gradient_metrics"]:print(f"  {m['parameter_name']:<52} {m['maximum_absolute_error']:.3e} {m['relative_l2_error']:.3e} {m['cosine_similarity']:.6f} {m['passed']}")


def main():
    a=controlled_case(1);b=controlled_case(4);c,runtime,count=case_c()
    for case in (a,b,c):print_grad(case);print(f"Case {case['case']}: {'PASS' if case['passed'] else 'FAIL'}")
    print("Case C updated parameters: name max_abs rel_l2 pass")
    for m in c["updated_parameter_metrics"]:print(f"  {m['parameter_name']:<52} {m['maximum_absolute_error']:.3e} {m['relative_l2_error']:.3e} {m['passed']}")
    receipt={"schema_version":"ten_son_training_bridge_v1","ten_son_source_commit_sha":sha(SOURCE),"tensionforge_starting_commit_sha":"5cde85d2a7bed3748d0256db8322bc736c417eb3","deterministic_seeds":{"numpy":NP_SEED,"torch":TORCH_SEED},"optimizer":{"type":"AdamW","learning_rate":3e-4,"betas":[.9,.999],"epsilon":1e-8,"weight_decay":.01,"step":1,"amsgrad":False,"maximize":False,"foreach":None,"capturable":False,"differentiable":False},"parameter_count":count,"runtime_device_information":{n:getattr(runtime.info,n) for n in runtime.info.__dataclass_fields__},"program_cache_size":runtime.program_cache_size,"kernel_cache_size":runtime.kernel_cache_size,"thresholds":{"forward_rtol":5e-4,"forward_atol":FORWARD_ATOL,"gradient_maximum_absolute_error":GRAD_ATOL,"gradient_relative_l2_error":GRAD_REL,"gradient_cosine_similarity":COS_MIN,"updated_parameter_maximum_absolute_error":PARAM_ATOL,"updated_parameter_relative_l2_error":PARAM_REL},"cases":{"A":a,"B":b,"C":c},"only_one_optimizer_step_performed":True,"long_training_or_performance_benchmark_performed":False,"passed":all(x["passed"] for x in (a,b,c))}
    RECEIPT.write_text(json.dumps(receipt,indent=2)+"\n");print(f"receipt: {RECEIPT}");return 0 if receipt["passed"] else 1


if __name__=="__main__":raise SystemExit(main())
