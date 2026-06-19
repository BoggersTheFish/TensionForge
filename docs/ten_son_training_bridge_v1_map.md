# Ten-SON Training Bridge v1 map

Source of truth: read-only `/home/boggersthefish/BoggersSpace/TensionLM` at `b4976f6e5846d0df498f6ee82c919b3215914415`. TensionForge starts at `5cde85d2a7bed3748d0256db8322bc736c417eb3`. Symbols are batch `B`, sequence `L`, slots `N`, selected slots `K`, slot width `D`, embedding width `E`, key width `Q`, proposal hidden width `P`, tension hidden width `T`, readout hidden width `R`, vocabulary `V`, and output classes `C`.

## Audited training path

`training.loop.train` seeds Python/NumPy/PyTorch through `set_seed`, constructs `TensionModel`, creates one `torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)` parameter group, generates a `DelayedRecallTask` batch, runs the complete sequence, adds the task loss and two diagnostic auxiliaries, calls `loss.backward()`, clips all parameter gradients with `torch.nn.utils.clip_grad_norm_`, and takes one optimizer step. AdamW therefore uses PyTorch defaults `betas=(0.9,0.999)`, `eps=1e-8`, `amsgrad=False`, `maximize=False`, and applies the configured weight decay to every parameter; there are no exclusions.

Token IDs index `TensionModel.embedding`. Every embedding is passed through the same workspace module in sequence and both configured microsteps reuse all parameters. The recurrent workspace is never detached. With no supplied state, `initial_workspace.unsqueeze(0).expand(B,-1,-1)` participates in the graph, so gradients sum over the batch. Slot keys are trainable and contribute through router scores, selected-key features, and the selected routing scores. Hard top-k indices are fixed in backward; no straight-through estimator exists. `routing_probabilities` and diagnostic post-tensions do not affect the task loss. Selected routing scores do affect the pre/post tension networks. Post-tensions are diagnostics only.

Delayed recall is masked classification. The generator puts a store marker at position 1, the value at 2, a recall marker at `min(L-2, 1+delay)`, and the only supervised target at the following position. `F.cross_entropy(logits[mask], targets[mask])` averages over the `B` selected examples. Logits at all other positions still influence later recurrent state, but their heads do not receive direct loss gradients. The final classifier is `head: R -> C`.

The real total loss also contains `0.001 * tension_balance_loss(pre_tension)` and `0.001 * slot_usage_balance_loss(selected_indices,N)` under the checked-in scientific configuration. The tension balance term is zero when mean pre-tension lies in `[0.05,0.95]`; the slot-usage term is nondifferentiable because it is computed only from integer indices. Case C includes both exact scalar terms even when their gradient is zero.

All `nn.LayerNorm` modules use `eps=1e-5`, biased population variance, trainable affine scale/bias, and default scale-one/bias-zero initialization. Softmax is always over the last dimension. Linear/embedding modules use PyTorch defaults; workspace and slot keys use `randn * 0.02`. Readout attention is scaled query/workspace dot product plus `log1p(-clamp(slot_tension,0,0.999))`, followed by last-dimension softmax and a weighted workspace sum. The clamp derivative is zero outside the open interval and `-1/(1-t)` inside it. Workspace max backward routes each feature gradient to PyTorch's indexed maximum (first maximum); exact claims use unique maxima.

## Parameter mapping

PyTorch linear weights are `[out,in]`; TensionForge stores them transposed as `[in,out]`. Gradient and updated-parameter comparisons transpose them back to PyTorch layout. All other layouts are unchanged.

| PyTorch state-dict name | PyTorch shape | TensionForge name | TensionForge shape | Gradient | Conversion |
|---|---:|---|---:|---|---|
| `embedding.weight` | `[V,E]` | same | `[V,E]` | yes | none |
| `workspace.initial_workspace` | `[N,D]` | `initial_workspace` | `[N,D]` | yes | strip prefix |
| `workspace.slot_keys` | `[N,Q]` | `slot_keys` | `[N,Q]` | yes | strip prefix |
| `workspace.summary_norm.{weight,bias}` | `[2D]` | `summary_norm.*` | `[2D]` | yes | strip prefix |
| `workspace.router.query_network.0.{weight,bias}` | `[Q,E+2D]`, `[Q]` | `router.query_network.0.*` | `[E+2D,Q]`, `[Q]` | yes | transpose weight |
| `workspace.router.query_network.1.{weight,bias}` | `[Q]` | `router.query_network.1.*` | `[Q]` | yes | none |
| `workspace.router.query_network.3.{weight,bias}` | `[Q,Q]`, `[Q]` | `router.query_network.3.*` | `[Q,Q]`, `[Q]` | yes | transpose weight |
| `workspace.proposal.microstep_embedding.weight` | `[M,8]` | same below `workspace` | `[M,8]` | yes | none |
| `workspace.proposal.network.0.{weight,bias}` | `[D+Q+E+2D+8]` | same below `workspace` | same | yes | none |
| `workspace.proposal.network.1.{weight,bias}` | `[P,D+Q+E+2D+8]`, `[P]` | same below `workspace` | transposed weight, same bias | yes | transpose weight |
| `workspace.proposal.network.3.{weight,bias}` | `[D,P]`, `[D]` | same below `workspace` | `[P,D]`, `[D]` | yes | transpose weight |
| `workspace.tension.microstep_embedding.weight` | `[M,8]` | same below `workspace` | `[M,8]` | yes | none |
| `workspace.tension.network.0.{weight,bias}` | `[2D+Q+E+2D+2+8]` | same below `workspace` | same | yes | none |
| `workspace.tension.network.1.{weight,bias}` | `[T,2D+Q+E+2D+2+8]`, `[T]` | same below `workspace` | transposed weight, same bias | yes | transpose weight |
| `workspace.tension.network.3.{weight,bias}` | `[1,T]`, `[1]` | same below `workspace` | `[T,1]`, `[1]` | yes | transpose weight |
| `workspace.readout.query.0.{weight,bias}` | `[R,E+2D]`, `[R]` | same below `workspace` | `[E+2D,R]`, `[R]` | yes | transpose weight |
| `workspace.readout.query.2.{weight,bias}` | `[D,R]`, `[D]` | same below `workspace` | `[R,D]`, `[D]` | yes | transpose weight |
| `workspace.readout.output.0.{weight,bias}` | `[D]` | same below `workspace` | `[D]` | yes | none |
| `workspace.readout.output.1.{weight,bias}` | `[R,D]`, `[R]` | same below `workspace` | `[D,R]`, `[R]` | yes | transpose weight |
| `workspace.readout.output.3.{weight,bias}` | `[R,R]`, `[R]` | same below `workspace` | `[R,R]`, `[R]` | yes | transpose weight |
| `head.{weight,bias}` | `[C,R]`, `[C]` | same | `[R,C]`, `[C]` | yes | transpose weight |

## Operation backward map

| Forward operation | Local derivative / saved values | Existing backward | Required addition | Risk |
|---|---|---|---|---|
| linear | `dx=dy W^T`, `dW=x^T dy`, `db=sum dy`; save `x,W` | `linear_backward_device` | accumulation across reuse | reduction order |
| exact GELU | `0.5(1+erf(x/sqrt2))+x exp(-x²/2)/sqrt(2pi)`; save `x` | none | GELU backward | transcendental tolerance |
| tanh/sigmoid | standard output-based derivatives | both exist | accumulation | low |
| LayerNorm | standard biased-variance affine derivative; save input, scale, row mean/invstd | none | input/scale/bias backward | reduction order |
| workspace mean/max | mean broadcasts `dy/N`; max scatters to first argmax; save workspace/argmax | none | combined reduction backward | ties |
| softmax | `dx=y*(dy-sum(dy*y))`; save output | none | row softmax backward | low |
| top-k values | scatter value gradient to selected score indices; indices fixed | none | indexed scatter-add | ties fixed by forward |
| batched gather | scatter-add output gradient to source indices | none | shared and batched variants | duplicate indices |
| functional scatter | selected source positions get zero input gradient; updates gather gradient | none | scatter backward | duplicate indices absent for top-k |
| concatenate | split gradient by column ranges | none | column split | low |
| broadcast | sum repeated rows; embedding row accumulates | none | row/table reductions | reduction order |
| delta norm | `dleft=dy*(left-right)/norm`; opposite for right | none | row norm backward | zero norm (PyTorch returns zero) |
| tension cell | `ds=dy*(1-t)`, `dp=dy*t`, `dt=sum_d dy*(p-s)` | scalar-gate version differs | broadcast-gate backward | reduction order |
| scaled batched dot | standard two-input matmul derivative | none | batched dot backward | reduction order |
| tension penalty | score identity; tension `-dy/(1-t)` inside clamp interval | none | penalty backward | clamp boundaries |
| weighted sum | `dw=dy dot values`, `dvalues=weights*dy` | none | weighted sum backward | low |
| embedding | indexed row gradient accumulation | none | embedding backward | repeated IDs |
| masked cross entropy | `(softmax(logits)-onehot)/B` on masked positions | none | loss and logits gradient | stable logsumexp |
| gradient clipping | multiply all gradients by `min(1,max_norm/(norm+1e-6))` | none | global norm/scale | device reduction ordering |
| AdamW | PyTorch decoupled decay and bias-corrected moments | `adamw_update_device` | apply to every mapped parameter | PyTorch scalar ordering |

## One-step order

Case C uses the checked-in deterministic delayed-recall configuration: generate IDs/targets/mask with the seeded PyTorch generator; gather token embeddings; expand the trainable initial workspace; for each token route once, run two shared-parameter microsteps, read out, and classify; stack logits; select the single supervised position per batch; mean cross entropy; add the two real auxiliary losses; clear gradients; reverse through classifier, all token recurrence, and embedding; compute the global gradient norm and apply the real clip; apply AdamW at step 1 to every parameter; compare parameters in PyTorch layout; stop. No tensor is detached on the normal `model_kind='tension'` path.
