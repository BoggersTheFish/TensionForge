# Ten-SON Bridge v0 source map

Source of truth: sibling repository `/home/boggersthefish/BoggersSpace/TensionLM`, commit `b4976f6e5846d0df498f6ee82c919b3215914415`. This audit maps the unmodified PyTorch `TensionWorkspace.forward_token` path. Symbols are `B` batch size, `N` slots, `K` selected slots, `D` slot dimension, `E` token embedding dimension, `Q` key/query dimension, `P` proposal hidden dimension, `T` tension hidden dimension, and `R` readout hidden dimension.

| PyTorch module or operation | Exact input shape(s) | Exact output shape(s) | Activation or mathematical rule | Existing TensionForge equivalent | Missing TensionForge operation | Expected parity |
|---|---|---|---|---|---|---|
| `TensionWorkspace.summarize`: slot reduction | workspace `[B,N,D]` | mean `[B,D]`, maximum `[B,D]` | `mean(dim=1)` and `max(dim=1).values` | None | Workspace mean/max reduction | Tolerance-based because reduction order can differ |
| Summary concatenation | mean `[B,D]`, maximum `[B,D]` | `[B,2D]` | Concatenate on the last dimension | Row concatenation exists, but not last-dimension concatenation | Last-dimension concatenation | Exact copy |
| `summary_norm` | `[B,2D]`; weight/bias `[2D]` | `[B,2D]` | LayerNorm over last dimension, PyTorch default `eps=1e-5`, biased variance (`unbiased=False`) | None | LayerNorm | Tolerance-based |
| Router input concatenation | token `[B,E]`, summary `[B,2D]` | `[B,E+2D]` | Concatenate on last dimension | None | Last-dimension concatenation | Exact copy |
| `router.query_network.0` | `[B,E+2D]`; PyTorch weight `[Q,E+2D]`, bias `[Q]` | `[B,Q]` | Affine transform | `linear_device`, whose weight layout is `[in,out]` | Parameter transpose during import only | Tolerance-based |
| `router.query_network.1` | `[B,Q]`; weight/bias `[Q]` | `[B,Q]` | LayerNorm, `eps=1e-5` | None | LayerNorm | Tolerance-based |
| `router.query_network.2` | `[B,Q]` | `[B,Q]` | `tanh` | `tanh_device` | None | Tolerance-based |
| `router.query_network.3` | `[B,Q]`; PyTorch weight `[Q,Q]`, bias `[Q]` | query `[B,Q]` | Affine transform | `linear_device` | Parameter transpose during import only | Tolerance-based |
| Router scores | query `[B,Q]`, slot keys `[N,Q]` | scores `[B,N]` | `query @ slot_keys.T / sqrt(Q)` | Affine operation can represent matmul; no scale primitive | Scalar multiply/scale | Tolerance-based |
| Router probabilities | scores `[B,N]` | `[B,N]` | `softmax(dim=-1)` | None | Last-dimension softmax | Tolerance-based |
| Router hard selection | scores `[B,N]` | selected scores `[B,K]`, indices `[B,K]` | `torch.topk(..., k=K, dim=-1)`, descending values, unsorted flag omitted (therefore sorted) | None | Deterministic row top-k | Exact indices for non-tied scores; values tolerance-based. Ties are implementation-dependent and not claimed portable. |
| Selected key gather | slot keys `[N,Q]`, indices `[B,K]` | `[B,K,Q]` | Advanced indexing `slot_keys[selected_indices]` | None | Batched gather from shared rows | Exact copy |
| Selected state gather | workspace `[B,N,D]`, indices `[B,K]` | `[B,K,D]` | `gather(dim=1)` | None | Batched slot gather | Exact copy |
| Proposal feature assembly | state `[B,K,D]`, keys `[B,K,Q]`, token `[B,E]`, summary `[B,2D]`, step embedding table `[M,8]` | `[B,K,D+Q+E+2D+8]` | Broadcast token/summary and concatenate with learned embedding for integer microstep | None | Broadcast and last-dimension concatenation / fixed-row embedding broadcast | Exact copy |
| `proposal.network.0` | `[B,K,3D+Q+E+8]`; weight/bias same last size | same | LayerNorm, `eps=1e-5` | None | LayerNorm on flattened leading dimensions | Tolerance-based |
| `proposal.network.1` | `[B,K,3D+Q+E+8]`; weight `[P,3D+Q+E+8]`, bias `[P]` | `[B,K,P]` | Affine transform | `linear_device` on `[B*K,*]` | Reshape is metadata-only; parameter transpose on import | Tolerance-based |
| `proposal.network.2` | `[B,K,P]` | `[B,K,P]` | `nn.GELU(approximate='none')`, i.e. exact erf formulation | None | Exact GELU | Tolerance-based |
| `proposal.network.3` | `[B,K,P]`; weight `[D,P]`, bias `[D]` | proposal `[B,K,D]` | Affine transform | `linear_device` | Parameter transpose during import | Tolerance-based |
| Tension delta norm | proposal/state `[B,K,D]` | `[B,K,1]` | `(proposal-state).norm(dim=-1, keepdim=True)`, L2 norm | None | Row difference L2 norm | Tolerance-based |
| Tension feature assembly | state/proposal `[B,K,D]`, keys `[B,K,Q]`, token `[B,E]`, summary `[B,2D]`, selected scores `[B,K]`, delta norm `[B,K,1]`, step table `[M,8]` | `[B,K,4D+Q+E+10]` | Broadcast and concatenate; score and norm are scalar features | None | Broadcast and last-dimension concatenation | Exact copy except computed norm |
| `tension.network.0` | `[B,K,4D+Q+E+10]`; weight/bias same last size | same | LayerNorm, `eps=1e-5` | None | LayerNorm | Tolerance-based |
| `tension.network.1` | feature tensor; weight `[T,input]`, bias `[T]` | `[B,K,T]` | Affine transform | `linear_device` | Parameter transpose during import | Tolerance-based |
| `tension.network.2` | `[B,K,T]` | `[B,K,T]` | Exact GELU | None | Exact GELU | Tolerance-based |
| `tension.network.3` and sigmoid | `[B,K,T]`; weight `[1,T]`, bias `[1]` | logits and tension `[B,K,1]` | Affine then sigmoid | `linear_device`, `sigmoid_device` | None beyond imported transpose | Tolerance-based |
| `TensionCell` | state/proposal `[B,K,D]`, tension `[B,K,1]`, implicit stability zero | updated `[B,K,D]` | `state + tension * (proposal - state)` | `tension_update_device` | Broadcasting support must match `[B*K,D]` / `[B*K,1]` | Tolerance-based |
| Selected slot scatter | workspace `[B,N,D]`, indices `[B,K]`, updated `[B,K,D]` | `[B,N,D]` | Functional `scatter(dim=1)` | None | Batched scatter/copy | Exact copy of unchanged values; updated values tolerance-based |
| Slot-tension scatter | zeros `[B,N,1]`, indices `[B,K]`, tension `[B,K,1]` | `[B,N,1]` | Functional `scatter(dim=1)`; latest microstep overwrites selected positions | None | Batched scatter/copy and zero fill | Tolerance-based on selected values |
| Post-update tension | updated state, same proposal/keys/token/selected scores, newly summarized workspace | `[B,K,1]` | Re-run identical tension network after scatter, without updating state | Same components as pre-tension | No additional primitive | Tolerance-based |
| Readout reductions/query input | workspace `[B,N,D]`, token `[B,E]` | `[B,E+2D]` | Mean/max over slots then concatenate token | Same missing reduction/concatenation above | No additional primitive | Tolerance-based |
| `readout.query` | `[B,E+2D]`; weights `[R,E+2D]`, `[D,R]` and biases | query `[B,D]` | Linear, exact GELU, linear | `linear_device` | GELU as above | Tolerance-based |
| Readout slot scores | query `[B,D]`, workspace `[B,N,D]` | `[B,N]` | `einsum('bd,bnd->bn') / sqrt(D)` | None | Batched query/slot dot product with scale | Tolerance-based |
| Tension readout penalty | scores `[B,N]`, slot tension `[B,N,1]` | adjusted scores `[B,N]` | Add `log1p(-clamp(slot_tension.squeeze(-1), 0, 0.999))` | None | Clamp/log1p score adjustment | Tolerance-based |
| Readout weights | adjusted scores `[B,N]` | `[B,N]` | `softmax(dim=-1)` | None | Softmax as above | Tolerance-based |
| Readout context | weights `[B,N]`, workspace `[B,N,D]` | `[B,D]` | `einsum('bn,bnd->bd')` | None | Batched weighted reduction | Tolerance-based |
| `readout.output` | context `[B,D]`; LayerNorm `[D]`; weights `[R,D]`, `[R,R]` and biases | readout `[B,R]` | LayerNorm (`eps=1e-5`), linear, exact GELU, linear | `linear_device` | LayerNorm/GELU as above | Tolerance-based |

## CPU scientific-validation dimensions

The checked-in `experiments/milestone1_v1.json` specifies: `N=32`, `K=6`, `D=32`, `E=32`, `Q=16`, `P=64`, `T=64`, `R=64`, fixed `microsteps=2`, and `max_microsteps=4`. The learned microstep embedding dimension is hard-coded to 8 in both proposal and tension networks. Bridge v0 uses fixed inference and does not enable adaptive inference or any ablation flags.

## Parameter layout

PyTorch `nn.Linear` stores `[out_features,in_features]`; TensionForge `linear_device` consumes `[in_features,out_features]`. The bridge imports each linear weight by a single float32 transpose and preserves all biases, LayerNorm parameters, learned microstep embeddings, slot keys, and initial workspace values unchanged. This is a storage-layout adaptation, not a model transformation.

## Diagnostics

The unmodified PyTorch method directly returns routing scores/probabilities/indices, stacked pre/post tension, slot tension, update norms, microstep count, and aggregate workspace norms. The parity harness derives the additional required summary, selected scores/states, proposals, per-microstep updated states/workspaces, and readout vector by tracing the same source modules without changing the source repository.
