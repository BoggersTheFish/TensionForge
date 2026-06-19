from __future__ import annotations

import numpy as np

from experiments.ten_son_training_common import (
    benchmark_configuration, build_models, generate_host_batches, upload_batch,
)
from experiments.ten_son_training_trajectory_parity import MODEL_SEED, run_trajectory
from tensionforge.models.ten_son_bridge import LINEAR_WEIGHTS
from tensionforge.runtime import TensionForgeRuntime


def test_pregenerated_batches_upload_identically():
    _,_,config=benchmark_configuration("development")
    batches=generate_host_batches(config,3,777)
    assert len({batch.sha256 for batch in batches})==3
    runtime=TensionForgeRuntime(profiling=False)
    uploaded=upload_batch(runtime,batches[0])
    reconstructed=np.stack([tensor.to_numpy() for tensor in uploaded.token_ids],axis=1)
    np.testing.assert_array_equal(reconstructed,batches[0].inputs)
    np.testing.assert_array_equal(uploaded.supervised_targets.to_numpy(),batches[0].targets[:,uploaded.supervised_position])


def test_optimizer_state_exports_and_resets():
    model_config,_,_=benchmark_configuration("development");runtime=TensionForgeRuntime(profiling=False)
    _,_,bridge,initial=build_models(model_config,MODEL_SEED,runtime)
    for moment in bridge.export_optimizer_state().values():
        assert np.count_nonzero(moment["first_moment"])==0
        assert np.count_nonzero(moment["second_moment"])==0
    bridge.reset_training_state(initial)
    exported=bridge.export_parameters()
    for original,value in initial.items():
        name=original.removeprefix("workspace.");expected=value.T if name in LINEAR_WEIGHTS or name=="head.weight" else value
        np.testing.assert_array_equal(exported[name],expected)


def test_reduced_three_step_trajectory():
    result,_=run_trajectory(3)
    assert result["first_failing_step"] is None
    assert len(result["rows"])==4
    assert result["passed"]
