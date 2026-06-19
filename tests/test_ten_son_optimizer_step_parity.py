from __future__ import annotations

from experiments.ten_son_training_step_parity import case_c


def test_real_delayed_recall_one_step_optimizer_parity():
    result,_,_=case_c()
    assert result["selected_index_equality"]
    assert result["aggregate_gradient_metrics"]["failed_parameters"]==0
    assert result["aggregate_updated_parameter_metrics"]["failed_parameters"]==0
    assert result["passed"],result
