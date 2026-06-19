from __future__ import annotations

from experiments.ten_son_training_step_parity import controlled_case


def test_one_token_workspace_gradient_parity():
    result=controlled_case(1)
    assert result["selected_index_equality"]
    assert result["passed"],result["aggregate_gradient_metrics"]


def test_four_token_recurrent_gradient_parity():
    result=controlled_case(4)
    assert result["selected_index_equality"]
    assert result["passed"],result["aggregate_gradient_metrics"]
