from __future__ import annotations

import pytest

from experiments.ten_son_forward_parity import (
    _load_torch_source,
    development_config,
    run_parity_case,
)


try:
    _, ModelConfig, _ = _load_torch_source()
except RuntimeError as exc:
    pytest.skip(str(exc), allow_module_level=True)


def test_one_token_pytorch_tensionforge_parity() -> None:
    result, _, _, _ = run_parity_case(development_config(ModelConfig), sequence_length=1)
    assert result["selected_index_equality"]
    assert result["passed"], result


def test_multi_token_recurrent_workspace_parity() -> None:
    result, _, _, _ = run_parity_case(development_config(ModelConfig), sequence_length=4)
    assert result["selected_index_equality"]
    assert result["passed"], result
