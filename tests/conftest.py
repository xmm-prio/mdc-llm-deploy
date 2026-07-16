from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from mdc_llm_deploy.export import export
from tests.support.models.qwen3 import dense_model

InputFactory = Callable[[int], dict[str, torch.Tensor]]
GraphFactory = Callable[[torch.nn.Module | None, int], torch.fx.GraphModule]


@pytest.fixture
def input_factory() -> InputFactory:
    """Build deterministic token inputs with a configurable sequence length."""

    def make_inputs(sequence_length: int = 8) -> dict[str, torch.Tensor]:
        input_ids = torch.arange(sequence_length).reshape(1, sequence_length) % 128
        return {"input_ids": input_ids}

    return make_inputs


@pytest.fixture
def graph_factory(input_factory: InputFactory) -> GraphFactory:
    """Export fresh dense or caller-provided models for isolated tests."""

    def make_graph(
        model: torch.nn.Module | None = None,
        sequence_length: int = 8,
    ) -> torch.fx.GraphModule:
        selected_model = (
            model if model is not None else dense_model(sequence_length)
        )
        return export(selected_model, input_factory(sequence_length))

    return make_graph
