from __future__ import annotations

import pytest
import torch
from torch import nn

from mdc_llm_deploy.errors import UnsupportedPatternError
from mdc_llm_deploy.export import export
from mdc_llm_deploy.graph.lifecycle import metadata


class TiedExportModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        layer = nn.Linear(4, 4)
        self.left = layer
        self.right = layer
        self.register_buffer("scale", torch.ones(4), persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.right(input_ids.float()) * self.scale


def test_export_maps_signature_fqns_and_preserves_aliases() -> None:
    model = TiedExportModel().eval()
    original_weight = model.left.weight

    graph = export(model, {"input_ids": torch.arange(4).reshape(1, 4)})

    mapping = metadata(graph).properties["state_fqn_to_graph_attribute"]
    assert mapping == {
        "left.bias": "left.bias",
        "left.weight": "left.weight",
        "right.bias": "right.bias",
        "right.weight": "right.weight",
        "scale": "scale",
    }
    assert graph.left.weight is graph.right.weight
    assert graph.left.bias is graph.right.bias
    assert graph.scale.device == model.scale.device
    assert "scale" in graph._non_persistent_buffers_set
    assert model.left.weight is original_weight
    assert model.left.weight is model.right.weight


@pytest.mark.parametrize(
    "offload",
    [
        {"": "disk"},
        {"layer": "cpu"},
    ],
)
def test_export_rejects_dynamic_weight_offload(offload: dict[str, str]) -> None:
    model = TiedExportModel().eval()
    if "disk" in offload.values():
        model.hf_device_map = offload
    else:
        hook_type = type("WeightPager", (), {"offload": True})
        model.right._hf_hook = hook_type()

    with pytest.raises(
        UnsupportedPatternError,
        match=r"Disk offload|Dynamic weight offload",
    ):
        export(model, {"input_ids": torch.arange(4).reshape(1, 4)})


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="requires two CUDA devices",
)
def test_export_supports_resident_multi_device_shards() -> None:
    class ResidentShards(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = nn.Linear(4, 4).to("cuda:0")
            self.second = nn.Linear(4, 4).to("cuda:1")
            self.hf_device_map = {
                "first": "cuda:0",
                "second": "cuda:1",
            }

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            hidden = self.first(input_ids.float().to("cuda:0"))
            return self.second(hidden.to("cuda:1"))

    model = ResidentShards().eval()
    identities = {
        name: id(parameter)
        for name, parameter in model.named_parameters(remove_duplicate=False)
    }

    graph = export(
        model,
        {"input_ids": torch.arange(4, device="cuda:0").reshape(1, 4)},
    )

    assert graph.first.weight.device == torch.device("cuda:0")
    assert graph.second.weight.device == torch.device("cuda:1")
    assert {
        name: id(parameter)
        for name, parameter in model.named_parameters(remove_duplicate=False)
    } == identities
    targets = {str(node.target) for node in graph.graph.nodes}
    assert any("_to_copy" in target or ".to." in target for target in targets)
