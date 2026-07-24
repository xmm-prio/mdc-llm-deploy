from __future__ import annotations

import onnx
import pytest
import torch

from mdc_llm_deploy.onnx.fusion.apply_rotary_pos_emb import (
    fuse_apply_rotary_pos_emb,
)
from mdc_llm_deploy.onnx.schema import (
    ROTARY_POSITION_EMBEDDING_OP,
    register_schemas,
)

from .qwen3_export_fixtures import (
    DTYPES,
    AttentionBackend,
    Qwen3ExportCase,
    Qwen3Family,
    export_static_generation,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda dtype: str(dtype).removeprefix("torch."))
@pytest.mark.parametrize("attention_backend", list(AttentionBackend))
def test_real_qwen3_prefill_and_decode_rope_fuse(
    attention_backend: AttentionBackend,
    dtype: torch.dtype,
) -> None:
    case = Qwen3ExportCase(
        Qwen3Family.DENSE_4B,
        attention_backend,
        dtype,
    )
    components = export_static_generation(case)
    register_schemas(ROTARY_POSITION_EMBEDDING_OP)

    for model in components.values():
        result = fuse_apply_rotary_pos_emb(model)

        assert result.fused_count == 1
        assert sum(
            node.op_type == ROTARY_POSITION_EMBEDDING_OP
            for node in model.graph.node
        ) == 1
        assert not any(node.op_type == "Neg" for node in model.graph.node)
        onnx.checker.check_model(model, full_check=True)
