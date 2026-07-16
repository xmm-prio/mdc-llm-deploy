from __future__ import annotations

from pathlib import Path

from mdc_llm_deploy.mdc_ops.operators import OPERATOR_SCHEMAS

ROOT = Path(__file__).parents[2]


def test_operator_documents_match_source_names() -> None:
    functions = {
        "ApplyRotaryPosEmb": "apply_rotary_pos_emb",
        "AscendDequant": "ascend_dequant",
        "AscendQuantV2": "ascend_quant_v2",
        "FusedInferAttentionScore": "fused_infer_attention_score",
        "MoeExpert": "moe_expert",
        "RmsNorm": "rms_norm",
    }

    assert set(functions) == set(OPERATOR_SCHEMAS)
    for key, function_name in functions.items():
        document = (ROOT / "docs" / "ops" / f"{key}.md").read_text(encoding="utf-8")
        schema = OPERATOR_SCHEMAS[key]
        assert schema.ge_name in document
        assert schema.onnx_name in document
        assert function_name in document or key == "MoeExpert"
        assert "验证" in document
