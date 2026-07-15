from __future__ import annotations

from pathlib import Path

from mdc_llm_deploy.mdc_ops.operators import OPERATOR_SCHEMAS

ROOT = Path(__file__).parents[1]
FULLWIDTH_COLON = chr(0xFF1A)


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
        assert f"operators.{function_name}`" in document
        assert f'OPERATOR_SCHEMAS["{key}"]' in document
        assert f"GE 原名{FULLWIDTH_COLON}`{schema.ge_name}`" in document
        assert f"ONNX OP{FULLWIDTH_COLON}`{schema.onnx_name}`" in document
        assert "不代表 GPU、NPU、parser、ATC 或真机已验证" in document


def test_prd_freezes_stage_zero_contracts() -> None:
    prd = (ROOT / "docs" / "PRD.md").read_text(encoding="utf-8")

    for phrase in (
        "Python 3.12",
        "schema_version=1",
        "标准中间 ONNX 生命周期",
        "CAPABILITY_MATRIX",
        "`PASS`",
        "`BLOCKED`",
        "`WAIVED`",
        "不可豁免门禁",
        "GPTQ",
        "FX 数值路径",
        "面向 MDC 的 ONNX",
        "已通过 ATC 编译",
    ):
        assert phrase in prd


def test_b_side_template_records_probe_without_claiming_execution() -> None:
    document = (ROOT / "docs" / "validation" / "b-side.md").read_text(
        encoding="utf-8"
    )

    assert f"状态{FULLWIDTH_COLON}`BLOCKED`" in document
    assert "尚未在 B 端执行命令" in document
    assert "artifact_returned_to_a: false" in document
    assert "code_changed_on_b: false" in document
    assert "timeout_seconds: 1800" in document
    for op_type in (
        "NPURmsNorm",
        "ApplyRoPE",
        "FusedInferAttentionScore",
        "NPUAscendQuantV2",
        "AscendDequant",
        "MoeExpert",
        "MatMul",
    ):
        assert op_type in document
