"""Generate deterministic Qwen3 floating FIA ATC validation bundles."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import onnx
import torch
from google.protobuf.message import Message
from onnx import TensorProto, ValueInfoProto, helper

from mdc_llm_deploy.onnx import process_onnx
from mdc_llm_deploy.onnx.schemas import (
    CANN_FIA_SOURCE_COMMIT,
    FUSED_INFER_ATTENTION_SCORE_OP,
)
from tests.integration.onnx.qwen3_export_fixtures import (
    AttentionBackend,
    Qwen3ExportCase,
    Qwen3Family,
    export_static_generation,
    export_static_prefill,
)

_BUNDLE_NAME: Final = "qwen3_fia"
_SOC_VERSION: Final = "MC62CM12AA"
_DTYPES: Final = (torch.float16, torch.bfloat16)
_STAGES: Final = ("prefill", "decode")
_MODEL_DIMENSIONS: Final = {
    "batch_size": 1,
    "prefill_length": 3,
    "decode_length": 1,
    "decode_kv_length": 4,
    "vocab_size": 32,
    "hidden_size": 32,
    "num_hidden_layers": 1,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
}


@dataclass(frozen=True, slots=True)
class HardwareCase:
    """Describe one member of the Qwen3 FIA hardware matrix."""

    family: Qwen3Family
    attention_backend: AttentionBackend
    dtype: torch.dtype
    stage: str

    @property
    def name(self) -> str:
        """Return a stable case identifier."""
        dtype_name = str(self.dtype).removeprefix("torch.")
        return (
            f"{self.family.value}-{self.attention_backend.value}-"
            f"{self.stage}-{dtype_name}"
        )

    @property
    def export_case(self) -> Qwen3ExportCase:
        """Return the shared Qwen3 export configuration."""
        return Qwen3ExportCase(self.family, self.attention_backend, self.dtype)


HARDWARE_CASES: Final = tuple(
    HardwareCase(family, backend, dtype, stage)
    for family in Qwen3Family
    for backend in AttentionBackend
    for dtype in _DTYPES
    for stage in _STAGES
)


def _export_case(case: HardwareCase) -> onnx.ModelProto:
    if case.stage == "prefill":
        return export_static_prefill(case.export_case)
    if case.stage == "decode":
        return export_static_generation(case.export_case)["decode"]
    raise ValueError(f"Unsupported hardware stage: {case.stage}")


def _strip_export_metadata(message: Message) -> None:
    """Remove exporter diagnostics that contain process-global trace identifiers."""
    for field, value in list(message.ListFields()):
        if field.name == "metadata_props":
            message.ClearField(field.name)
        elif field.message_type is not None and field.is_repeated:
            for child in value:
                _strip_export_metadata(child)
        elif field.message_type is not None:
            _strip_export_metadata(value)


def _shape(value: ValueInfoProto) -> list[int]:
    dimensions = value.type.tensor_type.shape.dim
    if any(dimension.dim_param or dimension.dim_value <= 0 for dimension in dimensions):
        raise ValueError(f"Input {value.name!r} must have a positive static shape")
    return [dimension.dim_value for dimension in dimensions]


def _input_recipe(value: ValueInfoProto, stage: str) -> dict[str, object]:
    name = value.name
    if name == "input_ids":
        return {"kind": "token_ids", "start": 1, "vocab_size": 32}
    if name == "attention_mask":
        return {"kind": "ones"}
    if name == "position_ids":
        return {"kind": "arange", "start": 0 if stage == "prefill" else 3}
    if "past_key_values.layers.0.key" in name:
        return {"kind": "normal", "seed": 1001, "mean": 0.0, "std": 0.02}
    if "past_key_values.layers.0.value" in name:
        return {"kind": "normal", "seed": 1002, "mean": 0.0, "std": 0.02}
    raise ValueError(f"No deterministic input recipe for {name!r}")


def _input_manifest(model: onnx.ModelProto, stage: str) -> list[dict[str, object]]:
    return [
        {
            "name": value.name,
            "dtype": TensorProto.DataType.Name(value.type.tensor_type.elem_type).lower(),
            "shape": _shape(value),
            "recipe": _input_recipe(value, stage),
        }
        for value in model.graph.input
    ]


def _validate_fia_abi(model: onnx.ModelProto) -> None:
    nodes = [
        node
        for node in model.graph.node
        if node.domain in ("", "ai.onnx")
        and node.op_type == FUSED_INFER_ATTENTION_SCORE_OP
    ]
    if len(nodes) != 1:
        raise ValueError(f"Expected one FIA node, got {len(nodes)}")
    node = nodes[0]
    if len(node.input) != 31 or len(node.output) != 2:
        raise ValueError(
            f"FIA ABI mismatch: {len(node.input)} inputs, {len(node.output)} outputs"
        )
    if not all(node.input[index] for index in (0, 1, 2, 4)):
        raise ValueError("FIA required Q/K/V/mask inputs must be populated")
    if node.input[3] or any(node.input[5:]):
        raise ValueError("Unsupported FIA optional input slot is populated")
    attributes = {
        attribute.name: helper.get_attribute_value(attribute)
        for attribute in node.attribute
    }
    if (
        attributes.get("input_layout") != b"BNSD"
        or attributes.get("num_heads") != 4
        or attributes.get("num_key_value_heads") != 2
    ):
        raise ValueError("FIA layout or head attributes do not match Qwen3 fixture")


def _case_manifest(
    case: HardwareCase,
    model: onnx.ModelProto,
    model_path: Path,
    digest: str,
) -> dict[str, object]:
    return {
        "name": case.name,
        "family": case.family.value,
        "attention_backend": case.attention_backend.value,
        "stage": case.stage,
        "dtype": str(case.dtype).removeprefix("torch."),
        "model": model_path.as_posix(),
        "model_sha256": digest,
        "model_byte_size": len(model.SerializeToString()),
        "inputs": _input_manifest(model, case.stage),
        "expected_fia": {
            "count": 1,
            "input_count": 31,
            "output_count": 2,
            "input_layout": "BNSD",
            "num_heads": 4,
            "num_key_value_heads": 2,
        },
    }


def _generate_bundle(temporary_dir: Path) -> dict[str, object]:
    models_dir = temporary_dir / "models"
    models_dir.mkdir(parents=True)
    cases: list[dict[str, object]] = []

    for case in HARDWARE_CASES:
        model = _export_case(case)
        process_onnx(model)
        _strip_export_metadata(model)
        onnx.checker.check_model(model, full_check=True)
        _validate_fia_abi(model)

        serialized = model.SerializeToString()
        digest = hashlib.sha256(serialized).hexdigest()
        relative_model_path = Path("models") / f"{digest}.onnx"
        model_path = temporary_dir / relative_model_path
        if not model_path.exists():
            model_path.write_bytes(serialized)
        cases.append(_case_manifest(case, model, relative_model_path, digest))

    return {
        "schema_version": 1,
        "name": _BUNDLE_NAME,
        "soc_version": _SOC_VERSION,
        "cann_fia_source_commit": CANN_FIA_SOURCE_COMMIT,
        "model_fixture": _MODEL_DIMENSIONS,
        "matrix_axes": {
            "family": [family.value for family in Qwen3Family],
            "attention_backend": [backend.value for backend in AttentionBackend],
            "stage": list(_STAGES),
            "dtype": [str(dtype).removeprefix("torch.") for dtype in _DTYPES],
        },
        "case_count": len(cases),
        "unique_model_count": len(tuple(models_dir.glob("*.onnx"))),
        "cases": cases,
    }


def generate(output_root: Path) -> Path:
    """Generate the complete content-addressed Qwen3 FIA hardware bundle."""
    output_root = output_root.resolve()
    bundle_dir = output_root / _BUNDLE_NAME
    temporary_dir = output_root / f".{_BUNDLE_NAME}.tmp"
    shutil.rmtree(temporary_dir, ignore_errors=True)
    temporary_dir.mkdir(parents=True)
    try:
        manifest = _generate_bundle(temporary_dir)
        (temporary_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        output_root.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(bundle_dir, ignore_errors=True)
        temporary_dir.replace(bundle_dir)
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    return bundle_dir


__all__ = ["HARDWARE_CASES", "HardwareCase", "generate"]
