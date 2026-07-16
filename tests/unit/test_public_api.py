from __future__ import annotations

import inspect
import subprocess
import sys
from dataclasses import fields


def test_root_public_api_surface_is_frozen() -> None:
    import mdc_llm_deploy

    assert set(mdc_llm_deploy.__all__) == {
        "AutoExportModel",
        "ExportModelConfig",
        "Qwen3Config",
        "Qwen3ForCausalLM",
        "Qwen3MoeConfig",
        "Qwen3MoeForCausalLM",
        "GraphStateError",
        "MdcDeployError",
        "OnnxExportError",
        "QuantizationConfig",
        "QuantizationConfigError",
        "UnsupportedPatternError",
        "__version__",
        "convert_to_decode",
        "export",
        "oneshot",
        "onnx_export",
    }
    assert mdc_llm_deploy.__version__ == "0.1.0"


def test_root_package_preserves_lazy_runtime_imports() -> None:
    command = (
        "import sys; "
        "import mdc_llm_deploy as package; "
        "_ = (package.QuantizationConfig, "
        "package.MdcDeployError, package.__version__); "
        "assert 'torch' not in sys.modules; "
        "assert 'onnx' not in sys.modules; "
        "_ = package.export; "
        "assert 'torch' in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", command],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_decode_conversion_has_one_public_implementation() -> None:
    from mdc_llm_deploy import convert_to_decode as root_decode
    from mdc_llm_deploy.export import convert_to_decode as package_decode
    from mdc_llm_deploy.export.api import convert_to_decode as legacy_decode
    from mdc_llm_deploy.export.decode import convert_to_decode as module_decode

    assert root_decode is package_decode is legacy_decode is module_decode


def test_export_has_one_public_implementation() -> None:
    from mdc_llm_deploy import export as root_export
    from mdc_llm_deploy.export import export as package_export
    from mdc_llm_deploy.export.api import export as module_export

    assert root_export is package_export is module_export


def test_quantization_and_onnx_export_have_one_public_implementation() -> None:
    from mdc_llm_deploy import oneshot as root_oneshot
    from mdc_llm_deploy import onnx_export as root_onnx_export
    from mdc_llm_deploy.onnx import (
        onnx_export as package_onnx_export,
    )
    from mdc_llm_deploy.onnx.api import (
        onnx_export as module_onnx_export,
    )
    from mdc_llm_deploy.quantization import (
        oneshot as package_oneshot,
    )
    from mdc_llm_deploy.quantization.api import (
        oneshot as module_oneshot,
    )

    assert root_oneshot is package_oneshot is module_oneshot
    assert (
        root_onnx_export
        is package_onnx_export
        is module_onnx_export
    )


def test_public_entrypoint_signatures_are_frozen() -> None:
    from mdc_llm_deploy import (
        convert_to_decode,
        export,
        oneshot,
        onnx_export,
    )

    assert tuple(inspect.signature(export).parameters) == (
        "model",
        "example_inputs",
    )
    assert tuple(
        inspect.signature(convert_to_decode).parameters
    ) == ("graph",)
    assert tuple(inspect.signature(oneshot).parameters) == (
        "graph",
        "config",
        "calibration_dataloader",
    )
    onnx_parameters = inspect.signature(
        onnx_export
    ).parameters
    assert tuple(onnx_parameters) == (
        "graph",
        "output_path",
        "external_data",
    )
    assert (
        onnx_parameters["external_data"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )
    assert onnx_parameters["external_data"].default is True


def test_public_exception_hierarchy_is_frozen() -> None:
    from mdc_llm_deploy import (
        GraphStateError,
        MdcDeployError,
        OnnxExportError,
        QuantizationConfigError,
        UnsupportedPatternError,
    )

    assert issubclass(GraphStateError, MdcDeployError)
    assert issubclass(UnsupportedPatternError, MdcDeployError)
    assert issubclass(OnnxExportError, MdcDeployError)
    assert issubclass(QuantizationConfigError, MdcDeployError)
    assert issubclass(QuantizationConfigError, ValueError)


def test_discovery_module_exposes_cohesive_metadata_api() -> None:
    from mdc_llm_deploy.export import api
    from mdc_llm_deploy.export.discovery import DiscoveryResult, discover_metadata

    assert tuple(field.name for field in fields(DiscoveryResult)) == (
        "input_abi",
        "output_abi",
        "boundaries",
        "properties",
    )
    assert callable(discover_metadata)
    assert not hasattr(api, "_discover_boundaries")
    assert not hasattr(api, "_model_properties")
    assert not hasattr(api, "_output_abi")
    assert not hasattr(api, "_tensor_abi")


def test_extracted_operators_keep_one_public_implementation() -> None:
    from mdc_llm_deploy.operators import (
        fused_infer_attention_score as package_attention,
    )
    from mdc_llm_deploy.operators import moe_expert as package_moe
    from mdc_llm_deploy.operators.runtime.attention import (
        fused_infer_attention_score as module_attention,
    )
    from mdc_llm_deploy.operators.runtime.moe import moe_expert as module_moe

    assert package_attention is module_attention
    assert package_moe is module_moe
