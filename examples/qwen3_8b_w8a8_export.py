"""Export a one-layer Qwen3-8B as static W8A8 prefill and decode ONNX graphs."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import torch
from torch.onnx import ONNXProgram
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from transformers.exporters import OnnxConfig, OnnxExporter

from mdc_llm_deploy.onnx import process_onnx
from mdc_llm_deploy.quantization import MinMaxConfig, quantize

MODEL_ID = "Qwen/Qwen3-8B"
SEQUENCE_LENGTH = 3072


def load_one_layer(model_id: str) -> PreTrainedModel:
    """Load pretrained Qwen3-8B weights into a one-layer model."""
    config = AutoConfig.from_pretrained(model_id)
    config.num_hidden_layers = 1
    config.use_cache = True
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.set_attn_implementation("eager")
    return model.eval()


def random_inputs(model: PreTrainedModel) -> dict[str, torch.Tensor]:
    """Create deterministic random token inputs."""
    generator = torch.Generator().manual_seed(0)
    input_ids = torch.randint(
        model.config.vocab_size,
        (1, SEQUENCE_LENGTH),
        generator=generator,
    )
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "position_ids": torch.arange(SEQUENCE_LENGTH).unsqueeze(0),
    }


def quantize_w8a8(
    model: PreTrainedModel,
    inputs: dict[str, torch.Tensor],
) -> None:
    """Run symmetric per-tensor W8A8 static quantization."""
    config = MinMaxConfig(
        weight=True,
        activation=True,
        weight_granularity="per_tensor",
        activation_granularity="per_tensor",
        weight_symmetric=True,
        activation_symmetric=True,
    )
    calibration_inputs = {**inputs, "use_cache": False}
    quantize(model, config, [calibration_inputs])


def export_graphs(
    model: PreTrainedModel,
    inputs: dict[str, torch.Tensor],
) -> dict[str, object]:
    """Export static prefill and decode graphs with real KV cache."""
    export_config = OnnxConfig(
        opset_version=21,
        optimize=False,
        dynamic=False,
        external_data=False,
    )
    generation_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
    }
    return OnnxExporter().export_for_generation(
        model,
        generation_inputs,
        export_config,
    )


def save_external(
    model: onnx.ModelProto,
    output_dir: Path,
    component: str,
) -> None:
    """Save one ONNX graph and all its weights as one external data file."""
    model_path = output_dir / f"{component}.onnx"
    data_name = f"{component}.data"
    data_path = output_dir / data_name
    model_path.unlink(missing_ok=True)
    data_path.unlink(missing_ok=True)
    onnx.save_model(
        model,
        model_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_name,
        size_threshold=0,
    )


def main(model_id: str, output_dir: Path) -> None:
    """Run loading, quantization, export, processing, and serialization."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading one layer from {model_id}")
    model = load_one_layer(model_id)
    inputs = random_inputs(model)

    print("Calibrating and converting symmetric per-tensor W8A8")
    quantize_w8a8(model, inputs)

    print("Exporting static prefill and decode graphs")
    programs = export_graphs(model, inputs)
    del model
    for component in ("prefill", "decode"):
        program = programs.pop(component)
        if not isinstance(program, ONNXProgram):
            raise TypeError(f"Expected ONNXProgram, got {type(program).__name__}")
        graph = program.model_proto
        print(f"Processing and saving {component}")
        process_onnx(graph)
        save_external(graph, output_dir, component)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID, help="Hugging Face model ID or local path")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/qwen3_8b_w8a8"),
        help="Directory for ONNX graphs and external weights",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.model, args.output_dir)
