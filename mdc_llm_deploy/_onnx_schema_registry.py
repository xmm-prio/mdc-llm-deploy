"""Install declarative schemas into ONNX's process-local registry."""

from __future__ import annotations

from collections.abc import Iterable
from threading import RLock
from typing import Final, Protocol, cast

from onnx import defs
from onnx.defs import OpSchema

_REGISTRY_LOCK: Final = RLock()
_NO_DEFAULT: Final = object()


class _SchemaCapabilities(Protocol):
    has_function: bool
    has_context_dependent_function: bool
    has_type_and_shape_inference_function: bool
    has_data_propagation_function: bool
    node_determinism: object


class OnnxSchemaConflictError(RuntimeError):
    """Report an incompatible process-local ONNX schema."""

    def __init__(self, schema: OpSchema) -> None:
        self.name = schema.name
        self.domain = schema.domain
        self.since_version = schema.since_version
        super().__init__(
            f"conflicting ONNX schema already registered for {_schema_key(schema)}"
        )


def _schema_key(schema: OpSchema) -> str:
    return f"{schema.domain!r}::{schema.name}@{schema.since_version}"


def _require_declarative(schema: OpSchema) -> None:
    schema_capabilities = cast(_SchemaCapabilities, schema)
    capabilities = (
        ("function body", schema_capabilities.has_function),
        (
            "context-dependent function",
            schema_capabilities.has_context_dependent_function,
        ),
        (
            "shape inference callback",
            schema_capabilities.has_type_and_shape_inference_function,
        ),
        (
            "data propagation callback",
            schema_capabilities.has_data_propagation_function,
        ),
    )
    unsupported = [name for name, enabled in capabilities if enabled]
    if unsupported:
        details = ", ".join(unsupported)
        raise RuntimeError(
            f"unsupported ONNX schema capabilities for {_schema_key(schema)}: {details}"
        )


def _parameter_abi(parameter: OpSchema.FormalParameter) -> tuple[object, ...]:
    return (
        parameter.name,
        parameter.type_str,
        parameter.option,
        parameter.is_homogeneous,
        parameter.min_arity,
        parameter.differentiation_category,
    )


def _attribute_abi(attribute: OpSchema.Attribute) -> tuple[object, ...]:
    default = attribute.default_value
    serialized_default: object = (
        default.SerializeToString(deterministic=True)
        if default.ByteSize()
        else _NO_DEFAULT
    )
    return (
        attribute.name,
        attribute.type,
        attribute.required,
        serialized_default,
    )


def _schema_abi(schema: OpSchema) -> tuple[object, ...]:
    _require_declarative(schema)
    schema_capabilities = cast(_SchemaCapabilities, schema)
    return (
        schema.name,
        schema.domain,
        schema.since_version,
        tuple(_parameter_abi(parameter) for parameter in schema.inputs),
        tuple(_parameter_abi(parameter) for parameter in schema.outputs),
        tuple(
            (
                constraint.type_param_str,
                tuple(sorted(set(constraint.allowed_type_strs))),
            )
            for constraint in sorted(
                schema.type_constraints,
                key=lambda constraint: constraint.type_param_str,
            )
        ),
        tuple(
            (name, _attribute_abi(attribute))
            for name, attribute in sorted(schema.attributes.items())
        ),
        schema.deprecated,
        schema.support_level,
        schema_capabilities.node_determinism,
    )


def _get_exact_schema(schema: OpSchema) -> OpSchema | None:
    try:
        existing = defs.get_schema(
            schema.name,
            schema.since_version,
            schema.domain,
        )
    except defs.SchemaError:
        return None
    return existing if existing.since_version == schema.since_version else None


def ensure_onnx_schemas(schemas: Iterable[OpSchema]) -> None:
    """Preflight and install exact schema ABIs in the process-local registry."""
    requested = tuple(schemas)
    if not requested:
        return

    with _REGISTRY_LOCK:
        unique: dict[
            tuple[str, str, int],
            tuple[OpSchema, tuple[object, ...]],
        ] = {}
        for schema in requested:
            identity = (schema.domain, schema.name, schema.since_version)
            expected_abi = _schema_abi(schema)
            prior = unique.get(identity)
            if prior is None:
                unique[identity] = (schema, expected_abi)
            elif prior[1] != expected_abi:
                raise OnnxSchemaConflictError(schema)

        missing: list[OpSchema] = []
        for schema, expected_abi in unique.values():
            existing = _get_exact_schema(schema)
            if existing is None:
                missing.append(schema)
            elif _schema_abi(existing) != expected_abi:
                raise OnnxSchemaConflictError(schema)

        for schema in missing:
            defs.register_schema(schema)
