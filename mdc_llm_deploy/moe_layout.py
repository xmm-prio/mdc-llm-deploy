"""Framework-independent expert-major MoE deployment layout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

QuantSlot = Literal["gate", "up", "down"]
Projection = Literal["gate_proj", "up_proj", "down_proj"]


@dataclass(frozen=True, slots=True)
class WeightSegment:
    """One projection segment in the flattened expert-weight tensor."""

    expert_id: int
    projection: Projection
    offset: int
    length: int
    rows: int
    columns: int


@dataclass(frozen=True, slots=True)
class MoeExpertLayout:
    """Describe an expert-major packed MoE weight matrix."""

    expert_count: int
    projections: tuple[Projection, ...] = (
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    quant_slots: tuple[QuantSlot, ...] = ("gate", "up", "down")

    def __post_init__(self) -> None:
        self._require_int("expert_count", self.expert_count)
        if self.expert_count <= 0:
            raise ValueError("expert_count must be positive")
        if self.projections != ("gate_proj", "up_proj", "down_proj"):
            raise ValueError("MoE projection order is fixed")
        if self.quant_slots != ("gate", "up", "down"):
            raise ValueError("MoE quantization slot order is fixed")

    @staticmethod
    def _require_int(name: str, value: object) -> int:
        if type(value) is not int:
            raise TypeError(f"{name} must be an integer")
        return value

    @property
    def quant_parameter_count(self) -> int:
        """Return per-projection quantization parameter count."""
        return self.expert_count * len(self.quant_slots)

    @property
    def packed_projection_count(self) -> int:
        """Return total flattened expert projection segments."""
        return self.expert_count * len(self.projections)

    @property
    def input_activation_projections(
        self,
    ) -> tuple[Projection, ...]:
        """Return projections sharing the expert input activation."""
        return self.projections[:-1]

    @property
    def output_projection(self) -> Projection:
        """Return the projection consuming the intermediate activation."""
        return self.projections[-1]

    def projection_for_fqn(
        self,
        fqn: str,
    ) -> Projection | None:
        """Return the exact projection named by a module FQN."""
        candidate = fqn.rsplit(".", 1)[-1]
        if candidate not in self.projections:
            return None
        return candidate

    def _validate_expert(self, expert_id: int) -> None:
        self._require_int("expert_id", expert_id)
        if not 0 <= expert_id < self.expert_count:
            raise ValueError(f"expert_id must be in [0, {self.expert_count})")

    def scale_index(self, expert_id: int, slot: QuantSlot) -> int:
        """Return one per-expert quantization parameter index."""
        self._validate_expert(expert_id)
        try:
            slot_index = self.quant_slots.index(slot)
        except ValueError as error:
            raise ValueError(f"Unsupported quantization slot: {slot}") from error
        return expert_id * len(self.quant_slots) + slot_index

    def weight_index(self, expert_id: int, projection: Projection) -> int:
        """Return one flattened projection segment index."""
        self._validate_expert(expert_id)
        try:
            projection_index = self.projections.index(projection)
        except ValueError as error:
            raise ValueError(f"Unsupported projection: {projection}") from error
        return expert_id * len(self.projections) + projection_index

    def quant_slot_for_projection(
        self,
        projection: Projection,
    ) -> QuantSlot:
        """Return the weight quantization slot for a projection."""
        if projection not in self.projections:
            raise ValueError(
                f"Unsupported projection: {projection}"
            )
        return cast(QuantSlot, projection.removesuffix("_proj"))

    def weight_segment(
        self,
        hidden_size: int,
        intermediate_size: int,
        expert_id: int,
        projection: Projection,
    ) -> WeightSegment:
        """Build one flattened weight segment descriptor."""
        self._require_int("hidden_size", hidden_size)
        self._require_int("intermediate_size", intermediate_size)
        if hidden_size <= 0 or intermediate_size <= 0:
            raise ValueError("MoE hidden and intermediate sizes must be positive")
        self._validate_expert(expert_id)
        index = self.projections.index(projection)
        length = hidden_size * intermediate_size
        rows, columns = (
            (hidden_size, intermediate_size)
            if projection == self.output_projection
            else (intermediate_size, hidden_size)
        )
        return WeightSegment(
            expert_id=expert_id,
            projection=projection,
            offset=index * length,
            length=length,
            rows=rows,
            columns=columns,
        )

    def weight_segments(
        self,
        hidden_size: int,
        intermediate_size: int,
    ) -> tuple[WeightSegment, ...]:
        """Return all packed projection segments in ABI order."""
        return tuple(
            self.weight_segment(
                hidden_size,
                intermediate_size,
                expert_id,
                projection,
            )
            for expert_id in range(self.expert_count)
            for projection in self.projections
        )

    def packed_weight_length(
        self,
        hidden_size: int,
        intermediate_size: int,
    ) -> int:
        """Return flattened packed-weight element count."""
        self._require_int("hidden_size", hidden_size)
        self._require_int("intermediate_size", intermediate_size)
        if hidden_size <= 0 or intermediate_size <= 0:
            raise ValueError("MoE hidden and intermediate sizes must be positive")
        return len(self.projections) * hidden_size * intermediate_size

    def routing_shape(self, token_count: int, top_k: int) -> tuple[int, int]:
        """Return routing shape for a caller-selected top-k."""
        self._require_int("token_count", token_count)
        self._require_int("top_k", top_k)
        if token_count <= 0:
            raise ValueError("token_count must be positive")
        if not 0 < top_k <= self.expert_count:
            raise ValueError("top_k must be in [1, expert_count]")
        return token_count, top_k

    def quant_parameter_order(self) -> tuple[str, ...]:
        """Return stable quantization metadata names."""
        return tuple(
            f"expert.{expert_id}.{slot}"
            for expert_id in range(self.expert_count)
            for slot in self.quant_slots
        )

    def expert_order(self) -> tuple[str, ...]:
        """Return stable packed expert metadata labels."""
        return tuple(str(expert_id) for expert_id in range(self.expert_count))


def infer_moe_layout(expert_weights_shape: tuple[int, ...]) -> MoeExpertLayout:
    """Infer layout from a rank-2 expert-major packed tensor shape."""
    if len(expert_weights_shape) != 2:
        raise ValueError("expert_weights must be expert-major rank 2")
    expert_count, packed_width = expert_weights_shape
    layout = MoeExpertLayout(expert_count)
    if packed_width <= 0:
        raise ValueError("packed expert width must be positive")
    return layout

__all__ = [
    "MoeExpertLayout",
    "Projection",
    "QuantSlot",
    "WeightSegment",
    "infer_moe_layout",
]
