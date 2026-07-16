"""Framework-independent Tiny MoE deployment ABI layout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

QuantSlot = Literal["gate", "up", "intermediate", "down"]
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
class TinyMoeAbiLayout:
    """Current fixed Tiny MoE ABI consumed by MDC operators."""

    routed_expert_count: int = 4
    shared_expert_count: int = 1
    routed_top_k: int = 2
    projections: tuple[Projection, ...] = (
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    quant_slots: tuple[QuantSlot, ...] = (
        "gate",
        "up",
        "intermediate",
        "down",
    )

    def __post_init__(self) -> None:
        self._require_int("routed_expert_count", self.routed_expert_count)
        self._require_int("shared_expert_count", self.shared_expert_count)
        self._require_int("routed_top_k", self.routed_top_k)
        if self.routed_expert_count != 4:
            raise ValueError("Tiny MoE ABI requires exactly four routed experts")
        if self.shared_expert_count != 1:
            raise ValueError("Tiny MoE ABI requires exactly one shared expert")
        if self.routed_top_k != 2:
            raise ValueError("Tiny MoE ABI requires routed top-k 2")
        if self.projections != ("gate_proj", "up_proj", "down_proj"):
            raise ValueError("Tiny MoE ABI projection order is fixed")
        if self.quant_slots != ("gate", "up", "intermediate", "down"):
            raise ValueError("Tiny MoE ABI quantization slot order is fixed")

    @staticmethod
    def _require_int(name: str, value: object) -> int:
        if type(value) is not int:
            raise TypeError(f"{name} must be an integer")
        return value

    @property
    def expert_count(self) -> int:
        """Return routed plus shared expert count."""
        return self.routed_expert_count + self.shared_expert_count

    @property
    def shared_expert_id(self) -> int:
        """Return the packed shared expert id."""
        return self.routed_expert_count

    @property
    def route_width(self) -> int:
        """Return routed selections plus shared expert."""
        return self.routed_top_k + self.shared_expert_count

    @property
    def quant_parameter_count(self) -> int:
        """Return global input plus per-expert quantization slots."""
        return 1 + self.expert_count * len(self.quant_slots)

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
        return cast(Projection, candidate)

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
        return 1 + expert_id * len(self.quant_slots) + slot_index

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
        index = self.weight_index(expert_id, projection)
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
        return (
            self.packed_projection_count
            * hidden_size
            * intermediate_size
        )

    def routing_shape(self, token_count: int) -> tuple[int, int]:
        """Return the operator routing tensor shape."""
        self._require_int("token_count", token_count)
        if token_count <= 0:
            raise ValueError("token_count must be positive")
        return token_count, self.route_width

    def quant_parameter_order(self) -> tuple[str, ...]:
        """Return stable quantization metadata names."""
        return (
            "input",
            *tuple(
                f"expert.{expert_id}.{slot}"
                for expert_id in range(self.expert_count)
                for slot in self.quant_slots
            ),
        )

    def expert_order(self) -> tuple[str, ...]:
        """Return stable packed expert metadata labels."""
        return (
            *tuple(
                str(expert_id)
                for expert_id in range(self.routed_expert_count)
            ),
            f"{self.shared_expert_id}(shared)",
        )


DEFAULT_MOE_LAYOUT = TinyMoeAbiLayout()

__all__ = [
    "DEFAULT_MOE_LAYOUT",
    "Projection",
    "QuantSlot",
    "TinyMoeAbiLayout",
    "WeightSegment",
]
