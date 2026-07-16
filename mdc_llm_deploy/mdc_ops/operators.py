"""MDC operator validation, reference kernels, and device dispatch."""
# mypy: disable-error-code="no-any-return,no-untyped-call"

from __future__ import annotations

from ..operator_schema import OPERATOR_SCHEMAS as OPERATOR_SCHEMAS
from .attention import (
    fused_infer_attention_score as fused_infer_attention_score,
)
from .moe import (
    moe_expert as moe_expert,
)
from .normalization import (
    apply_rotary_pos_emb as apply_rotary_pos_emb,
)
from .normalization import (
    rms_norm as rms_norm,
)
from .quantized_io import (
    ascend_dequant as ascend_dequant,
)
from .quantized_io import (
    ascend_quant_v2 as ascend_quant_v2,
)
from .registry import (
    operator_backend_status as operator_backend_status,
)
from .registry import (
    operator_schemas as operator_schemas,
)
from .registry import (
    registered_device_dispatches as registered_device_dispatches,
)
