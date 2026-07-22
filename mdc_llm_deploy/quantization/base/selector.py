"""Reusable module target selection."""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase


@dataclass(frozen=True, slots=True)
class TargetSelector:
    """Select module names with include and exclude glob patterns."""

    include: tuple[str, ...] = ("*",)
    exclude: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.include:
            raise ValueError("include must contain at least one pattern")
        patterns = (*self.include, *self.exclude)
        if any(not pattern for pattern in patterns):
            raise ValueError("target patterns must not be empty")

    def matches(self, module_name: str) -> bool:
        """Return whether a fully qualified module name is selected."""
        included = any(fnmatchcase(module_name, pattern) for pattern in self.include)
        excluded = any(fnmatchcase(module_name, pattern) for pattern in self.exclude)
        return included and not excluded


__all__ = ["TargetSelector"]
