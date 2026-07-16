"""FQN selector matching and modifier inheritance."""

from __future__ import annotations

from collections.abc import Iterable


def pattern_matches(fqn: str, pattern: str) -> bool:
    """Match a dot-component subsequence, never a raw substring."""
    if not pattern:
        return False
    components = fqn.split(".")
    expected = pattern.split(".")
    width = len(expected)
    return any(
        components[index : index + width] == expected
        for index in range(len(components) - width + 1)
    )


def selected(
    fqn: str,
    include: Iterable[str],
    exclude: Iterable[str],
) -> bool:
    """Apply include-all and exclude-wins selector semantics."""
    include_patterns = tuple(include)
    included = not include_patterns or any(
        pattern_matches(fqn, pattern) for pattern in include_patterns
    )
    excluded = any(pattern_matches(fqn, pattern) for pattern in exclude)
    return included and not excluded


def effective_selector(
    root_include: tuple[str, ...],
    root_exclude: tuple[str, ...],
    local_include: tuple[str, ...] | None,
    local_exclude: tuple[str, ...] | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Replace root selectors with explicitly provided local selectors."""
    return (
        root_include if local_include is None else local_include,
        root_exclude if local_exclude is None else local_exclude,
    )
