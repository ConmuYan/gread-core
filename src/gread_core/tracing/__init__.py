"""Trace node selection: bucket assignment and evidence diversity sampling."""

from __future__ import annotations

from gread_core.tracing.buckets import assign_buckets
from gread_core.tracing.diversity import diversity_sample
from gread_core.tracing.selector import SelectionResult, TraceSelector

__all__ = [
    "SelectionResult",
    "TraceSelector",
    "assign_buckets",
    "diversity_sample",
]
