"""
task_graders.py — per-task graders for all 10 tasks.

Each grader wraps the shared grade_trace() logic from grader.py but is
typed to its own task class so callers can import one function per task.

Usage:
    from grader.task_graders import grade_null_pointer, grade_off_by_one, ...
    report = grade_null_pointer(trace)
"""

from __future__ import annotations

from typing import Any, Dict, List

from grader.grader import grade_trace
from tasks.all_tasks import (
    NullPointerTask,
    OffByOneTask,
    DeadlockTask,
    SqlInjectionTask,
    IntegerOverflowTask,
    RecursionBaseTask,
    RaceConditionTask,
    MemoryLeakTask,
    SwallowedExceptionTask,
    BinarySearchTask,
)

# ── Singleton task instances (shared; they are frozen dataclasses) ────────────
_NULL_POINTER       = NullPointerTask()
_OFF_BY_ONE         = OffByOneTask()
_DEADLOCK           = DeadlockTask()
_SQL_INJECTION      = SqlInjectionTask()
_INTEGER_OVERFLOW   = IntegerOverflowTask()
_RECURSION_BASE     = RecursionBaseTask()
_RACE_CONDITION     = RaceConditionTask()
_MEMORY_LEAK        = MemoryLeakTask()
_SWALLOWED_EXCEPTION = SwallowedExceptionTask()
_BINARY_SEARCH      = BinarySearchTask()


# ── Per-task grader functions ─────────────────────────────────────────────────

def grade_null_pointer(trace: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Grade an episode for the NullPointerTask."""
    return grade_trace(trace, _NULL_POINTER)


def grade_off_by_one(trace: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Grade an episode for the OffByOneTask."""
    return grade_trace(trace, _OFF_BY_ONE)


def grade_deadlock(trace: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Grade an episode for the DeadlockTask."""
    return grade_trace(trace, _DEADLOCK)


def grade_sql_injection(trace: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Grade an episode for the SqlInjectionTask."""
    return grade_trace(trace, _SQL_INJECTION)


def grade_integer_overflow(trace: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Grade an episode for the IntegerOverflowTask."""
    return grade_trace(trace, _INTEGER_OVERFLOW)


def grade_recursion_base(trace: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Grade an episode for the RecursionBaseTask."""
    return grade_trace(trace, _RECURSION_BASE)


def grade_race_condition(trace: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Grade an episode for the RaceConditionTask."""
    return grade_trace(trace, _RACE_CONDITION)


def grade_memory_leak(trace: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Grade an episode for the MemoryLeakTask."""
    return grade_trace(trace, _MEMORY_LEAK)


def grade_swallowed_exception(trace: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Grade an episode for the SwallowedExceptionTask."""
    return grade_trace(trace, _SWALLOWED_EXCEPTION)


def grade_binary_search(trace: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Grade an episode for the BinarySearchTask."""
    return grade_trace(trace, _BINARY_SEARCH)


# ── Registry — map task name → grader function ────────────────────────────────
TASK_GRADERS: Dict[str, Any] = {
    "null_pointer":         grade_null_pointer,
    "off_by_one":           grade_off_by_one,
    "deadlock_detection":   grade_deadlock,
    "sql_injection":        grade_sql_injection,
    "integer_overflow":     grade_integer_overflow,
    "recursion_base_case":  grade_recursion_base,
    "race_condition":       grade_race_condition,
    "memory_leak":          grade_memory_leak,
    "swallowed_exception":  grade_swallowed_exception,
    "binary_search_boundary": grade_binary_search,
}


def grade_by_task_name(
    task_name: str, trace: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Grade a trace using the grader registered for task_name.

    Raises KeyError if task_name is not recognised.
    """
    grader_fn = TASK_GRADERS[task_name]
    return grader_fn(trace)
