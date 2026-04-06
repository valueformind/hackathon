"""Episode grader — evaluates a full trace across four explicit dimensions."""

from typing import Any, Dict, List

from tasks.task import YourTask

# Score weights for each dimension (sum to 1.0)
_W_CORRECTNESS  = 0.35
_W_COMPLETENESS = 0.25
_W_QUALITY      = 0.25
_W_ADHERENCE    = 0.15


def grade_trace(trace: List[Dict[str, Any]], task: YourTask) -> Dict[str, Any]:
    """
    Evaluate an episode trace across four dimensions:

    ┌─────────────────┬──────────────────────────────────────────────────┐
    │ Dimension       │ What it measures                                 │
    ├─────────────────┼──────────────────────────────────────────────────┤
    │ Correctness     │ Did the tests/code produce correct outcomes?     │
    │                 │ — bug found, deadlock detected, no spurious fails│
    ├─────────────────┼──────────────────────────────────────────────────┤
    │ Completeness    │ Was the episode fully worked through?            │
    │                 │ — finish reached, coverage target met,           │
    │                 │   all required action types used                 │
    ├─────────────────┼──────────────────────────────────────────────────┤
    │ Task quality    │ How good were the generated tests?               │
    │                 │ — assertion density, test count, flakiness,      │
    │                 │   coverage gain per run                          │
    ├─────────────────┼──────────────────────────────────────────────────┤
    │ Env adherence   │ Did the agent respect the environment contract?  │
    │                 │ — no invalid actions/payloads, no infra aborts   │
    └─────────────────┴──────────────────────────────────────────────────┘

    Returns a report with per-dimension scores [0.0–1.0],
    a weighted composite score, pass/fail verdict, and failure breakdown.
    """
    if not trace:
        return _empty_report(task)

    # ── raw episode data ───────────────────────────────────────────────
    total_reward = float(sum(step.get("reward", 0.0) for step in trace))
    final_state  = trace[-1].get("state", {}) or {}
    last_status  = trace[-1].get("info", {}).get("status", "")
    history      = final_state.get("history", []) or []

    last_test_result  = final_state.get("last_test_result", {}) or {}
    coverage          = float(final_state.get("coverage", 0.0) or 0.0)
    test_count        = len(final_state.get("tests", []) or [])
    tests             = final_state.get("tests", []) or []

    found_bug         = bool(last_test_result.get("found_bug", False))
    deadlock_detected = bool(last_test_result.get("deadlock_detected", False))
    passed_tests      = int(last_test_result.get("passed", 0))
    failed_tests      = int(last_test_result.get("failed", 0))
    timeout_count     = int(last_test_result.get("timeout_count", 0))
    flaky_rate        = float(last_test_result.get("flaky_rate", 0.0))
    failure_reason    = last_test_result.get("failure_reason", None)

    target_coverage   = getattr(task, "target_coverage", 0.8)

    # ── collect per-step action types and statuses ─────────────────────
    action_types_used: List[str] = [
        h.get("action", {}).get("action_type", "") for h in history
        if isinstance(h.get("action"), dict)
    ]
    statuses_seen: List[str] = [h.get("status", "") for h in history]

    invalid_count = sum(
        1 for s in statuses_seen
        if s in {"invalid_action_format", "invalid_action_type", "invalid_payload"}
    )
    env_abort = last_status in {"env_setup_issue", "max_steps_exceeded"}

    # ══════════════════════════════════════════════════════════════════
    # DIMENSION 1 — CORRECTNESS
    # Did the tests actually identify bugs / did the code end up correct?
    # ══════════════════════════════════════════════════════════════════
    correctness_score = 0.0
    correctness_notes: List[str] = []

    if found_bug or deadlock_detected:
        correctness_score += 0.5
        correctness_notes.append("bug or deadlock correctly detected (+0.5)")
    else:
        correctness_notes.append("no bug detected (0.0)")

    total_runs = passed_tests + failed_tests
    if total_runs > 0:
        pass_rate = passed_tests / total_runs
        correctness_score += 0.3 * pass_rate
        correctness_notes.append(f"test pass rate {pass_rate:.0%} (+{0.3 * pass_rate:.2f})")
    else:
        correctness_notes.append("no test run results available")

    if flaky_rate == 0.0:
        correctness_score += 0.2
        correctness_notes.append("no flaky tests (+0.2)")
    else:
        penalty = 0.2 * flaky_rate
        correctness_score -= penalty
        correctness_notes.append(f"flaky_rate={flaky_rate:.0%} (-{penalty:.2f})")

    correctness_score = max(0.0, min(1.0, correctness_score))

    # ══════════════════════════════════════════════════════════════════
    # DIMENSION 2 — COMPLETENESS
    # Was the episode fully worked through to a valid finish?
    # ══════════════════════════════════════════════════════════════════
    completeness_score = 0.0
    completeness_notes: List[str] = []

    episode_completed = last_status in {"success", "incomplete"}
    if episode_completed:
        completeness_score += 0.3
        completeness_notes.append("episode reached finish (+0.3)")
    else:
        completeness_notes.append("episode did not reach finish (0.0)")

    if coverage >= target_coverage:
        completeness_score += 0.4
        completeness_notes.append(f"coverage {coverage:.0%} >= target {target_coverage:.0%} (+0.4)")
    else:
        partial = (coverage / target_coverage) * 0.4
        completeness_score += partial
        completeness_notes.append(
            f"partial coverage {coverage:.0%} / {target_coverage:.0%} (+{partial:.2f})"
        )

    required_action_types = {"generate_tests", "run_tests", "finish"}
    used_set = set(action_types_used)
    covered = required_action_types & used_set
    action_completeness = len(covered) / len(required_action_types)
    completeness_score += 0.3 * action_completeness
    completeness_notes.append(
        f"required actions used: {covered} / {required_action_types} (+{0.3 * action_completeness:.2f})"
    )

    completeness_score = max(0.0, min(1.0, completeness_score))

    # ══════════════════════════════════════════════════════════════════
    # DIMENSION 3 — TASK QUALITY
    # How good were the generated tests themselves?
    # ══════════════════════════════════════════════════════════════════
    quality_score = 0.0
    quality_notes: List[str] = []

    # Test count: 3+ tests considered sufficient
    count_score = min(1.0, test_count / 3)
    quality_score += 0.3 * count_score
    quality_notes.append(f"test_count={test_count} → count_score={count_score:.2f} (+{0.3 * count_score:.2f})")

    # Assertion density: ratio of tests containing 'assert'
    if tests:
        assertion_ratio = sum(1 for t in tests if "assert" in t.lower()) / len(tests)
        quality_score += 0.3 * assertion_ratio
        quality_notes.append(f"assertion_ratio={assertion_ratio:.0%} (+{0.3 * assertion_ratio:.2f})")
    else:
        quality_notes.append("no tests to assess assertion density")

    # Coverage gain over the episode (compare first vs final run)
    first_run = next(
        (h for h in history if h.get("status") == "tests_ran"), None
    )
    first_coverage = 0.0
    if first_run:
        first_state_updates = first_run.get("action", {})  # not available; use 0.0 as baseline
        first_coverage = 0.0
    coverage_gain = max(0.0, coverage - first_coverage)
    gain_score = min(1.0, coverage_gain / target_coverage)
    quality_score += 0.2 * gain_score
    quality_notes.append(f"coverage_gain={coverage_gain:.0%} (+{0.2 * gain_score:.2f})")

    # Penalise flakiness
    quality_score -= 0.2 * flaky_rate
    quality_notes.append(f"flaky penalty={-0.2 * flaky_rate:.2f}")

    quality_score = max(0.0, min(1.0, quality_score))

    # ══════════════════════════════════════════════════════════════════
    # DIMENSION 4 — ENV ADHERENCE
    # Did the agent respect the environment contract at every step?
    # ══════════════════════════════════════════════════════════════════
    adherence_score = 1.0
    adherence_notes: List[str] = []
    total_steps = len(history)

    if total_steps > 0:
        # Deduct 0.15 per invalid step (format/type/payload errors)
        invalid_penalty = min(1.0, invalid_count * 0.15)
        adherence_score -= invalid_penalty
        if invalid_count:
            adherence_notes.append(f"{invalid_count} invalid action(s) (-{invalid_penalty:.2f})")
        else:
            adherence_notes.append("no invalid actions (0.0 penalty)")
    else:
        adherence_notes.append("no steps taken")

    if env_abort:
        adherence_score -= 0.4
        adherence_notes.append(f"env/infra abort: {last_status} (-0.4)")
    else:
        adherence_notes.append("no env abort (+0.0 penalty)")

    if timeout_count > 0:
        timeout_penalty = min(0.3, timeout_count * 0.1)
        adherence_score -= timeout_penalty
        adherence_notes.append(f"{timeout_count} timeout(s) (-{timeout_penalty:.2f})")
    else:
        adherence_notes.append("no timeouts")

    adherence_score = max(0.0, min(1.0, adherence_score))

    # ══════════════════════════════════════════════════════════════════
    # COMPOSITE SCORE + VERDICT
    # ══════════════════════════════════════════════════════════════════
    composite = (
        _W_CORRECTNESS  * correctness_score
        + _W_COMPLETENESS * completeness_score
        + _W_QUALITY      * quality_score
        + _W_ADHERENCE    * adherence_score
    )

    # Pass: composite >= 0.7 AND correctness > 0 AND coverage gate met
    passed = (
        composite >= 0.7
        and correctness_score > 0.0
        and coverage >= target_coverage
        and (found_bug or deadlock_detected)
    )

    # ── failure classifications ────────────────────────────────────────
    failure_classifications: List[str] = []
    if not passed:
        if env_abort:
            failure_classifications.append(last_status)
        if last_status == "incomplete":
            failure_classifications.append("premature_finish")
        if coverage < target_coverage:
            failure_classifications.append(f"low_coverage ({coverage:.0%} < {target_coverage:.0%})")
        if not (found_bug or deadlock_detected):
            failure_classifications.append("no_bug_found")
        if test_count == 0:
            failure_classifications.append("no_tests")
        if invalid_count > 0:
            failure_classifications.append(f"invalid_actions ({invalid_count})")
        if composite < 0.7:
            failure_classifications.append(f"low_composite_score ({composite:.2f} < 0.70)")

    return {
        "task":          task.name,
        "passed":        passed,
        "composite_score": round(composite, 3),
        "total_reward":  total_reward,
        "steps":         len(trace),
        # ── four dimensions ──────────────────────────────────────────
        "dimensions": {
            "correctness": {
                "score":  round(correctness_score, 3),
                "weight": _W_CORRECTNESS,
                "notes":  correctness_notes,
            },
            "completeness": {
                "score":  round(completeness_score, 3),
                "weight": _W_COMPLETENESS,
                "notes":  completeness_notes,
            },
            "task_quality": {
                "score":  round(quality_score, 3),
                "weight": _W_QUALITY,
                "notes":  quality_notes,
            },
            "env_adherence": {
                "score":  round(adherence_score, 3),
                "weight": _W_ADHERENCE,
                "notes":  adherence_notes,
            },
        },
        # ── raw metrics ───────────────────────────────────────────────
        "metrics": {
            "coverage":          coverage,
            "test_count":        test_count,
            "found_bug":         found_bug,
            "deadlock_detected": deadlock_detected,
            "passed_tests":      passed_tests,
            "failed_tests":      failed_tests,
            "flaky_rate":        flaky_rate,
            "timeout_count":     timeout_count,
            "invalid_actions":   invalid_count,
            "failure_reason":    failure_reason,
        },
        "failure_classifications": failure_classifications,
        "final_state": final_state,
    }


def _empty_report(task: YourTask) -> Dict[str, Any]:
    return {
        "task":            task.name,
        "passed":          False,
        "composite_score": 0.0,
        "total_reward":    0.0,
        "steps":           0,
        "dimensions": {
            "correctness":  {"score": 0.0, "weight": _W_CORRECTNESS,  "notes": ["no trace"]},
            "completeness": {"score": 0.0, "weight": _W_COMPLETENESS, "notes": ["no trace"]},
            "task_quality": {"score": 0.0, "weight": _W_QUALITY,      "notes": ["no trace"]},
            "env_adherence":{"score": 0.0, "weight": _W_ADHERENCE,    "notes": ["no trace"]},
        },
        "metrics": {
            "coverage": 0.0, "test_count": 0, "found_bug": False,
            "deadlock_detected": False, "passed_tests": 0, "failed_tests": 0,
            "flaky_rate": 0.0, "timeout_count": 0, "invalid_actions": 0,
            "failure_reason": None,
        },
        "failure_classifications": ["no_trace"],
        "final_state": {},
    }
