"""Run metrics: the numbers that make "did that change help?" answerable.

The important properties are that measurement is free when nobody asked for it,
that it never breaks the thing it measures, and that the expensive paths (model
calls, Figma round trips) are counted by construction rather than wherever
someone remembered to add a counter.
"""
from __future__ import annotations

import threading

from agent import metrics


def test_latency_is_kept_as_a_distribution_not_just_a_mean():
    """A mean hides the one 40s call that made a run feel broken."""
    timing = metrics.Timing()
    for seconds in [0.1] * 90 + [40.0] * 10:   # one call in ten is pathological
        timing.observe(seconds)

    assert timing.count == 100
    assert timing.percentile(0.50) == 0.1    # the typical call is fast
    assert timing.percentile(0.95) == 40.0   # and p95 says what actually happened
    assert timing.mean_seconds < timing.percentile(0.95)  # the mean splits the difference


def test_failures_are_counted_separately_from_calls():
    timing = metrics.Timing()
    timing.observe(0.2)
    timing.observe(0.3, ok=False)

    assert timing.count == 2 and timing.errors == 1


def test_samples_are_bounded_so_a_runaway_loop_cannot_grow_them():
    timing = metrics.Timing()
    for _ in range(metrics.MAX_SAMPLES + 500):
        timing.observe(0.01)

    assert timing.count == metrics.MAX_SAMPLES + 500  # still counted
    assert len(timing.samples) == metrics.MAX_SAMPLES  # but not retained


# ---- the active recorder ---------------------------------------------------


def test_recording_outside_a_run_is_harmless():
    """Instrumented code must not need a guard -- a guard gets forgotten."""
    metrics.current().observe_model(0.5)          # no run active: must not raise
    metrics.current().observe_bridge("exec", 0.1)


def test_a_recorder_is_restored_after_the_run_that_installed_it():
    outer = metrics.RunMetrics()
    with metrics.recording(outer):
        with metrics.recording(metrics.RunMetrics()) as inner:
            inner.observe_model(1.0)
        # The nested run (the benchmark drives several) must not leave its own
        # recorder installed behind it.
        assert metrics.current() is outer
    assert metrics.current() is not outer


def test_each_thread_records_into_its_own_run():
    """The dashboard runs each run on its own thread."""
    mine = metrics.RunMetrics()
    theirs = metrics.RunMetrics()

    def other():
        with metrics.recording(theirs):
            metrics.current().observe_model(2.0)

    with metrics.recording(mine):
        thread = threading.Thread(target=other)
        thread.start()
        thread.join()
        metrics.current().observe_model(1.0)

    assert mine.model.count == 1
    assert theirs.model.count == 1


def test_a_finished_recorder_stops_the_clock():
    with metrics.recording() as measured:
        pass
    elapsed = measured.elapsed_seconds

    assert measured.finished_at is not None
    assert measured.elapsed_seconds == elapsed  # not still counting


# ---- what the numbers say --------------------------------------------------


def test_retry_rate_is_attempts_per_step():
    measured = metrics.RunMetrics()
    measured.steps_completed, measured.steps_failed = 3, 1
    measured.step_attempts = 8

    assert measured.retry_rate == 2.0  # every step took two goes on average


def test_bridge_latency_is_split_by_request_type():
    """A slow screenshot and a slow script have different causes."""
    measured = metrics.RunMetrics()
    measured.observe_bridge("exec", 0.4)
    measured.observe_bridge("screenshot", 3.0)

    snapshot = measured.snapshot()
    assert snapshot["bridge"]["exec"]["count"] == 1
    assert snapshot["bridge"]["screenshot"]["mean_seconds"] == 3.0
    assert snapshot["bridge_calls"] == 2


def test_the_snapshot_is_json_safe():
    """It is served over the dashboard API and saved into benchmark results."""
    import json

    measured = metrics.RunMetrics()
    measured.observe_model(0.5)
    measured.record_failure("script-error")
    measured.record_gate_failure("geometry")
    measured.start_step("Add the hero section", index=2, total=7)

    payload = json.loads(json.dumps(measured.snapshot()))

    assert payload["failure_reasons"] == {"script-error": 1}
    assert payload["gate_failures"] == {"geometry": 1}
    assert payload["progress"] == {
        "step": "Add the hero section", "index": 2, "total": 7, "attempt": 0
    }


def test_the_summary_names_where_the_time_went():
    measured = metrics.RunMetrics()
    measured.observe_model(5.0)
    measured.observe_bridge("exec", 0.2)
    measured.steps_completed, measured.step_attempts = 2, 3
    measured.finish()

    line = measured.summary()

    assert "model call" in line and "Figma call" in line and "attempt" in line
