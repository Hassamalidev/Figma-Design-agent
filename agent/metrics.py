"""What a run actually costs, measured rather than guessed.

Every improvement claimed in CLAUDE.md is argued from reading the code. That is
the honest position while there are no numbers -- but it also means nobody can
tell whether a change helped, and the two most expensive things in a run (model
calls and Figma round trips) are exactly the two nobody is counting.

This module counts them. It is deliberately:

- **Free when nobody is looking.** `current()` returns a live recorder inside a
  run and a shared throwaway outside one, so instrumented code never has to ask
  whether metrics are enabled.
- **Out of every signature.** The recorder is thread-local, so `llm.complete`
  and `execute_figma_js` record into the active run without the loop threading
  a metrics object through eight call sites. The dashboard runs one run per
  thread, which is exactly the granularity we want.
- **Not a logger.** It answers "how many, how slow, why did it fail", and
  `summary()` prints one line at the end of a run. The narrative log is
  separate and stays where it is.

Latency is kept as samples, not just a mean: a mean hides the one 40-second
call that made a run feel broken, and p95 is the number you actually tune
against.
"""
from __future__ import annotations

import threading
import time
from collections import Counter
from dataclasses import dataclass, field

# Enough samples for a stable p95 over a long run, bounded so a runaway loop
# cannot grow this without limit.
MAX_SAMPLES = 2000


@dataclass
class Timing:
    """Call count, failures, and the latency distribution for one operation."""

    count: int = 0
    errors: int = 0
    total_seconds: float = 0.0
    samples: list[float] = field(default_factory=list, repr=False)

    def observe(self, seconds: float, ok: bool = True) -> None:
        self.count += 1
        self.total_seconds += seconds
        if not ok:
            self.errors += 1
        if len(self.samples) < MAX_SAMPLES:
            self.samples.append(seconds)

    @property
    def mean_seconds(self) -> float:
        return self.total_seconds / self.count if self.count else 0.0

    def percentile(self, fraction: float) -> float:
        """Nearest-rank percentile. Exact for the sample sizes a run produces."""
        if not self.samples:
            return 0.0
        ordered = sorted(self.samples)
        rank = max(0, min(len(ordered) - 1, round(fraction * len(ordered)) - 1))
        return ordered[rank]

    def snapshot(self) -> dict:
        return {
            "count": self.count,
            "errors": self.errors,
            "total_seconds": round(self.total_seconds, 2),
            "mean_seconds": round(self.mean_seconds, 3),
            "p50_seconds": round(self.percentile(0.50), 3),
            "p95_seconds": round(self.percentile(0.95), 3),
        }


@dataclass
class RunMetrics:
    """Everything measured about one run of the agent loop."""

    # -- model ---------------------------------------------------------------
    model: Timing = field(default_factory=Timing)
    # Transient endpoint failures the client absorbed (see llm.TRANSIENT_ERRORS).
    # A run that "felt slow" is usually this number, not the model.
    model_transient_retries: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    # -- the Figma bridge ----------------------------------------------------
    # Keyed by request type (exec / metadata / screenshot), because a slow
    # screenshot and a slow script have completely different causes.
    bridge: dict[str, Timing] = field(default_factory=dict)
    # How long the run waited for the target file to connect. Not a failure --
    # the user is opening Figma -- but it dominates wall clock and would
    # otherwise be blamed on the agent.
    plugin_wait_seconds: float | None = None

    # -- the loop ------------------------------------------------------------
    steps_planned: int = 0
    steps_completed: int = 0
    steps_failed: int = 0
    # Attempts across all steps. attempts > steps means retries happened, which
    # is the single best predictor of a slow, expensive run.
    step_attempts: int = 0
    gate_failures: Counter = field(default_factory=Counter)
    # Why steps failed, bucketed -- "script-error" and "no-script-run" call for
    # completely different fixes, and the raw log does not separate them.
    failure_reasons: Counter = field(default_factory=Counter)

    # -- live progress, for the dashboard ------------------------------------
    current_step: str = ""
    current_step_index: int = 0
    current_attempt: int = 0
    started_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None

    # -- recording -----------------------------------------------------------

    def observe_model(self, seconds: float, ok: bool = True) -> None:
        self.model.observe(seconds, ok)

    def observe_bridge(self, request_type: str, seconds: float, ok: bool = True) -> None:
        self.bridge.setdefault(request_type, Timing()).observe(seconds, ok)

    def observe_tokens(self, prompt: int, completion: int) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion

    def start_step(self, description: str, index: int, total: int) -> None:
        self.current_step = description
        self.current_step_index = index
        self.steps_planned = total
        self.current_attempt = 0

    def start_attempt(self) -> None:
        self.current_attempt += 1
        self.step_attempts += 1

    def record_failure(self, reason: str) -> None:
        self.failure_reasons[reason] += 1

    def record_gate_failure(self, gate: str) -> None:
        self.gate_failures[gate] += 1

    def finish(self) -> None:
        self.finished_at = time.monotonic()

    # -- reading -------------------------------------------------------------

    @property
    def elapsed_seconds(self) -> float:
        return (self.finished_at or time.monotonic()) - self.started_at

    @property
    def bridge_calls(self) -> int:
        return sum(t.count for t in self.bridge.values())

    @property
    def retry_rate(self) -> float:
        """Attempts per step. 1.0 is a run where nothing needed a second try."""
        done = self.steps_completed + self.steps_failed
        return self.step_attempts / done if done else 0.0

    def snapshot(self) -> dict:
        """A JSON-safe view, for the dashboard and for saved benchmark results."""
        return {
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "model": self.model.snapshot(),
            "model_transient_retries": self.model_transient_retries,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "bridge": {name: timing.snapshot() for name, timing in sorted(self.bridge.items())},
            "bridge_calls": self.bridge_calls,
            "plugin_wait_seconds": (
                round(self.plugin_wait_seconds, 1) if self.plugin_wait_seconds else None
            ),
            "steps_planned": self.steps_planned,
            "steps_completed": self.steps_completed,
            "steps_failed": self.steps_failed,
            "step_attempts": self.step_attempts,
            "retry_rate": round(self.retry_rate, 2),
            "gate_failures": dict(self.gate_failures),
            "failure_reasons": dict(self.failure_reasons),
            "progress": {
                "step": self.current_step,
                "index": self.current_step_index,
                "total": self.steps_planned,
                "attempt": self.current_attempt,
            },
        }

    def summary(self) -> str:
        """One line at the end of a run: where the time and the calls went."""
        model_time = self.model.total_seconds
        share = (model_time / self.elapsed_seconds * 100) if self.elapsed_seconds else 0
        parts = [
            f"{self.elapsed_seconds:.0f}s total",
            f"{self.model.count} model call(s) {model_time:.0f}s ({share:.0f}%)"
            f" p95 {self.model.percentile(0.95):.1f}s",
            f"{self.bridge_calls} Figma call(s)",
            f"{self.step_attempts} attempt(s) over "
            f"{self.steps_completed + self.steps_failed} step(s)"
            f" (x{self.retry_rate:.2f})",
        ]
        if self.prompt_tokens or self.completion_tokens:
            parts.append(f"{self.prompt_tokens}+{self.completion_tokens} tokens")
        if self.model.errors:
            parts.append(f"{self.model.errors} model error(s)")
        if self.failure_reasons:
            reasons = ", ".join(f"{k}x{v}" for k, v in self.failure_reasons.most_common())
            parts.append(f"failures: {reasons}")
        return " | ".join(parts)


# ---- the active recorder ---------------------------------------------------
#
# Thread-local rather than a parameter: the alternative is passing a metrics
# object through `run` -> `run_step` -> `converse_step` -> `dispatch` -> the
# tool functions, and into `llm.complete`, which would put measurement code in
# every signature it touches for no benefit. The dashboard already runs each
# run on its own thread.

_STATE = threading.local()
_DETACHED = RunMetrics()  # written to when no run is active; never read


def current() -> RunMetrics:
    """The active run's recorder, or a throwaway when no run is in progress.

    Never None, so instrumented code is a plain `metrics.current().observe(...)`
    with no guard -- a guard that would eventually be forgotten somewhere.
    """
    return getattr(_STATE, "metrics", _DETACHED)


class recording:
    """Make `metrics` the active recorder for this thread.

    Restores whatever was active before, so a nested run (the benchmark drives
    several) cannot leave the wrong recorder installed.
    """

    def __init__(self, metrics: RunMetrics | None = None):
        self.metrics = metrics or RunMetrics()
        self._previous: RunMetrics | None = None

    def __enter__(self) -> RunMetrics:
        self._previous = getattr(_STATE, "metrics", None)
        _STATE.metrics = self.metrics
        return self.metrics

    def __exit__(self, *exc) -> None:
        self.metrics.finish()
        if self._previous is None:
            del _STATE.metrics
        else:
            _STATE.metrics = self._previous
