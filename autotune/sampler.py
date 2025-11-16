"""Sampling strategies for camera auto-tuning."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from .properties import PropertyInfo, PropertyManager, PropertyState


@dataclass
class SamplingConfig:
    """Configuration for the sampler."""

    settle_delay: float = 0.2
    coarse_sample_count: int = 24
    refine_iterations: int = 2
    refine_step_shrink: float = 0.5
    safe_window_ratio: float = 0.1
    max_frames: int = 180
    max_seconds: float = 12.0


@dataclass
class Candidate:
    """A property candidate evaluated during sampling."""

    state: PropertyState
    score: float
    metrics: Dict[str, float]
    frames_used: int = 0
    phase: str = ""


@dataclass
class SamplingResult:
    """Summary of a sampling run."""

    best: Optional[Candidate]
    history: List[Candidate] = field(default_factory=list)


class Sampler:
    """Implements coarse sampling followed by coordinate-descent refinement."""

    def __init__(self, manager: PropertyManager, config: Optional[SamplingConfig] = None) -> None:
        self._manager = manager
        self._config = config or SamplingConfig()
        self._bases = [2, 3, 5, 7, 11, 13]
        self._start_time: Optional[float] = None
        self._frames_used = 0
        self._logger = logging.getLogger(__name__)

    def run(self, evaluate: Callable[[PropertyState], Tuple[float, Dict[str, float], int]]) -> SamplingResult:
        self._start_time = time.monotonic()
        self._frames_used = 0
        best: Optional[Candidate] = None
        history: List[Candidate] = []

        coarse_candidates = self._coarse_candidates()
        for index, state in enumerate(coarse_candidates):
            candidate = self._evaluate_candidate(state, evaluate, phase="coarse", index=index)
            if candidate is None:
                break
            history.append(candidate)
            if best is None or candidate.score > best.score or (
                math.isclose(candidate.score, best.score, rel_tol=1e-6)
                and self._tie_break(candidate.state, best.state)
            ):
                best = candidate

        if best is None:
            return SamplingResult(best=None, history=history)

        refined = self._refine(best.state, evaluate, history)
        if refined and refined.score > best.score:
            best = refined

        return SamplingResult(best=best, history=history)

    # ------------------------------------------------------------------
    # Internal helpers

    def _evaluate_candidate(
        self,
        state: PropertyState,
        evaluate: Callable[[PropertyState], Tuple[float, Dict[str, float], int]],
        phase: str,
        index: int,
    ) -> Optional[Candidate]:
        if not self._budget_allows():
            return None
        self._manager.apply_state(state)
        time.sleep(max(0.0, self._config.settle_delay))
        score, metrics, frames_used = evaluate(state)
        self._frames_used += frames_used
        candidate = Candidate(state=state.copy(), score=score, metrics=metrics, frames_used=frames_used, phase=f"{phase}-{index}")
        self._logger.info(
            "autotune.sample phase=%s score=%.2f frames=%d bad=%.0f",
            candidate.phase,
            candidate.score,
            candidate.frames_used,
            metrics.get("bad_frame", 0.0),
        )
        return candidate

    def _budget_allows(self) -> bool:
        if self._start_time is None:
            return True
        if self._frames_used >= self._config.max_frames:
            return False
        elapsed = time.monotonic() - self._start_time
        return elapsed < self._config.max_seconds

    def _coarse_candidates(self) -> Iterable[PropertyState]:
        properties = [info for info in self._manager.properties.values() if info.high_impact and info.writable]
        baseline = self._manager.get_state()
        if not properties:
            return [baseline]
        safe_states: List[PropertyState] = [baseline]
        for index in range(1, self._config.coarse_sample_count + 1):
            values: Dict[str, float] = {}
            auto_modes: Dict[str, bool] = {}
            for axis, info in enumerate(properties):
                base = self._bases[axis % len(self._bases)]
                sample = _halton(index, base)
                window_min, window_max = _safe_window(info, self._config.safe_window_ratio)
                values[info.name] = window_min + sample * (window_max - window_min)
                if info.auto_supported:
                    auto_modes[info.name] = False
            safe_states.append(PropertyState(values=values, auto_modes=auto_modes))
        return safe_states

    def _refine(
        self,
        seed_state: PropertyState,
        evaluate: Callable[[PropertyState], Tuple[float, Dict[str, float], int]],
        history: List[Candidate],
    ) -> Optional[Candidate]:
        current_state = seed_state.copy()
        current_score = history[-1].score if history else float("-inf")
        properties = [info for info in self._manager.properties.values() if info.high_impact and info.writable]
        if not properties:
            return None
        step_sizes = {info.name: max(info.step, (info.max_value - info.min_value) / 12.0) for info in properties}
        best_candidate: Optional[Candidate] = None

        for iteration in range(self._config.refine_iterations):
            improved = False
            for info in properties:
                step = step_sizes[info.name]
                for direction in (-1, 1):
                    test_state = current_state.copy()
                    target = test_state.values.get(info.name, info.current_value) + direction * step
                    test_state.values[info.name] = info.clamp(target)
                    if info.auto_supported:
                        test_state.auto_modes[info.name] = False
                    candidate = self._evaluate_candidate(test_state, evaluate, phase=f"refine{iteration}", index=direction)
                    if candidate is None:
                        return best_candidate
                    history.append(candidate)
                    if candidate.score > current_score:
                        current_state = candidate.state
                        current_score = candidate.score
                        best_candidate = candidate
                        improved = True
                        break
                if improved:
                    break
            for name in step_sizes:
                step_sizes[name] *= self._config.refine_step_shrink
            if not improved:
                break
        return best_candidate

    def _tie_break(self, candidate_state: PropertyState, best_state: PropertyState) -> bool:
        exposure_key = "exposure"
        gain_key = "gain"
        cand_exposure = candidate_state.values.get(exposure_key, float("inf"))
        best_exposure = best_state.values.get(exposure_key, float("inf"))
        if cand_exposure != best_exposure:
            return cand_exposure > best_exposure
        cand_gain = candidate_state.values.get(gain_key, float("inf"))
        best_gain = best_state.values.get(gain_key, float("inf"))
        return cand_gain < best_gain


def _halton(index: int, base: int) -> float:
    result = 0.0
    f = 1.0 / base
    i = index
    while i > 0:
        result += f * (i % base)
        i //= base
        f /= base
    return result


def _safe_window(info: PropertyInfo, ratio: float) -> Tuple[float, float]:
    span = info.max_value - info.min_value
    guard = span * ratio
    window_min = info.min_value + guard
    window_max = info.max_value - guard
    if window_min >= window_max:
        return info.min_value, info.max_value
    return window_min, window_max
