"""High level orchestration of camera auto-tuning."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2

from .metrics import compute_metrics
from .properties import PropertyManager, PropertyState
from .sampler import Sampler, SamplingConfig, SamplingResult

LOGGER = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Public result type returned by :func:`optimize_capture`."""

    sampling: SamplingResult
    baseline_score: float
    baseline_state: PropertyState


def optimize_capture(
    capture: cv2.VideoCapture,
    sampler_config: Optional[SamplingConfig] = None,
    warmup_frames: int = 6,
) -> OptimizationResult:
    """Auto-tune camera properties for the provided capture object."""

    manager = PropertyManager(capture)
    baseline_state = manager.get_state()
    LOGGER.info(
        "autotune.start properties=%d warmup=%d",
        len(manager.properties),
        warmup_frames,
    )

    try:
        _warm_up(capture, warmup_frames)
        manager.apply_state(baseline_state)
        baseline_score, baseline_metrics, baseline_frames = _capture_and_score(capture)
        LOGGER.info(
            "autotune.baseline score=%.2f frames=%d bad=%.0f",
            baseline_score,
            baseline_frames,
            baseline_metrics.get("bad_frame", 0.0),
        )

        sampler = Sampler(manager, sampler_config)

        def evaluate(_: PropertyState):
            return _capture_and_score(capture)

        sampling_result = sampler.run(evaluate)
        best_candidate = sampling_result.best
        if best_candidate is None or best_candidate.score <= baseline_score:
            LOGGER.info(
                "autotune.result outcome=no-improvement baseline=%.2f best=%s",
                baseline_score,
                "none" if best_candidate is None else f"{best_candidate.score:.2f}",
            )
            manager.apply_state(baseline_state)
            return OptimizationResult(sampling=sampling_result, baseline_score=baseline_score, baseline_state=baseline_state)

        manager.apply_state(best_candidate.state)
        LOGGER.info(
            "autotune.result outcome=applied score=%.2f",
            best_candidate.score,
        )
        return OptimizationResult(sampling=sampling_result, baseline_score=baseline_score, baseline_state=baseline_state)
    except Exception as exc:  # pragma: no cover - defensive against driver faults
        LOGGER.exception("autotune.error %s", exc)
        manager.apply_state(baseline_state)
        raise


def _warm_up(capture: cv2.VideoCapture, frames: int) -> None:
    for _ in range(max(0, frames)):
        ret, _ = capture.read()
        if not ret:
            break


def _capture_and_score(capture: cv2.VideoCapture) -> tuple[float, dict, int]:
    frames_used = 0
    metrics = {"bad_frame": 1.0, "composite_score": 0.0}
    score = 0.0
    for index in range(2):
        ret, frame = capture.read()
        if not ret:
            break
        frames_used += 1
        if index == 0:
            continue
        metrics = compute_metrics(frame)
        score = float(metrics.get("composite_score", 0.0))
    return score, metrics, frames_used
