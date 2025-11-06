"""Deterministic, center-weighted image quality metrics."""

from __future__ import annotations

from typing import Dict

import cv2
import numpy as np


def compute_metrics(frame) -> Dict[str, float]:
    """Compute deterministic metrics and a composite score for an image frame."""

    metrics: Dict[str, float] = {
        "bad_frame": 1.0,
        "composite_score": 0.0,
    }
    if frame is None or getattr(frame, "size", 0) == 0:
        return metrics

    frame = np.asarray(frame)
    if frame.ndim == 2:
        gray = frame
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32)
    weight = _center_weight(gray.shape)

    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap_mean = float(np.average(lap, weights=weight))
    lap_var = float(np.average((lap - lap_mean) ** 2, weights=weight))

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    hist_sum = float(hist.sum()) or 1.0
    hist /= hist_sum
    shadow = float(hist[:5].sum())
    highlight = float(hist[-5:].sum())
    midtone = float(hist[96:160].sum())
    exposure_penalty = min(1.0, shadow * 4.0 + highlight * 4.0 + max(0.0, 0.25 - midtone) * 2.0)
    clip_penalty = min(1.0, shadow + highlight)

    if frame.ndim == 3 and frame.shape[2] >= 3:
        channels = frame.astype(np.float32)
        channel_means = channels.reshape(-1, channels.shape[2]).mean(axis=0)
        mean_all = float(channel_means.mean()) or 1.0
        gray_world_deviation = float(np.sqrt(np.mean(((channel_means - mean_all) / mean_all) ** 2)))
    else:
        gray_world_deviation = 0.0

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    noise_proxy = float(np.average(np.abs(gray - blur), weights=weight))

    metrics.update(
        {
            "laplacian_variance": lap_var,
            "exposure_penalty": exposure_penalty,
            "clip_penalty": clip_penalty,
            "gray_world_deviation": gray_world_deviation,
            "noise_proxy": noise_proxy,
        }
    )

    sharp_component = min(1.0, lap_var / 80.0)
    exposure_component = max(0.0, 1.0 - exposure_penalty)
    color_component = max(0.0, 1.0 - min(1.0, gray_world_deviation / 0.25))
    noise_component = max(0.0, 1.0 - min(1.0, noise_proxy / 15.0))

    composite = 100.0 * (
        0.45 * sharp_component
        + 0.2 * exposure_component
        + 0.2 * color_component
        + 0.15 * noise_component
    )
    composite = max(0.0, min(100.0, composite))
    bad_frame = bool(
        lap_var < 5.0
        or clip_penalty > 0.85
        or np.isnan(composite)
        or gray.mean() < 8.0
    )

    metrics["composite_score"] = 0.0 if bad_frame else composite
    metrics["bad_frame"] = 1.0 if bad_frame else 0.0
    return metrics


def _center_weight(shape) -> np.ndarray:
    height, width = shape
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    sigma = 0.6
    weight = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    weight /= weight.sum()
    return weight.astype(np.float32)
