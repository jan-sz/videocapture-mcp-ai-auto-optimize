"""Runtime discovery of camera properties."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PropertyDefinition:
    """Metadata about a candidate camera property."""

    name: str
    prop_id: int
    fallback_range: Tuple[float, float, float]
    high_impact: bool = False
    auto_prop_id: Optional[int] = None
    auto_on_value: Optional[float] = None
    auto_off_value: Optional[float] = None
    read_only: bool = False


# The discovery process still needs a *catalog* of candidate controls to try.
# PROPERTY_SPECS provides that seed list with OpenCV constant names, sensible
# fallback ranges, and auto-mode metadata when a paired toggle exists. During
# runtime, discovery translates these specs into `PropertyDefinition` entries
# and then probes each one on the connected device to learn the *actual* range,
# writability, and default values. The fallbacks only kick in when a camera
# refuses to report limits, ensuring we always have conservative bounds for the
# sampler instead of guessing from scratch.
PROPERTY_SPECS = (
    {
        "name": "exposure",
        "attr": "CAP_PROP_EXPOSURE",
        "fallback": (-9.0, 1.0, 0.5),
        "high_impact": True,
        "auto_attr": "CAP_PROP_AUTO_EXPOSURE",
        "auto_on": 0.75,
        "auto_off": 0.25,
    },
    {
        "name": "gain",
        "attr": "CAP_PROP_GAIN",
        "fallback": (0.0, 64.0, 1.0),
        "high_impact": True,
    },
    {
        "name": "brightness",
        "attr": "CAP_PROP_BRIGHTNESS",
        "fallback": (0.0, 255.0, 1.0),
    },
    {
        "name": "contrast",
        "attr": "CAP_PROP_CONTRAST",
        "fallback": (0.0, 255.0, 1.0),
    },
    {
        "name": "hue",
        "attr": "CAP_PROP_HUE",
        "fallback": (-180.0, 180.0, 1.0),
    },
    {
        "name": "saturation",
        "attr": "CAP_PROP_SATURATION",
        "fallback": (0.0, 255.0, 1.0),
    },
    {
        "name": "gamma",
        "attr": "CAP_PROP_GAMMA",
        "fallback": (1.0, 500.0, 1.0),
    },
    {
        "name": "sharpness",
        "attr": "CAP_PROP_SHARPNESS",
        "fallback": (0.0, 255.0, 1.0),
    },
    {
        "name": "white_balance",
        "attr": "CAP_PROP_WHITE_BALANCE_BLUE_U",
        "fallback": (2800.0, 7500.0, 50.0),
        "high_impact": True,
        "auto_attr": "CAP_PROP_AUTO_WB",
        "auto_on": 1.0,
        "auto_off": 0.0,
    },
    {
        "name": "temperature",
        "attr": "CAP_PROP_TEMPERATURE",
        "fallback": (2800.0, 7500.0, 50.0),
    },
    {
        "name": "focus",
        "attr": "CAP_PROP_FOCUS",
        "fallback": (0.0, 255.0, 1.0),
        "high_impact": True,
        "auto_attr": "CAP_PROP_AUTOFOCUS",
        "auto_on": 1.0,
        "auto_off": 0.0,
    },
    {
        "name": "zoom",
        "attr": "CAP_PROP_ZOOM",
        "fallback": (0.0, 400.0, 1.0),
    },
    {
        "name": "iris",
        "attr": "CAP_PROP_IRIS",
        "fallback": (0.0, 255.0, 1.0),
    },
    {
        "name": "pan",
        "attr": "CAP_PROP_PAN",
        "fallback": (-180.0, 180.0, 1.0),
    },
    {
        "name": "tilt",
        "attr": "CAP_PROP_TILT",
        "fallback": (-90.0, 90.0, 1.0),
    },
    {
        "name": "roll",
        "attr": "CAP_PROP_ROLL",
        "fallback": (-90.0, 90.0, 1.0),
    },
    {
        "name": "backlight",
        "attr": "CAP_PROP_BACKLIGHT",
        "fallback": (0.0, 1.0, 1.0),
    },
    {
        "name": "color_enable",
        "attr": "CAP_PROP_COLOR_ENABLE",
        "fallback": (0.0, 1.0, 1.0),
    },
    {
        "name": "power_line_frequency",
        "attr": "CAP_PROP_POWERLINE_FREQUENCY",
        "fallback": (0.0, 2.0, 1.0),
    },
    {
        "name": "low_light_compensation",
        "attr": "CAP_PROP_LOWLIGHT",
        "fallback": (0.0, 1.0, 1.0),
    },
)


def _resolve_prop_id(name: str) -> Optional[int]:
    value = getattr(cv2, name, None)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _build_property_definitions() -> Tuple[PropertyDefinition, ...]:
    definitions: List[PropertyDefinition] = []
    for spec in PROPERTY_SPECS:
        prop_id = _resolve_prop_id(spec["attr"])
        if prop_id is None:
            continue
        auto_prop_id = None
        if "auto_attr" in spec:
            auto_prop_id = _resolve_prop_id(spec["auto_attr"])
        definitions.append(
            PropertyDefinition(
                name=spec["name"],
                prop_id=prop_id,
                fallback_range=spec["fallback"],
                high_impact=spec.get("high_impact", False),
                auto_prop_id=auto_prop_id,
                auto_on_value=spec.get("auto_on"),
                auto_off_value=spec.get("auto_off"),
            )
        )
    return tuple(definitions)


PROPERTY_DEFINITIONS: Tuple[PropertyDefinition, ...] = _build_property_definitions()


@dataclass
class PropertyInfo:
    """Detailed capability information for a camera property."""

    definition: PropertyDefinition
    supported: bool
    readable: bool
    writable: bool
    min_value: float
    max_value: float
    step: float
    default_value: float
    current_value: float
    auto_supported: bool
    auto_enabled: bool
    auto_on_value: Optional[float] = None
    auto_off_value: Optional[float] = None

    def clamp(self, value: float) -> float:
        return float(min(self.max_value, max(self.min_value, value)))

    @property
    def name(self) -> str:
        return self.definition.name

    @property
    def prop_id(self) -> int:
        return self.definition.prop_id

    @property
    def high_impact(self) -> bool:
        return self.definition.high_impact


@dataclass
class PropertyState:
    """A snapshot of property values and auto modes."""

    values: Dict[str, float]
    auto_modes: Dict[str, bool] = field(default_factory=dict)

    def copy(self) -> "PropertyState":
        return PropertyState(values=dict(self.values), auto_modes=dict(self.auto_modes))


class PropertyManager:
    """Manages discovery and safe setting of camera properties."""

    def __init__(self, capture: cv2.VideoCapture, settle_delay: float = 0.05) -> None:
        self._capture = capture
        self._settle_delay = settle_delay
        self._properties = discover_properties(capture)

    @property
    def properties(self) -> Dict[str, PropertyInfo]:
        return self._properties

    def get_state(self) -> PropertyState:
        values: Dict[str, float] = {}
        auto_modes: Dict[str, bool] = {}
        for name, info in self._properties.items():
            values[name] = info.current_value
            if info.auto_supported:
                auto_modes[name] = info.auto_enabled
        return PropertyState(values=values, auto_modes=auto_modes)

    def apply_state(self, state: PropertyState) -> None:
        for name, value in state.values.items():
            if name not in self._properties:
                continue
            info = self._properties[name]
            if not info.writable:
                continue
            desired_auto = state.auto_modes.get(name, info.auto_enabled)
            self._set_auto_mode(info, desired_auto)
            if desired_auto:
                continue
            target = info.clamp(value)
            if not _try_set(self._capture, info.prop_id, target):
                LOGGER.debug("Failed to set %s to %s", name, target)
            else:
                actual = _safe_get(self._capture, info.prop_id)
                if actual is not None:
                    info.current_value = actual
        time.sleep(self._settle_delay)

    def _set_auto_mode(self, info: PropertyInfo, enabled: bool) -> None:
        if not info.auto_supported or info.definition.auto_prop_id is None:
            return
        target = info.auto_on_value if enabled else info.auto_off_value
        if target is None:
            return
        if _try_set(self._capture, info.definition.auto_prop_id, target):
            info.auto_enabled = enabled


def discover_properties(capture: cv2.VideoCapture) -> Dict[str, PropertyInfo]:
    """Detect supported camera properties via runtime probing."""

    discovered: Dict[str, PropertyInfo] = {}
    original_values: Dict[int, float] = {}
    original_autos: Dict[int, float] = {}

    for definition in PROPERTY_DEFINITIONS:
        if definition.read_only:
            continue
        prop_id = definition.prop_id
        original = _safe_get(capture, prop_id)
        readable = original is not None
        writable = False
        supported = False
        min_value, max_value, step = definition.fallback_range
        auto_supported = False
        auto_enabled = False
        auto_on_value = definition.auto_on_value
        auto_off_value = definition.auto_off_value

        if readable:
            original_values[prop_id] = original  # type: ignore[arg-type]

        if definition.auto_prop_id is not None:
            auto_current = _safe_get(capture, definition.auto_prop_id)
            if auto_current is not None:
                original_autos[definition.auto_prop_id] = auto_current
                auto_supported = True
                if auto_on_value is None:
                    auto_on_value = auto_current
                auto_enabled = _is_auto_enabled(auto_current, auto_on_value, auto_off_value)

        if auto_supported and auto_off_value is not None:
            _try_set(capture, definition.auto_prop_id, auto_off_value)
            auto_enabled = False

        if readable:
            probe_step = max(definition.fallback_range[2], abs(original) * 0.05)
            if not math.isfinite(probe_step) or probe_step == 0.0:
                probe_step = definition.fallback_range[2]
            sample_targets = _candidate_targets(original, min_value, max_value, probe_step)
            writable = False
            last_good = original
            for target in sample_targets:
                if _try_set(capture, prop_id, target):
                    actual = _safe_get(capture, prop_id)
                    if actual is None:
                        continue
                    if math.isclose(actual, last_good, rel_tol=1e-3, abs_tol=max(step, 1e-2)):
                        continue
                    writable = True
                    last_good = actual
                    min_value = min(min_value, actual)
                    max_value = max(max_value, actual)
                if writable:
                    break
            if writable:
                supported = True
                _try_set(capture, prop_id, original)

        if supported:
            min_value, max_value = _refine_range(capture, prop_id, original or 0.0, min_value, max_value, definition.fallback_range[2])
            current = _safe_get(capture, prop_id)
            if current is None:
                current = original if original is not None else definition.fallback_range[0]
            discovered[definition.name] = PropertyInfo(
                definition=definition,
                supported=True,
                readable=readable,
                writable=writable,
                min_value=min_value,
                max_value=max_value,
                step=max(definition.fallback_range[2], 1e-3),
                default_value=original if original is not None else current,
                current_value=current,
                auto_supported=auto_supported,
                auto_enabled=auto_enabled,
                auto_on_value=auto_on_value,
                auto_off_value=auto_off_value,
            )
        elif readable and definition.high_impact:
            LOGGER.debug("Property %s readable but not writable", definition.name)

    for prop_id, value in original_values.items():
        _try_set(capture, prop_id, value)
    for prop_id, value in original_autos.items():
        _try_set(capture, prop_id, value)

    return discovered


def _candidate_targets(original: float, min_value: float, max_value: float, step: float) -> List[float]:
    candidates: List[float] = []
    for delta in (-step, step):
        candidate = original + delta
        candidate = max(min_value, min(max_value, candidate))
        if not math.isclose(candidate, original, rel_tol=1e-3, abs_tol=step * 0.5):
            candidates.append(candidate)
    if not candidates:
        candidates.append(max(min_value, min(max_value, original)))
    return candidates


def _refine_range(
    capture: cv2.VideoCapture,
    prop_id: int,
    origin: float,
    fallback_min: float,
    fallback_max: float,
    step: float,
) -> Tuple[float, float]:
    min_value = fallback_min
    max_value = fallback_max
    current = origin
    for direction in (-1, 1):
        probe = current
        last_good = current
        limit = fallback_min if direction < 0 else fallback_max
        for _ in range(12):
            probe += direction * step
            probe = max(fallback_min, min(fallback_max, probe))
            if math.isclose(probe, last_good, rel_tol=1e-3, abs_tol=step * 0.5):
                break
            if not _try_set(capture, prop_id, probe):
                break
            actual = _safe_get(capture, prop_id)
            if actual is None:
                break
            if direction < 0 and actual > last_good:
                break
            if direction > 0 and actual < last_good:
                break
            last_good = actual
            if direction < 0:
                min_value = min(min_value, actual)
            else:
                max_value = max(max_value, actual)
            if math.isclose(actual, limit, rel_tol=1e-3, abs_tol=step * 0.5):
                break
        _try_set(capture, prop_id, current)
    return min_value, max_value


def _safe_get(capture: cv2.VideoCapture, prop_id: int) -> Optional[float]:
    value = capture.get(prop_id)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return float(value)


def _try_set(capture: cv2.VideoCapture, prop_id: int, value: float) -> bool:
    if not capture.set(prop_id, float(value)):
        return False
    actual = _safe_get(capture, prop_id)
    if actual is None:
        return False
    tolerance = max(1e-3, abs(value) * 0.05)
    return math.isclose(actual, value, rel_tol=0.2, abs_tol=tolerance)


def _is_auto_enabled(current: float, auto_on: Optional[float], auto_off: Optional[float]) -> bool:
    if auto_on is None and auto_off is None:
        return bool(current)
    if auto_on is not None and math.isclose(current, auto_on, abs_tol=0.05):
        return True
    if auto_off is not None and math.isclose(current, auto_off, abs_tol=0.05):
        return False
    return current > (auto_off or 0.0)
