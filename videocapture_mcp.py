import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple

import cv2
from mcp.server.fastmcp import FastMCP, Image

from autotune import (
    OptimizationResult,
    PropertyInfo,
    PropertyManager,
    PropertyState,
    SamplingConfig,
    discover_properties,
    optimize_capture,
)

LOGGER = logging.getLogger(__name__)

# Store active video capture objects
active_captures: Dict[str, cv2.VideoCapture] = {}
baseline_cache: Dict[str, PropertyState] = {}
optimized_candidates: Dict[str, PropertyState] = {}
applied_settings: Dict[str, PropertyState] = {}

MAX_COARSE_SAMPLES = 48
MAX_FRAME_CAP = 360
MAX_SECONDS_CAP = 30.0
MAX_WARMUP_FRAMES = 12

# Define our application context
@dataclass
class AppContext:
    active_captures: Dict[str, cv2.VideoCapture]
    baseline_cache: Dict[str, PropertyState]
    optimized_candidates: Dict[str, PropertyState]
    applied_settings: Dict[str, PropertyState]

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle with camera resource cleanup"""
    # Initialize on startup
    #print("Starting VideoCapture MCP Server")
    try:
        # Pass the shared caches in the context
        yield AppContext(
            active_captures=active_captures,
            baseline_cache=baseline_cache,
            optimized_candidates=optimized_candidates,
            applied_settings=applied_settings,
        )
    finally:
        # Cleanup on shutdown
        #print("Shutting down VideoCapture MCP Server")
        for connection_id, cap in active_captures.items():
            cap.release()
        active_captures.clear()
        baseline_cache.clear()
        optimized_candidates.clear()
        applied_settings.clear()

# Initialize the FastMCP server with lifespan
mcp = FastMCP("VideoCapture",
              description="Provides access to camera and video streams via OpenCV",
              dependencies=["opencv-python", "numpy"],
              lifespan=app_lifespan)


def _require_capture(connection_id: str) -> cv2.VideoCapture:
    if connection_id not in active_captures:
        raise ValueError(f"No active connection with ID: {connection_id}")
    return active_captures[connection_id]


def _state_to_payload(state: PropertyState) -> Dict[str, Dict[str, float]]:
    return {
        "values": dict(state.values),
        "auto_modes": dict(state.auto_modes),
    }


def _filter_supported(manager: PropertyManager, state: PropertyState) -> Tuple[PropertyState, Tuple[str, ...]]:
    values: Dict[str, float] = {}
    auto_modes: Dict[str, bool] = {}
    skipped = []
    for name, value in state.values.items():
        info = manager.properties.get(name)
        if info is None or not info.writable:
            skipped.append(name)
            continue
        values[name] = info.clamp(value)
    for name, enabled in state.auto_modes.items():
        info = manager.properties.get(name)
        if info is None or not info.auto_supported:
            skipped.append(name)
            continue
        auto_modes[name] = bool(enabled)
    return PropertyState(values=values, auto_modes=auto_modes), tuple(sorted(set(skipped)))


def _record_applied_state(connection_id: str, manager: PropertyManager) -> PropertyState:
    state = manager.get_state()
    applied_settings[connection_id] = state
    return state


def _property_info_to_payload(info: PropertyInfo) -> Dict[str, object]:
    return {
        "name": info.name,
        "supported": info.supported,
        "readable": info.readable,
        "writable": info.writable,
        "min_value": info.min_value,
        "max_value": info.max_value,
        "step": info.step,
        "default_value": info.default_value,
        "current_value": info.current_value,
        "auto_supported": info.auto_supported,
        "auto_enabled": info.auto_enabled,
        "auto_on_value": info.auto_on_value,
        "auto_off_value": info.auto_off_value,
    }


def main():
    """Main entry point for the VideoCapture Server"""

    mcp.run()
    
@mcp.tool()
def quick_capture(device_index: int = 0, flip: bool = False) -> Image:
    """
    Quickly open a camera, capture a single frame, and close it.
    If the camera is already open, use the existing connection.
    
    Args:
        device_index: Camera index (0 is usually the default webcam)
        flip: Whether to horizontally flip the image
    
    Returns:
        The captured frame as an Image object
    """
    # Check if this device is already open
    device_key = None
    for key, cap in active_captures.items():
        if key.startswith(f"camera_{device_index}_"):
            device_key = key
            break
    
    # If device is not already open, open it temporarily
    temp_connection = False
    if device_key is None:
        device_key = open_camera(device_index)
        temp_connection = True
    
    try:
        # Capture the frame
        frame = capture_frame(device_key, flip)
        return frame
    finally:
        # Close the connection if we opened it temporarily
        if temp_connection:
            close_connection(device_key)

@mcp.tool()
def open_camera(device_index: int = 0, name: Optional[str] = None) -> str:
    """
    Open a connection to a camera device.
    
    Args:
        device_index: Camera index (0 is usually the default webcam)
        name: Optional name to identify this camera connection
    
    Returns:
        Connection ID for the opened camera
    """
    if name is None:
        name = f"camera_{device_index}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        raise ValueError(f"Failed to open camera at index {device_index}")
    
    active_captures[name] = cap
    return name

@mcp.tool()
def capture_frame(connection_id: str, flip: bool = False) -> Image:
    """
    Capture a single frame from the specified video source.
    
    Args:
        connection_id: ID of the previously opened video connection
        flip: Whether to horizontally flip the image
    
    Returns:
        The captured frame as an Image object
    """
    cap = _require_capture(connection_id)
    ret, frame = cap.read()
    
    if not ret:
        raise RuntimeError(f"Failed to capture frame from {connection_id}")
    
    if flip:
        frame = cv2.flip(frame, 1)  # 1 for horizontal flip
    
    
    # Encode the image as PNG
    _, png_data = cv2.imencode('.png', frame)
    
    # Return as MCP Image object
    return Image(data=png_data.tobytes(), 
                 format="png")

@mcp.tool()
def get_video_properties(connection_id: str) -> dict:
    """
    Get properties of the video source.
    
    Args:
        connection_id: ID of the previously opened video connection
    
    Returns:
        Dictionary of video properties
    """
    cap = _require_capture(connection_id)
    
    properties = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "brightness": cap.get(cv2.CAP_PROP_BRIGHTNESS),
        "contrast": cap.get(cv2.CAP_PROP_CONTRAST),
        "saturation": cap.get(cv2.CAP_PROP_SATURATION),
        "format": int(cap.get(cv2.CAP_PROP_FORMAT))
    }
    
    return properties


@mcp.tool()
def list_supported_properties(connection_id: str) -> dict:
    """Enumerate properties supported by the active capture."""

    cap = _require_capture(connection_id)
    discovered = discover_properties(cap)
    payload = [_property_info_to_payload(info) for info in discovered.values()]
    return {"connection_id": connection_id, "properties": payload}

@mcp.tool()
def set_video_property(connection_id: str, property_name: str, value: float) -> bool:
    """
    Set a property of the video source.
    
    Args:
        connection_id: ID of the previously opened video connection
        property_name: Name of the property to set (width, height, brightness, etc.)
        value: Value to set
    
    Returns:
        True if successful, False otherwise
    """
    cap = _require_capture(connection_id)
    
    property_map = {
        "width": cv2.CAP_PROP_FRAME_WIDTH,
        "height": cv2.CAP_PROP_FRAME_HEIGHT,
        "fps": cv2.CAP_PROP_FPS,
        "brightness": cv2.CAP_PROP_BRIGHTNESS,
        "contrast": cv2.CAP_PROP_CONTRAST,
        "saturation": cv2.CAP_PROP_SATURATION,
        "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
        "auto_focus": cv2.CAP_PROP_AUTOFOCUS
    }
    
    if property_name not in property_map:
        raise ValueError(f"Unknown property: {property_name}")
    
    return cap.set(property_map[property_name], value)


@mcp.tool()
def auto_optimize(
    connection_id: str,
    coarse_sample_count: int = 24,
    max_frames: int = 180,
    max_seconds: float = 12.0,
    warmup_frames: int = 6,
) -> dict:
    """Run camera auto-tuning and cache the resulting states."""

    cap = _require_capture(connection_id)
    manager = PropertyManager(cap)
    baseline_state = manager.get_state()
    baseline_cache[connection_id] = baseline_state.copy()
    high_impact_count = sum(1 for info in manager.properties.values() if info.high_impact and info.writable)

    coarse = max(0, min(coarse_sample_count, MAX_COARSE_SAMPLES))
    frame_cap = max(1, min(max_frames, MAX_FRAME_CAP))
    seconds_cap = max(1.0, min(max_seconds, MAX_SECONDS_CAP))
    warmup = max(0, min(warmup_frames, MAX_WARMUP_FRAMES))

    sampler_config = SamplingConfig(
        settle_delay=0.2,
        coarse_sample_count=coarse,
        refine_iterations=2,
        refine_step_shrink=0.5,
        safe_window_ratio=0.1,
        max_frames=frame_cap,
        max_seconds=seconds_cap,
    )

    start_time = time.monotonic()
    result: OptimizationResult
    try:
        result = optimize_capture(cap, sampler_config=sampler_config, warmup_frames=warmup)
    finally:
        # Always return the capture to its baseline configuration
        manager.apply_state(baseline_state)

    elapsed = time.monotonic() - start_time
    baseline_cache[connection_id] = result.baseline_state.copy()

    best_candidate = result.sampling.best
    best_payload = None
    if best_candidate and best_candidate.score > result.baseline_score:
        optimized_candidates[connection_id] = best_candidate.state.copy()
        best_payload = {
            "score": best_candidate.score,
            "metrics": dict(best_candidate.metrics),
            "frames_used": best_candidate.frames_used,
            "phase": best_candidate.phase,
            "state": _state_to_payload(best_candidate.state),
        }
    else:
        optimized_candidates.pop(connection_id, None)

    history_payload = [
        {
            "phase": candidate.phase,
            "score": candidate.score,
            "frames_used": candidate.frames_used,
            "metrics": dict(candidate.metrics),
            "state": _state_to_payload(candidate.state),
        }
        for candidate in result.sampling.history
    ]

    frames_used = sum(candidate["frames_used"] for candidate in history_payload)
    coarse_evaluated = sum(1 for candidate in history_payload if candidate["phase"].startswith("coarse"))
    expected_coarse = 1 if high_impact_count == 0 else sampler_config.coarse_sample_count + 1

    reason = "complete"
    if frames_used >= sampler_config.max_frames:
        reason = "frame_cap"
    elif elapsed >= sampler_config.max_seconds:
        reason = "time_cap"
    elif high_impact_count > 0 and coarse_evaluated < expected_coarse:
        reason = "sample_cap"

    if reason != "complete":
        LOGGER.info(
            "autotune.exit reason=%s connection=%s frames=%d samples=%d",
            reason,
            connection_id,
            frames_used,
            coarse_evaluated,
        )

    return {
        "connection_id": connection_id,
        "baseline": {
            "score": result.baseline_score,
            "state": _state_to_payload(result.baseline_state),
        },
        "best": best_payload,
        "history": history_payload,
        "caps": {
            "coarse_sample_count": sampler_config.coarse_sample_count,
            "max_frames": sampler_config.max_frames,
            "max_seconds": sampler_config.max_seconds,
            "warmup_frames": warmup,
        },
        "frames_used": frames_used,
        "elapsed_seconds": elapsed,
        "reason": reason,
    }


@mcp.tool()
def apply_settings(connection_id: str) -> dict:
    """Apply the most recent optimized settings to the capture."""

    candidate = optimized_candidates.get(connection_id)
    if candidate is None:
        raise ValueError("No optimized settings cached for this connection")

    cap = _require_capture(connection_id)
    manager = PropertyManager(cap)
    filtered_state, skipped = _filter_supported(manager, candidate)
    baseline_state = baseline_cache.get(connection_id)
    baseline_filtered: Optional[PropertyState] = None
    if baseline_state is not None:
        baseline_filtered, _ = _filter_supported(manager, baseline_state)

    try:
        manager.apply_state(filtered_state)
    except Exception as exc:
        if baseline_filtered is not None:
            manager.apply_state(baseline_filtered)
        raise exc

    optimized_candidates[connection_id] = filtered_state
    applied_state = _record_applied_state(connection_id, manager)

    return {
        "connection_id": connection_id,
        "applied_state": _state_to_payload(applied_state),
        "skipped": list(skipped),
    }


@mcp.tool()
def revert_settings(connection_id: str) -> dict:
    """Revert the capture to the cached baseline settings."""

    baseline_state = baseline_cache.get(connection_id)
    if baseline_state is None:
        raise ValueError("No baseline settings cached for this connection")

    cap = _require_capture(connection_id)
    manager = PropertyManager(cap)
    filtered_state, skipped = _filter_supported(manager, baseline_state)
    last_applied = applied_settings.get(connection_id)

    try:
        manager.apply_state(filtered_state)
    except Exception as exc:
        if last_applied is not None:
            manager.apply_state(last_applied)
        raise exc

    applied_state = _record_applied_state(connection_id, manager)

    return {
        "connection_id": connection_id,
        "applied_state": _state_to_payload(applied_state),
        "skipped": list(skipped),
    }

@mcp.tool()
def close_connection(connection_id: str) -> bool:
    """
    Close a video connection and release resources.
    
    Args:
        connection_id: ID of the connection to close
    
    Returns:
        True if successful
    """
    cap = _require_capture(connection_id)
    cap.release()
    del active_captures[connection_id]
    baseline_cache.pop(connection_id, None)
    optimized_candidates.pop(connection_id, None)
    applied_settings.pop(connection_id, None)
    return True

@mcp.tool()
def list_active_connections() -> list:
    """
    List all active video connections.
    
    Returns:
        List of active connection IDs
    """
    return list(active_captures.keys())

    mcp.run(transport='stdio')

# For: $ mcp run videocapture_mcp.py
def run():
    main()

if __name__ == "__main__":
    main()
