import cv2
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Optional, Dict
from mcp.server.fastmcp import FastMCP, Image


@dataclass
class ConnectionEntry:
    """Track an open capture along with its originating device index."""

    capture: cv2.VideoCapture
    device_index: int


# Store active video capture objects keyed by a deterministic connection ID
active_captures: Dict[str, ConnectionEntry] = {}
# Track counters per device index so generated IDs stay stable and non-ambiguous
device_counters: Dict[int, int] = {}

# Define our application context
@dataclass
class AppContext:
    active_captures: Dict[str, ConnectionEntry]

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle with camera resource cleanup"""
    # Initialize on startup
    #print("Starting VideoCapture MCP Server")
    try:
        # Pass the active_captures dictionary in the context
        yield AppContext(active_captures=active_captures)
    finally:
        # Cleanup on shutdown
        #print("Shutting down VideoCapture MCP Server")
        for connection_id, cap in active_captures.items():
            cap.capture.release()
        active_captures.clear()
        device_counters.clear()

# Initialize the FastMCP server with lifespan
mcp = FastMCP("VideoCapture", 
              description="Provides access to camera and video streams via OpenCV",
              dependencies=["opencv-python", "numpy"],
              lifespan=app_lifespan)

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
    for key, entry in active_captures.items():
        if entry.device_index == device_index:
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
        next_index = device_counters.get(device_index, 0) + 1
        device_counters[device_index] = next_index
        name = f"camera_{device_index}_{next_index:02d}"
    elif name in active_captures:
        raise ValueError(f"Connection name already in use: {name}")

    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        raise ValueError(f"Failed to open camera at index {device_index}")

    active_captures[name] = ConnectionEntry(capture=cap, device_index=device_index)
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
    if connection_id not in active_captures:
        raise ValueError(f"No active connection with ID: {connection_id}")
    
    cap = active_captures[connection_id].capture
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
    if connection_id not in active_captures:
        raise ValueError(f"No active connection with ID: {connection_id}")
    
    cap = active_captures[connection_id].capture
    
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
    if connection_id not in active_captures:
        raise ValueError(f"No active connection with ID: {connection_id}")
    
    cap = active_captures[connection_id].capture
    
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
def list_cameras(max_devices: int = 10) -> list:
    """
    Probe connected cameras by index and report which ones can be opened.

    Args:
        max_devices: Highest index (exclusive) to probe. The function checks
            indices in the range [0, max_devices).

    Returns:
        A list of dictionaries describing each successfully opened camera.
        Keys include:
        - index: The probed numeric device index.
        - backend: Backend name reported by OpenCV (if available).
        - width/height: Current capture resolution reported by the driver.
    """

    def _backend_name(cap: cv2.VideoCapture) -> str:
        try:
            return cap.getBackendName()
        except AttributeError:
            return "unknown"

    discovered = []
    seen = set()

    for index in range(max_devices):
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            continue

        backend = _backend_name(cap)
        key = (backend, index)
        if key in seen:
            cap.release()
            continue

        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        discovered.append(
            {
                "index": index,
                "backend": backend,
                "width": width,
                "height": height,
            }
        )
        seen.add(key)
        cap.release()

    return discovered

@mcp.tool()
def close_connection(connection_id: str) -> bool:
    """
    Close a video connection and release resources.
    
    Args:
        connection_id: ID of the connection to close
    
    Returns:
        True if successful
    """
    if connection_id not in active_captures:
        raise ValueError(f"No active connection with ID: {connection_id}")
    
    active_captures[connection_id].capture.release()
    del active_captures[connection_id]
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
