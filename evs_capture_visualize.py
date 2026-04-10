"""
Event Camera (EVS) Frame Capture and Live Visualization
--------------------------------------------------------
Connects to a LUCID TRT009S-E EVS camera, captures event frames, and displays
them live using OpenCV. Based on py_evs_xytp_frame_heatmap.py from the Arena SDK
examples, extended with live toggling, hardware noise filtering, and a larger display.

Five visualization modes (cycle with 'm'):
  - CDFrame        : accumulated event frames as a grayscale image
  - XYTPFrame      : per-event heatmap coloured by polarity (blue=ON, red=OFF)
  - DV-Events      : dv-processing EventVisualizer (colour-coded ON/OFF events)
  - DV-Accumulator : dv-processing accumulated potential surface (grayscale)
  - DV-TimeSurface : dv-processing time-surface with JET colourmap (recent=warm)

Noise reduction (three layers):
  1. Hardware: BiasEventThreshold, BiasRefractoryPeriod, EventBurstFilter via camera nodes
  2. Software (legacy): morphological opening/dilation for CDFrame/XYTPFrame modes
  3. Software (dv-processing): BackgroundActivityNoiseFilter for DV-* modes (toggle 'n')

Controls:
  q / ESC : quit
  s       : save current frame as PNG to evs_captures/
  m       : cycle through visualization modes
  n       : toggle dv-processing noise filter (DV-* modes)

Requirements:
  pip install arena_api-2.8.4-py3-none-any.whl numpy opencv-python dv-processing
  Run via ./run_evs.sh (sets LD_LIBRARY_PATH for ArenaSDK)
"""

import ctypes
import time
import os
from datetime import timedelta
import numpy as np
import cv2
import dv_processing as dv

from arena_api.system import system
from arena_api.buffer import BufferFactory
from arena_api.enums import PixelFormat

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CAMERA_IP       = '169.254.80.199'
NUM_BUFFERS     = 10
IMAGE_TIMEOUT_MS = 2000
ERC_RATE_LIMIT_MEV = 10.0   # max event rate sent by camera (Mev/s)
SAVE_DIR        = "evs_captures"
WINDOW_NAME     = "EVS Camera  —  TRT009S-E"
DISPLAY_SCALE   = 2.0       # upscale factor for the display window

# Hardware noise-filter values (0 = off, increase to reject weaker events)
BIAS_THRESHOLD_POS  = 10    # minimum contrast for a positive (ON) event
BIAS_THRESHOLD_NEG  = 10    # minimum contrast for a negative (OFF) event
BIAS_REFRACTORY     = 10    # refractory period – prevents rapid pixel re-firing
BURST_FILTER_ENABLE = True  # discard burst-noise events at the sensor

# CDFrame: morphological opening removes isolated hot pixels
DENOISE_KERNEL  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# XYTPFrame: dilate each event point into a small dot so sparse events are visible
DILATE_KERNEL   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))

MODE_CDFRAME      = "CDFrame"
MODE_XYTPFRAME    = "XYTPFrame"
MODE_DV_EVENTS    = "DV-Events"
MODE_DV_ACCUMULATOR = "DV-Accumulator"
MODE_DV_TIMESURFACE = "DV-TimeSurface"

ALL_MODES = [MODE_CDFRAME, MODE_XYTPFRAME, MODE_DV_EVENTS, MODE_DV_ACCUMULATOR, MODE_DV_TIMESURFACE]

# Modes that require XYTPFrame data from the camera (per-event x,y,t,p)
XYTP_MODES = {MODE_XYTPFRAME, MODE_DV_EVENTS, MODE_DV_ACCUMULATOR, MODE_DV_TIMESURFACE}

# dv-processing noise filter duration (microseconds) — events without
# neighbours within this window are considered background noise
DV_NOISE_DURATION_US = 2000

# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def connect_device(max_tries: int = 6, wait_secs: int = 10):
    """Wait up to max_tries * wait_secs seconds for a device to appear."""
    system.DEVICE_INFOS_TIMEOUT_MILLISEC = 1000
    system.add_unicast_discovery_device(CAMERA_IP)
    for attempt in range(1, max_tries + 1):
        devices = system.create_device()
        if devices:
            print(f"[+] Found {len(devices)} device(s)")
            return devices
        print(f"  Try {attempt}/{max_tries}: no device yet, waiting {wait_secs}s …")
        time.sleep(wait_secs)
    raise RuntimeError("No device found. Connect the EVS camera and retry.")


def configure_noise_filters(device) -> dict:
    """
    Apply hardware noise-reduction settings.
    Returns a dict of initial values for restoration on exit.
    """
    nm = device.nodemap
    saved = {}
    settings = {
        "BiasEventThresholdPositive": BIAS_THRESHOLD_POS,
        "BiasEventThresholdNegative": BIAS_THRESHOLD_NEG,
        "BiasRefractoryPeriod":       BIAS_REFRACTORY,
        "EventBurstFilterEnable":     BURST_FILTER_ENABLE,
    }
    for node_name, new_val in settings.items():
        try:
            node = nm[node_name]
            saved[node_name] = node.value
            node.value = new_val
            print(f"  {node_name}: {saved[node_name]} → {new_val}")
        except Exception as e:
            print(f"  [!] Could not set {node_name}: {e}")
    return saved


def configure_evs(device, mode: str) -> dict:
    """
    Apply EVS stream settings for the chosen output mode.
    Returns a dict of initial values for restoration on exit.
    """
    nm = device.nodemap
    tl = device.tl_stream_nodemap

    saved = {
        "AcquisitionMode": nm["AcquisitionMode"].value,
        "EventFormat":     nm["EventFormat"].value,
        "ErcEnable":       nm["ErcEnable"].value,
        "ErcRateLimit":    nm["ErcRateLimit"].value,
    }

    nm["AcquisitionMode"].value = "Continuous"
    tl["StreamBufferHandlingMode"].value = "NewestOnly"
    nm["EventFormat"].value = "EVT3_0"
    nm["ErcEnable"].value = True
    nm["ErcRateLimit"].value = ERC_RATE_LIMIT_MEV
    tl["StreamEvsOutputFormat"].value = mode

    if mode == MODE_CDFRAME:
        fps = tl["StreamFrameGeneratorFPS"].value
        accum_us = int(1_000_000 / fps)
        tl["StreamFrameGeneratorAccumTime"].value = accum_us
        print(f"[+] CDFrame  — FPS={fps:.1f}, accum={accum_us} µs")
    else:
        print("[+] XYTPFrame — timestamp heatmap")

    return saved


def restore_settings(device, *saved_dicts):
    nm = device.nodemap
    for saved in saved_dicts:
        for key, val in saved.items():
            try:
                nm[key].value = val
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def cdframe_to_bgr(buffer) -> np.ndarray:
    """
    Wrap a CDFrame buffer as a BGR NumPy array.
    CDFrames contain accumulated events as a standard grey/colour image.
    """
    bpp      = buffer.bits_per_pixel
    channels = max(1, bpp // 8)
    raw      = (ctypes.c_ubyte * buffer.size_filled).from_address(
                    ctypes.addressof(buffer.pbytes))
    arr = np.frombuffer(raw, dtype=np.uint8)

    if channels == 1:
        img = arr.reshape((buffer.height, buffer.width))
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif channels == 3:
        img = arr.reshape((buffer.height, buffer.width, 3))
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img = arr[:buffer.height * buffer.width].reshape((buffer.height, buffer.width))
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return bgr


def xytp_to_heatmap(buffer) -> np.ndarray:
    """
    Convert an XYTPFrame buffer to a polarity image.

    Each event carries (x, y, t, p) as float32:
      x, y  — pixel coordinates
      t     — timestamp (microseconds)
      p     — polarity: 1 = ON (brightness increased), 0 = OFF (brightness decreased)

    Colour encoding (standard event-camera convention):
      Blue  — ON  events: a bright edge just entered this pixel
      Red   — OFF events: a bright edge just left this pixel
      Black — no event this frame

    The "doubling" effect seen on moving objects is real: the leading edge of a
    moving bright object fires ON events (blue) and the trailing edge fires OFF
    events (red), producing a blue-front / red-back signature.

    Based on py_evs_xytp_frame_heatmap.py from the Arena SDK examples.
    """
    STEP = 4   # x, y, t, p — each float32 (LUCID_LucidXYTP128f = 128 bits/event)
    width, height   = buffer.width, buffer.height
    bytes_per_event = buffer.bits_per_pixel / 8   # 128 bits → 16 bytes
    num_events      = int(buffer.size_filled / bytes_per_event)

    if num_events == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)

    # ctypes.string_at reads exactly size_filled bytes from the pointer,
    # then frombuffer reinterprets those bytes as float32 — matches the
    # per-element access used in the SDK's py_evs_xytp_frame_heatmap.py example.
    raw    = ctypes.string_at(buffer.pdata, buffer.size_filled)
    events = np.frombuffer(raw, dtype=np.float32).reshape(num_events, STEP)

    xs = events[:, 0].astype(np.int32).clip(0, width  - 1)
    ys = events[:, 1].astype(np.int32).clip(0, height - 1)
    ps = events[:, 3].astype(np.int32)   # polarity: 0 or 1

    bgr = np.zeros((height, width, 3), dtype=np.uint8)

    # ON events  (p=1) → blue  (BGR: 255, 80, 0)
    on_mask = ps == 1
    bgr[ys[on_mask], xs[on_mask]] = (255, 80, 0)

    # OFF events (p=0) → red   (BGR: 0, 80, 255)
    off_mask = ~on_mask
    bgr[ys[off_mask], xs[off_mask]] = (0, 80, 255)

    return bgr


def buffer_to_eventstore(buffer) -> dv.EventStore:
    """
    Convert an XYTPFrame buffer into a dv.EventStore so we can use
    dv-processing algorithms (noise filters, visualizer, accumulator, etc.).
    """
    STEP = 4
    bytes_per_event = buffer.bits_per_pixel / 8
    num_events = int(buffer.size_filled / bytes_per_event)

    store = dv.EventStore()
    if num_events == 0:
        return store

    raw = ctypes.string_at(buffer.pdata, buffer.size_filled)
    events = np.frombuffer(raw, dtype=np.float32).reshape(num_events, STEP)

    width, height = buffer.width, buffer.height
    xs = events[:, 0].astype(np.int32).clip(0, width - 1)
    ys = events[:, 1].astype(np.int32).clip(0, height - 1)
    ts = events[:, 2]  # timestamps in microseconds (float32)
    ps = events[:, 3].astype(np.int32)

    # dv.EventStore.push_back expects (timestamp_us: int, x: int, y: int, polarity: bool)
    for i in range(num_events):
        store.push_back(int(ts[i]), int(xs[i]), int(ys[i]), bool(ps[i]))

    return store


def dv_filter_noise(store: dv.EventStore, resolution: tuple,
                    noise_filter: dv.noise.BackgroundActivityNoiseFilter) -> dv.EventStore:
    """Run dv-processing background activity noise filter on an EventStore."""
    noise_filter.accept(store)
    return noise_filter.generateEvents()


def dv_events_to_bgr(store: dv.EventStore, visualizer) -> np.ndarray:
    """Render events using dv-processing EventVisualizer (blue=ON, red=OFF)."""
    return visualizer.generateImage(store)


def dv_accumulator_to_bgr(store: dv.EventStore, accumulator) -> np.ndarray:
    """Render events as an accumulated potential surface (grayscale)."""
    accumulator.accept(store)
    frame = accumulator.generateFrame()
    return cv2.cvtColor(frame.image, cv2.COLOR_GRAY2BGR)


def dv_timesurface_to_bgr(store: dv.EventStore, timesurface) -> np.ndarray:
    """Render events as a time-surface (recent events brighter)."""
    timesurface.accept(store)
    frame = timesurface.generateFrame()
    return cv2.applyColorMap(frame.image, cv2.COLORMAP_JET)


def denoise(bgr: np.ndarray, mode: str) -> np.ndarray:
    """
    Mode-aware post-processing:
    - CDFrame  : morphological opening removes isolated hot pixels from the
                 accumulated image without destroying edges.
    - XYTPFrame: dilation grows each sparse event point into a small visible
                 dot (MORPH_OPEN would erase them entirely).
    """
    if mode == MODE_CDFRAME:
        return cv2.morphologyEx(bgr, cv2.MORPH_OPEN, DENOISE_KERNEL)
    else:
        return cv2.dilate(bgr, DILATE_KERNEL)


def draw_overlay(img: np.ndarray, info: dict) -> None:
    mode = info.get('mode', '')
    lines = [
        f"Mode : {mode}",
        f"Frame: {info.get('frame_id')}",
        f"Rate : {info.get('event_rate')}",
        f"FPS  : {info.get('gvsp_fps')}",
        f"BW   : {info.get('throughput')}",
        f"Render {info.get('render_ms', 0):.1f} ms",
        "[m] mode  [n] noise  [s] save  [q] quit",
    ]
    y = 30
    for line in lines:
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 255, 0), 1, cv2.LINE_AA)
        y += 26

    # Colour legend for XYTPFrame
    if mode == MODE_XYTPFRAME:
        h = img.shape[0]
        cv2.putText(img, "Blue = ON  (leading edge / brightness increase)",
                    (10, h - 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 80, 0), 1, cv2.LINE_AA)
        cv2.putText(img, "Red  = OFF (trailing edge / brightness decrease)",
                    (10, h - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 80, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "Patterns follow scene edges — diagonal edges appear diagonal",
                    (10, h - 8),  cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)


def fmt_rate(r: float) -> str:
    if r < 1e3: return f"{r:.0f} ev/s"
    if r < 1e6: return f"{r/1e3:.1f} Kev/s"
    if r < 1e9: return f"{r/1e6:.1f} Mev/s"
    return f"{r/1e9:.1f} Gev/s"

def fmt_bw(r: float) -> str:
    if r < 1e3: return f"{r:.0f} Bps"
    if r < 1e6: return f"{r/1e3:.1f} KBps"
    if r < 1e9: return f"{r/1e6:.1f} MBps"
    return f"{r/1e9:.1f} GBps"


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run():
    os.makedirs(SAVE_DIR, exist_ok=True)

    devices = connect_device()
    device  = system.select_device(devices)

    nm = device.nodemap
    tl = device.tl_stream_nodemap
    width  = nm["Width"].value
    height = nm["Height"].value
    print(f"[+] Sensor: {width} x {height}")

    mode        = MODE_CDFRAME      # CDFrame shows content even without motion
    saved_evs   = {}
    saved_noise = {}
    dv_noise_on = True             # software noise filtering via dv-processing

    disp_w = int(width  * DISPLAY_SCALE)
    disp_h = int(height * DISPLAY_SCALE)

    # dv-processing objects (initialised once, reused every frame)
    resolution = (width, height)
    dv_visualizer   = dv.visualization.EventVisualizer(resolution)
    dv_accumulator  = dv.Accumulator(resolution)
    dv_timesurface  = dv.TimeSurface(resolution)
    dv_noise_filter = dv.noise.BackgroundActivityNoiseFilter(
        resolution, backgroundActivityDuration=timedelta(microseconds=DV_NOISE_DURATION_US))

    def camera_mode_for(m):
        """Return the camera-level output format needed for a display mode."""
        return MODE_XYTPFRAME if m in XYTP_MODES else MODE_CDFRAME

    try:
        # EVT3_0 must be active before Bias nodes become writable
        saved_evs = configure_evs(device, camera_mode_for(mode))
        print("[+] Applying hardware noise filters:")
        saved_noise = configure_noise_filters(device)
        device.start_stream(NUM_BUFFERS)
        print(f"[+] Stream started  ({disp_w}×{disp_h} display).")
        print("    Controls: [m] cycle mode  [n] toggle noise filter  [s] save  [q]/ESC quit\n")

        save_idx = 0
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, disp_w, disp_h)

        while True:
            try:
                buffer = device.get_buffer(timeout=IMAGE_TIMEOUT_MS)
            except Exception as e:
                print(f"[!] get_buffer: {e}")
                continue

            if buffer.is_incomplete:
                print(f"[!] Incomplete frame {buffer.frame_id}, skipping")
                device.requeue_buffer(buffer)
                continue

            t0 = time.perf_counter()

            if mode == MODE_CDFRAME:
                bgr = cdframe_to_bgr(buffer)
                bgr = denoise(bgr, mode)
            elif mode == MODE_XYTPFRAME:
                bgr = xytp_to_heatmap(buffer)
                bgr = denoise(bgr, mode)
            else:
                # dv-processing modes: convert buffer → EventStore
                store = buffer_to_eventstore(buffer)
                if dv_noise_on and store.size() > 0:
                    store = dv_filter_noise(store, resolution, dv_noise_filter)
                if mode == MODE_DV_EVENTS:
                    bgr = dv_events_to_bgr(store, dv_visualizer)
                elif mode == MODE_DV_ACCUMULATOR:
                    bgr = dv_accumulator_to_bgr(store, dv_accumulator)
                elif mode == MODE_DV_TIMESURFACE:
                    bgr = dv_timesurface_to_bgr(store, dv_timesurface)

            render_ms = (time.perf_counter() - t0) * 1000

            info = {
                "mode":       mode + (" [NF]" if dv_noise_on and mode in XYTP_MODES else ""),
                "frame_id":   buffer.frame_id,
                "event_rate": fmt_rate(tl["StreamEvsEventRate"].value),
                "gvsp_fps":   f"{tl['StreamEvsGvspFrameRate'].value:.1f} fps",
                "throughput": fmt_bw(tl["StreamEvsLinkThroughput"].value),
                "render_ms":  render_ms,
            }

            # Upscale for display
            display = cv2.resize(bgr, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
            draw_overlay(display, info)
            cv2.imshow(WINDOW_NAME, display)
            device.requeue_buffer(buffer)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('s'):
                path = os.path.join(SAVE_DIR, f"frame_{save_idx:04d}_{mode}.png")
                cv2.imwrite(path, bgr)
                print(f"[+] Saved {path}")
                save_idx += 1
            elif key == ord('m'):
                # Cycle to next mode
                old_mode = mode
                idx = ALL_MODES.index(mode)
                mode = ALL_MODES[(idx + 1) % len(ALL_MODES)]
                # Only reconfigure the camera if the underlying format changed
                old_cam = camera_mode_for(old_mode)
                new_cam = camera_mode_for(mode)
                if old_cam != new_cam:
                    device.stop_stream()
                    restore_settings(device, saved_evs, saved_noise)
                    saved_evs = configure_evs(device, new_cam)
                    saved_noise = configure_noise_filters(device)
                    device.start_stream(NUM_BUFFERS)
                print(f"[+] Switched to {mode}")
            elif key == ord('n'):
                dv_noise_on = not dv_noise_on
                print(f"[+] dv-processing noise filter: {'ON' if dv_noise_on else 'OFF'}")

    finally:
        try:
            device.stop_stream()
        except Exception:
            pass
        restore_settings(device, saved_evs, saved_noise)
        cv2.destroyAllWindows()
        system.destroy_device()
        print("[+] Cleaned up. Done.")


if __name__ == "__main__":
    run()
