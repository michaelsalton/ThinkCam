import time

import cv2
import numpy as np
from PySide6.QtCore import QMutex, QMutexLocker, QThread, Signal

from arena_api.system import system

from thinkcam.constants import (
    BIAS_REFRACTORY,
    BIAS_THRESHOLD_NEG,
    BIAS_THRESHOLD_POS,
    BURST_FILTER_ENABLE,
    CAMERA_IP,
    ERC_RATE_LIMIT_MEV,
    IMAGE_TIMEOUT_MS,
    MODE_CDFRAME,
    MODE_XYTPFRAME,
    NUM_BUFFERS,
    XYTP_MODES,
)
from thinkcam.visualizer import EventVisualizer


def _camera_mode_for(display_mode: str) -> str:
    return MODE_XYTPFRAME if display_mode in XYTP_MODES else MODE_CDFRAME


class CameraWorker(QThread):
    frame_ready = Signal(np.ndarray, dict)
    status_message = Signal(str)
    error = Signal(str)
    connected = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lock = QMutex()
        self._running = False

        # Parameters (read under lock by the worker)
        self._mode = MODE_CDFRAME
        self._colormap = cv2.COLORMAP_JET
        self._sw_noise = True
        self._noise_duration_us = 2000
        self._pending_bias: dict | None = None
        self._mode_changed = False

    # ------------------------------------------------------------------
    # Thread-safe setters (called from GUI thread)
    # ------------------------------------------------------------------

    def set_mode(self, mode: str):
        with QMutexLocker(self._lock):
            if mode != self._mode:
                self._mode = mode
                self._mode_changed = True

    def set_colormap(self, colormap_id: int):
        with QMutexLocker(self._lock):
            self._colormap = colormap_id

    def set_sw_noise(self, enabled: bool):
        with QMutexLocker(self._lock):
            self._sw_noise = enabled

    def set_noise_duration(self, us: int):
        with QMutexLocker(self._lock):
            self._noise_duration_us = us

    def set_bias(self, node_name: str, value):
        with QMutexLocker(self._lock):
            if self._pending_bias is None:
                self._pending_bias = {}
            self._pending_bias[node_name] = value

    def stop(self):
        with QMutexLocker(self._lock):
            self._running = False

    # ------------------------------------------------------------------
    # Camera helpers
    # ------------------------------------------------------------------

    def _connect_device(self, max_tries=6, wait_secs=10):
        system.DEVICE_INFOS_TIMEOUT_MILLISEC = 1000
        system.add_unicast_discovery_device(CAMERA_IP)
        for attempt in range(1, max_tries + 1):
            self.status_message.emit(
                f"Searching for camera… (attempt {attempt}/{max_tries})"
            )
            devices = system.create_device()
            if devices:
                return devices
            time.sleep(wait_secs)
        return None

    def _configure_evs(self, device, cam_mode: str) -> dict:
        nm = device.nodemap
        tl = device.tl_stream_nodemap

        saved = {
            "AcquisitionMode": nm["AcquisitionMode"].value,
            "EventFormat": nm["EventFormat"].value,
            "ErcEnable": nm["ErcEnable"].value,
            "ErcRateLimit": nm["ErcRateLimit"].value,
        }

        nm["AcquisitionMode"].value = "Continuous"
        tl["StreamBufferHandlingMode"].value = "NewestOnly"
        nm["EventFormat"].value = "EVT3_0"
        nm["ErcEnable"].value = True
        nm["ErcRateLimit"].value = ERC_RATE_LIMIT_MEV
        tl["StreamEvsOutputFormat"].value = cam_mode

        if cam_mode == MODE_CDFRAME:
            fps = tl["StreamFrameGeneratorFPS"].value
            accum_us = int(1_000_000 / fps)
            tl["StreamFrameGeneratorAccumTime"].value = accum_us

        return saved

    def _configure_noise_filters(self, device) -> dict:
        nm = device.nodemap
        saved = {}
        settings = {
            "BiasEventThresholdPositive": BIAS_THRESHOLD_POS,
            "BiasEventThresholdNegative": BIAS_THRESHOLD_NEG,
            "BiasRefractoryPeriod": BIAS_REFRACTORY,
            "EventBurstFilterEnable": BURST_FILTER_ENABLE,
        }
        for node_name, new_val in settings.items():
            try:
                node = nm[node_name]
                saved[node_name] = node.value
                node.value = new_val
            except Exception:
                pass
        return saved

    def _restore_settings(self, device, *saved_dicts):
        nm = device.nodemap
        for saved in saved_dicts:
            for key, val in saved.items():
                try:
                    nm[key].value = val
                except Exception:
                    pass

    def _apply_pending_bias(self, device, pending: dict):
        nm = device.nodemap
        for node_name, value in pending.items():
            try:
                nm[node_name].value = value
            except Exception as e:
                self.status_message.emit(f"Bias error: {node_name}: {e}")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        self._running = True
        devices = self._connect_device()
        if not devices:
            self.error.emit("No camera found. Check connection and retry.")
            return

        device = system.select_device(devices)
        nm = device.nodemap
        tl = device.tl_stream_nodemap
        width = nm["Width"].value
        height = nm["Height"].value

        self.connected.emit(width, height)
        self.status_message.emit(f"Connected: {width}x{height}")

        visualizer = EventVisualizer(width, height)
        cam_mode = _camera_mode_for(self._mode)
        saved_evs = self._configure_evs(device, cam_mode)
        saved_noise = self._configure_noise_filters(device)
        device.start_stream(NUM_BUFFERS)

        prev_noise_dur = self._noise_duration_us

        try:
            while True:
                # Read parameters under lock
                with QMutexLocker(self._lock):
                    if not self._running:
                        break
                    mode = self._mode
                    colormap = self._colormap
                    sw_noise = self._sw_noise
                    noise_dur = self._noise_duration_us
                    mode_changed = self._mode_changed
                    self._mode_changed = False
                    pending_bias = self._pending_bias
                    self._pending_bias = None

                # Handle mode switch requiring camera reconfiguration
                if mode_changed:
                    new_cam = _camera_mode_for(mode)
                    if new_cam != cam_mode:
                        self.status_message.emit("Reconfiguring stream…")
                        device.stop_stream()
                        self._restore_settings(device, saved_evs, saved_noise)
                        cam_mode = new_cam
                        saved_evs = self._configure_evs(device, cam_mode)
                        saved_noise = self._configure_noise_filters(device)
                        device.start_stream(NUM_BUFFERS)

                # Apply pending bias changes
                if pending_bias:
                    self._apply_pending_bias(device, pending_bias)

                # Update noise filter duration if changed
                if noise_dur != prev_noise_dur:
                    visualizer.set_noise_duration(noise_dur)
                    prev_noise_dur = noise_dur

                # Acquire buffer
                try:
                    buffer = device.get_buffer(timeout=IMAGE_TIMEOUT_MS)
                except Exception:
                    continue

                if buffer.is_incomplete:
                    device.requeue_buffer(buffer)
                    continue

                # Grab frame_id before requeue
                frame_id = buffer.frame_id

                # Render
                t0 = time.perf_counter()
                bgr = visualizer.render(buffer, mode, colormap=colormap, sw_noise=sw_noise)
                render_ms = (time.perf_counter() - t0) * 1000.0
                device.requeue_buffer(buffer)

                # Read stats (buffer already requeued — use saved frame_id)
                stats = {
                    "mode": mode,
                    "frame_id": frame_id,
                    "event_rate": tl["StreamEvsEventRate"].value,
                    "gvsp_fps": tl["StreamEvsGvspFrameRate"].value,
                    "throughput": tl["StreamEvsLinkThroughput"].value,
                    "render_ms": render_ms,
                    "noise_on": sw_noise,
                }

                self.frame_ready.emit(bgr, stats)

        finally:
            try:
                device.stop_stream()
            except Exception:
                pass
            self._restore_settings(device, saved_evs, saved_noise)
            system.destroy_device()
