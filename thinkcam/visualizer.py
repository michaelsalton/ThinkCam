import ctypes
from datetime import timedelta

import cv2
import dv_processing as dv
import numpy as np

from thinkcam.constants import (
    DENOISE_KERNEL,
    DILATE_KERNEL,
    DV_NOISE_DURATION_US,
    MODE_CDFRAME,
    MODE_DV_ACCUMULATOR,
    MODE_DV_EVENTS,
    MODE_DV_TIMESURFACE,
    MODE_XYTPFRAME,
)


class EventVisualizer:
    """Renders arena_api buffers into BGR images using multiple modes."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self._resolution = (width, height)

        # dv-processing objects
        self._dv_vis = dv.visualization.EventVisualizer(self._resolution)
        self._dv_acc = dv.Accumulator(self._resolution)
        self._dv_ts = dv.TimeSurface(self._resolution)
        self._dv_noise = dv.noise.BackgroundActivityNoiseFilter(
            self._resolution,
            backgroundActivityDuration=timedelta(microseconds=DV_NOISE_DURATION_US),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(
        self,
        buffer,
        mode: str,
        colormap: int = cv2.COLORMAP_JET,
        sw_noise: bool = True,
    ) -> np.ndarray:
        if mode == MODE_CDFRAME:
            bgr = self._cdframe_to_bgr(buffer)
            return cv2.morphologyEx(bgr, cv2.MORPH_OPEN, DENOISE_KERNEL)

        if mode == MODE_XYTPFRAME:
            bgr = self._xytp_to_heatmap(buffer)
            return cv2.dilate(bgr, DILATE_KERNEL)

        # dv-processing modes
        store = self._buffer_to_eventstore(buffer)
        if sw_noise and store.size() > 0:
            self._dv_noise.accept(store)
            store = self._dv_noise.generateEvents()

        if mode == MODE_DV_EVENTS:
            return self._dv_vis.generateImage(store)
        elif mode == MODE_DV_ACCUMULATOR:
            self._dv_acc.accept(store)
            frame = self._dv_acc.generateFrame()
            return cv2.cvtColor(frame.image, cv2.COLOR_GRAY2BGR)
        elif mode == MODE_DV_TIMESURFACE:
            self._dv_ts.accept(store)
            frame = self._dv_ts.generateFrame()
            return cv2.applyColorMap(frame.image, colormap)

        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def set_noise_duration(self, duration_us: int):
        self._dv_noise = dv.noise.BackgroundActivityNoiseFilter(
            self._resolution,
            backgroundActivityDuration=timedelta(microseconds=duration_us),
        )

    # ------------------------------------------------------------------
    # Buffer conversion helpers
    # ------------------------------------------------------------------

    def _cdframe_to_bgr(self, buffer) -> np.ndarray:
        bpp = buffer.bits_per_pixel
        channels = max(1, bpp // 8)
        raw = (ctypes.c_ubyte * buffer.size_filled).from_address(
            ctypes.addressof(buffer.pbytes)
        )
        arr = np.frombuffer(raw, dtype=np.uint8)

        if channels == 1:
            img = arr.reshape((buffer.height, buffer.width))
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif channels == 3:
            img = arr.reshape((buffer.height, buffer.width, 3))
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img = arr[: buffer.height * buffer.width].reshape(
                (buffer.height, buffer.width)
            )
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    def _xytp_to_heatmap(self, buffer) -> np.ndarray:
        STEP = 4
        width, height = buffer.width, buffer.height
        bytes_per_event = buffer.bits_per_pixel / 8
        num_events = int(buffer.size_filled / bytes_per_event)

        if num_events == 0:
            return np.zeros((height, width, 3), dtype=np.uint8)

        raw = ctypes.string_at(buffer.pdata, buffer.size_filled)
        events = np.frombuffer(raw, dtype=np.float32).reshape(num_events, STEP)

        xs = events[:, 0].astype(np.int32).clip(0, width - 1)
        ys = events[:, 1].astype(np.int32).clip(0, height - 1)
        ps = events[:, 3].astype(np.int32)

        bgr = np.zeros((height, width, 3), dtype=np.uint8)
        on_mask = ps == 1
        bgr[ys[on_mask], xs[on_mask]] = (255, 80, 0)
        off_mask = ~on_mask
        bgr[ys[off_mask], xs[off_mask]] = (0, 80, 255)
        return bgr

    def _buffer_to_eventstore(self, buffer) -> dv.EventStore:
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
        ts = events[:, 2]
        ps = events[:, 3].astype(np.int32)

        for i in range(num_events):
            store.push_back(int(ts[i]), int(xs[i]), int(ys[i]), bool(ps[i]))

        return store
