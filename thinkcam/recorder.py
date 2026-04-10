import os
from datetime import datetime

import cv2
import numpy as np


class VideoRecorder:
    """Records BGR frames to an mp4 file via OpenCV VideoWriter."""

    def __init__(self, output_dir: str = "evs_captures"):
        self._output_dir = output_dir
        self._writer: cv2.VideoWriter | None = None
        self._path: str | None = None

    @property
    def is_recording(self) -> bool:
        return self._writer is not None

    @property
    def output_dir(self) -> str:
        return self._output_dir

    def set_output_dir(self, path: str):
        self._output_dir = path

    def start(self, width: int, height: int, fps: float = 30.0):
        os.makedirs(self._output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._path = os.path.join(self._output_dir, f"evs_recording_{ts}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(self._path, fourcc, fps, (width, height))

    def write_frame(self, bgr: np.ndarray):
        if self._writer is not None:
            self._writer.write(bgr)

    def stop(self) -> str | None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            path = self._path
            self._path = None
            return path
        return None
