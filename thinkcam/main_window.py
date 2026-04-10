import os
from datetime import datetime

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QWidget,
)

from thinkcam.camera_worker import CameraWorker
from thinkcam.controls import ControlPanel
from thinkcam.recorder import VideoRecorder
from thinkcam.status_bar import StatsStatusBar


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ThinkCam  \u2014  LUCID TRT009S-E EVS")

        self._cam_width = 0
        self._cam_height = 0
        self._last_bgr: np.ndarray | None = None
        self._show_overlay = False
        self._save_dir = "evs_captures"
        self._save_idx = 0

        self._recorder = VideoRecorder(self._save_dir)
        self._worker = CameraWorker()

        self._build_ui()
        self._connect_signals()
        self._setup_shortcuts()

        # Start camera worker
        self._worker.start()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)

        # Viewport
        self._viewport = QLabel()
        self._viewport.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._viewport.setMinimumSize(640, 480)
        self._viewport.setStyleSheet("background-color: #1a1a1a;")
        self._viewport.setText("Connecting to camera…")
        self._viewport.setStyleSheet(
            "background-color: #1a1a1a; color: #666; font-size: 16px;"
        )
        layout.addWidget(self._viewport, stretch=1)

        # Sidebar
        self._controls = ControlPanel()
        layout.addWidget(self._controls)

        # Status bar
        self._status_bar = StatsStatusBar()
        self.setStatusBar(self._status_bar)

    def _connect_signals(self):
        # Worker -> UI
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.connected.connect(self._on_connected)
        self._worker.error.connect(self._on_error)
        self._worker.status_message.connect(self._on_status)

        # Controls -> Worker
        self._controls.mode_changed.connect(self._worker.set_mode)
        self._controls.colormap_changed.connect(self._worker.set_colormap)
        self._controls.sw_noise_changed.connect(self._worker.set_sw_noise)
        self._controls.noise_duration_changed.connect(self._worker.set_noise_duration)
        self._controls.bias_changed.connect(self._worker.set_bias)

        # Controls -> UI
        self._controls.overlay_toggled.connect(self._on_overlay_toggled)
        self._controls.save_requested.connect(self._save_frame)
        self._controls.record_toggled.connect(self._toggle_recording)

    def _setup_shortcuts(self):
        QShortcut(QKeySequence("S"), self, self._save_frame)
        QShortcut(QKeySequence("Q"), self, self.close)
        QShortcut(QKeySequence("Escape"), self, self.close)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_connected(self, width: int, height: int):
        self._cam_width = width
        self._cam_height = height
        self._viewport.setStyleSheet("background-color: #1a1a1a;")
        self._viewport.setText("")

    def _on_error(self, msg: str):
        QMessageBox.critical(self, "Camera Error", msg)

    def _on_status(self, msg: str):
        self._status_bar.showMessage(msg, 3000)

    def _on_overlay_toggled(self, checked: bool):
        self._show_overlay = checked

    def _on_frame(self, bgr: np.ndarray, stats: dict):
        self._last_bgr = bgr.copy()

        # Record if active
        if self._recorder.is_recording:
            self._recorder.write_frame(bgr)

        # Optional overlay
        display = bgr
        if self._show_overlay:
            display = bgr.copy()
            self._draw_overlay(display, stats)

        # BGR -> QPixmap
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        # Scale to viewport
        scaled = pixmap.scaled(
            self._viewport.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        self._viewport.setPixmap(scaled)

        # Update status bar
        self._status_bar.update_stats(stats)

    def _draw_overlay(self, img: np.ndarray, stats: dict):
        lines = [
            f"Mode: {stats.get('mode', '')}",
            f"Frame: {stats.get('frame_id', 0)}",
            f"Render: {stats.get('render_ms', 0):.1f} ms",
        ]
        y = 30
        for line in lines:
            cv2.putText(
                img, line, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA,
            )
            y += 24

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _save_frame(self):
        if self._last_bgr is None:
            return
        os.makedirs(self._save_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self._save_dir, f"evs_{ts}_{self._save_idx:04d}.png")
        cv2.imwrite(path, self._last_bgr)
        self._save_idx += 1
        self._status_bar.showMessage(f"Saved {path}", 3000)

    def _toggle_recording(self, start: bool):
        if start:
            if self._cam_width == 0:
                return
            self._recorder.start(self._cam_width, self._cam_height)
            self._status_bar.showMessage("Recording started…", 2000)
        else:
            path = self._recorder.stop()
            if path:
                self._status_bar.showMessage(f"Saved recording: {path}", 5000)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        self._worker.stop()
        self._worker.wait(5000)
        self._recorder.stop()
        event.accept()
