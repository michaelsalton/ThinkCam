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
from thinkcam.derivative_plot import DerivativePlotWindow
from thinkcam.raw_recorder import RawEventRecorder
from thinkcam.recorder import VideoRecorder
from thinkcam.status_bar import StatsStatusBar


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ThinkCam  \u2014  LUCID TRT009S-E EVS")

        self._cam_width = 0
        self._cam_height = 0
        self._last_bgr: np.ndarray | None = None
        self._save_dir = "evs_captures"
        self._save_idx = 0

        self._flash_filter_enabled = False
        self._flash_threshold = 0.70
        # Need at least this fraction of pixels active before the imbalance
        # ratio is meaningful — otherwise random noise on a few pixels can
        # look highly imbalanced.
        self._flash_min_activity_frac = 0.01

        self._recorder = VideoRecorder(self._save_dir)
        self._raw_recorder = RawEventRecorder()
        self._worker = CameraWorker()
        # The worker submits raw event batches straight to the recorder from the
        # acquisition thread (the recorder's queue is thread-safe).
        self._worker.raw_recorder = self._raw_recorder
        self._plot_window = DerivativePlotWindow()

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

        # Controls -> UI
        self._controls.save_requested.connect(self._save_frame)
        self._controls.record_toggled.connect(self._toggle_recording)
        self._controls.raw_record_toggled.connect(self._toggle_raw_recording)
        self._controls.plots_requested.connect(self._show_plots)
        self._controls.flash_filter_toggled.connect(self._on_flash_filter_toggled)
        self._controls.flash_threshold_changed.connect(self._on_flash_threshold_changed)

    def _setup_shortcuts(self):
        QShortcut(QKeySequence("S"), self, self._save_frame)
        QShortcut(QKeySequence("P"), self, self._show_plots)
        QShortcut(QKeySequence("R"), self, self._toggle_raw_recording_shortcut)
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

    def _on_frame(self, bgr: np.ndarray, stats: dict):
        if self._is_global_flash(stats):
            bgr = np.full_like(bgr, 128)
            stats = {**stats, "flash_filtered": True}

        self._last_bgr = bgr.copy()

        if self._recorder.is_recording:
            self._recorder.write_frame(bgr)

        # Raw event batches are fed to the recorder inside the camera worker;
        # here we just surface its stats.
        if self._raw_recorder.is_recording:
            self._status_bar.update_raw_stats(self._raw_recorder.stats())

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        scaled = pixmap.scaled(
            self._viewport.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        self._viewport.setPixmap(scaled)

        self._status_bar.update_stats(stats)

        if self._plot_window.isVisible():
            self._plot_window.push(
                stats.get("frame_id", 0),
                stats.get("pos_count", 0),
                stats.get("neg_count", 0),
            )

    def _is_global_flash(self, stats: dict) -> bool:
        if not self._flash_filter_enabled:
            return False
        pos = stats.get("pos_count", 0)
        neg = stats.get("neg_count", 0)
        total = pos + neg
        min_activity = self._cam_width * self._cam_height * self._flash_min_activity_frac
        if total < min_activity:
            return False
        imbalance = abs(pos - neg) / total
        return imbalance >= self._flash_threshold

    def _on_flash_filter_toggled(self, enabled: bool):
        self._flash_filter_enabled = enabled

    def _on_flash_threshold_changed(self, value: float):
        self._flash_threshold = value

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

    def _show_plots(self):
        self._plot_window.show()
        self._plot_window.raise_()
        self._plot_window.activateWindow()

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

    def _toggle_raw_recording(self, start: bool):
        if start:
            if self._cam_width == 0:
                # Camera not connected yet — revert the button.
                self._controls.set_raw_recording(False)
                return
            session = self._raw_recorder.start(
                self._cam_width, self._cam_height, self._controls.take_label()
            )
            self._status_bar.showMessage(f"Raw recording → {session}", 2000)
        else:
            session = self._raw_recorder.stop()
            self._status_bar.clear_raw_stats()
            if session:
                self._status_bar.showMessage(f"Saved raw take: {session}", 5000)

    def _toggle_raw_recording_shortcut(self):
        # Flip the control button; it emits raw_record_toggled -> _toggle_raw_recording.
        self._controls.set_raw_recording(not self._raw_recorder.is_recording)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        self._worker.stop()
        self._worker.wait(5000)
        self._recorder.stop()
        # Stop after the worker so no more batches arrive mid-flush; this also
        # writes the metadata sidecar.
        self._raw_recorder.stop()
        self._plot_window.close()
        event.accept()
