from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class ControlPanel(QWidget):
    save_requested = Signal()
    record_toggled = Signal(bool)
    raw_record_toggled = Signal(bool)
    plots_requested = Signal()
    flash_filter_toggled = Signal(bool)
    flash_threshold_changed = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(200)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self._save_btn = QPushButton("Save PNG")
        self._save_btn.clicked.connect(self.save_requested.emit)
        layout.addWidget(self._save_btn)

        self._record_btn = QPushButton("Record Video")
        self._record_btn.setCheckable(True)
        self._record_btn.toggled.connect(self._on_record_toggled)
        layout.addWidget(self._record_btn)

        self._take_label_edit = QLineEdit()
        self._take_label_edit.setPlaceholderText("Take label (optional)")
        layout.addWidget(self._take_label_edit)

        self._raw_record_btn = QPushButton("Record RAW")
        self._raw_record_btn.setCheckable(True)
        self._raw_record_btn.setToolTip(
            "Record the lossless raw event stream (x, y, t, p) to "
            "recordings/<timestamp>_<label>/. Independent of video recording."
        )
        self._raw_record_btn.toggled.connect(self._on_raw_record_toggled)
        layout.addWidget(self._raw_record_btn)

        self._plots_btn = QPushButton("Show Plots")
        self._plots_btn.clicked.connect(self.plots_requested.emit)
        layout.addWidget(self._plots_btn)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep)

        self._flash_check = QCheckBox("Suppress flashes")
        self._flash_check.setToolTip(
            "Replace frames dominated by one polarity (lights on/off) "
            "with neutral gray."
        )
        self._flash_check.toggled.connect(self.flash_filter_toggled.emit)
        layout.addWidget(self._flash_check)

        self._threshold_label = QLabel("Imbalance ≥ 70%")
        self._threshold_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self._threshold_label)

        self._threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self._threshold_slider.setRange(50, 95)
        self._threshold_slider.setValue(70)
        self._threshold_slider.valueChanged.connect(self._on_threshold_changed)
        layout.addWidget(self._threshold_slider)

        layout.addStretch()

    def _on_record_toggled(self, checked: bool):
        self._record_btn.setText("Stop Recording" if checked else "Record Video")
        style = "background-color: #cc3333; color: white;" if checked else ""
        self._record_btn.setStyleSheet(style)
        self.record_toggled.emit(checked)

    def _on_raw_record_toggled(self, checked: bool):
        self._raw_record_btn.setText("Stop RAW" if checked else "Record RAW")
        style = "background-color: #cc3333; color: white;" if checked else ""
        self._raw_record_btn.setStyleSheet(style)
        # Lock the label while a take is rolling so it can't change mid-recording.
        self._take_label_edit.setEnabled(not checked)
        self.raw_record_toggled.emit(checked)

    def take_label(self) -> str:
        return self._take_label_edit.text()

    def set_raw_recording(self, recording: bool):
        """Reflect external state (e.g. keyboard shortcut) on the button."""
        if self._raw_record_btn.isChecked() != recording:
            self._raw_record_btn.setChecked(recording)

    def _on_threshold_changed(self, value: int):
        self._threshold_label.setText(f"Imbalance ≥ {value}%")
        self.flash_threshold_changed.emit(value / 100.0)
