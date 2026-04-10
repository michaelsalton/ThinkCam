from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from thinkcam.constants import ALL_MODES, COLORMAP_OPTIONS, XYTP_MODES


class ControlPanel(QWidget):
    mode_changed = Signal(str)
    colormap_changed = Signal(int)
    sw_noise_changed = Signal(bool)
    noise_duration_changed = Signal(int)
    bias_changed = Signal(str, object)
    overlay_toggled = Signal(bool)
    save_requested = Signal()
    record_toggled = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(300)
        self._build_ui()

    def _build_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        layout.addWidget(self._build_mode_group())
        layout.addWidget(self._build_colormap_group())
        layout.addWidget(self._build_noise_group())
        layout.addWidget(self._build_bias_group())
        layout.addWidget(self._build_display_group())
        layout.addWidget(self._build_export_group())
        layout.addStretch()

        scroll.setWidget(container)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    # ------------------------------------------------------------------
    # Mode
    # ------------------------------------------------------------------

    def _build_mode_group(self) -> QGroupBox:
        group = QGroupBox("Visualization Mode")
        layout = QVBoxLayout(group)
        self._mode_buttons = QButtonGroup(self)

        for i, mode in enumerate(ALL_MODES):
            btn = QRadioButton(mode)
            if i == 0:
                btn.setChecked(True)
            self._mode_buttons.addButton(btn, i)
            layout.addWidget(btn)

        self._mode_buttons.idToggled.connect(self._on_mode_toggled)
        return group

    def _on_mode_toggled(self, btn_id: int, checked: bool):
        if checked:
            mode = ALL_MODES[btn_id]
            self.mode_changed.emit(mode)
            self._update_colormap_enabled(mode)

    def _update_colormap_enabled(self, mode: str):
        enabled = mode in XYTP_MODES
        self._colormap_combo.setEnabled(enabled)

    # ------------------------------------------------------------------
    # Colormap
    # ------------------------------------------------------------------

    def _build_colormap_group(self) -> QGroupBox:
        group = QGroupBox("Colormap")
        layout = QVBoxLayout(group)

        self._colormap_combo = QComboBox()
        self._colormap_names = list(COLORMAP_OPTIONS.keys())
        self._colormap_combo.addItems(self._colormap_names)
        self._colormap_combo.setEnabled(False)  # CDFrame default
        self._colormap_combo.currentIndexChanged.connect(self._on_colormap_changed)
        layout.addWidget(self._colormap_combo)

        return group

    def _on_colormap_changed(self, index: int):
        name = self._colormap_names[index]
        self.colormap_changed.emit(COLORMAP_OPTIONS[name])

    # ------------------------------------------------------------------
    # Noise
    # ------------------------------------------------------------------

    def _build_noise_group(self) -> QGroupBox:
        group = QGroupBox("Software Noise Filter")
        layout = QVBoxLayout(group)

        self._noise_cb = QCheckBox("Enable (dv-processing)")
        self._noise_cb.setChecked(True)
        self._noise_cb.toggled.connect(self._on_noise_toggled)
        layout.addWidget(self._noise_cb)

        dur_layout = QHBoxLayout()
        dur_layout.addWidget(QLabel("Duration:"))
        self._dur_label = QLabel("2000 µs")
        dur_layout.addWidget(self._dur_label)
        layout.addLayout(dur_layout)

        self._dur_slider = QSlider(Qt.Orientation.Horizontal)
        self._dur_slider.setRange(500, 10000)
        self._dur_slider.setValue(2000)
        self._dur_slider.setSingleStep(100)
        self._dur_slider.valueChanged.connect(self._on_dur_changed)
        layout.addWidget(self._dur_slider)

        return group

    def _on_noise_toggled(self, checked: bool):
        self.sw_noise_changed.emit(checked)
        self._dur_slider.setEnabled(checked)

    def _on_dur_changed(self, value: int):
        self._dur_label.setText(f"{value} µs")
        self.noise_duration_changed.emit(value)

    # ------------------------------------------------------------------
    # Hardware bias
    # ------------------------------------------------------------------

    def _build_bias_group(self) -> QGroupBox:
        group = QGroupBox("Hardware Bias")
        layout = QVBoxLayout(group)

        self._bias_sliders = {}
        bias_defs = [
            ("BiasEventThresholdPositive", "Threshold +", 0, 50, 10),
            ("BiasEventThresholdNegative", "Threshold −", 0, 50, 10),
            ("BiasRefractoryPeriod", "Refractory", 0, 50, 10),
        ]

        for node_name, label, lo, hi, default in bias_defs:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            val_label = QLabel(str(default))
            val_label.setFixedWidth(30)
            row.addWidget(val_label)
            layout.addLayout(row)

            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(lo, hi)
            slider.setValue(default)
            slider.valueChanged.connect(
                lambda v, n=node_name, lbl=val_label: self._on_bias_slider(n, v, lbl)
            )
            layout.addWidget(slider)
            self._bias_sliders[node_name] = slider

        self._burst_cb = QCheckBox("Burst Filter")
        self._burst_cb.setChecked(True)
        self._burst_cb.toggled.connect(
            lambda v: self.bias_changed.emit("EventBurstFilterEnable", v)
        )
        layout.addWidget(self._burst_cb)

        return group

    def _on_bias_slider(self, node_name: str, value: int, label: QLabel):
        label.setText(str(value))
        self.bias_changed.emit(node_name, value)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def _build_display_group(self) -> QGroupBox:
        group = QGroupBox("Display")
        layout = QVBoxLayout(group)

        self._overlay_cb = QCheckBox("Show overlay")
        self._overlay_cb.setChecked(False)
        self._overlay_cb.toggled.connect(self.overlay_toggled.emit)
        layout.addWidget(self._overlay_cb)

        return group

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _build_export_group(self) -> QGroupBox:
        group = QGroupBox("Export")
        layout = QVBoxLayout(group)

        self._save_btn = QPushButton("Save PNG")
        self._save_btn.clicked.connect(self.save_requested.emit)
        layout.addWidget(self._save_btn)

        self._record_btn = QPushButton("Record Video")
        self._record_btn.setCheckable(True)
        self._record_btn.toggled.connect(self._on_record_toggled)
        layout.addWidget(self._record_btn)

        return group

    def _on_record_toggled(self, checked: bool):
        self._record_btn.setText("Stop Recording" if checked else "Record Video")
        style = "background-color: #cc3333; color: white;" if checked else ""
        self._record_btn.setStyleSheet(style)
        self.record_toggled.emit(checked)
