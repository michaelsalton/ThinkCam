from PySide6.QtWidgets import QLabel, QStatusBar


def _fmt_rate(r: float) -> str:
    if r < 1e3:
        return f"{r:.0f} ev/s"
    if r < 1e6:
        return f"{r / 1e3:.1f} Kev/s"
    if r < 1e9:
        return f"{r / 1e6:.1f} Mev/s"
    return f"{r / 1e9:.1f} Gev/s"


def _fmt_bw(r: float) -> str:
    if r < 1e3:
        return f"{r:.0f} Bps"
    if r < 1e6:
        return f"{r / 1e3:.1f} KBps"
    if r < 1e9:
        return f"{r / 1e6:.1f} MBps"
    return f"{r / 1e9:.1f} GBps"


class StatsStatusBar(QStatusBar):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._mode_label = QLabel("Mode: --")
        self._events_label = QLabel("Events: --")
        self._fps_label = QLabel("FPS: --")
        self._bw_label = QLabel("BW: --")
        self._render_label = QLabel("Render: --")
        self._frame_label = QLabel("Frame: --")

        for label in (
            self._mode_label,
            self._events_label,
            self._fps_label,
            self._bw_label,
            self._render_label,
            self._frame_label,
        ):
            label.setStyleSheet("padding: 0 8px;")
            self.addPermanentWidget(label)

    def update_stats(self, stats: dict):
        mode = stats.get("mode", "--")
        nf = " [NF]" if stats.get("noise_on") else ""
        self._mode_label.setText(f"Mode: {mode}{nf}")
        self._events_label.setText(f"Events: {_fmt_rate(stats.get('event_rate', 0))}")
        self._fps_label.setText(f"FPS: {stats.get('gvsp_fps', 0):.1f}")
        self._bw_label.setText(f"BW: {_fmt_bw(stats.get('throughput', 0))}")
        self._render_label.setText(f"Render: {stats.get('render_ms', 0):.1f} ms")
        self._frame_label.setText(f"Frame: {stats.get('frame_id', 0)}")
